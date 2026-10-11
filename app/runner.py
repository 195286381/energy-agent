"""把 Deep Agents 的流式输出整理成前端好画的事件：思考、工具调用、工具结果、委派、校验、终答。"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import build_agent, mcp_tools
from app.answer_guard import FEEDBACK_PREFIX
from app.config import Settings, get_settings

Emit = Callable[[dict], Awaitable[None]]
MAX_RESULT_CHARS = 6000


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def _pretty(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except (ValueError, TypeError):
        return text


def model_labels(s: Settings) -> dict[str, str]:
    if s.model_mode == "offline":
        return {"main": "离线规则模型", "alarm-triage": "离线规则模型", "fault-diagnostician": "离线规则模型"}
    return {"main": s.model_pro, "alarm-triage": s.model_flash, "fault-diagnostician": s.model_pro}


async def run_alarm(alarm_id: str, emit: Emit, s: Settings | None = None, inject_hallucination: bool = False) -> None:
    s = s or get_settings()
    started = time.time()
    await emit({"type": "run_start", "alarm_id": alarm_id, "mode": s.model_mode, "models": model_labels(s),
                "inject_hallucination": inject_hallucination})

    ns_agent: dict[tuple, str] = {(): "main"}
    pending_tasks: list[str] = []
    tool_names: dict[str, str] = {}
    stats = {"tool_calls": 0, "model_calls": 0}

    def agent_of(ns: tuple) -> str:
        if ns not in ns_agent:
            ns_agent[ns] = pending_tasks.pop(0) if pending_tasks else "subagent"
        return ns_agent[ns]

    try:
        async with mcp_tools(s) as tools:
            agent = build_agent(tools, s, inject_hallucination=inject_hallucination and s.model_mode == "offline")
            user = f"新告警 {alarm_id}，请分诊、诊断并给出处置建议和企业微信推送草稿。"
            config = {"configurable": {"thread_id": f"run-{uuid.uuid4().hex[:8]}"}, "recursion_limit": 120}
            async for ns, mode, data in agent.astream(
                {"messages": [{"role": "user", "content": user}]}, config,
                stream_mode=["updates", "custom"], subgraphs=True,
            ):
                who = agent_of(tuple(ns))
                if mode == "custom":
                    if isinstance(data, dict) and data.get("type") == "guard":
                        await emit({**data, "agent": who})
                    continue
                for node, upd in (data or {}).items():
                    if not isinstance(upd, dict) or "messages" not in upd:
                        continue
                    msgs = upd["messages"]
                    msgs = getattr(msgs, "value", msgs)
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    for m in msgs:
                        await _emit_message(m, node, who, emit, pending_tasks, tool_names, stats)
    except Exception as e:  # noqa: BLE001 —— 错误也要推给前端
        await emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
    await emit({"type": "done", "elapsed_s": round(time.time() - started, 1), **stats})


async def _emit_message(m: Any, node: str, who: str, emit: Emit, pending_tasks: list[str],
                        tool_names: dict[str, str], stats: dict) -> None:
    if isinstance(m, AIMessage):
        text = _text(m.content).strip()
        if node.startswith("AnswerGuard"):
            await emit({"type": "answer", "agent": who, "text": text, "revised": True})
            return
        stats["model_calls"] += 1
        if m.tool_calls:
            if text:
                await emit({"type": "thought", "agent": who, "text": text})
            for tc in m.tool_calls:
                stats["tool_calls"] += 1
                tool_names[tc["id"]] = tc["name"]
                if tc["name"] == "task":
                    pending_tasks.append(tc["args"].get("subagent_type", "subagent"))
                    await emit({"type": "delegate", "agent": who, "id": tc["id"],
                                "subagent": tc["args"].get("subagent_type"), "description": tc["args"].get("description", "")})
                elif tc["name"] == "write_todos":
                    await emit({"type": "todos", "agent": who, "id": tc["id"], "todos": tc["args"].get("todos", [])})
                else:
                    await emit({"type": "tool_call", "agent": who, "id": tc["id"], "name": tc["name"], "args": tc["args"]})
        elif text:
            await emit({"type": "answer", "agent": who, "text": text})
    elif isinstance(m, ToolMessage):
        name = tool_names.get(m.tool_call_id, m.name or "")
        text = _text(m.content)
        if name == "write_todos":
            return
        if name == "task":
            await emit({"type": "subagent_result", "agent": who, "id": m.tool_call_id, "text": text})
            return
        if name == "draft_wecom_message":
            try:
                await emit({"type": "push_draft", "agent": who, "content": json.loads(text)["markdown"]["content"]})
            except (ValueError, KeyError, TypeError):
                pass
        result = _pretty(text)
        await emit({"type": "tool_result", "agent": who, "id": m.tool_call_id, "name": name,
                    "result": result[:MAX_RESULT_CHARS] + ("\n…（已截断）" if len(result) > MAX_RESULT_CHARS else ""),
                    "is_error": m.status == "error" or text.startswith("Error")})
    elif isinstance(m, HumanMessage) and _text(m.content).startswith(FEEDBACK_PREFIX):
        await emit({"type": "guard_feedback", "agent": who, "text": _text(m.content)})

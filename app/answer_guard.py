"""AnswerGuard：终答事实校验中间件（参考 i-epower 的 AnswerGuardMiddleware 思路的简化版）。

规则：终答里出现的每个数值，都必须能在本智能体看到的工具观测（或任务输入）里找到，
或者能由两个观测值相减 / 求变化率得到。
- 全部通过 → 放行；
- 有无依据的数值 → 带着违规项让模型重写一次；
- 重写后仍不通过 → 删掉含无依据数值的句子，并标注。

主智能体和子智能体都挂这个中间件：子智能体的结论先在自己的观测上校验，
主智能体再以子智能体返回的文本为观测校验终答，形成一条证据链。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.config import get_stream_writer

FEEDBACK_PREFIX = "【事实校验未通过】"

# 不参与校验的片段：设备/告警编号、日期、时刻、单体位置、优先级、列表序号
_STRIP_PATTERNS = [
    r"[A-Z]{2,}(?:-[A-Z0-9]+)+",
    r"M\d{2}-\d{2}",
    r"\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2})?",
    r"\d{1,2}-\d{2}\s+\d{1,2}:\d{2}",
    r"\d{1,2}:\d{2}",
    r"\bP[1-3]\b",
    r"(?m)^\s*\d+[.、)]\s",
    r"[一二三]级",
    r"(?:最近|近|过去)\s*\d+\s*(?:天|小时|次)",
    r"\d+\s*小时内",
]
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")  # 只比较绝对值，「下降 45.8%」和「-45.8」算同一个数


def _strip(text: str) -> str:
    for p in _STRIP_PATTERNS:
        text = re.sub(p, " ", text)
    return text


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def _headline_numbers(obj: Any, out: list[float]) -> None:
    """JSON 工具结果里的标量数值（不含 samples 这类长序列），用来算差值和变化率。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k != "samples":
                _headline_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj[:20]:
            _headline_numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(abs(float(obj)))


def _evidence_values(texts: list[str]) -> tuple[set[float], set[float]]:
    vals: set[float] = set()
    derived: set[float] = set()
    for t in texts:
        for v in _numbers(t):
            vals.add(round(v, 2))
        # 只允许同一个工具结果里的「主要数值」之间相减或求变化率，避免派生值太多导致什么都能对上
        try:
            obj = json.loads(t)
        except (ValueError, TypeError):
            continue
        heads: list[float] = []
        _headline_numbers(obj, heads)
        heads = sorted(set(heads))[:40]
        for i, a in enumerate(heads):
            for b in heads[i + 1:]:
                derived.add(round(b - a, 1))
                if a:
                    derived.add(round((b - a) / a * 100, 0))
                derived.add(round((b - a) / b * 100, 0))
    return vals, derived


def _supported(v: float, evidence: tuple[set[float], set[float]]) -> bool:
    raw, derived = evidence
    if abs(v) <= 3 and float(v).is_integer():
        return True  # 计数、序号这类小整数不校验
    if any(abs(v - e) <= 0.051 for e in raw):  # 原值，只容忍四舍五入误差
        return True
    # 派生值（差值、变化率）只保留到 1 位小数，更精确的数必须是原值
    return round(v, 1) == v and v in derived


def check(answer: str, evidence_texts: list[str]) -> tuple[int, list[str]]:
    evidence = _evidence_values(evidence_texts)
    stripped = _strip(answer)
    nums = _numbers(stripped)
    bad = sorted({_fmt(v) for v in nums if not _supported(v, evidence)})
    return len(nums), bad


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def _redact(answer: str, bad: list[str]) -> str:
    parts = re.split(r"(?<=[。；！\n])", answer)
    kept = []
    for p in parts:
        nums = {_fmt(v) for v in _numbers(_strip(p))}
        if nums & set(bad):
            kept.append("（此句含未经工具数据证实的数值，已删除）" + ("\n" if p.endswith("\n") else ""))
        else:
            kept.append(p)
    return "".join(kept)


class AnswerGuardMiddleware(AgentMiddleware):
    def __init__(self, agent_label: str):
        super().__init__()
        self.agent_label = agent_label

    @property
    def name(self) -> str:  # 主、子智能体各一份实例
        return f"AnswerGuard[{self.agent_label}]"

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        messages = state["messages"]
        last = messages[-1]
        if not isinstance(last, AIMessage) or last.tool_calls or not isinstance(last.content, str) or not last.content.strip():
            return None

        evidence = [_content_text(m.content) for m in messages if isinstance(m, (ToolMessage, HumanMessage))]
        checked, bad = check(last.content, evidence)
        retried = any(isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.startswith(FEEDBACK_PREFIX) for m in messages)

        writer = _writer()
        if not bad:
            writer({"type": "guard", "agent": self.agent_label, "checked": checked, "unsupported": [], "action": "pass"})
            return None
        if not retried:
            writer({"type": "guard", "agent": self.agent_label, "checked": checked, "unsupported": bad, "action": "rewrite"})
            feedback = (
                f"{FEEDBACK_PREFIX}以下数值在工具结果里找不到依据：{', '.join(bad)}。"
                "请只用工具返回的数值重写答复；没有数据的地方写「待确认」。"
            )
            return {"messages": [HumanMessage(content=feedback)], "jump_to": "model"}
        writer({"type": "guard", "agent": self.agent_label, "checked": checked, "unsupported": bad, "action": "redact"})
        return {"messages": [AIMessage(content=_redact(last.content, bad), id=last.id)]}


def _content_text(content: Any) -> str:
    """MCP 工具结果是内容块列表，普通工具是字符串，统一成文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def _writer():
    try:
        return get_stream_writer()
    except Exception:  # 不在 graph 流式上下文里（如单测）
        return lambda _x: None

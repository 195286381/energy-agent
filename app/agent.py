"""储能运维智能体：Deep Agents 主智能体 + 两个专职子智能体。

    值班长（主，pro）
      ├─ alarm-triage        告警分诊（flash，快）：等级、优先级、抖动、关联
      └─ fault-diagnostician 故障诊断（pro）：按 Skill 多步取证，给根因和证据
    工具：业务数据全部来自只读 MCP 服务；主智能体额外有「企业微信推送草稿」工具。
    中间件：Skills（原生）、TODO、文件系统（/skills 只读）、调用次数上限、AnswerGuard 事实校验。
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver

from app.answer_guard import AnswerGuardMiddleware
from app.config import ROOT, Settings, get_settings
from app.notify import draft_wecom_message
from app.offline_model import OfflineOpsModel

MAIN_PROMPT = """你是储能电站的运维值班智能体（值班长），负责处理新进来的告警。

工作流程：
1. 用 write_todos 列出处置计划。
2. 把告警交给 alarm-triage 子智能体分诊（等级、优先级、是否抖动、是否需要深度诊断）。
3. 需要深度诊断时，交给 fault-diagnostician 子智能体。委派时在描述里写清告警 ID、设备、告警类型，
   并附上分诊返回的告警详情 JSON（放在 ```json 代码块里）。
4. 读 alarm-report Skill，按里面的格式写最终答复，并调用 draft_wecom_message 生成推送草稿。

红线：
- 数值只能来自工具或子智能体返回的结果，不许估算或编造；没有数据就写「待确认」。
- 你只能读数据、给建议，不能也不要声称下发了任何控制指令。
- 用中文，简洁。"""

TRIAGE_PROMPT = """你是告警分诊员。只做分诊，不做根因分析。

步骤：get_alarm_detail → get_alarm_history(近 30 天) 和 get_device_info → list_active_alarms。
判断：
- 优先级：一级 → P1，二级 → P2，三级 → P3。
- 同一告警短时间内发生 ≥3 次且每次自动恢复 → 抖动告警，合并处理，降为 P3，不需要深度诊断。
- 三级告警但同设备近 30 天有过同类预警 → 可能在发展，需要深度诊断。
- 同一储能单元有其他未复归告警 → 列为关联告警。

输出：先用两三句话说结论，最后附一个 ```json 代码块，字段：alarm_id, device_id, alarm_type, title,
level, level_name, priority, flapping, need_diagnosis, related_alarms, summary, evidence(列表), actions(列表), detail(告警详情原样)。
数值只能来自工具结果。"""

DIAG_PROMPT = """你是储能故障诊断工程师。先读与告警类型对应的诊断 Skill（SKILL.md），按里面的步骤调用工具取证。

注意：BMS 只上送簇级单体极值（最高/最低单体电压、温度及位置），没有全量单体数据，结论要按这个粒度说。

输出：
- 一句话诊断结论 + 置信度（高/中/低）；
- 证据列表，每条带数值并注明来自哪个工具；
- 不超过 3 条处置建议（立即做什么、派单查什么、什么情况升级）。
最后附一个 ```json 代码块，字段：conclusion, conclusion_short, confidence, evidence(列表), actions(列表)。
数值只能来自工具结果，不许估算；只给建议，不下发控制指令。"""

TRIAGE_TOOLS = {"get_alarm_detail", "get_alarm_history", "get_device_info", "list_active_alarms"}


def build_models(s: Settings, inject_hallucination: bool = False) -> dict[str, BaseChatModel]:
    if s.model_mode == "offline":
        return {
            "main": OfflineOpsModel(role="main", step_delay=s.offline_step_delay),
            "triage": OfflineOpsModel(role="triage", step_delay=s.offline_step_delay),
            "diagnosis": OfflineOpsModel(role="diagnosis", step_delay=s.offline_step_delay,
                                         inject_hallucination=inject_hallucination),
        }
    from langchain_openai import ChatOpenAI

    def ds(model: str) -> ChatOpenAI:
        kwargs = {}
        if model.startswith("deepseek-v4"):
            # 和 i-epower 一致：V4 默认开思考，工具调用轮关掉以降低延迟
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return ChatOpenAI(model=model, base_url=s.deepseek_base_url, api_key=s.deepseek_api_key,
                          temperature=0.2, use_responses_api=False, timeout=120, max_retries=2, **kwargs)

    return {"main": ds(s.model_pro), "triage": ds(s.model_flash), "diagnosis": ds(s.model_pro)}


@asynccontextmanager
async def mcp_tools(s: Settings) -> AsyncIterator[list]:
    """每次运行开一个 MCP 会话（stdio 时即一个只读 MCP 子进程），查询预算按会话计。"""
    if s.mcp_transport == "http":
        conn = {"transport": "streamable_http", "url": s.mcp_url}
    else:
        conn = {"transport": "stdio", "command": sys.executable, "args": ["-m", "app.mcp_server"], "cwd": str(ROOT)}
    client = MultiServerMCPClient({"bess": conn})
    async with client.session("bess") as session:
        yield await load_mcp_tools(session)


def build_agent(tools: list, s: Settings | None = None, checkpointer=None, inject_hallucination: bool = False):
    s = s or get_settings()
    models = build_models(s, inject_hallucination)
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/skills/": FilesystemBackend(root_dir=s.skills_dir, virtual_mode=True)},
    )
    triage_tools = [t for t in tools if t.name in TRIAGE_TOOLS]
    subagents = [
        {
            "name": "alarm-triage",
            "description": "告警分诊：确认告警等级和处理优先级，识别抖动/重复告警和关联告警，判断是否需要深度诊断。输入告警 ID。",
            "system_prompt": TRIAGE_PROMPT,
            "tools": triage_tools,
            "model": models["triage"],
            "middleware": [AnswerGuardMiddleware("alarm-triage")],
        },
        {
            "name": "fault-diagnostician",
            "description": "故障诊断：按诊断 Skill 多步调用只读数据工具，找出根因，给出带数值的证据和处置建议。描述里要带告警详情 JSON。",
            "system_prompt": DIAG_PROMPT,
            "tools": tools,
            "model": models["diagnosis"],
            "skills": ["/skills/diagnosis/"],
            "middleware": [AnswerGuardMiddleware("fault-diagnostician")],
        },
    ]
    return create_deep_agent(
        model=models["main"],
        tools=[draft_wecom_message],
        system_prompt=MAIN_PROMPT,
        subagents=subagents,
        skills=["/skills/ops/"],
        backend=backend,
        permissions=[FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")],
        middleware=[
            TodoListMiddleware(),
            AnswerGuardMiddleware("main"),
            ModelCallLimitMiddleware(run_limit=20),
            ToolCallLimitMiddleware(run_limit=40),
        ],
        checkpointer=checkpointer or InMemorySaver(),
        name="bess-ops-agent",
    )

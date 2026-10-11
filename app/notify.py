"""企业微信推送：智能体只生成草稿，值班员在界面上确认后才真正发送。"""

from __future__ import annotations

import json
import urllib.request

from langchain_core.tools import tool

LEVEL_COLOR = {1: "warning", 2: "warning", 3: "comment"}
LEVEL_TEXT = {1: "一级告警", 2: "二级告警", 3: "三级告警"}


def build_markdown(level: int, title: str, conclusion: str, actions: list[str], priority: str,
                   station: str = "示范储能电站", mention_leader: bool = True) -> str:
    lines = [
        f'### <font color="{LEVEL_COLOR.get(level, "comment")}">【{LEVEL_TEXT.get(level, "告警")}】</font>{title}',
        f"> 电站：{station}",
        f"> 优先级：<font color=\"warning\">{priority}</font>",
        f"> 结论：{conclusion}",
        "",
        "**处置建议**",
    ]
    lines += [f"{i}. {a}" for i, a in enumerate(actions[:3], 1)]
    lines.append("")
    lines.append("<font color=\"comment\">由运维智能体生成，建议仅供参考，控制操作需人工执行</font>")
    if mention_leader:
        lines.append("<@值班长>")
    return "\n".join(lines)


@tool
def draft_wecom_message(level: int, title: str, conclusion: str, actions: list[str], priority: str,
                        mention_leader: bool = True) -> dict:
    """生成企业微信告警推送草稿（不会直接发送，需值班员在界面确认）。

    level: 告警等级 1/2/3；title: 设备 + 告警名；conclusion: 一句话结论；
    actions: 处置建议，最多 3 条；priority: P1/P2/P3；mention_leader: 是否 @值班长。
    """
    content = build_markdown(level, title, conclusion, actions, priority, mention_leader=mention_leader)
    return {"status": "draft", "msgtype": "markdown", "markdown": {"content": content}}


def send_wecom(webhook_url: str, content: str) -> dict:
    body = json.dumps({"msgtype": "markdown", "markdown": {"content": content}}).encode()
    req = urllib.request.Request(webhook_url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

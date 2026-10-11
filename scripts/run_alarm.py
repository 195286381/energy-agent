"""命令行跑告警诊断。

    python scripts/run_alarm.py ALM-20261011-0012       # 跑一条，打印过程
    python scripts/run_alarm.py --all --out traces.json   # 跑全部告警并保存事件（用于离线回放页面）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import station_sim as sim  # noqa: E402
from app.runner import run_alarm  # noqa: E402

COLORS = {"main": "\033[36m", "alarm-triage": "\033[33m", "fault-diagnostician": "\033[35m"}


def show(e: dict) -> None:
    c = COLORS.get(e.get("agent", ""), "")
    tag = f"{c}[{e.get('agent', '-')}]\033[0m"
    t = e["type"]
    if t == "thought":
        print(tag, "💭", e["text"])
    elif t == "tool_call":
        print(tag, "🔧", e["name"], json.dumps(e["args"], ensure_ascii=False))
    elif t == "tool_result":
        print(tag, "  ↳", e["result"].replace("\n", " ")[:160])
    elif t == "delegate":
        print(tag, "➡️ 委派给", e["subagent"])
    elif t == "guard":
        print(tag, "🛡 事实校验", e["action"], f"检查 {e['checked']} 个数值", e["unsupported"] or "")
    elif t == "answer":
        print(tag, "✅ 答复：\n" + e["text"] + "\n")
    elif t in ("run_start", "done", "error"):
        print("==", t, {k: v for k, v in e.items() if k != "type"})


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("alarm_id", nargs="?", default="ALM-20261011-0012")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--inject", action="store_true", help="离线模式下演示事实校验：让诊断智能体编一个数")
    a = ap.parse_args()
    ids = list(sim.ALARMS) if a.all else [a.alarm_id]
    traces = {}
    for aid in ids:
        events: list[dict] = []

        async def emit(e: dict) -> None:
            events.append(e)
            if not a.quiet:
                show(e)

        await run_alarm(aid, emit, inject_hallucination=a.inject)
        traces[aid] = events
    if a.out:
        Path(a.out).write_text(json.dumps(traces, ensure_ascii=False), encoding="utf-8")
        print("已保存", a.out)


if __name__ == "__main__":
    asyncio.run(main())

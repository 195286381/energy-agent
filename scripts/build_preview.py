"""生成可离线打开的回放页（preview/index.html）：先在本地真实跑一遍全部告警，把事件流嵌进页面。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODEL_MODE", "offline")

from app import station_sim as sim  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.runner import model_labels, run_alarm  # noqa: E402


async def collect() -> dict:
    traces = {}
    for aid in sim.ALARMS:
        for inject in (False, True):
            if inject and aid == "ALM-20261011-0021":
                continue
            events: list[dict] = []

            async def emit(e: dict) -> None:
                events.append(e)

            await run_alarm(aid, emit, inject_hallucination=inject)
            traces[aid + (":inject" if inject else "")] = events
    return traces


def main() -> None:
    s = get_settings()
    traces = asyncio.run(collect())
    alarms = sorted(sim.ALARMS.values(), key=lambda a: (a["level"], a["first_time"]))
    meta = {
        "mode": s.model_mode, "models": model_labels(s), "wecom_configured": False,
        "station": {k: v for k, v in sim.STATION.items() if k != "units"},
        "as_of": sim.NOW.strftime("%Y-%m-%d %H:%M"),
        "alarms": [{**a, "level_name": sim.LEVEL_NAME[a["level"]]} for a in alarms],
    }
    html = (ROOT / "app/static/index.html").read_text(encoding="utf-8")
    if html.startswith("<!doctype html>"):
        html = html.split("\n", 1)[1]
    inject = ("<script>window.__DEMO_META__ = " + json.dumps(meta, ensure_ascii=False)
              + ";\nwindow.__DEMO_TRACES__ = " + json.dumps(traces, ensure_ascii=False).replace("</", "<\\/") + ";</script>\n")
    html = html.replace("<script>\n(() => {", inject + "<script>\n(() => {", 1)
    out = ROOT / "preview/index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print("已生成", out, f"{out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()

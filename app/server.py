"""FastAPI + SSE 服务。

    uvicorn app.server:app --port 8000

接口：
    GET  /api/meta                       模型模式、电站、告警列表
    POST /api/runs {alarm_id}            发起一次告警处置，返回 run_id（后台运行）
    GET  /api/runs/{run_id}/events       SSE 事件流，可用 Last-Event-ID / ?after= 断线续看
    POST /api/runs/{run_id}/push         值班员确认后发送企业微信（需配置 WECOM_WEBHOOK_URL）
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app import station_sim as sim
from app.config import get_settings
from app.notify import send_wecom
from app.runner import model_labels, run_alarm

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="储能运维智能体 Demo")


class Run:
    def __init__(self, alarm_id: str):
        self.alarm_id = alarm_id
        self.events: list[dict] = []
        self.cond = asyncio.Condition()
        self.done = False
        self.push_content: str | None = None

    async def emit(self, e: dict) -> None:
        async with self.cond:
            e = {"seq": len(self.events), **e}
            self.events.append(e)
            if e["type"] == "push_draft":
                self.push_content = e["content"]
            if e["type"] == "done":
                self.done = True
            self.cond.notify_all()


RUNS: dict[str, Run] = {}


class RunRequest(BaseModel):
    alarm_id: str
    inject_hallucination: bool = False


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
async def meta() -> dict:
    s = get_settings()
    alarms = sorted(sim.ALARMS.values(), key=lambda a: (a["level"], a["first_time"]))
    return {
        "mode": s.model_mode,
        "models": model_labels(s),
        "wecom_configured": bool(s.wecom_webhook_url),
        "station": {k: v for k, v in sim.STATION.items() if k != "units"},
        "as_of": sim.NOW.strftime("%Y-%m-%d %H:%M"),
        "alarms": [{**a, "level_name": sim.LEVEL_NAME[a["level"]]} for a in alarms],
    }


@app.post("/api/runs")
async def create_run(req: RunRequest) -> dict:
    if req.alarm_id not in sim.ALARMS:
        raise HTTPException(404, "告警不存在")
    run_id = uuid.uuid4().hex[:12]
    run = RUNS[run_id] = Run(req.alarm_id)
    asyncio.create_task(run_alarm(req.alarm_id, run.emit, inject_hallucination=req.inject_hallucination))
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, request: Request, after: int = -1) -> StreamingResponse:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "run 不存在")
    last = request.headers.get("last-event-id")
    cursor = int(last) + 1 if last and last.isdigit() else after + 1

    async def stream():
        nonlocal cursor
        while True:
            async with run.cond:
                while cursor >= len(run.events) and not run.done:
                    try:
                        await asyncio.wait_for(run.cond.wait(), timeout=15)
                    except asyncio.TimeoutError:
                        break
                batch = run.events[cursor:]
                finished = run.done
            if not batch and not finished:
                yield ": keep-alive\n\n"
                continue
            for e in batch:
                yield f"id: {e['seq']}\nevent: {e['type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"
            cursor += len(batch)
            if finished and cursor >= len(run.events):
                return

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/runs/{run_id}/push")
async def push(run_id: str) -> dict:
    run = RUNS.get(run_id)
    if run is None or not run.push_content:
        raise HTTPException(404, "没有可发送的推送草稿")
    s = get_settings()
    if not s.wecom_webhook_url:
        raise HTTPException(400, "未配置 WECOM_WEBHOOK_URL，草稿未发送")
    resp = await asyncio.to_thread(send_wecom, s.wecom_webhook_url, run.push_content)
    return {"sent": resp.get("errcode") == 0, "wecom": resp}

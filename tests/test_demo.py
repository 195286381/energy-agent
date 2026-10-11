"""离线模式回归测试：不需要 DeepSeek Key。  pytest -q"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MODEL_MODE"] = "offline"

from app import station_sim as sim  # noqa: E402
from app.answer_guard import check  # noqa: E402
from app.runner import run_alarm  # noqa: E402


def _run(alarm_id: str, inject: bool = False) -> list[dict]:
    events: list[dict] = []

    async def emit(e):
        events.append(e)

    asyncio.run(run_alarm(alarm_id, emit, inject_hallucination=inject))
    return events


def test_guard_flags_unsupported_numbers():
    evidence = ['{"first": 120.0, "last": 63.8, "change_pct": -46.8}']
    assert check("流量从 120.0 降到 63.8 L/min（下降 46.8%）", evidence)[1] == []
    assert check("差了 56.2 L/min", evidence)[1] == []            # 同一结果里两个数相减
    assert check("SOH 只剩 73.46%", evidence)[1] == ["73.46"]
    assert check("BESS-02-C05 在 05:30，2026-10-11", evidence)[1] == []  # 编号、时刻不算数值


def test_alarms_consistent_with_telemetry():
    a = sim.ALARMS["ALM-20261011-0012"]
    peak = max(v for _, v in sim.series(a["device_id"], "cell_temp_max", 8))
    assert a["value"] == peak and peak >= a["threshold"]


@pytest.mark.parametrize("alarm_id", list(sim.ALARMS))
def test_offline_run_end_to_end(alarm_id):
    ev = _run(alarm_id)
    types = [e["type"] for e in ev]
    assert "error" not in types, [e for e in ev if e["type"] == "error"]
    assert types[-1] == "done"
    assert any(e["type"] == "push_draft" for e in ev)
    final = [e for e in ev if e["type"] == "answer" and e["agent"] == "main"]
    assert final and "## 处置建议" in final[-1]["text"]
    assert all(e["action"] == "pass" for e in ev if e["type"] == "guard")
    delegated = {e["subagent"] for e in ev if e["type"] == "delegate"}
    assert "alarm-triage" in delegated
    # 抖动告警不做深度诊断，其余都要诊断
    assert ("fault-diagnostician" in delegated) == (alarm_id != "ALM-20261011-0021")


def test_injected_hallucination_is_caught():
    ev = _run("ALM-20261011-0012", inject=True)
    guards = [e for e in ev if e["type"] == "guard" and e["agent"] == "fault-diagnostician"]
    assert guards[0]["action"] == "rewrite" and guards[0]["unsupported"] == ["73.46"]
    final = [e for e in ev if e["type"] == "answer" and e["agent"] == "main"][-1]["text"]
    assert "73.46" not in final

"""只读 MCP 服务：智能体读取电站业务数据的唯一通道。

和 i-epower 的做法一致：智能体不直连业务库，所有读取走独立的只读 MCP 服务，
每个工具声明所需 scope，服务端按调用方 scope 校验并限制查询预算。
Demo 里调用方 scope 从环境变量 MCP_CALLER_SCOPES 读（生产应来自 Token Exchange
换出来的短期 MCP Token 的 claims）。

运行方式：
    python -m app.mcp_server                # stdio（智能体默认以子进程方式拉起）
    python -m app.mcp_server --http 8765    # Streamable HTTP
"""

from __future__ import annotations

import functools
import os
import sys
from datetime import timedelta

from mcp.server.fastmcp import FastMCP

from app import station_sim as sim

mcp = FastMCP("bess-readonly")

_DEFAULT_SCOPES = "bess:alarms:read bess:telemetry:read bess:assets:read bess:maintenance:read bess:knowledge:read bess:weather:read"
_CALLER_SCOPES = set(os.environ.get("MCP_CALLER_SCOPES", _DEFAULT_SCOPES).split())
_QUERY_BUDGET = int(os.environ.get("MCP_QUERY_BUDGET", "60"))
_calls = 0


class ToolDenied(Exception):
    pass


def readonly_tool(scope: str):
    """注册只读工具：校验 scope、扣查询预算，出错时返回结构化错误而不是抛给模型。"""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            global _calls
            if scope not in _CALLER_SCOPES:
                return {"error": "forbidden", "message": f"调用方缺少权限 {scope}"}
            _calls += 1
            if _calls > _QUERY_BUDGET:
                return {"error": "budget_exceeded", "message": f"本会话查询预算 {_QUERY_BUDGET} 次已用完"}
            try:
                return fn(*args, **kwargs)
            except KeyError as e:
                return {"error": "not_found", "message": f"没有找到 {e}"}

        wrapper.__doc__ = (fn.__doc__ or "") + f"\n\n所需权限：{scope}（只读）"
        return mcp.tool()(wrapper)

    return deco


@readonly_tool("bess:alarms:read")
def list_active_alarms(station_id: str = "ES-DEMO") -> dict:
    """列出电站当前未复归的告警（按等级排序）。"""
    if station_id != sim.STATION["station_id"]:
        raise KeyError(station_id)
    alarms = sorted(sim.ALARMS.values(), key=lambda a: (a["level"], a["first_time"]))
    return {
        "station": sim.STATION["name"],
        "as_of": sim.NOW.strftime("%Y-%m-%d %H:%M"),
        "count": len(alarms),
        "alarms": [
            {k: a[k] for k in ("alarm_id", "device_id", "title", "level", "first_time", "occurrences")}
            for a in alarms
        ],
    }


@readonly_tool("bess:alarms:read")
def get_alarm_detail(alarm_id: str) -> dict:
    """查询单条告警详情：设备、等级、触发值、阈值、首次/最近发生时间、发生次数。"""
    a = dict(sim.ALARMS[alarm_id])
    a["level_name"] = sim.LEVEL_NAME[a["level"]]
    return a


@readonly_tool("bess:alarms:read")
def get_alarm_history(device_id: str, days: int = 30) -> dict:
    """查询设备最近 N 天的历史告警（用于判断抖动、重复发生）。"""
    days = max(1, min(days, 90))
    cutoff = (sim.NOW - timedelta(days=days)).strftime("%Y-%m-%d")
    items = [h for h in sim.ALARM_HISTORY.get(device_id, []) if h["time"][:10] >= cutoff]
    return {"device_id": device_id, "days": days, "count": len(items), "items": items}


@readonly_tool("bess:assets:read")
def get_device_info(device_id: str) -> dict:
    """查询设备台账和拓扑：类型、额定参数、上下级设备、关联的 PCS/液冷/BMS、告警阈值。"""
    info = sim.device_info(device_id)
    if info is None:
        raise KeyError(device_id)
    return info


@readonly_tool("bess:telemetry:read")
def query_telemetry(device_id: str, metric: str, hours: int = 8) -> dict:
    """查询设备某个测点最近 N 小时的时序数据，返回统计值、变化率和抽样点（5 分钟原始粒度）。

    常用测点：电池簇 cell_temp_max / cell_temp_min / volt_diff_mv / soc；
    液冷 coolant_flow / pump_current / outlet_temp / inlet_temp；
    PCS insulation_resistance / cabinet_humidity / active_power；BMS packet_loss_pct / comm_status。
    """
    if sim.device_info(device_id) is None:
        raise KeyError(device_id)
    if metric not in sim.METRIC_UNITS:
        return {"error": "bad_metric", "message": f"未知测点 {metric}", "available": sorted(sim.METRIC_UNITS)}
    hours = max(1, min(hours, 48))
    pts = sim.series(device_id, metric, hours)
    vals = [v for _, v in pts]
    first, last = vals[0], vals[-1]
    mx = max(pts, key=lambda x: x[1])
    mn = min(pts, key=lambda x: x[1])
    step = max(1, len(pts) // 16)
    return {
        "device_id": device_id,
        "metric": metric,
        "unit": sim.METRIC_UNITS[metric],
        "window": f"最近 {hours} 小时",
        "points": len(pts),
        "first": first,
        "last": last,
        "max": mx[1],
        "max_time": mx[0].strftime("%H:%M"),
        "min": mn[1],
        "min_time": mn[0].strftime("%H:%M"),
        "avg": round(sum(vals) / len(vals), 2),
        "change_pct": round((last - first) / first * 100, 1) if first else None,
        "samples": [[t.strftime("%H:%M"), v] for t, v in pts[::step]],
    }


@readonly_tool("bess:telemetry:read")
def get_cluster_extremes(unit_or_cluster_id: str, at: str = "") -> dict:
    """查询簇级单体极值（BMS 只上送簇级最高/最低单体电压、温度及位置，没有全量单体数据）。

    传储能单元 ID（如 BESS-02）返回该单元所有簇的对比；传簇 ID（如 BESS-03-C02）
    额外返回最近 7 次充电末端的压差和最低单体位置。at 为 HH:MM，默认当前时刻。
    """
    when = sim.NOW
    if at:
        hh, mm = at.split(":")
        when = sim.NOW.replace(hour=int(hh), minute=int(mm) // 5 * 5)
    if unit_or_cluster_id in sim.STATION["units"]:
        clusters = [f"{unit_or_cluster_id}-C{c:02d}" for c in range(1, sim.CLUSTERS_PER_UNIT + 1)]
        rows = [sim.cluster_snapshot(c, when) for c in clusters]
        temps = [r["cell_temp_max"] for r in rows]
        return {
            "unit_id": unit_or_cluster_id,
            "time": when.strftime("%H:%M"),
            "clusters": rows,
            "temp_max_of_unit": max(temps),
            "temp_max_median": sorted(temps)[len(temps) // 2],
            "reference_temp_c": 40,
            "clusters_over_reference": sum(t >= 40 for t in temps),
        }
    if sim.device_info(unit_or_cluster_id) is None or "-C" not in unit_or_cluster_id:
        raise KeyError(unit_or_cluster_id)
    snap = sim.cluster_snapshot(unit_or_cluster_id, when)
    hist = sim.charge_end_history(unit_or_cluster_id)
    snap["charge_end_history"] = hist
    snap["volt_diff_trend_mv_per_cycle"] = round((hist[-1]["volt_diff_mv"] - hist[0]["volt_diff_mv"]) / (len(hist) - 1), 1)
    return snap


@readonly_tool("bess:maintenance:read")
def get_maintenance_records(device_id: str) -> dict:
    """查询设备检修/保养记录，含下次保养到期日（判断是否超期）。"""
    recs = sim.MAINTENANCE.get(device_id, [])
    overdue = [r for r in recs if r.get("next_due") and r["next_due"] < sim.NOW.strftime("%Y-%m-%d")]
    latest = recs[0] if recs else None
    return {
        "device_id": device_id,
        "records": recs,
        "overdue": bool(latest and latest in overdue),
        "today": sim.NOW.strftime("%Y-%m-%d"),
    }


@readonly_tool("bess:knowledge:read")
def search_fault_cases(query: str, top_k: int = 3) -> dict:
    """检索历史故障案例库（生产环境走 PGVector 向量检索，Demo 用关键词打分）。"""
    terms = [t for t in query.replace("，", " ").replace(",", " ").split() if t]
    scored = []
    for c in sim.FAULT_CASES:
        text = " ".join([c["title"], c["symptom"], c["root_cause"], " ".join(c["keywords"])])
        score = sum(1 for t in terms if t in text)
        if score:
            scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], x[1]["grade"]))
    return {
        "query": query,
        "results": [{**{k: v for k, v in c.items() if k != "keywords"}, "match": s} for s, c in scored[: max(1, min(top_k, 5))]],
    }


@readonly_tool("bess:weather:read")
def get_weather(station_id: str = "ES-DEMO", hours: int = 12) -> dict:
    """查询电站所在地最近 N 小时气象（气温、湿度、降雨），每 30 分钟一个点。"""
    if station_id != sim.STATION["station_id"]:
        raise KeyError(station_id)
    return {"station_id": station_id, "points": sim.weather(max(1, min(hours, 24)))}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--http":
        mcp.settings.port = int(sys.argv[2])
        mcp.run(transport="streamable-http")
    else:
        mcp.run()

"""模拟储能电站：拓扑、遥测、告警、检修记录、故障案例、气象。

数据是确定性生成的（固定随机种子 + 固定"当前时刻"），同一个告警每次跑出来的
数据都一样，方便回放和评测。生产环境里这些数据来自只读 MCP 服务背后的
时序库 / 业务库，这里只是替身。

BMS 只上送簇级单体极值（最高/最低单体电压、温度及其位置），没有全量单体数据，
和真实约束保持一致。
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import datetime, timedelta, timezone

TZ = timezone(timedelta(hours=8))
NOW = datetime(2026, 10, 11, 8, 0, tzinfo=TZ)
STEP_MIN = 5

STATION = {
    "station_id": "ES-DEMO",
    "name": "示范储能电站",
    "capacity": "50MW/100MWh",
    "location": "江苏 · 盐城",
    "commissioned": "2024-06-28",
    "units": [f"BESS-0{i}" for i in range(1, 5)],
}

CLUSTERS_PER_UNIT = 8
HOT_CLUSTER_FACTOR = 1.2
HEAT_RATE = 0.36


# ---------------------------------------------------------------------------
# 拓扑
# ---------------------------------------------------------------------------

def _unit_of(device_id: str) -> str | None:
    if device_id.startswith("BESS-0"):
        return device_id[:7]
    if device_id.startswith("PCS-0"):
        return "BESS-0" + device_id[5]
    return None


def device_info(device_id: str) -> dict | None:
    unit = _unit_of(device_id)
    if device_id in STATION["units"]:
        n = device_id[-1]
        return {
            "device_id": device_id,
            "type": "储能单元",
            "rated": "12.5MW/25MWh",
            "children": [f"{device_id}-C{c:02d}" for c in range(1, CLUSTERS_PER_UNIT + 1)],
            "related": {"pcs": f"PCS-0{n}", "bms": f"{device_id}-BMS", "tms": f"{device_id}-TMS"},
            "cell": "LFP 280Ah，每簇 16 模组 × 26 串",
        }
    if device_id.endswith("-TMS") and unit:
        return {
            "device_id": device_id,
            "type": "液冷机组",
            "parent": unit,
            "serves": [f"{unit}-C{c:02d}" for c in range(1, CLUSTERS_PER_UNIT + 1)],
            "rated": "额定流量 120 L/min，制冷量 60kW",
            "metrics": ["coolant_flow", "pump_current", "outlet_temp", "inlet_temp"],
        }
    if device_id.endswith("-BMS") and unit:
        return {
            "device_id": device_id,
            "type": "电池管理系统（簇级上送）",
            "parent": unit,
            "uplink": f"SW-{unit[-2:]} 交换机端口 3",
            "metrics": ["packet_loss_pct", "comm_status"],
            "comm_escalate_after_min": 5,
        }
    if device_id.startswith("PCS-0") and unit:
        return {
            "device_id": device_id,
            "type": "储能变流器",
            "unit": unit,
            "rated": "12.5MW，1500V 直流侧",
            "metrics": ["insulation_resistance", "cabinet_humidity", "active_power"],
            "insulation_alarm_threshold_kohm": 100,
            "insulation_trip_threshold_kohm": 50,
        }
    if unit and "-C" in device_id:
        return {
            "device_id": device_id,
            "type": "电池簇",
            "parent": unit,
            "cooled_by": f"{unit}-TMS",
            "pcs": f"PCS-0{unit[-1]}",
            "metrics": ["cell_temp_max", "cell_temp_min", "cell_volt_max", "cell_volt_min", "volt_diff_mv", "soc"],
            "thresholds": {"cell_temp_max_l2": 45, "cell_temp_max_l1": 50, "volt_diff_mv_l3": 150, "volt_diff_mv_l2": 250},
        }
    return None


# ---------------------------------------------------------------------------
# 遥测
# ---------------------------------------------------------------------------

def _rng(*key: object) -> random.Random:
    h = hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _hour(ts: datetime) -> float:
    return ts.hour + ts.minute / 60


def _charge_factor(ts: datetime) -> float:
    """0-1：凌晨 02:00-06:00 充电，18:00-21:00 放电。"""
    h = _hour(ts)
    if 2 <= h < 6:
        return math.sin((h - 2) / 4 * math.pi) * 0.6 + 0.4
    if 18 <= h < 21:
        return math.sin((h - 18) / 3 * math.pi) * 0.6 + 0.4
    return 0.0


def _minutes_since(ts: datetime, hh: int, mm: int) -> float:
    start = ts.replace(hour=hh, minute=mm)
    if start > ts:
        start -= timedelta(days=1)
    return (ts - start).total_seconds() / 60


def _in_today_window(ts: datetime, start_h: float, end_h: float) -> bool:
    return ts.date() == NOW.date() and start_h <= _hour(ts) < end_h


def _tms_flow(unit: str, ts: datetime) -> float:
    base = 120.0
    if unit == "BESS-02" and ts.date() == NOW.date() and _hour(ts) >= 3 + 10 / 60:
        m = (ts - ts.replace(hour=3, minute=10)).total_seconds() / 60
        base = 120 - min(m / 110, 1.0) * 55  # 03:10 起逐步跌到 65
    return base


def _value(device_id: str, metric: str, ts: datetime) -> float:
    r = _rng(device_id, metric, ts.isoformat())
    noise = r.uniform(-1, 1)
    unit = _unit_of(device_id) or ""
    cf = _charge_factor(ts)

    if metric == "coolant_flow":
        return round(_tms_flow(unit, ts) + noise * 1.5, 1)
    if metric == "pump_current":
        return round(8.5 * _tms_flow(unit, ts) / 120 + noise * 0.1, 2)
    if metric == "outlet_temp":
        loss = (120 - _tms_flow(unit, ts)) / 55
        return round(18 + cf * 3 + loss * 6 + noise * 0.3, 1)
    if metric == "inlet_temp":
        loss = (120 - _tms_flow(unit, ts)) / 55
        return round(24 + cf * 6 + loss * 9 + noise * 0.3, 1)

    if metric in ("cell_temp_max", "cell_temp_min"):
        base = _thermal(device_id).get(ts)
        if base is None:
            base = 27.0
        if metric == "cell_temp_min":
            base -= 4.2 + (base - 27) * 0.25
        return round(base + noise * 0.25, 1)

    if metric in ("cell_volt_max", "cell_volt_min", "volt_diff_mv"):
        diff = 42 + noise * 3
        if device_id == "BESS-03-C02":
            diff = 60 + noise * 3
            if 5 + 40 / 60 <= _hour(ts) < 6 and ts.date() == NOW.date():
                diff = 182 + noise * 1.5
        vmax = 3.32 + cf * 0.18
        if metric == "cell_volt_max":
            return round(vmax, 3)
        if metric == "cell_volt_min":
            return round(vmax - diff / 1000, 3)
        return round(diff, 0)

    if metric == "soc":
        h = _hour(ts)
        if 2 <= h < 6:
            soc = 10 + (h - 2) / 4 * 85
        elif 6 <= h < 18:
            soc = 95
        elif 18 <= h < 21:
            soc = 95 - (h - 18) / 3 * 85
        else:
            soc = 10
        return round(soc + noise * 0.3, 1)

    if metric == "insulation_resistance":
        val = 650.0
        if device_id == "PCS-01" and ts.date() == NOW.date():
            h = _hour(ts)
            if 1.5 <= h < 4.33:
                val = 650 - (h - 1.5) / 2.83 * 565
            elif h >= 4.33:
                val = 85 + (h - 4.33) / 3.67 * 55
        return round(val + noise * 3, 0)
    if metric == "cabinet_humidity":
        val = 58.0
        if device_id == "PCS-01" and ts.date() == NOW.date() and _hour(ts) >= 1.5:
            h = _hour(ts)
            val = 58 + min((h - 1.5) / 2, 1) * 35 - max(0, h - 5) * 2.5
        return round(val + noise, 1)
    if metric == "active_power":
        h = _hour(ts)
        sign = -1 if 2 <= h < 6 else 1
        return round(sign * cf * 12.5 if cf else 0.0, 2)

    if metric == "packet_loss_pct":
        val = 0.2
        if device_id == "BESS-04-BMS" and _in_today_window(ts, 6, 8):
            val = 3.5 + abs(noise) * 4
        return round(val + abs(noise) * 0.2, 1)
    if metric == "comm_status":
        if device_id == "BESS-04-BMS" and ts.minute in (5, 25, 45) and _in_today_window(ts, 6, 8):
            return 0.0
        return 1.0

    # 气象
    if metric == "ambient_temp":
        return round(19 + 4 * math.sin((_hour(ts) - 9) / 24 * 2 * math.pi) + noise * 0.3, 1)
    if metric == "humidity":
        val = 65.0
        if ts.date() == NOW.date() and _hour(ts) >= 1.0:
            val = 96 - max(0, _hour(ts) - 5) * 3
        return round(val + noise, 1)
    if metric == "rainfall_mm_h":
        if ts.date() == NOW.date() and 1.0 <= _hour(ts) < 5.5:
            return round(6 + noise * 2, 1)
        return 0.0
    raise KeyError(metric)


METRIC_UNITS = {
    "coolant_flow": "L/min", "pump_current": "A", "outlet_temp": "°C", "inlet_temp": "°C",
    "cell_temp_max": "°C", "cell_temp_min": "°C", "cell_volt_max": "V", "cell_volt_min": "V",
    "volt_diff_mv": "mV", "soc": "%", "insulation_resistance": "kΩ", "cabinet_humidity": "%",
    "active_power": "MW", "packet_loss_pct": "%", "comm_status": "1=正常 0=中断",
    "ambient_temp": "°C", "humidity": "%", "rainfall_mm_h": "mm/h",
}



_THERMAL_CACHE: dict[str, dict[datetime, float]] = {}


def _thermal(cluster_id: str) -> dict[datetime, float]:
    """一阶热模型：充放电发热，液冷按流量带走热量。两天预热后取值。"""
    if cluster_id in _THERMAL_CACHE:
        return _THERMAL_CACHE[cluster_id]
    unit = _unit_of(cluster_id) or ""
    idx = int(cluster_id[-2:]) if cluster_id[-2:].isdigit() else 1
    heat = HEAT_RATE * (1 + idx * 0.004)
    if cluster_id == "BESS-02-C05":
        heat *= HOT_CLUSTER_FACTOR
    temp = 21.0
    out: dict[datetime, float] = {}
    ts = NOW - timedelta(hours=72)
    while ts <= NOW:
        flow_ratio = _tms_flow(unit, ts) / 120
        coolant = 20 + (1 - flow_ratio) * 8
        k = 0.03 * flow_ratio ** 1.6
        temp += STEP_MIN * (heat * _charge_factor(ts) - k * (temp - coolant))
        out[ts] = temp
        ts += timedelta(minutes=STEP_MIN)
    _THERMAL_CACHE[cluster_id] = out
    return out


def series(device_id: str, metric: str, hours: int = 24) -> list[tuple[datetime, float]]:
    start = NOW - timedelta(hours=hours)
    out = []
    ts = start + timedelta(minutes=STEP_MIN)
    while ts <= NOW:
        out.append((ts, _value(device_id, metric, ts)))
        ts += timedelta(minutes=STEP_MIN)
    return out


# ---------------------------------------------------------------------------
# 簇级极值（BMS 只上送这些）
# ---------------------------------------------------------------------------

def _position(device_id: str, kind: str) -> str:
    if device_id == "BESS-02-C05" and kind == "temp_max":
        return "M09-14"
    if device_id == "BESS-03-C02" and kind == "volt_min":
        return "M07-12"
    r = _rng(device_id, kind, NOW.isoformat())
    return f"M{r.randint(1, 16):02d}-{r.randint(1, 26):02d}"


def cluster_snapshot(cluster_id: str, at: datetime) -> dict:
    return {
        "cluster_id": cluster_id,
        "time": at.strftime("%H:%M"),
        "cell_temp_max": _value(cluster_id, "cell_temp_max", at),
        "cell_temp_max_pos": _position(cluster_id, "temp_max"),
        "cell_temp_min": _value(cluster_id, "cell_temp_min", at),
        "cell_volt_max": _value(cluster_id, "cell_volt_max", at),
        "cell_volt_min": _value(cluster_id, "cell_volt_min", at),
        "cell_volt_min_pos": _position(cluster_id, "volt_min"),
        "volt_diff_mv": _value(cluster_id, "volt_diff_mv", at),
    }


def charge_end_history(cluster_id: str, cycles: int = 7) -> list[dict]:
    """最近几次充电末端（压差最大时刻）的压差和最低单体位置。"""
    out = []
    for i in range(cycles - 1, -1, -1):
        day = (NOW - timedelta(days=i)).date()
        if cluster_id == "BESS-03-C02":
            diff = [120, 131, 140, 152, 160, 171, 182][cycles - 1 - i] if cycles == 7 else 120 + (cycles - 1 - i) * 10
            pos = "M07-12"
        else:
            r = _rng(cluster_id, "hist", day.isoformat())
            diff = 40 + r.randint(0, 9)
            pos = _position(cluster_id, f"volt_min{day}")
        out.append({"date": day.isoformat(), "volt_diff_mv": diff, "cell_volt_min_pos": pos})
    return out


# ---------------------------------------------------------------------------
# 告警
# ---------------------------------------------------------------------------

ALARMS = {
    "ALM-20261011-0012": {
        "alarm_id": "ALM-20261011-0012",
        "device_id": "BESS-02-C05",
        "type": "cell_temp_high",
        "title": "电池簇单体最高温度过高",
        "level": 2,
        "value": 47.8,
        "threshold": 45,
        "unit": "°C",
        "first_time": "2026-10-11 05:32",
        "last_time": "2026-10-11 06:05",
        "occurrences": 1,
        "status": "active",
    },
    "ALM-20261011-0015": {
        "alarm_id": "ALM-20261011-0015",
        "device_id": "BESS-03-C02",
        "type": "cell_volt_diff_high",
        "title": "电池簇单体压差过大",
        "level": 3,
        "value": 182,
        "threshold": 150,
        "unit": "mV",
        "first_time": "2026-10-11 05:41",
        "last_time": "2026-10-11 05:58",
        "occurrences": 1,
        "status": "active",
    },
    "ALM-20261011-0018": {
        "alarm_id": "ALM-20261011-0018",
        "device_id": "PCS-01",
        "type": "insulation_low",
        "title": "PCS 直流侧绝缘阻抗低",
        "level": 2,
        "value": 85,
        "threshold": 100,
        "unit": "kΩ",
        "first_time": "2026-10-11 03:46",
        "last_time": "2026-10-11 07:40",
        "occurrences": 1,
        "status": "active",
    },
    "ALM-20261011-0021": {
        "alarm_id": "ALM-20261011-0021",
        "device_id": "BESS-04-BMS",
        "type": "bms_comm_lost",
        "title": "BMS 通讯中断",
        "level": 3,
        "value": 0,
        "threshold": None,
        "unit": "",
        "first_time": "2026-10-11 06:05",
        "last_time": "2026-10-11 07:45",
        "occurrences": 6,
        "max_duration_s": 48,
        "auto_recovered": True,
        "status": "active",
    },
}


def _sync_alarms_with_telemetry() -> None:
    """告警的数值和时间从模拟遥测里算出来，保证告警和数据自洽。"""
    def first_cross(dev, metric, pred):
        for ts, v in series(dev, metric, 8):
            if pred(v):
                return ts, v
        return None, None

    a = ALARMS["ALM-20261011-0012"]
    pts = series(a["device_id"], "cell_temp_max", 8)
    t0, _ = first_cross(a["device_id"], "cell_temp_max", lambda v: v >= 45)
    hot = [(t, v) for t, v in pts if v >= 45]
    peak = max(pts, key=lambda x: x[1])
    a.update(first_time=t0.strftime("%Y-%m-%d %H:%M"), last_time=hot[-1][0].strftime("%Y-%m-%d %H:%M"),
             value=peak[1], peak_time=peak[0].strftime("%H:%M"))

    a = ALARMS["ALM-20261011-0018"]
    pts = series(a["device_id"], "insulation_resistance", 8)
    t0, _ = first_cross(a["device_id"], "insulation_resistance", lambda v: v < 100)
    low = min(pts, key=lambda x: x[1])
    a.update(first_time=t0.strftime("%Y-%m-%d %H:%M"), value=low[1], lowest_time=low[0].strftime("%H:%M"),
             current_value=pts[-1][1])


LEVEL_NAME = {1: "一级（紧急）", 2: "二级（重要）", 3: "三级（一般）"}

ALARM_HISTORY = {
    "BESS-02-C05": [],
    "BESS-02-TMS": [
        {"time": "2026-09-02 14:20", "title": "液冷机组流量低（预警）", "level": 3, "duration_min": 25, "closed_by": "自动恢复"},
    ],
    "BESS-03-C02": [
        {"time": "2026-09-27 05:50", "title": "电池簇单体压差预警（120mV）", "level": 3, "duration_min": 12, "closed_by": "自动恢复"},
    ],
    "PCS-01": [
        {"time": "2026-07-14 04:10", "title": "PCS 直流侧绝缘阻抗低（78kΩ）", "level": 2, "duration_min": 420, "closed_by": "雨停后恢复，现场检查柜内加热器"},
    ],
    "BESS-04-BMS": [
        {"time": "2026-10-11 06:05", "title": "BMS 通讯中断", "level": 3, "duration_s": 31, "closed_by": "自动恢复"},
        {"time": "2026-10-11 06:25", "title": "BMS 通讯中断", "level": 3, "duration_s": 22, "closed_by": "自动恢复"},
        {"time": "2026-10-11 06:45", "title": "BMS 通讯中断", "level": 3, "duration_s": 48, "closed_by": "自动恢复"},
        {"time": "2026-10-11 07:05", "title": "BMS 通讯中断", "level": 3, "duration_s": 19, "closed_by": "自动恢复"},
        {"time": "2026-10-11 07:25", "title": "BMS 通讯中断", "level": 3, "duration_s": 35, "closed_by": "自动恢复"},
        {"time": "2026-10-11 07:45", "title": "BMS 通讯中断", "level": 3, "duration_s": 27, "closed_by": "自动恢复"},
    ],
}

MAINTENANCE = {
    "BESS-02-TMS": [
        {"date": "2026-06-18", "type": "季度保养", "content": "清洗过滤器、补充冷却液", "next_due": "2026-09-18"},
        {"date": "2026-03-20", "type": "季度保养", "content": "清洗过滤器、检查水泵", "next_due": "2026-06-20"},
    ],
    "BESS-02-C05": [
        {"date": "2026-05-10", "type": "年检", "content": "簇内连接排紧固力矩抽检，正常", "next_due": "2027-05-10"},
    ],
    "BESS-03-C02": [
        {"date": "2026-04-22", "type": "年检", "content": "内阻抽检，M07 模组内阻偏高 8%，列为关注", "next_due": "2027-04-22"},
    ],
    "PCS-01": [
        {"date": "2026-07-15", "type": "故障检修", "content": "绝缘低后检查：柜内加热除湿器故障，已更换", "next_due": None},
        {"date": "2026-04-02", "type": "年检", "content": "直流电缆接头绝缘检测正常", "next_due": "2027-04-02"},
    ],
    "BESS-04-BMS": [
        {"date": "2026-08-30", "type": "缺陷处理", "content": "SW-04 交换机端口 3 水晶头氧化，重新压接", "next_due": None},
    ],
}

FAULT_CASES = [
    {
        "case_id": "CASE-0231",
        "title": "液冷过滤器堵塞导致整舱电芯温升",
        "station": "某 100MWh 电站",
        "symptom": "同一储能舱多簇最高温度同步上升，液冷流量下降 40% 以上，水泵电流下降",
        "root_cause": "Y 型过滤器堵塞，超期未清洗",
        "action": "降功率运行 → 停机清洗过滤器 → 补液排气，恢复后温升正常",
        "keywords": ["温度", "温升", "液冷", "流量", "过滤器", "水泵", "cell_temp_high"],
        "grade": "A",
    },
    {
        "case_id": "CASE-0187",
        "title": "单簇温度高：簇内风扇/冷板局部失效",
        "station": "某 50MWh 电站",
        "symptom": "仅一簇温度高，同舱其他簇正常，液冷流量正常",
        "root_cause": "该簇冷板支路阀门半关",
        "action": "调整支路阀门开度",
        "keywords": ["温度", "单簇", "冷板", "阀门", "cell_temp_high"],
        "grade": "B",
    },
    {
        "case_id": "CASE-0302",
        "title": "单体一致性衰减导致充电末端压差逐日扩大",
        "station": "某 200MWh 电站",
        "symptom": "同一单体位置在充电末端反复为最低，压差每周上升约 10mV",
        "root_cause": "单体容量衰减 / 自放电偏大",
        "action": "开展容量标定与均衡，必要时更换模组",
        "keywords": ["压差", "一致性", "均衡", "充电末端", "cell_volt_diff_high"],
        "grade": "A",
    },
    {
        "case_id": "CASE-0115",
        "title": "降雨后 PCS 直流侧绝缘阻抗低",
        "station": "某 100MWh 电站",
        "symptom": "降雨、高湿后绝缘阻抗降至告警阈值以下，湿度下降后自行回升",
        "root_cause": "柜内凝露 + 加热除湿失效",
        "action": "检查加热除湿器与柜体密封，排查直流电缆接头；低于跳闸阈值则停机",
        "keywords": ["绝缘", "阻抗", "降雨", "湿度", "凝露", "insulation_low"],
        "grade": "A",
    },
    {
        "case_id": "CASE-0266",
        "title": "BMS 通讯频繁短时中断",
        "station": "某 50MWh 电站",
        "symptom": "通讯中断反复出现、每次一分钟内自动恢复，丢包率升高",
        "root_cause": "交换机端口/网线接触不良",
        "action": "合并告警，安排巡检时检查网线与端口，不需要停机",
        "keywords": ["通讯", "中断", "丢包", "交换机", "bms_comm_lost"],
        "grade": "B",
    },
]


_sync_alarms_with_telemetry()


def weather(hours: int = 12) -> list[dict]:
    out = []
    for ts, _ in series("WEATHER", "humidity", hours)[::6]:
        out.append({
            "time": ts.strftime("%m-%d %H:%M"),
            "ambient_temp": _value("WEATHER", "ambient_temp", ts),
            "humidity": _value("WEATHER", "humidity", ts),
            "rainfall_mm_h": _value("WEATHER", "rainfall_mm_h", ts),
        })
    return out

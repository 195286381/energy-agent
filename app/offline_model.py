"""离线模拟模型：没有 DeepSeek API Key 时，让整条 Deep Agents 流程照样跑通。

它是一个「规则版大脑」：读对话里已经拿到的工具结果，按 Skill 里写的诊断步骤
决定下一步调用哪个工具，最后用工具返回的真实数值拼出结论。所以：
- 工具调用、子智能体委派、Skills、事实校验、SSE 推流全部是真的在跑；
- 只有「推理」这一步换成了确定性规则，用来演示和做回归测试。
接上 DeepSeek（设置 DEEPSEEK_API_KEY）后，由真实模型自己决定步骤。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.answer_guard import FEEDBACK_PREFIX, _redact

ALARM_RE = re.compile(r"ALM-\d{8}-\d{4}")
PRIORITY_DESC = {"P1": "立即处理", "P2": "2 小时内响应", "P3": "当班巡检处理"}


# ---------------------------------------------------------------------------
# 读对话
# ---------------------------------------------------------------------------

def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def _parse(content: Any) -> Any:
    t = _text(content)
    try:
        return json.loads(t)
    except (ValueError, TypeError):
        return t


def _history(messages: list[BaseMessage]) -> list[dict]:
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    out = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                out.append({"name": tc["name"], "args": tc["args"], "result": _parse(results.get(tc["id"]))})
    return out


def _find(hist: list[dict], name: str, **match: Any) -> Any:
    for h in reversed(hist):
        if h["name"] == name and all(h["args"].get(k) == v for k, v in match.items()):
            return h["result"]
    return None


def _json_block(text: str) -> dict:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except ValueError:
        return {}


def _first_human(messages: list[BaseMessage]) -> str:
    for m in messages:
        if isinstance(m, HumanMessage) and not _text(m.content).startswith(FEEDBACK_PREFIX):
            return _text(m.content)
    return ""


class _Calls:
    def __init__(self, role: str, n: int):
        self.role, self.n, self.items = role, n, []

    def add(self, name: str, **args: Any) -> "_Calls":
        self.items.append({"name": name, "args": args, "id": f"call_{self.role}_{self.n}_{len(self.items)}", "type": "tool_call"})
        return self


def _say(text: str, calls: _Calls | None = None) -> AIMessage:
    return AIMessage(content=text, tool_calls=calls.items if calls else [])


# ---------------------------------------------------------------------------
# 告警分诊（flash 模型的角色）
# ---------------------------------------------------------------------------

def _triage(messages: list[BaseMessage]) -> AIMessage:
    task = _first_human(messages)
    alarm_id = (ALARM_RE.findall(task) or [""])[0]
    hist = _history(messages)
    step = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    c = _Calls("triage", step)

    detail = _find(hist, "get_alarm_detail")
    if detail is None:
        return _say(f"先查告警 {alarm_id} 的详情。", c.add("get_alarm_detail", alarm_id=alarm_id))
    dev = detail["device_id"]
    if _find(hist, "get_alarm_history") is None:
        return _say(
            f"{dev} 触发「{detail['title']}」。再查这台设备近 30 天的告警记录和台账，看是不是抖动或老问题。",
            c.add("get_alarm_history", device_id=dev, days=30).add("get_device_info", device_id=dev),
        )
    if _find(hist, "list_active_alarms") is None:
        return _say("看一下全站当前还有哪些未复归告警，判断有没有关联。", c.add("list_active_alarms", station_id="ES-DEMO"))

    history = _find(hist, "get_alarm_history")
    info = _find(hist, "get_device_info") or {}
    active = _find(hist, "list_active_alarms")
    unit = dev[:7] if dev.startswith("BESS") else info.get("unit", "")
    related = [a for a in active["alarms"] if a["alarm_id"] != alarm_id and (a["device_id"].startswith(unit) or a["device_id"] == f"PCS-0{unit[-1:]}")]

    level = detail["level"]
    priority = {1: "P1", 2: "P2", 3: "P3"}[level]
    flapping = detail.get("occurrences", 1) >= 3 and detail.get("auto_recovered", False)
    prior = [h for h in history["items"] if not h["time"].startswith("2026-10-11")]

    if detail.get("threshold") is not None:
        lines = [f"{detail['level_name']}告警，触发值 {detail['value']}{detail['unit']}，阈值 {detail['threshold']}{detail['unit']}，首次 {detail['first_time'][11:]}。"]
    else:
        lines = [f"{detail['level_name']}告警，首次 {detail['first_time'][11:]}。"]
    evidence = []
    if flapping:
        lines.append(f"{detail['first_time'][11:]}–{detail['last_time'][11:]} 共发生 {detail['occurrences']} 次，每次都自动恢复，最长 {detail['max_duration_s']} 秒，属于抖动告警，合并为一条处理。")
        evidence.append(f"告警历史：{detail['first_time'][11:]}–{detail['last_time'][11:]} 共 {detail['occurrences']} 次短时中断，最长 {detail['max_duration_s']} 秒，均自动恢复（get_alarm_history）")
        if info.get("uplink"):
            evidence.append(f"BMS 上联 {info['uplink']}（get_device_info）")
    if prior:
        lines.append(f"同设备近 30 天有 {len(prior)} 次相关记录：{prior[0]['time'][:10]}「{prior[0]['title']}」，可能是老问题在发展。")
    if detail.get("current_value") is not None and detail.get("threshold") is not None and detail["current_value"] >= detail["threshold"]:
        lines.append(f"当前值 {detail['current_value']}{detail['unit']} 已回到阈值以上，但告警未复归，需要确认原因。")
    lines.append(f"关联告警：{'、'.join(a['alarm_id'] + ' ' + a['title'] for a in related)}。" if related else "同一储能单元没有其他未复归告警。")

    need_diag = not flapping
    priority_final = priority
    if flapping:
        priority_final = "P3"
    result = {
        "alarm_id": alarm_id,
        "device_id": dev,
        "alarm_type": detail["type"],
        "title": detail["title"],
        "level": level,
        "level_name": detail["level_name"],
        "priority": priority_final,
        "flapping": flapping,
        "need_diagnosis": need_diag,
        "related_alarms": [a["alarm_id"] for a in related],
        "summary": "".join(lines),
        "evidence": evidence,
        "conclusion_short": f"{detail['occurrences']} 次短时中断均自动恢复，判定为通讯抖动，合并处理" if flapping else "",
        "actions": [
            "合并为一条告警处理，不需要停机",
            f"巡检时检查 {info.get('uplink', '上联交换机端口')} 的网线和水晶头",
            f"若中断持续超过 {info.get('comm_escalate_after_min', 5)} 分钟（失去电池监控）立即升级",
        ] if flapping else [],
        "detail": detail,
    }
    verdict = "不需要深度诊断" if not need_diag else "需要深度诊断"
    text = f"分诊完成：{detail['level_name']} · {priority_final} · {'抖动告警' if flapping else '非抖动'} · {verdict}。\n" + "".join(lines)
    return _say(text + "\n\n```json\n" + json.dumps(result, ensure_ascii=False) + "\n```")


# ---------------------------------------------------------------------------
# 故障诊断（pro 模型的角色）
# ---------------------------------------------------------------------------

SKILL_PATH = {
    "cell_temp_high": "/skills/diagnosis/cell-temp-high/SKILL.md",
    "cell_volt_diff_high": "/skills/diagnosis/cell-volt-diff-high/SKILL.md",
    "insulation_low": "/skills/diagnosis/insulation-low/SKILL.md",
    "bms_comm_lost": "/skills/diagnosis/bms-comm-lost/SKILL.md",
}


def _plan(a: dict) -> list[tuple[str, list[tuple[str, dict]]]]:
    dev, t = a["device_id"], a["type"]
    unit = dev[:7]
    steps: list[tuple[str, list[tuple[str, dict]]]] = [
        (f"告警类型是 {t}，先读对应的诊断 Skill，按里面的步骤走。", [("read_file", {"file_path": SKILL_PATH[t]})]),
        (f"查 {dev} 的台账，拿到拓扑关系和告警阈值。", [("get_device_info", {"device_id": dev})]),
    ]
    if t == "cell_temp_high":
        steps += [
            (f"先判断是整舱问题还是单簇问题：对比 {unit} 所有簇在峰值时刻 {a.get('peak_time', '')} 的最高温度。",
             [("get_cluster_extremes", {"unit_or_cluster_id": unit, "at": a.get("peak_time", "")})]),
            (f"多个簇同时偏高，像是散热问题。查 {unit}-TMS 液冷机组的流量和水泵电流。",
             [("query_telemetry", {"device_id": f"{unit}-TMS", "metric": "coolant_flow", "hours": 8}),
              ("query_telemetry", {"device_id": f"{unit}-TMS", "metric": "pump_current", "hours": 8})]),
            (f"流量和水泵电流同步下降，查 {unit}-TMS 的保养记录，看过滤器是不是超期没洗。",
             [("get_maintenance_records", {"device_id": f"{unit}-TMS"})]),
            ("找历史上相似的案例佐证。", [("search_fault_cases", {"query": "液冷 流量 温升 过滤器", "top_k": 3})]),
        ]
    elif t == "cell_volt_diff_high":
        steps += [
            ("看最近 7 次充电末端的压差趋势和最低单体位置。", [("get_cluster_extremes", {"unit_or_cluster_id": dev})]),
            ("同一个位置反复最低、压差逐次变大。再排除温差的影响。",
             [("query_telemetry", {"device_id": dev, "metric": "cell_temp_max", "hours": 8})]),
            ("查检修记录，看内阻抽检有没有提到这个模组。", [("get_maintenance_records", {"device_id": dev})]),
            ("找相似案例。", [("search_fault_cases", {"query": "压差 一致性 充电末端", "top_k": 3})]),
        ]
    elif t == "insulation_low":
        steps += [
            ("看绝缘阻抗和柜内湿度的走势。",
             [("query_telemetry", {"device_id": dev, "metric": "insulation_resistance", "hours": 8}),
              ("query_telemetry", {"device_id": dev, "metric": "cabinet_humidity", "hours": 8})]),
            ("阻抗和湿度一起变化，查同时段的天气。", [("get_weather", {"station_id": "ES-DEMO", "hours": 12})]),
            ("查这台 PCS 以前有没有同类问题、加热除湿器修没修过。",
             [("get_alarm_history", {"device_id": dev, "days": 90}), ("get_maintenance_records", {"device_id": dev})]),
            ("找相似案例。", [("search_fault_cases", {"query": "绝缘 降雨 湿度", "top_k": 3})]),
        ]
    else:  # bms_comm_lost
        steps += [
            ("看丢包率和中断记录。",
             [("query_telemetry", {"device_id": dev, "metric": "packet_loss_pct", "hours": 4}),
              ("get_alarm_history", {"device_id": dev, "days": 1})]),
            ("查端口以前的处理记录。", [("get_maintenance_records", {"device_id": dev})]),
        ]
    return steps


def _diag_temp(a: dict, hist: list[dict]) -> dict:
    dev, unit = a["device_id"], a["device_id"][:7]
    info = _find(hist, "get_device_info")
    ext = _find(hist, "get_cluster_extremes")
    flow = _find(hist, "query_telemetry", metric="coolant_flow")
    pump = _find(hist, "query_telemetry", metric="pump_current")
    mnt = _find(hist, "get_maintenance_records")
    case = (_find(hist, "search_fault_cases") or {"results": [{}]})["results"][0]
    l1 = info["thresholds"]["cell_temp_max_l1"]
    rec = mnt["records"][0]
    hottest = max(ext["clusters"], key=lambda r: r["cell_temp_max"])
    return {
        "conclusion": f"{unit} 液冷流量不足导致整舱多簇温升，{dev} 散热最差、最先越限；最可能原因是液冷过滤器堵塞（保养已超期）",
        "conclusion_short": f"{unit} 液冷流量不足，整舱温升，{dev} 最先越限，疑似过滤器堵塞",
        "confidence": "高",
        "evidence": [
            f"{ext['time']} 时 {unit} 有 {ext['clusters_over_reference']} 个簇最高温度超过 {ext['reference_temp_c']}°C，单元最高 {ext['temp_max_of_unit']}°C（{hottest['cluster_id']}，位置 {hottest['cell_temp_max_pos']}），中位数 {ext['temp_max_median']}°C，是整舱问题不是单簇问题（get_cluster_extremes）",
            f"{unit}-TMS 冷却液流量从 {flow['first']} 降到 {flow['last']} L/min（{flow['change_pct']}%），最低 {flow['min']} L/min（query_telemetry）",
            f"水泵电流从 {pump['first']} A 降到 {pump['last']} A，与流量同步下降，指向管路阻力增大或水泵出力不足（query_telemetry）",
            f"{unit}-TMS 上次保养 {rec['date']}，应在 {rec['next_due']} 前再保养，{'已超期' if mnt['overdue'] else '未超期'}（get_maintenance_records）",
            f"相似案例 {case.get('case_id')}「{case.get('title')}」：根因 {case.get('root_cause')}（search_fault_cases）",
        ],
        "actions": [
            f"立即降低 {unit} 充放电功率，液冷流量恢复前暂停该单元充放电",
            f"派单检查 {unit}-TMS：清洗过滤器、检查水泵和冷却液液位，排气后确认流量恢复到额定值",
            f"{dev} 最高温度达到 {l1}°C（一级阈值）立即停机；液冷恢复后复核 {dev} 是否仍高于同单元其他簇",
        ],
    }


def _diag_volt(a: dict, hist: list[dict]) -> dict:
    dev = a["device_id"]
    info = _find(hist, "get_device_info")
    ext = _find(hist, "get_cluster_extremes")
    temp = _find(hist, "query_telemetry", metric="cell_temp_max")
    mnt = _find(hist, "get_maintenance_records")
    case = (_find(hist, "search_fault_cases") or {"results": [{}]})["results"][0]
    h = ext["charge_end_history"]
    pos = h[-1]["cell_volt_min_pos"]
    same_pos = all(x["cell_volt_min_pos"] == pos for x in h)
    l2 = info["thresholds"]["volt_diff_mv_l2"]
    rec = mnt["records"][0]
    return {
        "conclusion": f"{dev} 的 {pos} 单体一致性衰减（容量衰减或自放电偏大）：最近 {len(h)} 次充电末端它{'都' if same_pos else '多次'}是最低单体，压差逐次扩大；温度正常，不是温差或采样问题",
        "conclusion_short": f"{dev} 的 {pos} 单体一致性衰减，压差逐次扩大，暂不影响运行",
        "confidence": "中",
        "evidence": [
            f"最近 {len(h)} 次充电末端压差：{' → '.join(str(x['volt_diff_mv']) for x in h)} mV，每次平均上升 {ext['volt_diff_trend_mv_per_cycle']} mV，最低单体都在 {pos}（get_cluster_extremes）",
            f"同期簇内最高温度 {temp['max']}°C，温度正常，排除温差引起的压差（query_telemetry）",
            f"{rec['date']} {rec['type']}：{rec['content']}（get_maintenance_records）",
            f"相似案例 {case.get('case_id')}「{case.get('title')}」：根因 {case.get('root_cause')}（search_fault_cases）",
        ],
        "actions": [
            "不需要停机；适当下调该簇充电截止 SOC，减小充电末端压差",
            f"一周内安排 {dev} 容量标定和主动均衡，重点检查 {pos} 所在模组",
            f"每天跟踪充电末端压差，到二级阈值 {l2} mV 时升级处理（距现在还差 {l2 - h[-1]['volt_diff_mv']} mV）",
        ],
    }


def _diag_insulation(a: dict, hist: list[dict]) -> dict:
    dev = a["device_id"]
    info = _find(hist, "get_device_info")
    ins = _find(hist, "query_telemetry", metric="insulation_resistance")
    hum = _find(hist, "query_telemetry", metric="cabinet_humidity")
    wx = _find(hist, "get_weather")["points"]
    ah = _find(hist, "get_alarm_history")
    mnt = _find(hist, "get_maintenance_records")
    rain = [p for p in wx if p["rainfall_mm_h"] > 0]
    max_h = max(p["humidity"] for p in wx)
    thr, trip = info["insulation_alarm_threshold_kohm"], info["insulation_trip_threshold_kohm"]
    prev = ah["items"][0] if ah["items"] else None
    fix = next((r for r in mnt["records"] if r["type"] == "故障检修"), None)
    ev = [
        f"绝缘阻抗从 {ins['first']} kΩ 降到 {ins['min']} kΩ（{ins['min_time']}），当前回升到 {ins['last']} kΩ（query_telemetry）",
        f"柜内湿度最高 {hum['max']}%（{hum['max_time']}），当前 {hum['last']}%（query_telemetry）",
    ]
    if rain:
        ev.append(f"气象：{rain[0]['time'][6:]}–{rain[-1]['time'][6:]} 有降雨，环境湿度最高 {max_h}%（get_weather）")
    if prev:
        ev.append(f"{prev['time'][:10]} 出过同类告警「{prev['title']}」，处理：{prev['closed_by']}（get_alarm_history）")
    if fix:
        ev.append(f"{fix['date']} {fix['content']}（get_maintenance_records）")
    return {
        "conclusion": f"降雨高湿导致 {dev} 直流侧受潮凝露，绝缘阻抗最低 {ins['min']} kΩ，湿度下降后回升到 {ins['last']} kΩ；更像受潮而不是绝缘破损，但之前出过同类问题，柜内加热除湿器可能再次失效",
        "conclusion_short": f"{dev} 降雨受潮致绝缘阻抗低，已回升到 {ins['last']} kΩ，需查加热除湿器",
        "confidence": "中",
        "evidence": ev,
        "actions": [
            f"暂不需要停机：当前 {ins['last']} kΩ 已高于告警阈值 {thr} kΩ，也高于跳闸阈值 {trip} kΩ",
            f"派单检查 {dev} 柜内加热除湿器是否工作、柜门密封和电缆进线封堵",
            f"若阻抗再次跌破 {trip} kΩ，或天晴后仍不回升，按绝缘破损排查直流电缆接头并停机",
        ],
    }


def _diag_comm(a: dict, hist: list[dict]) -> dict:
    dev = a["device_id"]
    loss = _find(hist, "query_telemetry", metric="packet_loss_pct")
    info = _find(hist, "get_device_info")
    return {
        "conclusion": f"{dev} 通讯抖动，丢包率最高 {loss['max']}%，多为上联网络接触不良",
        "conclusion_short": f"{dev} 通讯抖动，需检查上联端口",
        "confidence": "中",
        "evidence": [f"丢包率最高 {loss['max']}%（{loss['max_time']}）（query_telemetry）"],
        "actions": ["合并告警，不需要停机", f"检查 {info.get('uplink', '上联端口')}"],
    }


FINALIZERS = {"cell_temp_high": _diag_temp, "cell_volt_diff_high": _diag_volt,
              "insulation_low": _diag_insulation, "bms_comm_lost": _diag_comm}


def _diagnosis(messages: list[BaseMessage], inject_hallucination: bool = False) -> AIMessage:
    task = _first_human(messages)
    alarm = _json_block(task)
    if not alarm:
        return _say("任务描述里没有告警详情（```json 块），无法按离线规则诊断。")
    hist = _history(messages)
    step = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    plan = _plan(alarm)
    if step < len(plan):
        thought, calls = plan[step]
        c = _Calls("diag", step)
        for name, args in calls:
            c.add(name, **args)
        return _say(thought, c)
    d = FINALIZERS[alarm["type"]](alarm, hist)
    fake = ["设备健康度评分已降到 73.46 分，劣化明显（推测，没有对应工具数据）"] if inject_hallucination else []
    text = (
        f"诊断结论：{d['conclusion']}（置信度：{d['confidence']}）\n\n证据：\n"
        + "\n".join(f"- {e}" for e in d["evidence"] + fake)
        + "\n\n建议：\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(d["actions"], 1))
    )
    return _say(text + "\n\n```json\n" + json.dumps(d, ensure_ascii=False) + "\n```")


# ---------------------------------------------------------------------------
# 主智能体（值班长）
# ---------------------------------------------------------------------------

def _todos(stage: int, need_diag: bool = True) -> list[dict]:
    items = ["告警分诊（等级、抖动、关联）", "多步诊断，找根因和证据", "按报告规范整理处置建议", "生成企业微信推送草稿"]
    if not need_diag:
        items[1] = "多步诊断（分诊判定不需要，跳过）"
    out = []
    for i, it in enumerate(items):
        status = "completed" if i < stage else ("in_progress" if i == stage else "pending")
        out.append({"content": it, "status": status})
    return out


def _main(messages: list[BaseMessage]) -> AIMessage:
    task = _first_human(messages)
    alarm_id = (ALARM_RE.findall(task) or [""])[0]
    hist = _history(messages)
    step = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    c = _Calls("main", step)
    tasks = [h for h in hist if h["name"] == "task"]
    triage_done = next((h for h in tasks if h["args"].get("subagent_type") == "alarm-triage"), None)
    diag_done = next((h for h in tasks if h["args"].get("subagent_type") == "fault-diagnostician"), None)

    if not any(h["name"] == "write_todos" for h in hist):
        return _say(f"收到告警 {alarm_id}。先列处置计划，再按步骤执行。", c.add("write_todos", todos=_todos(0)))
    if triage_done is None:
        return _say(
            "第一步交给分诊智能体（flash 模型，快）判断等级、是否抖动、是否需要深度诊断。",
            c.add("task", subagent_type="alarm-triage",
                  description=f"对告警 {alarm_id} 做分诊：确认等级和处理优先级，判断是否抖动或重复告警、有没有关联告警、是否需要深度诊断。"),
        )
    tri = _json_block(_text(triage_done["result"]))
    need = tri.get("need_diagnosis", True)
    if need and diag_done is None:
        return _say(
            f"分诊：{tri.get('level_name')} · {tri.get('priority')}，需要深度诊断。交给故障诊断智能体（pro 模型）按 Skill 多步排查。",
            c.add("write_todos", todos=_todos(1)).add(
                "task", subagent_type="fault-diagnostician",
                description=f"诊断告警 {alarm_id}（{tri.get('title')}，设备 {tri.get('device_id')}，告警类型：{tri.get('alarm_type')}）。"
                            f"分诊摘要：{tri.get('summary')}\n找出根因、给出带数值的证据和处置建议。告警详情：\n```json\n"
                            + json.dumps(tri.get("detail", {}), ensure_ascii=False) + "\n```"),
        )
    if _find(hist, "read_file") is None:
        msg = "诊断完成。" if need else "分诊判定为抖动告警，不需要深度诊断。"
        return _say(msg + "按报告规范整理结论。",
                    c.add("write_todos", todos=_todos(2, need)).add("read_file", file_path="/skills/ops/alarm-report/SKILL.md"))

    diag = _json_block(_text(diag_done["result"])) if diag_done else {}
    conclusion = diag.get("conclusion_short") or tri.get("conclusion_short") or tri.get("summary", "")
    actions = diag.get("actions") or tri.get("actions") or []
    if _find(hist, "draft_wecom_message") is None:
        return _say(
            "生成企业微信推送草稿，等值班员确认后发送。",
            c.add("write_todos", todos=_todos(3, need)).add(
                "draft_wecom_message", level=tri.get("level", 3), title=f"{tri.get('device_id')} {tri.get('title')}",
                conclusion=conclusion, actions=actions[:3], priority=tri.get("priority", "P3"),
                mention_leader=tri.get("priority") != "P3"),
        )

    pr = tri.get("priority", "P3")
    related = tri.get("related_alarms") or []
    lines = [
        "## 分诊",
        f"{tri.get('level_name')} · {pr}（{PRIORITY_DESC.get(pr, '')}）· {'抖动告警，已合并' if tri.get('flapping') else '非抖动'} · "
        + ("关联告警 " + "、".join(related) if related else "同单元无其他未复归告警"),
        "",
        tri.get("summary", ""),
        "",
        "## 诊断结论",
        (f"{diag['conclusion']}（置信度：{diag['confidence']}）" if diag else "抖动类告警，按分诊结论处理，不做深度诊断。"),
        "",
        "## 关键证据",
        *[f"- {e}" for e in (diag.get("evidence") or tri.get("evidence") or [])],
        "",
        "## 处置建议",
        *[f"{i}. {a}" for i, a in enumerate(actions, 1)],
        "",
        "## 推送",
        f"已生成企业微信推送草稿（{pr}{'，@值班长' if pr != 'P3' else ''}），等值班员在界面确认后发送。",
        "",
        "> 智能体只读取数据、只给建议，没有下发任何控制指令。",
    ]
    return _say("\n".join(lines))


BRAINS = {"main": _main, "triage": _triage, "diagnosis": _diagnosis}


def _after_guard(messages: list[BaseMessage]) -> AIMessage | None:
    """事实校验要求重写时：去掉被点名的数值所在句子后重新给出答复。"""
    last = messages[-1]
    if not (isinstance(last, HumanMessage) and _text(last.content).startswith(FEEDBACK_PREFIX)):
        return None
    prev = next(m for m in reversed(messages) if isinstance(m, AIMessage))
    bad = re.findall(r"[\d.]+", _text(last.content).split("：", 1)[1].split("。")[0])
    # 规则版「重写」：去掉含无依据数值的那一行证据，其余不变
    lines = [ln for ln in _text(prev.content).split("\n") if not (ln.startswith("- ") and any(b in ln for b in bad))]
    text = "\n".join(lines)
    return AIMessage(content=text if text != _text(prev.content) else _redact(text, bad))


class OfflineOpsModel(BaseChatModel):
    role: str = "main"
    step_delay: float = 0.0
    inject_hallucination: bool = False  # 演示用：让诊断智能体编一个没有依据的数值，看事实校验拦截

    @property
    def _llm_type(self) -> str:
        return "offline-ops-rules"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "OfflineOpsModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        if self.role == "diagnosis":
            msg = _after_guard(messages) or _diagnosis(messages, self.inject_hallucination)
        else:
            msg = _after_guard(messages) or BRAINS[self.role](messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        if self.step_delay:
            await asyncio.sleep(self.step_delay)
        return self._generate(messages, stop, run_manager, **kwargs)

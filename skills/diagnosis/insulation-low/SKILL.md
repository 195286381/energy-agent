---
name: insulation-low
description: PCS 直流侧绝缘阻抗低告警的诊断步骤（区分受潮凝露和真实绝缘破损），告警类型 insulation_low 时使用
---

# 绝缘阻抗低诊断

## 步骤
1. `query_telemetry(PCS, insulation_resistance)` 和 `cabinet_humidity`：看最低值、当前值、是否在回升。
2. `get_weather`：同时段有没有降雨、高湿。
3. `get_alarm_history` 和 `get_maintenance_records`：以前有没有同类事件、加热除湿器是否修过。
4. `search_fault_cases`：关键词「绝缘 降雨 湿度」。

## 判断
- 降雨/高湿同步出现，湿度下降后阻抗回升 → 受潮凝露，优先查加热除湿器和柜体密封。
- 天气正常、阻抗持续下降 → 怀疑电缆或组件绝缘破损，需要停机排查。

## 处置原则
- 低于跳闸阈值（台账 insulation_trip_threshold_kohm）→ 建议停机。
- 回到告警阈值以上也不要直接关单，要求现场确认除湿设备状态。

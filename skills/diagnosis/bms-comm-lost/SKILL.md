---
name: bms-comm-lost
description: BMS 通讯中断告警的诊断步骤（识别抖动、判断是否影响监控），告警类型 bms_comm_lost 时使用
---

# BMS 通讯中断诊断

1. `get_alarm_history(<BMS>, days=1)`：短时间内反复出现且每次自动恢复 → 抖动。
2. `query_telemetry(<BMS>, packet_loss_pct)`：丢包率是否升高。
3. `get_device_info(<BMS>)`：上联交换机和端口；`get_maintenance_records` 看端口以前是否处理过。

处置：抖动类合并为一条，不需要停机，安排巡检检查网线和交换机端口；
如果中断持续超过 5 分钟，需要升级处理（失去电池监控）。

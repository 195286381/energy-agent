# 储能运维智能体 Demo

告警进来以后，智能体自动完成**分诊 → 多步诊断 → 处置建议 → 企业微信推送草稿**，网页上能看到每一步的推理、工具调用和子智能体委派。

- 框架：LangChain Deep Agents（`create_deep_agent`），原生 Skills、TODO、子智能体
- 模型：DeepSeek 云端 API，pro 做值班长和故障诊断，flash 做告警分诊；没有 Key 时自动切到离线规则模型
- 数据：只读 MCP 服务（9 个工具，scope 校验 + 查询预算），背后是一个模拟的 50MW/100MWh 电站
- 防幻觉：AnswerGuard 中间件，终答里的每个数值都要能在工具结果里找到依据，否则打回重写一次，再不过就删句
- 服务：FastAPI + SSE，事件可按序号断线续看

## 架构

```
告警 ──► 值班长（主智能体，deepseek-v4-pro）
          ├─ write_todos 列计划
          ├─ task ► alarm-triage（deepseek-v4-flash）
          │          get_alarm_detail / get_alarm_history / get_device_info / list_active_alarms
          │          → 等级、优先级、是否抖动、是否需要深度诊断
          ├─ task ► fault-diagnostician（deepseek-v4-pro）
          │          先读诊断 Skill（/skills/diagnosis/*），按步骤调只读 MCP 工具取证
          │          → 根因、带数值的证据、处置建议
          ├─ read_file /skills/ops/alarm-report/SKILL.md（报告格式）
          └─ draft_wecom_message → 推送草稿（值班员在页面确认后才发送）

每个智能体都挂 AnswerGuard：子智能体先在自己的工具结果上校验，
主智能体再以子智能体返回的文本为依据校验终答。
```

## 本地运行

需要 Python 3.11+。

```bash
cd energy-agent-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # 填 DEEPSEEK_API_KEY；不填就是离线规则模型
uvicorn app.server:app --port 8000
# 打开 http://localhost:8000
```

命令行直接跑：

```bash
python scripts/run_alarm.py ALM-20261011-0012            # 单条告警
python scripts/run_alarm.py ALM-20261011-0012 --inject   # 演示事实校验拦截编造的数值（离线模式）
python scripts/run_alarm.py --all --out traces.json       # 全部告警
pytest -q                                                 # 回归测试（离线，不需要 Key）
python scripts/build_preview.py                           # 生成可直接双击打开的回放页 preview/index.html
```

单独起 MCP 服务（HTTP）：`python -m app.mcp_server --http 8765`，然后在 `.env` 里设 `MCP_TRANSPORT=http`。

## 模拟的四条告警

| 告警 | 设备 | 埋的根因 | 智能体应得出 |
| --- | --- | --- | --- |
| 单体最高温度过高（二级） | BESS-02-C05 | 液冷流量从 120 掉到约 64 L/min，过滤器保养超期 | 整舱散热问题，C05 最先越限；降功率 + 派单洗过滤器 |
| 单体压差过大（三级） | BESS-03-C02 | M07-12 单体连续 7 次充电末端最低，压差逐次 +10mV | 一致性衰减，不停机，安排标定均衡 |
| 绝缘阻抗低（二级） | PCS-01 | 凌晨降雨高湿，柜内凝露，雨停后回升 | 受潮而非破损，查加热除湿器 |
| BMS 通讯中断（三级） | BESS-04-BMS | 2 小时 6 次短时中断，均自动恢复 | 抖动告警，合并降级，不做深度诊断 |

BMS 只上送簇级单体极值（和真实约束一致），诊断结论按这个粒度给。

## 离线规则模型是什么

没有 DeepSeek Key 时用 `app/offline_model.py`：它读对话里已有的工具结果，按 Skill 写的步骤决定下一步调哪个工具，结论里的数值全部取自工具返回。Deep Agents 的编排、子智能体、Skills、MCP 调用、事实校验、SSE 都是真的在跑，只有「推理」换成了确定性规则。它的用途是演示和回归测试，换成 DeepSeek 后由模型自己决定步骤。

## 和生产方案的差距

| 环节 | Demo | 生产（设计文档） |
| --- | --- | --- |
| 业务数据 | 模拟电站，MCP 工具直接读内存数据 | 只读 MCP 服务对接时序库/业务库，Token Exchange 换短期 MCP Token |
| 权限 | `MCP_CALLER_SCOPES` 环境变量 | 来自 Token claims，按资产范围逐行裁剪 |
| 会话 / checkpointer | 内存（InMemorySaver） | PostgreSQL（AsyncPostgresSaver / Store） |
| 案例检索 | 关键词打分 | PGVector |
| 推送 | 生成草稿，配了 `WECOM_WEBHOOK_URL` 才能手动发 | 企业微信 + 升级策略 |
| 告警入口 | 页面上点选 | 订阅平台告警流自动触发 |

## 目录

```
app/
  station_sim.py    模拟电站：拓扑、遥测（一阶热模型）、告警、检修、案例、气象
  mcp_server.py     只读 MCP 服务（FastMCP，stdio / Streamable HTTP）
  agent.py          Deep Agents 组装：主智能体 + 两个子智能体 + 中间件
  answer_guard.py   AnswerGuard 事实校验中间件
  offline_model.py  离线规则模型
  notify.py         企业微信推送草稿 / 发送
  runner.py         把流式输出整理成前端事件
  server.py         FastAPI + SSE
  static/index.html 值班台页面
skills/
  diagnosis/        四类告警的诊断 Skill
  ops/alarm-report  报告与推送格式 Skill
scripts/            命令行运行、生成回放页
tests/              离线回归测试
```

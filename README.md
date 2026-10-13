# 企业税负优化 Agent Harness

[![CI](https://github.com/tk-yuan/tax-agent-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/tk-yuan/tax-agent-harness/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-108%20passed-brightgreen)
![python](https://img.shields.io/badge/python-3.13-blue)

> 一个**从零手写**的多智能体运行时：任务规划、子 Agent 委派、上下文压缩、
> 可插拔中间件、Skills 渐进披露、分层记忆、执行链路追踪。
>
> 不基于 LangGraph 或任何 Agent 框架。原因很直接：**本项目要交付的就是运行时本身**，
> 而框架恰恰把这些替你做了。用框架搭出来的多智能体，被问一句
> "你上下文压缩的触发时机怎么定的"就答不上来。

---

## 它做什么

输入一家企业的画像（从业人数、资产总额、应纳税所得额、资产购置计划），
输出**税负最小的政策组合方案**，并附带政策依据、测算数字与独立复核结论。

```
用户目标
   │
   ├─ 任务规划：拆成任务清单，标注依赖（核查时效 → 设计组合 → 独立复核）
   ├─ 政策检索：从 499 篇政策原文中召回相关条款（bge-m3 向量 + 交叉编码精排）
   ├─ 子 Agent 委派：政策研究员 / 税负测算员 / 独立复核员，各自独立上下文
   ├─ 方案求解：穷举全部合法组合，取累计税额最小者（复用毕设模型）
   ├─ 上下文压缩：大结果落盘，上下文只留摘要与文件路径
   └─ 收工前判定：任务清单未清空、或完成条件未达成 → 不许收工
```

## 数据与测算基础

| | |
|---|---|
| 政策语料 | 499 篇财政部/税务总局政策公告原文 |
| 政策库 | 89 条结构化政策（适用条件、税率、有效期） |
| 企业算例 | **292** 个真实上市公司画像（2021–2025 披露数据） |
| 税负模型 | 8 种政策机制的企业所得税模拟，含政策时效（current / stale 两种口径） |
| 对照基线 | 人工经验法、单期贪心、局部搜索、随机搜索 |

税负模型复用自毕业设计的实验管线——**这里的数字与论文同源**，不是另起一套。

---

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env            # 填 SILICONFLOW_API_KEY 与 DEEPSEEK_API_KEY
```

**CLI 方式**（最快看到结果）：

```bash
python scripts/run.py "帮 size_large_104_920008 这家企业做税负筹划，给出最优方案和节税金额"
```

跑完会打印结论、任务清单、子 Agent 调用记录，并把完整执行链路写进
`traces/<id>.json`（可程序化断言）与 `traces/<id>.html`（浏览器打开看时间线）。

**服务方式**（实时看执行过程 + 在线审批）：

```bash
python scripts/serve.py         # 打开 http://127.0.0.1:8000
```

浏览器里是控制台：左侧实时事件流（模型调用 / 工具调用 / 子 Agent 委派 / 上下文压缩），
右侧任务清单、**审批面板**、落盘产出与运行统计。高风险工具会在这里弹出审批请求，
不批就不往下走。

| 端点 | 说明 |
|---|---|
| `GET /` | 单页控制台 |
| `POST /runs` | 提交任务，返回 `run_id` |
| `GET /runs/{id}/events` | SSE 实时事件流 |
| `POST /runs/{id}/approvals/{aid}` | 提交审批决定 |
| `GET /checkpoints` / `POST /runs/{id}/resume` | 断点列表 / 断点续跑 |
| `GET /tools` | 工具池（含哪些需要审批） |
| `GET /traces/{trace_id}` | 离线链路时间线 |

## 实跑结果

一次完整运行的记录（算例 `size_large_104_920008`，成电光信，177 人，总资产 5.10 亿元，高新技术企业）：

| | |
|---|---|
| 耗时 / 轮数 | 19–28 轮，主流程 `completed`（`HARNESS_MAX_TURNS` 默认 40，实测下限就落在这个区间） |
| 子 Agent | 政策研究员（7 轮完成）、税负测算员（6 轮完成）、独立复核员 |
| 工具调用 | 40–90 次（含委派、检索、测算、落盘） |
| 产出 | 3 份报告落盘至 `workspace/reports/` |

**结论**（穷举 36 种合法组合）：

| 设备 | 购置年 | 处理方式 | 理由 |
|---|---|---|---|
| EQ1 | 2022 | 加速折旧 | 一次性扣除政策窗口为 2024–2027，**该资产不在窗口内** |
| EQ2 | 2023 | 加速折旧 | 同上 |
| EQ3 | 2024 | 加速折旧 | 在窗口内，但加速折旧更优 |
| EQ4 | 2025 | **一次性扣除** | 在窗口内且单位价值 ≤ 500 万 |

- 最优方案累计应纳税额 **31,355,199.77 元**
- 较人工经验法（32,022,280.69 元）**节税 667,080.92 元**
- 较无筹划基线（42,023,500.93 元）节税 1,066.8 万元
- 合规校验通过；`current` 与 `stale` 两种口径结果一致，**证明方案未依赖任何失效政策**

**值得注意的一点**：Agent 自行识别出「一次性扣除政策窗口是 2024–2027，
2022/2023 年购置的资产不适用」——这个约束来自检索到的政策原文与结构化政策库的交叉验证，
而不是被硬编码进提示词。

---

## 架构

```
runtime/            主循环与四大机制
  loop.py           Agent 主循环（s01）—— 只做一件事：推进对话
                    一轮内多个工具调用**并发执行**（保序），单调用不走线程池
  tools.py          工具注册与动态工具池（s02）
  permissions.py    路径白名单 + 工具分级 + 审计留痕（s03，并发安全）
  approval.py       人工审批门：高风险操作阻塞等人批复，超时自动拒绝
  hooks.py          四节点可插拔中间件（s04）
  hooks_impl.py     标准中间件集合（把下面这些能力挂上去）
  middleware.py     缓存 / 重试 / PII 检测 / 耗时统计
tasks/
  todo.py           任务状态机 + 依赖图（s05 / s10）
  subagent.py       子 Agent 委派与最小权限收窄（s06 / s13）
context/
  compaction.py     上下文压缩：溢出落盘 + 历史蒸馏（s08）
  memory.py         分层记忆：衰减 / 强化 / 淘汰（s09）
skills/loader.py    Skills 渐进披露（s07）
observability/
  trace.py          trace_id + span + HTML 时间线 + 实时事件回调
storage/
  store.py          分层存储：短期 KV（断点/会话）+ 长期记忆接口
server/
  service.py        任务管理：后台线程执行 + 事件队列 + 审批跨线程握手
  app.py            FastAPI + ASGI：SSE 事件流、审批端点、单页控制台
plugins/
  mcp.py            MCP 外部工具动态并入（s14）
  workflow.py       确定性编排 + journal 续跑（s16）
  builtin_workflows.py
  goal.py           独立完成度判定器（s17）
tools/
  tax.py            税筹域工具（政策检索 / 税负测算 / 组合优化 / 提交归档）
  builtin.py        虚拟文件系统工具
  retrieval.py      精简检索内核
```

### 虚拟文件系统

工作区不是一个大目录，而是按用途分区——权限、清理策略、可读性都不同：

| 分区 | 用途 |
|---|---|
| `workspace/reports/` | 筹划报告、方案对比等交付物 |
| `workspace/artifacts/` | 超长工具结果的落盘位置（上下文只留摘要与路径） |
| `workspace/logs/` | 执行日志快照与审计片段 |
| `workspace/policies/` | 政策原文摘录、政策组合与时效核查材料 |
| `workspace/memories/` | 跨会话经验沉淀的快照 |
| `workspace/submissions/` | 已提交归档的方案（有对外后果，只增不改） |

## 工程质量

### 补充中间件

除权限与审计外，四节点 Hook 还挂了这些（对应 JD 那句「支持缓存、重试、
日志采集、PII 检测、权限校验、敏感操作拦截和审计上报」）：

| 中间件 | 做法 | 边界 |
|---|---|---|
| **缓存** | 只读 + 参数确定的工具（检索、测算）结果可复用 | **写操作、提交归档、需审批的一律不缓存**——缓存它们等于把第二次调用静默吞掉 |
| **重试** | 工具**抛异常**时指数退避重试（默认 2 次） | 返回错误字符串的不重试——那是业务失败，重试多少次都一样 |
| **PII 检测** | 扫描入参与工具结果，身份证/银行卡/手机号/邮箱打码 | 默认只脱敏不拦截——拦截会让任务直接失败 |
| **耗时统计** | 按工具聚合调用次数与 avg/max 耗时 | — |

### 回归测试

Agent 是概率系统，"能跑"和"跑得对"是两件事。改了提示词、调了 Hook、
换了模型——**单元测试全绿，但 Agent 可能已经不再调用某个关键工具**。

`regression/scenarios.jsonl` 把完整执行链路写成可断言的用例：

```bash
python scripts/run_regression.py            # 跑全部场景
```

断言维度：必须/不能调用哪些工具、退出原因、结论必须包含（或不含）什么、
各类 span 的最低数量、是否出现工具错误、审批门如何响应。

**代价要说清楚**：每个场景是一次真实完整运行（几十轮模型调用），
所以场景要少而精，不能当单元测试天天跑。

**核心设计主张：能力用中间件扩展，不用 if 分支堆叠。**

主循环 `loop.py` 里没有一处"如果开启了记忆就……"。所有机制都挂在四个节点上：

| 节点 | 挂了什么 |
|---|---|
| `before_model` | 注入记忆与任务清单 → 重写 system prompt → 触发上下文压缩 |
| `before_tool` | 权限校验 → **人工审批门** → 审计留痕 |
| `after_tool` | 超长结果落盘，上下文只留摘要与路径 |
| `before_finish` | 目标闭环：任务没做完不许收工 |

## 人工审批：为什么是阻塞的

本域里唯一有对外后果的动作是 **`submit_plan`（方案正式提交归档）**，它被标记为高风险，每次调用都要过审批门。

```
Agent 想调 submit_plan
   → before_tool Hook 命中审批规则
   → 推审批请求给前端（SSE）/ 控制台提问
   → ★ 执行线程阻塞在这里 ★
   → 人点「批准」→ 放行；点「拒绝」或超时 → 工具返回被拒绝，Agent 转而说明原因
```

**阻塞是刻意的**：有后果的动作不能「先做了再补审批」，否则审批只是事后通知，约束不了任何东西。
超时（默认 300 秒）**自动拒绝**——挂死比拒绝糟得多，拒绝是可审计的终态。

审批规则可查（`GET /tools` 会列出哪些工具需要审批），不是黑箱。

## 与教程 s01–s17 的对应

本项目是把 Agent 运行时的各个机制**装进同一个循环**（即教程 s15 的形态），
并在税筹域上跑通：

| 教程 | 本项目 | 落点 |
|---|---|---|
| s01 主循环 | ✅ | `runtime/loop.py` |
| s02 工具分发 | ✅ | `runtime/tools.py` |
| s03 权限 | ✅ | `runtime/permissions.py` |
| s04 Hooks | ✅ | `runtime/hooks.py` + `hooks_impl.py` |
| s05 任务清单 | ✅ | `tasks/todo.py` |
| s06 子 Agent | ✅ | `tasks/subagent.py` |
| s07 Skills | ✅ | `skills/loader.py` + `skills/*/SKILL.md` |
| s08 上下文压缩 | ✅ | `context/compaction.py` |
| s09 记忆 | ✅ | `context/memory.py` |
| s10 任务系统 | ✅ | `tasks/todo.py`（依赖图） |
| s11 后台任务 | ⬜ | 未做——本项目无长驻任务需求 |
| s12 cron 调度 | ⬜ | 未做——政策巡检留作扩展 |
| s13 Agent 团队 | ✅ | `tasks/subagent.py`（三角色） |
| s14 MCP | ✅ | `plugins/mcp.py` |
| s15 集成 | ✅ | `assemble.py`（这就是目标形态） |
| s16 Workflow | ✅ | `plugins/workflow.py` |
| s17 Goal Loop | ✅ | `plugins/goal.py` + `before_finish` Hook |

**未做的两讲是刻意的**：s11 后台任务、s12 cron 调度在本域里没有真实需求，
硬塞进来只会多两个讲不出所以然的模块。宁可少两块，不要说不清。

## 设计取舍

详见 [docs/DESIGN.md](docs/DESIGN.md)。几条要点：

- **手写运行时而非用框架**：控制上下文压缩的触发时机、Hook 的注入点、中断恢复，
  这些正是框架替你决定的部分。
- **子 Agent 的核心价值是上下文隔离**：子 Agent 在自己的上下文里翻几十页政策，
  主 Agent 只收到结论。不这么做，主上下文二十轮内就被淹。
- **记忆不是"做了个向量检索"**：实现了衰减、强化、淘汰三个机制，
  否则"记忆"只是个越写越大的文件。
- **压缩的裁剪点必须落在组边界**：从 `assistant.tool_calls` 与 `tool` 结果之间切开，
  下一次请求会被 API 直接拒绝。这个 bug 是被单元测试抓出来的。
- **Workflow 不允许模型提交代码**：模型只能给流程名、参数和续跑 id。
  否则这就是一个远程代码执行漏洞。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | **设计取舍**：每个模块为什么这么做、替代方案是什么、代价在哪 |
| [docs/LEARNING.md](docs/LEARNING.md) | **学习指南**：按面试问题组织——「上下文压缩怎么做的」「记忆凭什么算记忆」 |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | **术语速查**：简历/代码里每个词的三段式（是什么 → 为什么用它 → 面试怎么答） |
| [regression/scenarios.jsonl](regression/scenarios.jsonl) | 回归场景：把完整执行链路写成可断言的用例 |

## 测试

```bash
python -m pytest tests/ -q          # 108 个单元测试
python scripts/run_regression.py    # 3 个端到端回归场景（真实完整运行，耗时）

覆盖：工具池权限收窄、Hook 的拒绝语义与异常处理、路径越界拦截、
任务依赖阻塞、记忆衰减/强化/淘汰、压缩的组边界安全。

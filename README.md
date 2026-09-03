# kaggle-agent

一个 Kaggle 长周期战役的 **harness 仓库** —— 沉淀的是「让 agent 把事情做完的脚手架」，
而不是一个能自己跑比赛的框架。

三场真实战役跑下来的结论是：**杠杆从来不在模型代码里**。它在能跨 session 存活的规则、
让独立产出自由组合的契约、把笔记本变成 fleet 的远端执行器，以及一个拒绝被公榜欺骗的
诚实判据。

## 战绩与记录

| 比赛 | 结果 | harness 记录 |
|---|---|---|
| Playground S6E6 · Predicting Stellar Class | **private 4th / 2817**（0.97060） | [`competitions/playground-series-s6e6/HARNESS.md`](competitions/playground-series-s6e6/HARNESS.md) |
| NeuroGolf 2026 · ONNX code golf | **247 → 7625.77**（400 targets，禁过拟合 + 禁 exploit） | [`competitions/neurogolf-2026/HARNESS.md`](competitions/neurogolf-2026/HARNESS.md) |
| Multi-Step Tool Attacks · red-team | 见目录 | [`competitions/multi-step-tool-attacks/README.md`](competitions/multi-step-tool-attacks/README.md) |

这些记录写的是**过程中 harness 在哪一刻起了作用、改变了哪个数字**，不是组件说明书。
每场比赛一份、按比赛作用域命名，避免并行 session 在同一路径上互相覆盖。

## 仓库里真正在用的东西

| 位置 | 是什么 | 凭什么留下 |
|---|---|---|
| [`CLAUDE.md`](CLAUDE.md) | 规则层 | 唯一保证每个 session 开局都在 context 里的文本。踩过一次以上的坑当天写进去 —— P100 是 sm_60、cuDF 已弃 Pascal、GPU 并发上限 2、slug 中毒 |
| [`harness/`](harness/) | 跨比赛通用工具 | `experience_db.py`（证据分级账本 + 后验奖励调度）与四段第三方产物审计流水线 |
| [`src/kaggle_agent/tools/kernel_fleet.py`](src/kaggle_agent/tools/kernel_fleet.py) | Kaggle kernel fleet 驱动 | 唯一从实战反向沉淀回框架层的模块。GPU 配额调度、slug 自动换号、产物 shape 校验重拉、6 条失败签名诊断 |
| [`.claude/skills/`](.claude/skills/) | 协议层 | `agent-field-lessons`（11 条 fleet 教训）、`deli-auto-research`（长周期防停滞） |
| `competitions/<slug>/` | 各比赛的流水线与记录 | 实验日志、gap 分析、kernel 定义 |

## 实际怎么工作

不是 `kagent run`。真实的循环是 **Claude Code 直接驱动比赛目录下的脚本，由 `CLAUDE.md`
和各比赛的 `CURSOR.md` 约束**：

```bash
# 以 S6E6 为例：改完训练代码 → 同步到 Kaggle 代码 dataset
bash kaggle/sync_code.sh "描述"

# 远端训练（本地绝不训练：577k 行单次 fit 在笔记本上要几小时）
bash kaggle/run_remote.sh          # push → 轮询 → 拉回 artifacts/*.npy

# 本地只做 stacking（纯 numpy，秒级）与提交
python stack.py --models lgb_multi,xgb_multi,cat_multi,... --output submissions/stack.csv
kaggle competitions submit -c playground-series-s6e6 -f submissions/stack.csv -m "描述"
```

### ⛔ 硬规则：活跃比赛不入库

仓库的 `origin` 是**公开** GitHub remote。活跃比赛的解法一律不 commit / push，直到比赛结束。
详见 [`CLAUDE.md`](CLAUDE.md) 顶部的 Git policy。

## `src/kaggle_agent/` 的现状

本仓库同名的自主 agent 框架（`Orchestrator` 状态机、Cursor 文件交接协议、
`PlaybookManager` / `SkillManager` / `ReflectionEngine`，以及 `config.yaml`、`knowledge/`、
`kagent` 命令）**未被任何一场比赛使用过** —— 三场比赛的 `agent_tasks/` 与 `experiments/`
均为空，从未生成 `state.json`，且 `competitions/` 下无任何文件 import 它。

保留代码（既有资产，非比赛产出），此处仅作说明。唯一从实战反向沉淀回来的是
`tools/kernel_fleet.py`，而它是在它所替代的 bash 已经在生产里挂掉之后才写的 —— 这正是
整个 harness 的模式：**起作用的部件都是事后为一次具体失败而写的；照规格事前设计的那些
没被用上。**

## 目录结构

```
kaggle-agent/
├── CLAUDE.md                    # 规则层（最重要的单个文件）
├── harness/                     # 跨比赛通用工具
│   ├── experience_db.py         #   证据分级账本 + 调度器
│   └── README.md                #   四段审计流水线
├── competitions/
│   ├── playground-series-s6e6/  #   HARNESS.md · CURSOR.md · LEADER_GAP_ANALYSIS.md · 流水线
│   ├── neurogolf-2026/          #   HARNESS.md · findings/ · harness/verify_scoring.py
│   └── multi-step-tool-attacks/ #   README.md · FINDINGS.md · harness/eval_replay.py
├── .claude/skills/              # agent-field-lessons · deli-auto-research
├── src/kaggle_agent/            # 见上节：仅 tools/ 在用
├── knowledge/                   # playbooks + skills（其加载器未启用）
└── tests/
```

## 安装与测试

```bash
pip install -e .            # 运行时依赖
pip install -e ".[dev]"     # 外加 pytest / ruff / black / mypy

pytest                      # 全部
pytest tests/unit/          # 仅单元测试
ruff check . && black .
```

注：`tests/` 覆盖的是上节所述那个未启用的框架。`kernel_fleet.py` 目前**没有测试**，
尽管它的 docstring 声称纯逻辑部分是为「无需 Kaggle 凭据即可测试」而拆分的。

## 设计文档

[docs/superpowers/specs/2025-06-10-kaggle-agent-design.md](docs/superpowers/specs/2025-06-10-kaggle-agent-design.md)
—— 上节所述那个未启用框架的原始设计。保留作为历史记录。

## License

MIT

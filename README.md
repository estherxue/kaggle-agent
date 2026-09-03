# Kaggle Agent

一个能自动参加 Kaggle 比赛、从实战中持续积累经验并不断进化的 agent。

## 核心特性

- **全自动比赛**：从下载数据到提交结果的端到端自动化
- **人工指导注入**：用户可随时插话给建议，指导被采纳后沉淀为经验
- **双层经验系统**：文本 Playbook（战略层）+ 代码技能库（战术层）
- **可插拔 LLM**：支持 OpenRouter、Ollama 等多种 provider

## 快速开始

### 1. 安装

```bash
pip install -e .
```

### 2. 配置

复制 `config.yaml` 并根据需要修改，设置环境变量：

```bash
export OPENAI_API_KEY="your-api-key"
# 可选：Kaggle API 认证
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-key"
```

### 3. 运行

```bash
# 开始跑比赛
kagent run titanic

# 随时插入指导
kagent guide titanic "试试 LightGBM"

# 查看状态
kagent status titanic
```

## 项目结构

```
kaggle-agent/
├── src/kaggle_agent/     # 核心代码
├── knowledge/            # 经验库（可编辑！）
│   ├── playbooks/        # 文本经验
│   └── skills/           # 可复用代码
├── competitions/         # 比赛工作区
└── tests/               # 测试
```

## Harness 记录

每场比赛的 harness 复盘写在各自目录下，**按比赛作用域命名**，避免并行会话在同一路径上冲突：

| 比赛 | 记录 |
|---|---|
| Playground S6E6（Predicting Stellar Class，private 4th/2817） | [`competitions/playground-series-s6e6/HARNESS.md`](competitions/playground-series-s6e6/HARNESS.md) |
| NeuroGolf 2026（247 → 7625.77） | [`competitions/neurogolf-2026/HARNESS.md`](competitions/neurogolf-2026/HARNESS.md) |
| Multi-Step Tool Attacks | [`competitions/multi-step-tool-attacks/README.md`](competitions/multi-step-tool-attacks/README.md) |

写的是**过程中 harness 在哪一刻起了什么作用、改变了哪个数字**，不是组件说明书。
跨比赛可复用的通用工具在 [`harness/`](harness/)（evidence-graded ledger + 四段审计流水线）。

> **`src/kaggle_agent/` 的现状**：本仓库同名的自主 agent 框架（`Orchestrator` 状态机、
> Cursor 文件交接协议、`PlaybookManager`/`SkillManager`/`ReflectionEngine`）**未被任何一场
> 比赛使用过** —— 三场比赛的 `agent_tasks/` 与 `experiments/` 均为空，从未生成 `state.json`，
> 且 `competitions/` 下无任何文件 import 它。实际起作用的是 Claude Code 直接驱动脚本、由
> `CLAUDE.md` 约束。唯一从实战反向沉淀回框架层的是 `tools/kernel_fleet.py`。
> 代码保留（既有资产，非本次比赛产出），此处仅作说明。

## 设计文档

详见 [docs/superpowers/specs/2025-06-10-kaggle-agent-design.md](docs/superpowers/specs/2025-06-10-kaggle-agent-design.md)

## License

MIT

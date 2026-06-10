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

## 设计文档

详见 [docs/superpowers/specs/2025-06-10-kaggle-agent-design.md](docs/superpowers/specs/2025-06-10-kaggle-agent-design.md)

## License

MIT

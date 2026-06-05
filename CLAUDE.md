# MindFlow — CLAUDE.md

## 运行环境

- Python 虚拟环境: `/home/max-rayyy/ai-workspace/.venv`
- 所有命令需先激活: `source /home/max-rayyy/ai-workspace/.venv/bin/activate`
- Python 版本: 3.10+
- 工作目录: `/home/max-rayyy/ai-workspace/mindflow`

## 项目信息

- 名称: MindFlow — 个人认知操作系统
- 目标: 学习 LangChain + Function Calling + Memory，构建日常可用的社交媒体知识 Agent
- 技术栈: LangGraph, Chroma, SQLite, DeepSeek API, python-telegram-bot

## 开发流程: SDD (Specification-Driven Development)

本项目采用规格驱动开发，核心原则：

1. **文档即真相** — `docs/superpowers/specs/` 中的设计规格 + `docs/superpowers/plans/` 中的实现计划是唯一真相来源
2. **代码与文档同步** — 当用户做出技术决策变更（如选用 DeepSeek、切换模型等），必须同步更新相关文档
3. **无上下文时先读文档** — 每次新会话开始，若无上下文，先读取 `docs/superpowers/` 下的规格和计划，了解当前状态
4. **任务完成标记** — 每完成一个 Task/Phase，在实现计划中将对应步骤标记为 `[x]`（已完成）

## 文档结构

```
docs/superpowers/
├── specs/
│   └── 2025-06-05-mindflow-design.md    # 设计规格（产品定义/架构/数据模型）
└── plans/
    └── 2025-06-05-mindflow-phase1.md     # Phase 1 实现计划（10 个 Task）
```

## 当前进度

- **Phase 1: Digest MVP** 进行中
  - [x] Task 1: 项目初始化与环境配置
  - [x] Task 2: LLM 客户端封装
  - [x] Task 3: 解析器系统（基类 + 路由）
  - [ ] Task 4: 知乎 & 公众号解析器 ← 当前

## 已做出的技术决策（与原始规格的不同之处）

| 决策 | 原计划 | 实际选择 | 原因 |
|------|--------|---------|------|
| LLM 提供商 | OpenAI | **DeepSeek** | 国内访问快、便宜、支持 Function Calling |
| 对话模型 | gpt-4o-mini | **deepseek-v4-flash** | 用户指定 |
| Embedding 模型 | text-embedding-3-small | **本地 sentence-transformers** | DeepSeek 不提供 Embedding API |
| Bot 平台 | Telegram（暂定） | **Telegram** | 微信个人号不可行 |

## 提交规范

- 提交信息用中文或英文皆可
- 使用约定式提交格式: `feat:` / `fix:` / `chore:` / `docs:` / `refactor:`
- **不要在 commit message 中加 `Co-Authored-By` 或其他署名**

## 项目结构

```
src/
├── config.py          # 全局配置 (Settings dataclass)
├── llm/client.py      # DeepSeek API 封装 (chat / function_call / embedding)
├── parsers/           # 多平台内容解析器（base + router + zhihu + wechat + generic）
├── memory/            # 记忆系统 (Chroma + SQLite)
├── tools/             # LangChain Function Calling Tools
├── agents/digest/     # LangGraph Agent 工作流
└── bot/               # Telegram Bot
```

## API 配置

- LLM 提供商: DeepSeek (`api.deepseek.com`)
- 模型: `deepseek-v4-flash`
- Embedding: 本地 sentence-transformers（DeepSeek 不提供 Embedding API）

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

## 提交规范

- 提交信息用中文或英文皆可
- 使用约定式提交格式: `feat:` / `fix:` / `chore:` / `docs:` / `refactor:`
- **不要在 commit message 中加 `Co-Authored-By` 或其他署名**

## 项目结构

```
src/
├── config.py          # 全局配置 (Settings dataclass)
├── llm/client.py      # DeepSeek API 封装 (chat / function_call / embedding)
├── parsers/           # 多平台内容解析器
├── memory/            # 记忆系统 (Chroma + SQLite)
├── tools/             # LangChain Function Calling Tools
├── agents/digest/     # LangGraph Agent 工作流
└── bot/               # Telegram Bot
```

## API 配置

- LLM 提供商: DeepSeek (`api.deepseek.com`)
- 模型: `deepseek-v4-flash`
- Embedding: 本地 sentence-transformers（DeepSeek 不提供 Embedding API）

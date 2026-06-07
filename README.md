# 🧠 灵犀（MindFlow）— 个人认知操作系统

> 把你每天刷到的各种内容，自动消化成你的「第二大脑」——可检索、可对比、可主动提醒。

Bot: [@mind_lingxi_bot](https://t.me/mind_lingxi_bot)

## 这是什么

灵犀帮助你收集碎片化的社交媒体阅读（知乎、公众号、微博、小红书、B站等），自动提取观点、打标签、建立知识关联，最终在正确的时机主动把知识推送到你面前。

**目前 Phase 1：Digest MVP** — 通过 Telegram Bot 发送链接，自动解析并返回知识卡片。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key 和 Bot Token
```

需要配置：

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | LLM API Key（支持 OpenAI 兼容 API，如 DeepSeek） |
| `OPENAI_BASE_URL` | API 地址（DeepSeek: `https://api.deepseek.com`） |
| `LLM_MODEL` | 使用的模型（如 `deepseek-v4-flash`） |
| `TELEGRAM_BOT_TOKEN` | 从 @BotFather 获取的 Bot Token |

### 3. 启动

```bash
python -m src.bot.main
```

### 4. 使用

在 Telegram 中给 [@mind_lingxi_bot](https://t.me/mind_lingxi_bot) 发送任意文章链接，如：

- 知乎问题/专栏：`https://www.zhihu.com/question/...`
- 微信公众号：`https://mp.weixin.qq.com/s/...`
- 微博、小红书、B站等

Bot 会自动解析并返回知识卡片，包含：
- 📌 标题
- 🏷️ 自动标签
- 🧠 200字摘要
- 💡 核心观点 + 论证链 + 论据
- 📊 立场判断（支持/反对/中立/混合）

## 项目结构

```
src/
├── config.py              # 全局配置管理
├── llm/                   # LLM 客户端（DeepSeek API 封装）
│   └── client.py          # chat / function_call / embedding
├── parsers/               # 多平台内容解析器
│   ├── base.py            # 抽象基类 + ParsedArticle
│   ├── router.py          # URL 正则匹配 → 平台分发
│   ├── zhihu.py           # 知乎解析器
│   ├── wechat.py          # 微信公众号解析器
│   └── generic.py         # 通用解析器（Jina Reader + Readability 兜底）
├── memory/                # 三层记忆系统
│   ├── vector_store.py    # Chroma 向量存储（语义搜索）
│   └── metadata_store.py  # SQLite 元数据存储（精确查询）
├── tools/                 # Function Calling Tools
│   ├── extract_viewpoint.py  # 观点提取（摘要+观点+立场+标签）
│   └── generate_tags.py      # 轻量标签生成
├── agents/digest/         # LangGraph Digest Agent
│   ├── state.py           # 工作流 State 定义
│   ├── nodes.py           # 五个节点实现
│   └── graph.py           # StateGraph 组装 + 条件路由
└── bot/                   # Telegram Bot
    ├── handlers.py        # 消息处理器（/start, /help, /stats, 链接）
    └── main.py            # Bot 启动入口
```

## 工作流

```
用户发链接 → URL识别 → 平台解析 → LLM观点提取 → 向量存储 → 知识卡片
                │           │            │              │
                └─正则匹配    ├─知乎       ├─摘要          ├─SQLite（元数据）
                            ├─公众号      ├─观点          └─Chroma（向量）
                            ├─微博        ├─立场
                            └─通用兜底    └─标签
```

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| Agent 框架 | **LangGraph** | 有状态的图工作流（节点 + 条件路由） |
| LLM 工具 | **LangChain Tools + Function Calling** | 结构化观点提取 |
| LLM 服务 | **DeepSeek** (`deepseek-v4-flash`) | 对话 + 结构化输出 |
| Embedding | **BGE-small-zh**（本地） | 中文语义向量化 |
| 向量数据库 | **Chroma** | 语义搜索 |
| 元数据存储 | **SQLite** | 精确查询 + 用户偏好 |
| 内容抓取 | **httpx + BeautifulSoup + Jina Reader** | 多平台内容提取 |
| 交互界面 | **python-telegram-bot** | Telegram Bot API |

## Phase 1 完成情况

- [x] Task 1: 项目初始化与环境配置
- [x] Task 2: LLM 客户端封装
- [x] Task 3: 解析器系统（基类 + 路由）
- [x] Task 4: 知乎 & 公众号解析器
- [x] Task 5: 通用解析器（兜底方案）
- [x] Task 6: 记忆系统（向量存储 + 元数据存储）
- [x] Task 7: Function Calling Tools（观点提取 + 标签生成）
- [x] Task 8: LangGraph Digest Agent（状态 + 节点 + 图）
- [x] Task 9: Telegram Bot
- [x] Task 10: 集成验证与 README

## License

MIT

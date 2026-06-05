# MindFlow Phase 1: Digest MVP 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 Telegram Bot，用户发送链接 → 自动解析内容 → LLM 提取观点/标签/立场 → 存入向量库 → 返回知识卡片

**Architecture:** LangGraph 编排五步流水线（URL路由 → 内容解析 → 观点提取 → 向量存储 → 卡片回复）。三层记忆：Checkpoint（短期）+ Chroma（中期）+ SQLite Preference（长期）。LangChain Tools 封装 Function Calling 实现结构化观点提取。

**Tech Stack:** Python 3.10+, LangGraph, LangChain, Chroma, SQLite, httpx, BeautifulSoup, python-telegram-bot, OpenAI 兼容 API

---

## 文件职责总览

| 文件 | 职责 | 大小预期 |
|------|------|---------|
| `src/config.py` | 加载 .env 和 YAML 配置 | ~40 行 |
| `src/llm/client.py` | OpenAI 兼容 API 封装，支持 function calling | ~50 行 |
| `src/parsers/base.py` | 解析器抽象基类 + ParsedArticle 数据类 | ~30 行 |
| `src/parsers/router.py` | URL 正则匹配 → 分发到对应解析器 | ~30 行 |
| `src/parsers/zhihu.py` | 知乎专用解析器 | ~60 行 |
| `src/parsers/wechat.py` | 微信公众号解析器 | ~50 行 |
| `src/parsers/generic.py` | 通用解析器（Jina Reader + BS4 兜底） | ~40 行 |
| `src/memory/vector_store.py` | Chroma 集合管理 + 向量操作 | ~50 行 |
| `src/memory/metadata_store.py` | SQLite 元数据 CRUD | ~60 行 |
| `src/tools/extract_viewpoint.py` | 观点提取 Function Calling Tool | ~60 行 |
| `src/tools/generate_tags.py` | 标签生成 Tool | ~30 行 |
| `src/agents/digest/state.py` | LangGraph State TypedDict | ~20 行 |
| `src/agents/digest/nodes.py` | 各节点实现函数 | ~150 行 |
| `src/agents/digest/graph.py` | LangGraph StateGraph 组装 | ~60 行 |
| `src/bot/handlers.py` | Telegram 消息处理 | ~80 行 |
| `src/bot/main.py` | Bot 入口启动 | ~20 行 |

---

### Task 1: 项目初始化与环境配置

**目标：** 搭建项目骨架，安装依赖，创建配置文件

**Files:**
- Create: `mindflow/requirements.txt`
- Create: `mindflow/.env.example`
- Create: `mindflow/.gitignore`
- Create: `mindflow/src/__init__.py`
- Create: `mindflow/src/config.py`
- Create: `mindflow/src/agents/__init__.py`
- Create: `mindflow/src/agents/digest/__init__.py`
- Create: `mindflow/src/agents/brain/__init__.py`
- Create: `mindflow/src/parsers/__init__.py`
- Create: `mindflow/src/memory/__init__.py`
- Create: `mindflow/src/tools/__init__.py`
- Create: `mindflow/src/bot/__init__.py`
- Create: `mindflow/src/llm/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```bash
cat > /home/max-rayyy/ai-workspace/mindflow/requirements.txt << 'EOF'
# Agent 框架
langgraph>=1.2.0
langchain>=1.3.0
langchain-core>=1.4.0

# LLM
openai>=2.0.0

# 向量数据库
chromadb>=0.5.0

# 内容抓取
httpx>=0.28.0
beautifulsoup4>=4.12.0
readability-lxml>=0.8.0

# Telegram Bot
python-telegram-bot>=21.0

# 配置与工具
python-dotenv>=1.0.0
pyyaml>=6.0

# 测试
pytest>=8.0.0
pytest-asyncio>=0.24.0
EOF
```

- [ ] **Step 2: 安装依赖**

```bash
cd /home/max-rayyy/ai-workspace && source .venv/bin/activate && pip install -r mindflow/requirements.txt
```

预期：所有包安装成功，无报错

- [ ] **Step 3: 创建 .env.example**

```bash
cat > /home/max-rayyy/ai-workspace/mindflow/.env.example << 'EOF'
# LLM 配置
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Telegram Bot
TELEGRAM_BOT_TOKEN=your-bot-token-here

# 数据目录
DATA_DIR=./data
EOF
```

- [ ] **Step 4: 创建 .gitignore**

```bash
cat > /home/max-rayyy/ai-workspace/mindflow/.gitignore << 'EOF'
.env
data/
__pycache__/
*.pyc
.pytest_cache/
EOF
```

- [ ] **Step 5: 创建所有 __init__.py 文件**

```bash
touch /home/max-rayyy/ai-workspace/mindflow/src/__init__.py
touch /home/max-rayyy/ai-workspace/mindflow/src/agents/__init__.py
touch /home/max-rayyy/ai-workspace/mindflow/src/agents/digest/__init__.py
touch /home/max-rayyy/ai-workspace/mindflow/src/agents/brain/__init__.py
touch /home/max-rayyy/ai-workspace/mindflow/src/parsers/__init__.py
touch /home/max-rayyy/ai-workspace/mindflow/src/memory/__init__.py
touch /home/max-rayyy/ai-workspace/mindflow/src/tools/__init__.py
touch /home/max-rayyy/ai-workspace/mindflow/src/bot/__init__.py
touch /home/max-rayyy/ai-workspace/mindflow/src/llm/__init__.py
```

- [ ] **Step 6: 编写 src/config.py**

```python
"""全局配置管理。从 .env 和 config.yaml 加载配置。"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LLMConfig:
    """LLM 服务配置"""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))


@dataclass
class BotConfig:
    """Telegram Bot 配置"""
    token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))


@dataclass
class StorageConfig:
    """存储配置"""
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_DIR", "data"))
    chroma_dir: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_DIR", "data") / "chroma")
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_DIR", "data") / "mindflow.db")

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """全局配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


# 全局单例
settings = Settings()
```

- [ ] **Step 7: 验证配置能正常加载**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && cp .env.example .env && source ../../.venv/bin/activate && python -c "from src.config import settings; print('LLM model:', settings.llm.model); print('Data dir:', settings.storage.data_dir)"
```

预期：打印出 LLM model 和 data dir 路径

- [ ] **Step 8: Commit**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && git add -A && git commit -m "chore: project setup with config, env, and dependencies

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: LLM 客户端封装

**目标：** 封装 OpenAI 兼容 API 调用，支持普通对话和 Function Calling

**Files:**
- Create: `mindflow/src/llm/client.py`
- Create: `mindflow/tests/test_llm/__init__.py`
- Create: `mindflow/tests/test_llm/test_client.py`

- [ ] **Step 1: 编写 src/llm/client.py**

```python
"""LLM 客户端封装。提供统一的 OpenAI 兼容 API 调用接口。"""
from openai import OpenAI
from src.config import settings

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """获取全局 LLM 客户端单例"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )
    return _client


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str:
    """发送对话请求，返回文本回复。

    Args:
        messages: OpenAI 格式的消息列表 [{"role": "...", "content": "..."}]
        model: 模型名称，默认使用配置中的 model
        temperature: 生成温度
        max_tokens: 最大 token 数

    Returns:
        LLM 的文本回复
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model or settings.llm.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def function_call(
    messages: list[dict],
    tools: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
) -> dict:
    """发送带 function calling 的请求，返回结构化的 tool call 参数。

    Args:
        messages: OpenAI 格式的消息列表
        tools: OpenAI 格式的 tool 定义列表
        model: 模型名称
        temperature: 生成温度

    Returns:
        解析后的 tool call 参数字典

    Raises:
        ValueError: 如果 LLM 没有返回 tool call
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model or settings.llm.model,
        messages=messages,
        tools=tools,
        temperature=temperature,
    )

    choice = response.choices[0]
    if choice.message.tool_calls:
        import json
        return json.loads(choice.message.tool_calls[0].function.arguments)

    # 如果没有 tool call，尝试从 content 中解析 JSON
    content = choice.message.content or ""
    if content.strip().startswith("{"):
        import json
        return json.loads(content)

    raise ValueError(f"LLM 未返回 function call。原始回复: {content[:200]}")


def get_embedding(text: str, model: str | None = None) -> list[float]:
    """获取文本的向量嵌入。

    Args:
        text: 要向量化的文本
        model: embedding 模型名称

    Returns:
        浮点数向量列表
    """
    client = get_client()
    response = client.embeddings.create(
        model=model or settings.llm.embedding_model,
        input=text,
    )
    return response.data[0].embedding
```

- [ ] **Step 2: 编写测试 tests/test_llm/test_client.py**

```python
"""LLM 客户端测试（需要真实 API key，可作为集成测试跳过）"""
import pytest
from src.llm.client import chat, function_call, get_embedding, get_client


@pytest.mark.skip(reason="需要真实 API key")
def test_chat_returns_string():
    result = chat([
        {"role": "user", "content": "用一句话回复：你好"}
    ])
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.skip(reason="需要真实 API key")
def test_function_call_returns_dict():
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "description": "回显输入",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    }]
    result = function_call(
        [{"role": "user", "content": "请调用 echo 工具，text 是 hello"}],
        tools,
    )
    assert isinstance(result, dict)
    assert "text" in result


def test_get_client_returns_singleton():
    c1 = get_client()
    c2 = get_client()
    assert c1 is c2
```

- [ ] **Step 3: Commit**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && git add -A && git commit -m "feat: add LLM client with chat, function_call, and embedding support

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 解析器系统（基类 + 路由）

**目标：** 定义解析器接口，实现 URL 路由逻辑，这是 Function Calling 学习的第一步

**Files:**
- Create: `mindflow/src/parsers/base.py`
- Create: `mindflow/src/parsers/router.py`
- Create: `mindflow/tests/test_parsers/__init__.py`
- Create: `mindflow/tests/test_parsers/test_router.py`

- [ ] **Step 1: 编写 src/parsers/base.py**

```python
"""解析器抽象基类。所有平台解析器必须继承 BaseParser。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ParsedArticle:
    """解析后的统一文章结构"""
    title: str
    author: str = "未知"
    content: str = ""
    platform: str = "generic"
    metadata: dict = field(default_factory=dict)
    parse_method: str = "unknown"

    def to_text(self) -> str:
        """转为 LLM 可读的文本表示"""
        return f"标题：{self.title}\n作者：{self.author}\n平台：{self.platform}\n\n正文：\n{self.content}"


class BaseParser(ABC):
    """解析器基类，所有平台解析器继承此类"""

    # 子类必须定义的平台标识
    platform_name: str = "generic"

    @abstractmethod
    def can_parse(self, url: str) -> bool:
        """判断能否解析该 URL"""
        ...

    @abstractmethod
    async def parse(self, url: str) -> ParsedArticle:
        """解析 URL 返回 ParsedArticle

        Raises:
            ParseError: 解析失败时抛出
        """
        ...


class ParseError(Exception):
    """解析错误"""
    def __init__(self, message: str, platform: str = "", url: str = ""):
        self.platform = platform
        self.url = url
        super().__init__(message)
```

- [ ] **Step 2: 编写 src/parsers/router.py**

```python
"""URL 路由器。根据 URL 模式匹配，分发到对应的平台解析器。"""
import re
from src.parsers.base import BaseParser, ParsedArticle, ParseError

# URL 模式规则表
PLATFORM_PATTERNS: list[tuple[str, str]] = [
    ("zhihu", r"zhihu\.com/(question/\d+|answer/\d+|p/\d+)"),
    ("zhihu", r"zhuanlan\.zhihu\.com/p/\d+"),
    ("wechat", r"mp\.weixin\.qq\.com/s/"),
    ("weibo", r"(weibo\.com/|m\.weibo\.cn/)"),
    ("xiaohongshu", r"(xhslink\.com/|xiaohongshu\.com/)"),
    ("bilibili", r"bilibili\.com/(read/|video/)"),
]


def detect_platform(url: str) -> str:
    """识别 URL 所属平台

    Args:
        url: 文章链接

    Returns:
        平台标识字符串: zhihu/wechat/weibo/xiaohongshu/bilibili/generic
    """
    for platform_name, pattern in PLATFORM_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return platform_name
    return "generic"


class URLRouter:
    """URL 路由器。持有所有解析器实例，负责 URL → 解析器的分发。"""

    def __init__(self):
        self._parsers: dict[str, BaseParser] = {}

    def register(self, parser: BaseParser) -> None:
        """注册一个解析器"""
        self._parsers[parser.platform_name] = parser

    def get_parser(self, platform: str) -> BaseParser | None:
        """根据平台标识获取解析器"""
        return self._parsers.get(platform)

    def route(self, url: str) -> str:
        """识别 URL 所属平台

        Returns:
            平台标识字符串，保证至少返回 'generic'
        """
        return detect_platform(url)
```

- [ ] **Step 3: 编写测试 tests/test_parsers/test_router.py**

```python
"""URL 路由测试"""
import pytest
from src.parsers.router import detect_platform, URLRouter


class TestDetectPlatform:
    def test_zhihu_question(self):
        assert detect_platform("https://www.zhihu.com/question/12345678") == "zhihu"

    def test_zhihu_zhuanlan(self):
        assert detect_platform("https://zhuanlan.zhihu.com/p/12345678") == "zhihu"

    def test_wechat(self):
        assert detect_platform("https://mp.weixin.qq.com/s/abc123def") == "wechat"

    def test_weibo(self):
        assert detect_platform("https://weibo.com/1234567890/abcdef") == "weibo"

    def test_weibo_mobile(self):
        assert detect_platform("https://m.weibo.cn/detail/1234567890") == "weibo"

    def test_xiaohongshu(self):
        assert detect_platform("https://www.xiaohongshu.com/explore/abc123") == "xiaohongshu"

    def test_bilibili_read(self):
        assert detect_platform("https://www.bilibili.com/read/cv12345678") == "bilibili"

    def test_generic_url(self):
        assert detect_platform("https://example.com/article/hello-world") == "generic"

    def test_empty_url(self):
        assert detect_platform("") == "generic"


class TestURLRouter:
    def test_route_returns_platform(self):
        router = URLRouter()
        result = router.route("https://mp.weixin.qq.com/s/abc")
        assert result == "wechat"

    def test_get_parser_returns_none_when_not_registered(self):
        router = URLRouter()
        assert router.get_parser("zhihu") is None
```

- [ ] **Step 4: 运行测试验证**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && source ../../.venv/bin/activate && python -m pytest tests/test_parsers/test_router.py -v
```

预期：全部 10 个测试通过

- [ ] **Step 5: Commit**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && git add -A && git commit -m "feat: add parser base class and URL router with platform detection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 知乎 & 微信公众号解析器

**目标：** 实现两个最常见的国内平台解析器，验证解析器架构

**Files:**
- Create: `mindflow/src/parsers/zhihu.py`
- Create: `mindflow/src/parsers/wechat.py`
- Create: `mindflow/tests/test_parsers/test_zhihu.py`
- Create: `mindflow/tests/test_parsers/test_wechat.py`

- [ ] **Step 1: 编写 src/parsers/zhihu.py**

```python
"""知乎解析器。支持知乎问答和专栏文章。"""
import httpx
from bs4 import BeautifulSoup
from src.parsers.base import BaseParser, ParsedArticle, ParseError


class ZhihuParser(BaseParser):
    platform_name = "zhihu"

    def can_parse(self, url: str) -> bool:
        return "zhihu.com" in url

    async def parse(self, url: str) -> ParsedArticle:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html,application/xhtml+xml",
                    },
                    follow_redirects=True,
                )
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # 提取标题
            title = ""
            title_tag = (
                soup.select_one("h1.QuestionHeader-title")
                or soup.select_one("h1.Post-Title")
                or soup.select_one("h1")
            )
            if title_tag:
                title = title_tag.get_text(strip=True)

            # 提取正文 — 知乎的回答/文章在 .RichText 或 .Post-RichText 中
            content_parts = []
            content_blocks = soup.select(".RichText, .Post-RichText, .RichContent-inner")
            for block in content_blocks:
                text = block.get_text(separator="\n", strip=True)
                if text:
                    content_parts.append(text)

            content = "\n\n".join(content_parts)

            if not content:
                raise ParseError("未能提取到知乎内容", platform="zhihu", url=url)

            return ParsedArticle(
                title=title or "知乎文章",
                author="知乎用户",
                content=content[:5000],  # 限制长度，避免 token 爆炸
                platform="zhihu",
                parse_method="beautifulsoup",
                metadata={"url": url},
            )

        except httpx.HTTPError as e:
            raise ParseError(f"知乎请求失败: {e}", platform="zhihu", url=url)
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"知乎解析异常: {e}", platform="zhihu", url=url)
```

- [ ] **Step 2: 编写 src/parsers/wechat.py**

```python
"""微信公众号文章解析器。"""
import httpx
from bs4 import BeautifulSoup
from src.parsers.base import BaseParser, ParsedArticle, ParseError


class WechatParser(BaseParser):
    platform_name = "wechat"

    def can_parse(self, url: str) -> bool:
        return "mp.weixin.qq.com" in url

    async def parse(self, url: str) -> ParsedArticle:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html,application/xhtml+xml",
                    },
                    follow_redirects=True,
                )
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # 提取标题
            title = ""
            title_tag = soup.select_one("#activity-name")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # 提取作者
            author = "公众号"
            author_tag = soup.select_one("#js_name")
            if author_tag:
                author = author_tag.get_text(strip=True)

            # 提取正文
            content_tag = soup.select_one("#js_content")
            if not content_tag:
                raise ParseError("未找到公众号文章正文", platform="wechat", url=url)

            # 移除隐藏元素
            for hidden in content_tag.select('[style*="visibility: hidden"]'):
                hidden.decompose()

            content = content_tag.get_text(separator="\n", strip=True)

            if not content:
                raise ParseError("公众号文章正文为空", platform="wechat", url=url)

            return ParsedArticle(
                title=title or "微信公众号文章",
                author=author,
                content=content[:5000],
                platform="wechat",
                parse_method="beautifulsoup",
                metadata={"url": url},
            )

        except httpx.HTTPError as e:
            raise ParseError(f"公众号请求失败: {e}", platform="wechat", url=url)
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"公众号解析异常: {e}", platform="wechat", url=url)
```

- [ ] **Step 3: 编写测试 tests/test_parsers/test_zhihu.py**

```python
"""知乎解析器测试"""
import pytest
from src.parsers.zhihu import ZhihuParser


class TestZhihuParser:
    def setup_method(self):
        self.parser = ZhihuParser()

    def test_can_parse_zhihu_url(self):
        assert self.parser.can_parse("https://www.zhihu.com/question/12345678") is True

    def test_can_parse_zhuanlan_url(self):
        assert self.parser.can_parse("https://zhuanlan.zhihu.com/p/12345678") is True

    def test_cannot_parse_other_url(self):
        assert self.parser.can_parse("https://mp.weixin.qq.com/s/abc") is False

    @pytest.mark.asyncio
    async def test_parse_raises_error_for_bad_url(self):
        with pytest.raises(Exception):
            await self.parser.parse("https://www.zhihu.com/notexist/99999999999")
```

- [ ] **Step 4: 编写测试 tests/test_parsers/test_wechat.py**

```python
"""微信公众号解析器测试"""
import pytest
from src.parsers.wechat import WechatParser


class TestWechatParser:
    def setup_method(self):
        self.parser = WechatParser()

    def test_can_parse_wechat_url(self):
        assert self.parser.can_parse("https://mp.weixin.qq.com/s/abc123def456") is True

    def test_cannot_parse_other_url(self):
        assert self.parser.can_parse("https://zhuanlan.zhihu.com/p/12345678") is False

    @pytest.mark.asyncio
    async def test_parse_raises_error_for_bad_url(self):
        with pytest.raises(Exception):
            await self.parser.parse("https://mp.weixin.qq.com/s/notexist")
```

- [ ] **Step 5: 运行测试验证**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && source ../../.venv/bin/activate && python -m pytest tests/test_parsers/ -v
```

预期：全部测试通过

- [ ] **Step 6: Commit**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && git add -A && git commit -m "feat: add Zhihu and WeChat parsers with tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: 通用解析器（兜底方案）

**目标：** 实现通用解析器，优先用 Jina Reader 兜底，回退到 BeautifulSoup

**Files:**
- Create: `mindflow/src/parsers/generic.py`
- Create: `mindflow/tests/test_parsers/test_generic.py`

- [ ] **Step 1: 编写 src/parsers/generic.py**

```python
"""通用网页解析器。Jina Reader 优先，BeautifulSoup 兜底。"""
import httpx
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDoc
from src.parsers.base import BaseParser, ParsedArticle, ParseError


class GenericParser(BaseParser):
    platform_name = "generic"

    def can_parse(self, url: str) -> bool:
        # 通用解析器接受任何 URL
        return True

    async def parse(self, url: str) -> ParsedArticle:
        content = await self._try_jina_reader(url)
        if content:
            return content

        content = await self._try_readability(url)
        if content:
            return content

        raise ParseError("所有解析方式均失败", platform="generic", url=url)

    async def _try_jina_reader(self, url: str) -> ParsedArticle | None:
        """尝试使用 Jina Reader 解析"""
        try:
            jina_url = f"https://r.jina.ai/{url}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    jina_url,
                    headers={"Accept": "text/markdown"},
                    follow_redirects=True,
                )
                if response.status_code != 200:
                    return None

                text = response.text.strip()
                if not text or len(text) < 50:
                    return None

                # Jina Reader 返回 Markdown 格式，第一行通常是标题
                lines = text.split("\n")
                title = lines[0].lstrip("# ").strip() if lines else "未命名文章"
                content = "\n".join(lines[1:]) if len(lines) > 1 else text

                return ParsedArticle(
                    title=title[:200],
                    author="",
                    content=content[:5000],
                    platform="generic",
                    parse_method="jina_reader",
                    metadata={"url": url},
                )
        except Exception:
            return None

    async def _try_readability(self, url: str) -> ParsedArticle | None:
        """尝试使用 Readability + BeautifulSoup 解析"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    },
                    follow_redirects=True,
                )
                response.raise_for_status()

            # 使用 readability 提取正文
            doc = ReadabilityDoc(response.text)
            title = doc.title() or ""
            html_content = doc.summary()

            soup = BeautifulSoup(html_content, "html.parser")
            content = soup.get_text(separator="\n", strip=True)

            if not content or len(content) < 100:
                # readability 失败，回退到 body 文本
                soup = BeautifulSoup(response.text, "html.parser")
                body = soup.find("body")
                if body:
                    # 移除 script/style
                    for tag in body(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()
                    content = body.get_text(separator="\n", strip=True)

            if not content or len(content) < 50:
                return None

            return ParsedArticle(
                title=title[:200] or "未命名文章",
                author="",
                content=content[:5000],
                platform="generic",
                parse_method="readability",
                metadata={"url": url},
            )

        except Exception:
            return None
```

- [ ] **Step 2: 编写测试 tests/test_parsers/test_generic.py**

```python
"""通用解析器测试"""
import pytest
from src.parsers.generic import GenericParser


class TestGenericParser:
    def setup_method(self):
        self.parser = GenericParser()

    def test_can_parse_any_url(self):
        assert self.parser.can_parse("https://example.com") is True
        assert self.parser.can_parse("https://any-random-url.com/article") is True

    @pytest.mark.asyncio
    async def test_parse_raises_error_for_completely_invalid_url(self):
        with pytest.raises(Exception):
            await self.parser.parse("not-even-a-url-!!!")
```

- [ ] **Step 3: 运行测试验证**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && source ../../.venv/bin/activate && python -m pytest tests/test_parsers/ -v
```

预期：全部测试通过

- [ ] **Step 4: Commit**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && git add -A && git commit -m "feat: add generic parser with Jina Reader and readability fallback

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 记忆系统（向量存储 + 元数据存储）

**目标：** 实现 Chroma 向量存储和 SQLite 元数据存储，这是 Memory 学习的核心

**Files:**
- Create: `mindflow/src/memory/vector_store.py`
- Create: `mindflow/src/memory/metadata_store.py`
- Create: `mindflow/tests/test_memory/__init__.py`
- Create: `mindflow/tests/test_memory/test_vector_store.py`
- Create: `mindflow/tests/test_memory/test_metadata_store.py`

- [ ] **Step 1: 编写 src/memory/vector_store.py**

```python
"""Chroma 向量存储封装。用于文章全文和观点的语义检索。"""
import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.config import settings
from src.llm.client import get_embedding


class VectorStore:
    """Chroma 向量存储管理器"""

    def __init__(self, collection_name: str = "articles"):
        self.client = chromadb.PersistentClient(
            path=str(settings.storage.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_article(
        self,
        article_id: str,
        title: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """将文章向量化后存入 Chroma。

        Args:
            article_id: 文章唯一 ID
            title: 文章标题
            content: 文章正文（用于向量化）
            metadata: 附加元数据
        """
        # 将标题和正文拼接后向量化
        text_to_embed = f"{title}\n\n{content}"[:8000]
        embedding = get_embedding(text_to_embed)

        self.collection.add(
            ids=[article_id],
            embeddings=[embedding],
            documents=[text_to_embed],
            metadatas=[metadata or {}],
        )

    def search_similar(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """语义搜索相似文章。

        Args:
            query: 搜索查询文本
            n_results: 返回结果数

        Returns:
            [{"article_id": ..., "content": ..., "metadata": ..., "score": ...}, ...]
        """
        query_embedding = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                items.append({
                    "article_id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i],  # cosine distance → similarity
                })
        return items

    def count(self) -> int:
        """返回集合中的文档数"""
        return self.collection.count()
```

- [ ] **Step 2: 编写 src/memory/metadata_store.py**

```python
"""SQLite 元数据存储。管理文章的元数据、观点和用户偏好。"""
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from src.config import settings


class MetadataStore:
    """SQLite 元数据管理器"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = str(db_path or settings.storage.db_path)
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    platform TEXT NOT NULL,
                    title TEXT NOT NULL,
                    author TEXT DEFAULT '未知',
                    clean_content TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    stance TEXT DEFAULT 'neutral',
                    extracted_at TEXT NOT NULL,
                    source_metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS viewpoints (
                    id TEXT PRIMARY KEY,
                    article_id TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    evidence TEXT DEFAULT '[]',
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (article_id) REFERENCES articles(id)
                );

                CREATE TABLE IF NOT EXISTS article_tags (
                    article_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (article_id, tag),
                    FOREIGN KEY (article_id) REFERENCES articles(id)
                );

                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

    def insert_article(
        self,
        url: str,
        platform: str,
        title: str,
        clean_content: str,
        author: str = "未知",
        summary: str = "",
        stance: str = "neutral",
        source_metadata: dict | None = None,
    ) -> str:
        """插入文章元数据，返回 article_id"""
        import json
        article_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO articles
                   (id, url, platform, title, author, clean_content, summary, stance, extracted_at, source_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    article_id,
                    url,
                    platform,
                    title,
                    author,
                    clean_content,
                    summary,
                    stance,
                    now,
                    json.dumps(source_metadata or {}, ensure_ascii=False),
                ),
            )
        return article_id

    def insert_viewpoints(self, article_id: str, viewpoints: list[dict]) -> None:
        """批量插入观点"""
        import json
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            for vp in viewpoints:
                vp_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO viewpoints (id, article_id, claim, reasoning, evidence, confidence, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        vp_id,
                        article_id,
                        vp.get("claim", ""),
                        vp.get("reasoning", ""),
                        json.dumps(vp.get("evidence", []), ensure_ascii=False),
                        vp.get("confidence", 1.0),
                        now,
                    ),
                )

    def insert_tags(self, article_id: str, tags: list[str]) -> None:
        """插入文章标签"""
        with self._get_conn() as conn:
            for tag in tags:
                conn.execute(
                    "INSERT OR IGNORE INTO article_tags (article_id, tag) VALUES (?, ?)",
                    (article_id, tag),
                )

    def get_article(self, article_id: str) -> dict | None:
        """根据 ID 获取文章"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM articles WHERE id = ?", (article_id,)
            ).fetchone()
            return dict(row) if row else None

    def article_exists(self, url: str) -> str | None:
        """检查 URL 是否已存在，返回 article_id 或 None"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id FROM articles WHERE url = ?", (url,)
            ).fetchone()
            return row["id"] if row else None

    def set_preference(self, key: str, value: str) -> None:
        """设置用户偏好"""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now),
            )

    def get_preference(self, key: str, default: str = "") -> str:
        """获取用户偏好"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def get_article_count(self) -> int:
        """返回文章总数"""
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM articles").fetchone()
            return row["cnt"] if row else 0
```

- [ ] **Step 3: 编写测试 tests/test_memory/test_vector_store.py**

```python
"""向量存储测试"""
import pytest
from src.memory.vector_store import VectorStore


class TestVectorStore:
    def setup_method(self):
        # 使用独立的测试集合，避免干扰
        self.store = VectorStore(collection_name="test_articles")

    def test_count_starts_at_zero(self):
        assert self.store.count() >= 0

    def test_search_similar_returns_list(self):
        results = self.store.search_similar("测试查询")
        assert isinstance(results, list)
```

- [ ] **Step 4: 编写测试 tests/test_memory/test_metadata_store.py**

```python
"""元数据存储测试"""
import tempfile
from pathlib import Path
from src.memory.metadata_store import MetadataStore


class TestMetadataStore:
    def setup_method(self):
        # 使用临时数据库
        self.tmp_db = Path(tempfile.mktemp(suffix=".db"))
        self.store = MetadataStore(db_path=self.tmp_db)

    def test_insert_and_get_article(self):
        article_id = self.store.insert_article(
            url="https://example.com/test",
            platform="generic",
            title="测试文章",
            clean_content="这是测试内容",
            summary="摘要",
            stance="neutral",
        )
        assert article_id is not None
        assert len(article_id) > 0

        article = self.store.get_article(article_id)
        assert article is not None
        assert article["title"] == "测试文章"
        assert article["platform"] == "generic"

    def test_article_exists(self):
        self.store.insert_article(
            url="https://example.com/unique",
            platform="generic",
            title="唯一文章",
            clean_content="内容",
        )
        found_id = self.store.article_exists("https://example.com/unique")
        assert found_id is not None

        not_found = self.store.article_exists("https://example.com/never-added")
        assert not_found is None

    def test_insert_viewpoints(self):
        article_id = self.store.insert_article(
            url="https://example.com/vp-test",
            platform="generic",
            title="带观点的文章",
            clean_content="有观点",
        )
        viewpoints = [
            {"claim": "AI 将改变世界", "reasoning": "因为技术进步", "evidence": ["例子1"], "confidence": 0.9},
        ]
        self.store.insert_viewpoints(article_id, viewpoints)

    def test_insert_tags(self):
        article_id = self.store.insert_article(
            url="https://example.com/tag-test",
            platform="generic",
            title="带标签的文章",
            clean_content="内容",
        )
        self.store.insert_tags(article_id, ["AI", "机器学习", "深度学习"])

    def test_set_and_get_preference(self):
        self.store.set_preference("interested_topics", '["AI", "哲学"]')
        value = self.store.get_preference("interested_topics")
        assert "AI" in value

    def test_duplicate_url_updates_existing(self):
        """重复 URL 应该更新而不是新增"""
        url = "https://example.com/duplicate-test"
        id1 = self.store.insert_article(
            url=url, platform="generic", title="第一版", clean_content="旧内容"
        )
        id2 = self.store.insert_article(
            url=url, platform="generic", title="第二版", clean_content="新内容"
        )
        # 同一个 URL 返回同一个 ID（如果实现正确的话）
        count = self.store.get_article_count()
        assert count == 1

    def teardown_method(self):
        import os
        if self.tmp_db.exists():
            os.unlink(self.tmp_db)
```

- [ ] **Step 5: 运行测试验证**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && source ../../.venv/bin/activate && python -m pytest tests/test_memory/ -v
```

预期：全部测试通过

- [ ] **Step 6: Commit**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && git add -A && git commit -m "feat: add vector store (Chroma) and metadata store (SQLite) memory systems

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Function Calling Tools（观点提取 + 标签生成）

**目标：** 实现 LangChain 风格的 Function Calling Tools，这是 Function Calling 学习的核心

**Files:**
- Create: `mindflow/src/tools/extract_viewpoint.py`
- Create: `mindflow/src/tools/generate_tags.py`
- Create: `mindflow/tests/test_tools/__init__.py`
- Create: `mindflow/tests/test_tools/test_extract_viewpoint.py`

- [ ] **Step 1: 编写 src/tools/extract_viewpoint.py**

```python
"""观点提取 Tool。使用 Function Calling 从文章中提取结构化观点。"""
from src.llm.client import function_call

EXTRACT_VIEWPOINT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_article_viewpoints",
        "description": "从文章中提取核心观点、论证链和立场。仔细阅读全文，识别作者的主要论断和论证逻辑。",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "文章200字以内的精炼摘要，涵盖核心内容",
                },
                "viewpoints": {
                    "type": "array",
                    "description": "文章中提取的核心观点列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "核心论断：作者想要论证的主要命题",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "论证逻辑链：作者如何论证这个论断",
                            },
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "支撑论据：作者使用的数据、案例、引用等",
                            },
                        },
                        "required": ["claim", "reasoning", "evidence"],
                    },
                },
                "stance": {
                    "type": "string",
                    "enum": ["support", "oppose", "neutral", "mixed"],
                    "description": "文章对所讨论主题的整体立场",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5个标签，格式为 '领域-主题'（如 'AI-Agent'、'商业-SaaS'、'社会-教育'），从具体到抽象",
                    "minItems": 3,
                    "maxItems": 5,
                },
            },
            "required": ["summary", "viewpoints", "stance", "tags"],
        },
    },
}

SYSTEM_PROMPT = """你是一个专业的文章分析助手。当用户提供一篇文章时，你需要：
1. 仔细阅读全文，理解作者的核心观点和论证逻辑
2. 使用 extract_article_viewpoints 函数返回结构化分析结果
3. 摘要要精炼准确（200字以内）
4. 观点提取要抓住作者论证的核心，不要添加你自己的观点
5. 立场判断基于作者在文章中的实际表达
6. 标签要准确反映文章的领域和主题

请确保你的分析客观、准确、完整。"""


def extract_viewpoints(
    article_text: str,
    model: str | None = None,
) -> dict:
    """从文章文本中提取结构化观点。

    Args:
        article_text: 清洗后的文章全文
        model: 使用的模型

    Returns:
        {
            "summary": str,
            "viewpoints": [{"claim": str, "reasoning": str, "evidence": [str]}],
            "stance": str,
            "tags": [str],
        }

    Raises:
        ValueError: 如果 LLM 返回格式不对
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请分析以下文章：\n\n{article_text}"},
    ]

    result = function_call(
        messages=messages,
        tools=[EXTRACT_VIEWPOINT_TOOL],
        model=model,
        temperature=0.1,
    )

    # 验证必要字段
    required_fields = ["summary", "viewpoints", "stance", "tags"]
    for field in required_fields:
        if field not in result:
            raise ValueError(f"LLM 返回结果缺少必要字段: {field}")

    return result
```

- [ ] **Step 2: 编写 src/tools/generate_tags.py**

```python
"""标签生成 Tool。为文章生成分类标签。"""
from src.llm.client import chat

TAG_SYSTEM_PROMPT = """你是一个内容分类专家。给定文章的标题和摘要，生成 3-5 个精准的分类标签。
标签格式：'领域-主题'（如 'AI-Agent'、'商业-SaaS'、'科技-芯片'）
要求：从具体到抽象排列，每个标签不超过 15 个字。"""


def generate_tags(title: str, summary: str, model: str | None = None) -> list[str]:
    """根据标题和摘要生成标签。

    Args:
        title: 文章标题
        summary: 文章摘要
        model: 使用的模型

    Returns:
        标签字符串列表
    """
    messages = [
        {"role": "system", "content": TAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"标题：{title}\n\n摘要：{summary}\n\n请生成 3-5 个标签，用逗号分隔。"},
    ]

    response = chat(messages=messages, model=model, temperature=0.3, max_tokens=200)
    # 解析响应：按逗号、顿号或换行分割
    import re
    tags = re.split(r"[,，、\n]+", response.strip())
    # 清理空白和引号
    tags = [t.strip().strip("'\"") for t in tags if t.strip()]
    return tags[:5]  # 最多 5 个
```

- [ ] **Step 3: 编写测试 tests/test_tools/test_extract_viewpoint.py**

```python
"""观点提取 Tool 测试"""
import pytest
from src.tools.extract_viewpoint import EXTRACT_VIEWPOINT_TOOL, extract_viewpoints


class TestExtractViewpointTool:
    def test_tool_schema_is_valid_openai_format(self):
        """验证 Tool Schema 符合 OpenAI Function Calling 格式"""
        assert EXTRACT_VIEWPOINT_TOOL["type"] == "function"
        func = EXTRACT_VIEWPOINT_TOOL["function"]
        assert func["name"] == "extract_article_viewpoints"
        assert "parameters" in func
        assert "summary" in func["parameters"]["properties"]
        assert "viewpoints" in func["parameters"]["properties"]
        assert "stance" in func["parameters"]["properties"]
        assert "tags" in func["parameters"]["properties"]

    def test_tool_schema_has_correct_enum(self):
        """验证 stance 枚举值正确"""
        stance_prop = EXTRACT_VIEWPOINT_TOOL["function"]["parameters"]["properties"]["stance"]
        assert "enum" in stance_prop
        assert "support" in stance_prop["enum"]
        assert "oppose" in stance_prop["enum"]
        assert "neutral" in stance_prop["enum"]
        assert "mixed" in stance_prop["enum"]

    def test_tool_schema_has_tag_limits(self):
        """验证标签数量限制"""
        tags_prop = EXTRACT_VIEWPOINT_TOOL["function"]["parameters"]["properties"]["tags"]
        assert tags_prop["minItems"] == 3
        assert tags_prop["maxItems"] == 5

    @pytest.mark.skip(reason="需要真实 API key")
    def test_extract_viewpoints_from_real_article(self):
        """测试真实解析（需要 API key）"""
        article = """
        标题：远程办公的利与弊

        过去三年，远程办公从应急方案变成了常态。支持者认为，远程办公提升了员工的工作效率，
        减少了通勤时间，让员工有更多时间陪伴家人。然而，反对者指出，远程办公削弱了团队协作，
        增加了沟通成本，并导致了职业孤独感。从长远来看，混合办公模式可能是最优解——
        既保留了远程办公的灵活性，又维持了面对面协作的优势。
        """
        result = extract_viewpoints(article)
        assert "summary" in result
        assert "viewpoints" in result
        assert len(result["viewpoints"]) >= 1
        assert "stance" in result
        assert "tags" in result
        assert 3 <= len(result["tags"]) <= 5
```

- [ ] **Step 4: 运行测试验证**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && source ../../.venv/bin/activate && python -m pytest tests/test_tools/test_extract_viewpoint.py -v -k "not real_article"
```

预期：3 个单元测试通过（1 个跳过）

- [ ] **Step 5: Commit**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && git add -A && git commit -m "feat: add viewpoint extraction and tag generation tools with function calling

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: LangGraph Digest Agent（状态 + 节点 + 图）

**目标：** 组装 LangGraph 工作流，这是 LangChain/LangGraph 学习的核心

**Files:**
- Create: `mindflow/src/agents/digest/state.py`
- Create: `mindflow/src/agents/digest/nodes.py`
- Create: `mindflow/src/agents/digest/graph.py`
- Create: `mindflow/tests/test_agents/__init__.py`
- Create: `mindflow/tests/test_agents/test_digest.py`

- [ ] **Step 1: 编写 src/agents/digest/state.py**

```python
"""Digest Agent 的 LangGraph State 定义"""
from typing import TypedDict, Optional


class DigestState(TypedDict, total=False):
    """Digest Agent 的工作流状态。

    所有字段都是 Optional 的，因为工作流逐步填充状态。
    """
    # 输入
    url: str

    # 路由结果
    platform: str

    # 解析结果
    parsed_title: str
    parsed_author: str
    parsed_content: str
    parse_error: str  # 非空表示解析失败

    # LLM 提取结果
    summary: str
    viewpoints: list[dict]
    stance: str
    tags: list[str]

    # 存储结果
    article_id: str

    # 最终输出
    reply_card: str
```

- [ ] **Step 2: 编写 src/agents/digest/nodes.py**

```python
"""Digest Agent 各节点的实现函数。每个节点对应工作流中的一个步骤。"""
from src.parsers.router import URLRouter, detect_platform
from src.parsers.zhihu import ZhihuParser
from src.parsers.wechat import WechatParser
from src.parsers.generic import GenericParser
from src.parsers.base import ParseError
from src.memory.vector_store import VectorStore
from src.memory.metadata_store import MetadataStore
from src.tools.extract_viewpoint import extract_viewpoints

# 模块级别的单例，在 graph 组装时初始化
_router: URLRouter | None = None
_vector_store: VectorStore | None = None
_metadata_store: MetadataStore | None = None


def _get_router() -> URLRouter:
    global _router
    if _router is None:
        _router = URLRouter()
        _router.register(ZhihuParser())
        _router.register(WechatParser())
        _router.register(GenericParser())
    return _router


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_metadata_store() -> MetadataStore:
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store


async def url_router_node(state: dict) -> dict:
    """节点 1: URL 路由 — 识别平台类型，检查是否已存在"""
    url = state["url"]

    # 检查是否已经解析过这个 URL
    existing_id = _get_metadata_store().article_exists(url)
    if existing_id:
        article = _get_metadata_store().get_article(existing_id)
        if article:
            return {
                "platform": article["platform"],
                "parsed_title": article["title"],
                "parsed_author": article["author"],
                "parsed_content": article["clean_content"],
                "summary": article["summary"],
                "article_id": existing_id,
                "parse_error": "",
            }

    platform = _get_router().route(url)
    return {"platform": platform}


async def parse_node(state: dict) -> dict:
    """节点 2: 内容解析 — 使用对应平台的解析器抓取内容"""
    # 如果已经解析过（从 url_router_node 带过来的缓存数据），跳过
    if state.get("parsed_content"):
        return {}

    url = state["url"]
    platform = state.get("platform", "generic")
    parser = _get_router().get_parser(platform)

    if parser is None:
        parser = GenericParser()

    try:
        article = await parser.parse(url)
        return {
            "parsed_title": article.title,
            "parsed_author": article.author,
            "parsed_content": article.content,
            "parse_error": "",
        }
    except ParseError as e:
        return {"parse_error": str(e)}
    except Exception as e:
        return {"parse_error": f"解析失败: {e}"}


async def extract_viewpoint_node(state: dict) -> dict:
    """节点 3: 观点提取 — 使用 LLM Function Calling 提取结构化观点"""
    title = state.get("parsed_title", "")
    content = state.get("parsed_content", "")

    if not content:
        return {"reply_card": "❌ 文章内容为空，无法提取观点"}

    article_text = f"标题：{title}\n\n正文：\n{content}"

    try:
        result = extract_viewpoints(article_text)
        return {
            "summary": result.get("summary", ""),
            "viewpoints": result.get("viewpoints", []),
            "stance": result.get("stance", "neutral"),
            "tags": result.get("tags", []),
        }
    except Exception as e:
        return {"reply_card": f"❌ 观点提取失败: {e}"}


async def store_node(state: dict) -> dict:
    """节点 4: 存储 — 将结果保存到 Chroma 和 SQLite"""
    url = state["url"]

    # 如果已有 article_id（即之前解析过的），跳过存储
    if state.get("article_id"):
        return {}

    article_id = _get_metadata_store().insert_article(
        url=url,
        platform=state.get("platform", "generic"),
        title=state.get("parsed_title", "未命名"),
        author=state.get("parsed_author", "未知"),
        clean_content=state.get("parsed_content", ""),
        summary=state.get("summary", ""),
        stance=state.get("stance", "neutral"),
    )

    # 存储观点
    viewpoints = state.get("viewpoints", [])
    if viewpoints:
        _get_metadata_store().insert_viewpoints(article_id, viewpoints)

    # 存储标签
    tags = state.get("tags", [])
    if tags:
        _get_metadata_store().insert_tags(article_id, tags)

    # 向量化存储
    try:
        _get_vector_store().add_article(
            article_id=article_id,
            title=state.get("parsed_title", ""),
            content=state.get("parsed_content", ""),
            metadata={
                "platform": state.get("platform", ""),
                "summary": state.get("summary", ""),
                "tags": ",".join(state.get("tags", [])),
            },
        )
    except Exception:
        pass  # 向量存储失败不阻塞流程

    return {"article_id": article_id}


async def format_card_node(state: dict) -> dict:
    """节点 5: 卡片格式化 — 生成最终的知识卡片"""
    title = state.get("parsed_title", "未命名")
    platform = state.get("platform", "generic")
    summary = state.get("summary", "暂无摘要")
    stance = state.get("stance", "neutral")
    tags = state.get("tags", [])
    viewpoints = state.get("viewpoints", [])

    # 平台中文名
    platform_names = {
        "zhihu": "知乎",
        "wechat": "公众号",
        "weibo": "微博",
        "xiaohongshu": "小红书",
        "bilibili": "B站",
        "generic": "网页",
    }
    platform_cn = platform_names.get(platform, platform)

    # 立场图标和文字
    stance_map = {
        "support": "✅ 支持",
        "oppose": "❌ 反对",
        "neutral": "➖ 中立",
        "mixed": "🔄 混合立场",
    }
    stance_text = stance_map.get(stance, stance)

    # 构建标签行
    tags_line = " ".join(f"`#{t}`" for t in tags) if tags else "无标签"

    # 构建观点部分
    vp_lines = ""
    if viewpoints:
        for i, vp in enumerate(viewpoints, 1):
            claim = vp.get("claim", "")
            reasoning = vp.get("reasoning", "")
            evidence = vp.get("evidence", [])
            vp_lines += f"\n**观点 {i}：**{claim}\n  > 论证：{reasoning}"
            if evidence:
                evd_text = "；".join(evidence)
                vp_lines += f"\n  > 论据：{evd_text}"
            vp_lines += "\n"

    card = f"""📌 **{title}**

🏷️ {tags_line}
📱 来源：{platform_cn} ｜ 📊 立场：{stance_text}

🧠 **摘要**
{summary}
{vp_lines}
"""
    return {"reply_card": card}
```

- [ ] **Step 3: 编写 src/agents/digest/graph.py**

```python
"""Digest Agent 的 LangGraph StateGraph 组装。

工作流: URL路由 → 内容解析 → 观点提取 → 存储 → 卡片回复
"""
from langgraph.graph import StateGraph, END
from src.agents.digest.state import DigestState
from src.agents.digest.nodes import (
    url_router_node,
    parse_node,
    extract_viewpoint_node,
    store_node,
    format_card_node,
)


def should_continue_after_parse(state: dict) -> str:
    """条件边：解析失败时直接跳到格式化（显示错误），否则继续提取观点"""
    if state.get("parse_error"):
        return "format_card"
    return "extract_viewpoint"


def should_continue_after_extract(state: dict) -> str:
    """条件边：如果观点提取失败（已有 reply_card），直接结束"""
    if state.get("reply_card"):
        return END
    return "store"


def build_digest_graph() -> StateGraph:
    """构建 Digest Agent 的 LangGraph 工作流。

    Returns:
        编译好的 StateGraph，可直接调用 .ainvoke()
    """
    workflow = StateGraph(DigestState)

    # 添加节点
    workflow.add_node("url_router", url_router_node)
    workflow.add_node("parse", parse_node)
    workflow.add_node("extract_viewpoint", extract_viewpoint_node)
    workflow.add_node("store", store_node)
    workflow.add_node("format_card", format_card_node)

    # 设置入口
    workflow.set_entry_point("url_router")

    # 添加边
    workflow.add_edge("url_router", "parse")

    # 条件边：解析成功 → 提取观点；解析失败 → 格式化错误卡片
    workflow.add_conditional_edges(
        "parse",
        should_continue_after_parse,
        {
            "extract_viewpoint": "extract_viewpoint",
            "format_card": "format_card",
        },
    )

    # 条件边：观点提取成功 → 存储；失败 → 结束
    workflow.add_conditional_edges(
        "extract_viewpoint",
        should_continue_after_extract,
        {
            "store": "store",
            END: END,
        },
    )

    workflow.add_edge("store", "format_card")
    workflow.add_edge("format_card", END)

    return workflow.compile()


# 全局单例
digest_agent = build_digest_graph()
```

- [ ] **Step 4: 编写测试 tests/test_agents/test_digest.py**

```python
"""Digest Agent 测试"""
import pytest
from src.agents.digest.graph import build_digest_graph, digest_agent


class TestDigestGraph:
    def test_graph_can_be_built(self):
        """验证图可以正常构建"""
        graph = build_digest_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        """验证图包含预期节点"""
        graph = build_digest_graph()
        nodes = graph.get_graph().nodes
        # 节点包含入口节点 __start__ 和我们定义的节点
        assert len(nodes) >= 5

    def test_global_agent_is_compiled(self):
        """验证全局 digest_agent 已被编译"""
        assert digest_agent is not None

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="需要真实 API key 和网络")
    async def test_full_digest_flow(self):
        """集成测试：完整的工作流（需要 API key）"""
        result = await digest_agent.ainvoke({
            "url": "https://example.com/simple-article",
        })
        assert "reply_card" in result
        assert len(result["reply_card"]) > 0
```

- [ ] **Step 5: 运行测试验证**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && source ../../.venv/bin/activate && python -m pytest tests/test_agents/test_digest.py -v -k "not full_digest"
```

预期：2 个单元测试通过（1 个跳过）

- [ ] **Step 6: Commit**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && git add -A && git commit -m "feat: add LangGraph Digest Agent with 5-node workflow

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Telegram Bot

**目标：** 实现 Telegram Bot，连接用户交互和 Digest Agent

**Files:**
- Create: `mindflow/src/bot/handlers.py`
- Create: `mindflow/src/bot/main.py`

- [ ] **Step 1: 编写 src/bot/handlers.py**

```python
"""Telegram Bot 消息处理器"""
import re
from telegram import Update
from telegram.ext import ContextTypes
from src.agents.digest.graph import digest_agent


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /start 命令"""
    welcome = """🧠 *MindFlow* — 你的第二大脑

直接发送任意文章链接给我，我会：

1. 🔍 自动识别平台并抓取内容
2. 🧠 提取核心观点和论证链
3. 🏷️ 自动分类打标签
4. 📊 判断文章立场
5. 💾 存入你的个人知识库

以后你还可以问我：
• "关于 XXX，我收藏了哪些观点？"
• "最近有哪些和我观点相反的文章？"

发送链接试试吧 👇"""
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /help 命令"""
    help_text = """📚 *MindFlow 使用指南*

*核心功能：* 发送任意文章链接，我会自动解析并提取观点。

*支持平台：*
• 知乎 (zhihu.com)
• 微信公众号 (mp.weixin.qq.com)
• 微博 (weibo.com)
• 小红书 (xiaohongshu.com)
• B站专栏 (bilibili.com/read)
• 其他网页（通用解析）

*命令列表：*
/start - 开始使用
/help - 查看帮助
/stats - 查看知识库统计

*使用技巧：*
• 直接转发或粘贴链接
• 解析通常在 10-30 秒内完成
• 重复的链接会自动跳过"""
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /stats 命令 — 查看知识库统计"""
    from src.memory.metadata_store import MetadataStore
    from src.memory.vector_store import VectorStore

    meta = MetadataStore()
    vec = VectorStore()

    article_count = meta.get_article_count()
    vector_count = vec.count()

    stats_text = f"""📊 *MindFlow 知识库统计*

📄 已收藏文章：{article_count} 篇
🧮 向量索引数：{vector_count} 条

继续发送链接来丰富你的知识库！"""
    await update.message.reply_text(stats_text, parse_mode="Markdown")


# URL 正则模式
URL_PATTERN = re.compile(
    r"https?://[^\s]+",
    re.IGNORECASE,
)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理普通消息 — 检测 URL 并启动 Digest 工作流"""
    text = update.message.text.strip() if update.message.text else ""

    # 提取 URL
    urls = URL_PATTERN.findall(text)
    if not urls:
        await update.message.reply_text(
            "👋 请发送一个文章链接给我，我来帮你解析！\n\n"
            "支持：知乎、微信公众号、微博、小红书、B站等平台\n"
            "也可以发送 /help 查看详细说明"
        )
        return

    url = urls[0]  # 取第一个 URL
    await update.message.reply_text(f"🔍 正在解析...\n{url}")

    try:
        # 运行 Digest Agent
        result = await digest_agent.ainvoke({"url": url})
        reply_card = result.get("reply_card", "❌ 解析失败，请稍后重试")

        await update.message.reply_text(
            reply_card,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
    except Exception as e:
        await update.message.reply_text(
            f"❌ 处理出错：{str(e)[:200]}\n\n请稍后重试或检查链接是否有效。"
        )
```

- [ ] **Step 2: 编写 src/bot/main.py**

```python
"""MindFlow Telegram Bot 入口"""
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from src.config import settings
from src.bot.handlers import start_command, help_command, stats_command, handle_message


def main() -> None:
    """启动 Telegram Bot"""
    print("🤖 MindFlow Bot 正在启动...")
    print(f"📊 LLM 模型: {settings.llm.model}")
    print(f"💾 数据目录: {settings.storage.data_dir}")

    # 创建 Application
    app = Application.builder().token(settings.bot.token).build()

    # 注册命令处理器
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))

    # 注册消息处理器（处理所有非命令文本消息）
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot 已启动，等待消息...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 修复 main.py 中缺少的 import**

确认 `src/bot/main.py` 中有 `from telegram import Update`（已在 handlers 中导入，main.py 用 `allowed_updates=Update.ALL_TYPES` 需要它）。将 main.py 修改为：

```python
"""MindFlow Telegram Bot 入口"""
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from src.config import settings
from src.bot.handlers import start_command, help_command, stats_command, handle_message


def main() -> None:
    """启动 Telegram Bot"""
    print("🤖 MindFlow Bot 正在启动...")
    print(f"📊 LLM 模型: {settings.llm.model}")
    print(f"💾 数据目录: {settings.storage.data_dir}")

    app = Application.builder().token(settings.bot.token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot 已启动，等待消息...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && git add -A && git commit -m "feat: add Telegram Bot with start/help/stats commands and URL handler

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: 集成验证与 README

**目标：** 验证整体项目可运行，编写 README 文档

**Files:**
- Create: `mindflow/README.md`

- [ ] **Step 1: 验证项目导入**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && source ../../.venv/bin/activate && python -c "
from src.config import settings
from src.llm.client import get_client
from src.parsers.router import URLRouter
from src.parsers.zhihu import ZhihuParser
from src.parsers.wechat import WechatParser
from src.parsers.generic import GenericParser
from src.memory.vector_store import VectorStore
from src.memory.metadata_store import MetadataStore
from src.tools.extract_viewpoint import extract_viewpoints
from src.agents.digest.graph import digest_agent

print('✅ 所有模块导入成功')
print(f'LLM 模型: {settings.llm.model}')
print(f'数据目录: {settings.storage.data_dir}')
"
```

预期：打印 "✅ 所有模块导入成功" 及配置信息

- [ ] **Step 2: 运行全部单元测试**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && source ../../.venv/bin/activate && python -m pytest tests/ -v --tb=short -k "not real_api and not full_digest and not real_article"
```

预期：全部测试通过（可能需要 API key 的测试会被跳过）

- [ ] **Step 3: 编写 README.md**

```bash
cat > /home/max-rayyy/ai-workspace/mindflow/README.md << 'EOF'
# 🧠 MindFlow — 个人认知操作系统

> 把你每天刷到的各种内容，自动消化成你的「第二大脑」——可检索、可对比、可主动提醒。

## 这是什么

MindFlow 帮助你收集碎片化的社交媒体阅读（知乎、公众号、微博、小红书、B站等），
自动提取观点、打标签、建立知识关联，最终在正确的时机主动把知识推送到你面前。

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
- `OPENAI_API_KEY`：LLM API Key（支持 OpenAI 兼容 API，如 DeepSeek）
- `OPENAI_BASE_URL`：API 地址
- `LLM_MODEL`：使用的模型
- `TELEGRAM_BOT_TOKEN`：Telegram Bot Token

### 3. 运行

```bash
python -m src.bot.main
```

### 4. 使用

在 Telegram 中给你的 Bot 发送任意文章链接，如：
- 知乎问题/专栏：`https://www.zhihu.com/question/...`
- 微信公众号：`https://mp.weixin.qq.com/s/...`
- 微博、小红书、B站等

Bot 会自动解析并返回知识卡片。

## 项目结构

```
src/
├── config.py          # 配置管理
├── llm/               # LLM 客户端
├── parsers/           # 多平台解析器
├── memory/            # 记忆系统（Chroma + SQLite）
├── tools/             # Function Calling Tools
├── agents/digest/     # LangGraph Agent
└── bot/               # Telegram Bot
```

## 技术栈

- **Agent 框架：** LangGraph + LangChain
- **向量数据库：** Chroma
- **元数据存储：** SQLite
- **LLM：** OpenAI 兼容 API
- **交互：** Telegram Bot API

## 学习路径

这个项目是为学习以下技术而设计的：

| 技术 | 对应模块 |
|------|---------|
| **LangGraph** | `agents/digest/` — 有状态的图工作流 |
| **Function Calling** | `tools/` — 结构化观点提取 |
| **Memory 系统** | `memory/` — 三层记忆架构 |

详见 `docs/superpowers/specs/2025-06-05-mindflow-design.md`

## License

MIT
EOF
```

- [ ] **Step 4: 最终提交**

```bash
cd /home/max-rayyy/ai-workspace/mindflow && git add -A && git commit -m "docs: add README and complete integration verification

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 📋 Phase 1 完成检查清单

完成所有 Task 后，验证以下条件：

- [ ] 用户通过 Telegram 发送知乎链接 → 返回知识卡片
- [ ] 用户通过 Telegram 发送公众号链接 → 返回知识卡片
- [ ] 知识卡片包含：摘要、核心观点、立场、标签
- [ ] 重复链接不会重复解析
- [ ] 解析失败的链接返回友好错误提示
- [ ] 数据持久化到 SQLite 和 Chroma
- [ ] 所有单元测试通过

---

*计划结束。开始实现时，请按 Task 1 → Task 10 顺序执行，每个 Task 内的 Step 顺序执行。*

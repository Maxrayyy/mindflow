"""Digest Agent 各节点的实现函数。

工作流五步流水线:
  1. url_router      — 识别平台 + 去重检查
  2. parse           — 调用平台专用解析器抓取内容
  3. extract_viewpoint — LLM Function Calling 提取结构化观点
  4. store           — 持久化到 SQLite + Chroma
  5. format_card     — 生成 Markdown 知识卡片

每个节点都是一个 async 函数，接收 State dict，返回部分 State dict。
返回的 dict 会被 LangGraph 合并回完整 State。
"""

from src.parsers.router import URLRouter, detect_platform
from src.parsers.zhihu import ZhihuParser
from src.parsers.wechat import WechatParser
from src.parsers.generic import GenericParser
from src.parsers.base import ParseError
from src.memory.vector_store import VectorStore
from src.memory.metadata_store import MetadataStore
from src.tools.extract_viewpoint import extract_viewpoints

# ===== 模块级单例 — 惰性初始化 =====
# 不在模块加载时创建实例，避免导入时触发 Chroma 初始化等副作用。
# 首次调用 _get_xxx() 时才创建，之后复用。

_router: URLRouter | None = None
_vector_store: VectorStore | None = None
_metadata_store: MetadataStore | None = None


def _get_router() -> URLRouter:
    """获取 URLRouter 单例，自动注册三个解析器"""
    global _router
    if _router is None:
        _router = URLRouter()
        _router.register(ZhihuParser())
        _router.register(WechatParser())
        _router.register(GenericParser())
    return _router


def _get_vector_store() -> VectorStore:
    """获取 VectorStore 单例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_metadata_store() -> MetadataStore:
    """获取 MetadataStore 单例"""
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store


# ===== 节点实现 =====

async def url_router_node(state: dict) -> dict:
    """节点 1: URL 路由 — 识别平台，检查是否已解析过。

    如果 URL 已经存在于数据库中，直接复用已有数据，
    跳过后续的解析和提取步骤。

    Returns:
        {"platform": "zhihu"} 或完整的缓存数据
    """
    url = state["url"]

    # 去重：检查之前是否解析过这个 URL
    existing_id = _get_metadata_store().article_exists(url)
    if existing_id:
        article = _get_metadata_store().get_article(existing_id)
        if article:
            # URL 已存在 → 直接返回缓存数据，跳过解析和提取
            return {
                "platform": article["platform"],
                "parsed_title": article["title"],
                "parsed_author": article["author"],
                "parsed_content": article["clean_content"],
                "summary": article["summary"],
                "stance": article.get("stance", "neutral"),
                "article_id": existing_id,
                "parse_error": "",
            }

    # 新 URL → 识别平台
    platform = detect_platform(url)
    return {"platform": platform}


async def parse_node(state: dict) -> dict:
    """节点 2: 内容解析 — 调用平台专用解析器抓取页面内容。

    如果 parse_error 已经存在（来自上游），直接跳过。
    如果 parsed_content 已经存在（来自缓存命中），直接跳过。
    """
    # 已有数据（缓存命中或解析错误）→ 跳过
    if state.get("parse_error") or state.get("parsed_content"):
        return {}

    url = state["url"]
    platform = state.get("platform", "generic")

    # 获取对应平台的解析器，未注册则用通用解析器兜底
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
        return {"parse_error": f"解析失败: {e}"}
    except Exception as e:
        return {"parse_error": f"解析异常: {e}"}


async def extract_viewpoint_node(state: dict) -> dict:
    """节点 3: 观点提取 — 用 LLM Function Calling 提取结构化观点。

    这是整个工作流的核心步骤。把文章全文喂给 LLM，
    LLM 按 Schema 返回结构化的摘要/观点/立场/标签。

    如果 parsed_content 为空 → 设置错误信息，工作流直接跳到 format_card。
    """
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
    """节点 4: 持久化 — 将结果保存到 SQLite 和 Chroma。

    SQLite 存元数据（可精确查询），Chroma 存向量（可语义搜索）。
    如果已有 article_id（缓存命中），跳过存储。

    向量存储失败不会阻塞整个流程 —— 元数据存储成功了就行。
    """
    url = state["url"]

    # 缓存命中 → 跳过
    if state.get("article_id"):
        return {}

    # 写入 SQLite
    article_id = _get_metadata_store().insert_article(
        url=url,
        platform=state.get("platform", "generic"),
        title=state.get("parsed_title", "未命名"),
        author=state.get("parsed_author", "未知"),
        clean_content=state.get("parsed_content", ""),
        summary=state.get("summary", ""),
        stance=state.get("stance", "neutral"),
    )

    # 写入观点
    viewpoints = state.get("viewpoints", [])
    if viewpoints:
        _get_metadata_store().insert_viewpoints(article_id, viewpoints)

    # 写入标签
    tags = state.get("tags", [])
    if tags:
        _get_metadata_store().insert_tags(article_id, tags)

    # 写入 Chroma 向量索引
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
        # 向量存储失败不阻塞流程——文章已经存入 SQLite
        pass

    return {"article_id": article_id}


async def format_card_node(state: dict) -> dict:
    """节点 5: 卡片格式化 — 生成返回给用户的 Markdown 知识卡片。

    这个过程不需要 LLM，纯字符串拼接。
    如果上游已经设置了 reply_card（错误信息或缓存结果），直接返回。
    """
    # 如果已经有 reply_card（错误场景），直接返回
    if state.get("reply_card"):
        return {}

    title = state.get("parsed_title", "未命名")
    platform = state.get("platform", "generic")
    summary = state.get("summary", "暂无摘要")
    stance = state.get("stance", "neutral")
    tags = state.get("tags", [])
    viewpoints = state.get("viewpoints", [])
    article_id = state.get("article_id", "")

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

    # 标签行
    tags_line = " ".join(f"`#{t}`" for t in tags) if tags else "无标签"

    # 观点部分
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

    card = (
        f"📌 **{title}**\n\n"
        f"🏷️ {tags_line}\n"
        f"📱 来源：{platform_cn} ｜ 📊 立场：{stance_text}\n\n"
        f"🧠 **摘要**\n{summary}\n"
        f"{vp_lines}"
    )

    return {"reply_card": card}

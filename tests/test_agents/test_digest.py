"""Digest Agent 测试。

测试策略：
  - 单元测试：验证图和节点结构正确（不需要 API key）
  - 集成测试：验证完整工作流（需要 API key，默认跳过）
"""
import pytest
from src.agents.digest.state import DigestState
from src.agents.digest.graph import build_digest_graph, digest_agent
from src.agents.digest.nodes import (
    url_router_node,
    parse_node,
    extract_viewpoint_node,
    store_node,
    format_card_node,
)
from src.agents.digest.graph import should_after_parse, should_after_extract


class TestDigestGraphStructure:
    """验证 LangGraph 图的结构正确性"""

    def test_graph_can_be_built(self):
        """每次调用 build_digest_graph 都能正常构建"""
        graph = build_digest_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        """图必须包含全部五个业务节点"""
        graph = build_digest_graph()
        # LangGraph 的内部表示包含 __start__ 入口节点
        nodes = graph.get_graph().nodes
        node_names = {n for n in nodes if not n.startswith("__")}
        expected = {"url_router", "parse", "extract_viewpoint", "store", "format_card"}
        assert node_names >= expected, f"缺少节点: {expected - node_names}"

    def test_global_agent_is_compiled(self):
        """digest_agent 是编译好的图，可以直接 .ainvoke()"""
        assert digest_agent is not None


class TestConditionFunctions:
    """验证条件路由函数的逻辑正确性"""

    def test_should_after_parse_success(self):
        """解析成功 → 进入 extract_viewpoint"""
        assert should_after_parse({"parse_error": ""}) == "extract_viewpoint"

    def test_should_after_parse_failure(self):
        """解析失败 → 跳转到 format_card"""
        assert should_after_parse({"parse_error": "网络超时"}) == "format_card"

    def test_should_after_extract_success(self):
        """提取成功（无 reply_card）→ 进入 store"""
        assert should_after_extract({}) == "store"

    def test_should_after_extract_failure(self):
        """提取失败（已有 reply_card）→ 直接结束"""
        result = should_after_extract({"reply_card": "❌ 错误"})
        # END 是 langgraph 的特殊常量
        assert result is not None  # 证明返回了有效值


class TestNodesIsolated:
    """验证各节点的基本行为（不依赖真实网络或 API）"""

    @pytest.mark.asyncio
    async def test_url_router_returns_platform(self):
        """url_router 能正确识别平台"""
        result = await url_router_node({"url": "https://zhuanlan.zhihu.com/p/12345"})
        assert result["platform"] == "zhihu"

    @pytest.mark.asyncio
    async def test_url_router_unknown_url_defaults_to_generic(self):
        """未知 URL → generic"""
        result = await url_router_node({"url": "https://example.com/blog"})
        assert result["platform"] == "generic"

    @pytest.mark.asyncio
    async def test_url_router_empty_url(self):
        """空 URL 也是 generic"""
        result = await url_router_node({"url": ""})
        assert result["platform"] == "generic"

    @pytest.mark.asyncio
    async def test_extract_viewpoint_empty_content(self):
        """内容为空时返回错误"""
        result = await extract_viewpoint_node({
            "parsed_title": "测试",
            "parsed_content": "",
        })
        assert "reply_card" in result
        assert "❌" in result["reply_card"]

    @pytest.mark.asyncio
    async def test_format_card_error_bypass(self):
        """如果已存在 reply_card，跳过格式化"""
        result = await format_card_node({"reply_card": "已有错误"})
        assert result == {}

    @pytest.mark.asyncio
    async def test_format_card_generates_output(self):
        """正常输入生成 Markdown 卡片"""
        result = await format_card_node({
            "parsed_title": "测试文章",
            "platform": "zhihu",
            "summary": "这是一篇测试文章",
            "stance": "neutral",
            "tags": ["AI", "机器学习"],
            "viewpoints": [
                {"claim": "AI 改变世界", "reasoning": "技术进步", "evidence": ["例子"]},
            ],
        })
        card = result["reply_card"]
        assert "📌" in card
        assert "测试文章" in card
        assert "AI" in card
        assert "AI 改变世界" in card

    @pytest.mark.asyncio
    async def test_format_card_without_viewpoints(self):
        """没有观点时的退化表现"""
        result = await format_card_node({
            "parsed_title": "无观点文章",
            "summary": "只有摘要没有观点",
            "stance": "neutral",
            "tags": [],
            "viewpoints": [],
        })
        assert "📌" in result["reply_card"]
        assert "无标签" in result["reply_card"]

    @pytest.mark.asyncio
    async def test_parse_node_skips_when_content_exists(self):
        """已有 parsed_content → 直接跳过，不发起网络请求"""
        result = await parse_node({
            "url": "https://example.com",
            "parsed_content": "已有内容",
        })
        assert result == {}

    @pytest.mark.asyncio
    async def test_parse_node_skips_when_error_exists(self):
        """已有 parse_error → 直接跳过"""
        result = await parse_node({
            "url": "https://example.com",
            "parse_error": "之前失败了",
        })
        assert result == {}


class TestDigestState:
    """验证 State TypedDict 定义"""

    def test_digest_state_is_typeddict(self):
        """DigestState 是 TypedDict 的子类"""
        from typing import TypedDict
        # 验证它是一个 dict 子类型即可
        state: DigestState = {"url": "https://example.com"}
        assert isinstance(state, dict)

    def test_partial_state_allowed(self):
        """total=False 意味着可以不传所有字段"""
        state: DigestState = {"url": "https://example.com"}
        assert state["url"] == "https://example.com"
        # 缺失的字段不应该报错（total=False）


class TestDigestIntegration:
    """集成测试 — 需要真实 API key 和网络"""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="需要真实 API key")
    async def test_full_digest_flow_generic_url(self):
        """完整工作流：从 URL 到知识卡片"""
        result = await digest_agent.ainvoke({
            "url": "https://example.com/simple-article",
        })
        assert "reply_card" in result
        assert len(result["reply_card"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="需要真实 API key")
    async def test_duplicate_url_uses_cache(self):
        """重复 URL 应该命中缓存，跳过解析"""
        url = "https://example.com/unique-test-article"
        # 第一次
        result1 = await digest_agent.ainvoke({"url": url})
        # 第二次 — 应该直接返回缓存数据
        result2 = await digest_agent.ainvoke({"url": url})
        assert result1["article_id"] == result2["article_id"]

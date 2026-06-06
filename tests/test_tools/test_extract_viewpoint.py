"""观点提取 Tool 测试。

测试策略：
  - 单元测试：验证 Tool Schema 的结构正确性（不需要 API key）
  - 集成测试：验证真实 LLM 调用（需要 API key，默认跳过）
"""
import pytest
from src.tools.extract_viewpoint import (
    EXTRACT_VIEWPOINT_TOOL,
    extract_viewpoints,
    SYSTEM_PROMPT,
)


class TestExtractViewpointToolSchema:
    """验证 Function Calling Schema 的结构正确性"""

    def test_tool_type_is_function(self):
        """Schema 必须是 'function' 类型"""
        assert EXTRACT_VIEWPOINT_TOOL["type"] == "function"

    def test_tool_name_is_correct(self):
        """函数名必须与 Schema 中的一致"""
        assert EXTRACT_VIEWPOINT_TOOL["function"]["name"] == "extract_article_viewpoints"

    def test_tool_has_parameters(self):
        """必须有 parameters 定义"""
        assert "parameters" in EXTRACT_VIEWPOINT_TOOL["function"]

    def test_parameters_has_all_required_fields(self):
        """parameters 必须包含 summary, viewpoints, stance, tags"""
        props = EXTRACT_VIEWPOINT_TOOL["function"]["parameters"]["properties"]
        assert "summary" in props
        assert "viewpoints" in props
        assert "stance" in props
        assert "tags" in props

    def test_summary_is_string_type(self):
        props = EXTRACT_VIEWPOINT_TOOL["function"]["parameters"]["properties"]
        assert props["summary"]["type"] == "string"

    def test_viewpoints_is_array_type(self):
        props = EXTRACT_VIEWPOINT_TOOL["function"]["parameters"]["properties"]
        assert props["viewpoints"]["type"] == "array"

    def test_viewpoint_items_have_required_fields(self):
        """每个 viewpoint 必须包含 claim, reasoning, evidence"""
        items = EXTRACT_VIEWPOINT_TOOL["function"]["parameters"]["properties"]["viewpoints"]["items"]
        assert "claim" in items["properties"]
        assert "reasoning" in items["properties"]
        assert "evidence" in items["properties"]
        assert set(items["required"]) == {"claim", "reasoning", "evidence"}

    def test_stance_enum_values_are_correct(self):
        """stance 只能是 support/oppose/neutral/mixed 之一"""
        stance_prop = EXTRACT_VIEWPOINT_TOOL["function"]["parameters"]["properties"]["stance"]
        assert "enum" in stance_prop
        assert set(stance_prop["enum"]) == {"support", "oppose", "neutral", "mixed"}

    def test_tags_has_min_and_max_items(self):
        """标签必须在 3-5 个之间"""
        tags_prop = EXTRACT_VIEWPOINT_TOOL["function"]["parameters"]["properties"]["tags"]
        assert tags_prop["minItems"] == 3
        assert tags_prop["maxItems"] == 5

    def test_required_fields_at_top_level(self):
        """顶层 required 必须包含所有四个字段"""
        required = EXTRACT_VIEWPOINT_TOOL["function"]["parameters"]["required"]
        assert set(required) == {"summary", "viewpoints", "stance", "tags"}

    def test_tool_has_description(self):
        """必须有 description 帮助 LLM 理解何时调用"""
        assert len(EXTRACT_VIEWPOINT_TOOL["function"]["description"]) > 10


class TestSystemPrompt:
    """验证系统提示词的质量"""

    def test_system_prompt_is_not_empty(self):
        assert len(SYSTEM_PROMPT) > 50

    def test_system_prompt_mentions_tool_name(self):
        """提示词应该提到要使用的函数名"""
        assert "extract_article_viewpoints" in SYSTEM_PROMPT


class TestExtractViewpointsIntegration:
    """集成测试 — 需要真实 API key"""

    @pytest.mark.skip(reason="需要真实 API key")
    def test_extract_from_simple_article(self):
        """测试从一篇简单的文章提取观点"""
        article = """标题：远程办公的利与弊

        过去三年，远程办公从应急方案变成了常态。支持者认为，远程办公提升了员工的工作效率，
        减少了通勤时间，让员工有更多时间陪伴家人。斯坦福大学的一项研究显示，远程办公使
        员工生产力提升了 13%。

        然而，反对者指出，远程办公削弱了团队协作，增加了沟通成本，并导致了职业孤独感。
        Buffer 的调查显示，20% 的远程工作者认为孤独感是最大的挑战。

        从长远来看，混合办公模式可能是最优解——既保留了远程办公的灵活性，又维持了
        面对面协作的优势。微软的研究表明，73% 的员工希望保留远程办公的灵活性，
        但 67% 也渴望更多的面对面时间。
        """

        result = extract_viewpoints(article)

        # 验证返回结构
        assert isinstance(result, dict)
        assert "summary" in result
        assert "viewpoints" in result
        assert "stance" in result
        assert "tags" in result

        # 验证摘要
        assert len(result["summary"]) > 0

        # 验证观点
        assert isinstance(result["viewpoints"], list)
        assert len(result["viewpoints"]) >= 1
        for vp in result["viewpoints"]:
            assert "claim" in vp
            assert "reasoning" in vp
            assert "evidence" in vp
            assert len(vp["claim"]) > 0

        # 验证立场
        assert result["stance"] in ["support", "oppose", "neutral", "mixed"]

        # 验证标签数量
        assert 3 <= len(result["tags"]) <= 5

    @pytest.mark.skip(reason="需要真实 API key")
    def test_extract_from_short_text(self):
        """测试最短输入也能正常工作"""
        result = extract_viewpoints("标题：AI 改变世界\n正文：人工智能正在深刻改变各个行业。")
        assert isinstance(result, dict)
        assert "summary" in result
        assert len(result["viewpoints"]) >= 1
        assert 3 <= len(result["tags"]) <= 5

"""Digest Agent 的 LangGraph State 定义。

State 是 LangGraph 工作流中所有节点共享的数据容器。
每个节点读取 State 中的输入字段，写入输出字段。
工作流结束时，State 包含了完整的处理结果。

为什么用 TypedDict？
  LangGraph 用 TypedDict 来定义 State 的结构 —
  每个字段有明确的 key 和类型，编译器能检查错误。
  total=False 表示所有字段都是可选的（Optional），
  因为工作流逐步填充状态，开始时空字段不应该报错。
"""
from typing import TypedDict


class DigestState(TypedDict, total=False):
    """Digest Agent 的工作流状态。

    字段按工作流的执行顺序排列（输入 → 路由 → 解析 → 提取 → 存储 → 输出）。
    每个节点只负责写入自己产出的字段，读取上游节点产出的字段。
    """

    # ===== 输入 =====
    url: str
    """用户发送的文章 URL（工作流的唯一输入）"""

    # ===== 路由结果 =====
    platform: str
    """识别出的平台标识: zhihu / wechat / weibo / xiaohongshu / bilibili / generic"""

    # ===== 解析结果 =====
    parsed_title: str
    """解析出的文章标题"""

    parsed_author: str
    """解析出的作者名"""

    parsed_content: str
    """清洗后的文章正文文本"""

    parse_error: str
    """解析错误信息，非空字符串表示解析失败"""

    # ===== LLM 提取结果 ===== #
    summary: str
    """LLM 生成的文章摘要（200字以内）"""

    viewpoints: list[dict]
    """LLM 提取的核心观点列表，每项含 claim/reasoning/evidence"""

    stance: str
    """文章立场: support / oppose / neutral / mixed"""

    tags: list[str]
    """自动生成的分类标签（3-5个）"""

    # ===== 存储结果 =====
    article_id: str
    """存入数据库后获得的文章 ID"""

    # ===== 最终输出 =====
    reply_card: str
    """格式化后返回给用户的 Markdown 知识卡片"""

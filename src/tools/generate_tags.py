"""标签生成 Tool。从标题和摘要生成分类标签。

这是一个轻量级的工具 — 不需要 Function Calling，
直接用 chat() 让 LLM 返回逗号分隔的标签列表即可。

与 extract_viewpoint.py 的区别：
  - 本模块：输入是「标题 + 摘要」，输出只有标签列表，轻量快速
  - extract_viewpoint.py：输入是全文，一次性输出摘要+观点+立场+标签（更重但更全面）

使用场景：
  - 已经通过其他方式获得摘要，只需要补标签时
  - Digest Agent 的 extract_viewpoint 已经包含了标签生成，所以本模块是可选工具
"""
from src.llm.client import chat

TAG_SYSTEM_PROMPT = """你是一个内容分类专家。给定文章的标题和摘要，生成 3-5 个精准的分类标签。

要求：
1. 标签格式建议为 '领域-主题'（如 'AI-Agent'、'商业-SaaS'、'科技-芯片'）
2. 也可以直接用关键词（如 '远程办公'、'团队管理'）
3. 从具体到抽象排列
4. 每个标签不超过 15 个字

请直接返回标签，用中文逗号（，）分隔。不要输出任何其他内容。

示例：
输入：标题「GPT-5 发布：推理能力大幅提升」，摘要「OpenAI 发布了 GPT-5...」
输出：AI-GPT、大模型、深度学习、科技-人工智能"""


def generate_tags(
    title: str,
    summary: str,
    model: str | None = None,
) -> list[str]:
    """根据标题和摘要生成分类标签。

    Args:
        title: 文章标题
        summary: 文章摘要
        model: 使用的模型名，默认用配置里的

    Returns:
        标签字符串列表，最多 5 个

    示例:
        >>> generate_tags("GPT-5 发布", "OpenAI 发布了新一代模型...")
        ['AI-GPT', '大模型', '深度学习', '科技-人工智能']
    """
    messages = [
        {"role": "system", "content": TAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"标题：{title}\n\n摘要：{summary}\n\n请生成 3-5 个标签，用中文逗号分隔。",
        },
    ]

    response = chat(
        messages=messages,
        model=model,
        temperature=0.3,
        max_tokens=200,
    )

    # 解析标签：按中文/英文逗号、顿号、换行分割
    import re
    tags = re.split(r"[,，、\n]+", response.strip())
    # 清理每个标签：去首尾空白和引号
    tags = [t.strip().strip("'\"") for t in tags if t.strip()]
    return tags[:5]  # 最多 5 个

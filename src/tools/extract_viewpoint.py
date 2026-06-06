"""观点提取 Tool。使用 Function Calling 从文章中提取结构化观点。

这是 Function Calling 学习的核心模块 — 不给 LLM 自由发挥，
而是规定好 JSON Schema，让 LLM 按你的格式返回结构化数据。

与 generate_tags.py 的区别：
  - 本模块：一次性提取摘要 + 观点 + 立场 + 标签（一个 function call 全搞定）
  - generate_tags.py：单独从标题+摘要生成标签（更轻量，适合已有摘要的场景）
"""
from src.llm.client import function_call

# === Function Calling Schema ===
# 这个 schema 定义了 LLM 必须遵守的输出格式。
# OpenAI/DeepSeek 会根据这个 schema 确保返回的 JSON 结构正确。

EXTRACT_VIEWPOINT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_article_viewpoints",
        "description": (
            "从文章中提取核心观点、论证链和立场。"
            "仔细阅读全文，识别作者的主要论断和论证逻辑。"
            "不要添加你自己的观点，只提取作者表达的内容。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "文章200字以内的精炼摘要，涵盖核心内容",
                },
                "viewpoints": {
                    "type": "array",
                    "description": "文章中提取的核心观点列表。一篇文章通常有 1-3 个核心观点",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "核心论断：作者想要论证的主要命题，一句话说清楚",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "论证逻辑链：作者如何一步步论证这个论断",
                            },
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "支撑论据：作者使用的数据、案例、引用、类比等",
                            },
                        },
                        "required": ["claim", "reasoning", "evidence"],
                    },
                },
                "stance": {
                    "type": "string",
                    "enum": ["support", "oppose", "neutral", "mixed"],
                    "description": (
                        "文章对所讨论主题的整体立场。\n"
                        "  - support: 作者明确支持或赞同\n"
                        "  - oppose: 作者明确反对或批评\n"
                        "  - neutral: 作者客观陈述，无明显立场\n"
                        "  - mixed: 作者既支持又保留，态度复杂"
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "3-5个标签，从具体到抽象排列。\n"
                        "格式建议：'领域-主题'（如 'AI-Agent'、'商业-SaaS'、'社会-教育'）\n"
                        "也可以直接用关键词（如 '远程办公'、'团队管理'）"
                    ),
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
3. 摘要要精炼准确，控制在 200 字以内
4. 观点提取要抓住作者论证的核心，不要添加你自己的观点
5. 立场判断基于作者在文章中的实际表达，而不是你对这个话题的看法
6. 标签要准确反映文章的领域和主题，从具体到抽象排列

请确保你的分析客观、准确、完整。"""


def extract_viewpoints(
    article_text: str,
    model: str | None = None,
) -> dict:
    """从文章文本中提取结构化观点。

    这是 Digest Agent 工作流中最关键的一步 ——
    用 Function Calling 让 LLM 返回结构化 JSON，
    而不是自由文本。

    Args:
        article_text: 清洗后的文章全文（包含标题更好，如 "标题：xxx\n\n正文：xxx"）
        model: 使用的模型名，默认用配置里的

    Returns:
        {
            "summary": str,           # 200字摘要
            "viewpoints": [           # 核心观点列表
                {
                    "claim": str,      # 核心论断
                    "reasoning": str,  # 论证逻辑链
                    "evidence": [str], # 支撑论据
                }
            ],
            "stance": str,            # support / oppose / neutral / mixed
            "tags": [str],            # 3-5 个标签
        }

    Raises:
        ValueError: LLM 返回格式不对或缺少必要字段
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请分析以下文章：\n\n{article_text}"},
    ]

    result = function_call(
        messages=messages,
        tools=[EXTRACT_VIEWPOINT_TOOL],
        model=model,
        temperature=0.1,  # 低温度 = 确定性输出，保证一致性
    )

    # 验证返回结果包含所有必要字段
    required_fields = ["summary", "viewpoints", "stance", "tags"]
    for field in required_fields:
        if field not in result:
            raise ValueError(
                f"LLM 返回结果缺少必要字段: {field}\n"
                f"实际返回的字段: {list(result.keys())}"
            )

    return result

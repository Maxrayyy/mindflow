"""
LLM 客户端封装 — 统一调用 DeepSeek（或其他 OpenAI 兼容 API）的入口。

三个核心能力：
  1. chat()           → 普通对话，返回文本
  2. function_call()  → 让 LLM 按你规定的 JSON 结构返回数据（Agent 的基石！）
  3. get_embedding()  → 把文本变成数字向量（本地 sentence-transformers 模型）

为什么 Embedding 用本地模型而不是 API？
  DeepSeek 不提供 Embedding API。用本地 BGE 模型，零成本、离线可用。
"""
from openai import OpenAI
from src.config import settings

# 全局单例
_client: OpenAI | None = None
_embedding_model = None  # 惰性加载


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )
    return _client


def get_embedding(text: str, model: str | None = None) -> list[float]:
    """获取文本的向量嵌入（本地 BGE 模型）。

    首次调用会自动加载模型到内存（~100MB），之后复用。
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    vector = _embedding_model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str:
    """发送对话请求，返回纯文本回复。

    这是最基础的使用方式 — 你说一句话，LLM 回一句话。

    Args:
        messages: 对话历史，格式 [{"role": "system/user/assistant", "content": "..."}]
        model:    模型名，默认用配置里的（deepseek-chat）
        temperature: 0=固定答案, 1=随机创意（观点提取我们用 0.1 保证一致性）
        max_tokens: 最大输出长度

    Returns:
        LLM 的文本回复

    示例:
        >>> chat([{"role": "user", "content": "用一句话解释什么是 AI Agent"}])
        "AI Agent 是能够自主决策并使用工具完成复杂任务的人工智能系统。"
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
    """发送带 Function Calling 的请求，让 LLM 返回结构化 JSON。

    这是 Agent 最核心的能力！你不是让 LLM 随便说，
    而是规定好 JSON 格式，让 LLM 按你的格式填充数据。

    Args:
        messages:    对话消息
        tools:       function 定义列表（OpenAI format）
        model:       模型名
        temperature: 越低越确定（观点提取要确定，用 0.1）

    Returns:
        解析后的 JSON 字典，例如:
        {"summary": "本文讨论了...", "stance": "support", "tags": ["AI", "Agent"]}

    Raises:
        ValueError: 如果 LLM 没有返回 function call

    示例:
        >>> tools = [{"type": "function", "function": {"name": "summarize", ...}}]
        >>> function_call([{"role": "user", "content": "总结这篇文章..."}], tools)
        {"summary": "...", "key_points": ["..."]}
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model or settings.llm.model,
        messages=messages,
        tools=tools,
        temperature=temperature,
    )

    choice = response.choices[0]

    # LLM 决定调用 function → 返回结构化 JSON
    if choice.message.tool_calls:
        import json
        return json.loads(choice.message.tool_calls[0].function.arguments)

    # 有时候 LLM 不通过 tool_calls 而是直接在 content 里返回 JSON
    content = choice.message.content or ""
    if content.strip().startswith("{"):
        import json
        return json.loads(content)

    raise ValueError(
        f"LLM 未返回 function call 或有效 JSON。\n"
        f"原始回复前 200 字: {content[:200]}"
    )



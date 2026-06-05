"""
LLM 客户端封装 — 统一调用 DeepSeek（或其他 OpenAI 兼容 API）的入口。

三个核心能力：
  1. chat()           → 普通对话，返回文本
  2. function_call()  → 让 LLM 按你规定的 JSON 结构返回数据（Agent 的基石！）
  3. get_embedding()  → 把文本变成数字向量（用于语义搜索）

依赖：
  openai 这个包（OpenAI SDK）能直接调 DeepSeek，因为 DeepSeek 的 API 格式
  和 OpenAI 完全一样 — 只需改 base_url 就行。

  类比：你换了一个遥控器品牌，但按钮和红外码完全一样，所以不用重学。
"""
from openai import OpenAI
from src.config import settings

# 全局单例 — 整个应用只创建一个 OpenAI 客户端
# 为什么是单例？避免每次调用都重新创建连接（浪费资源）
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


def get_embedding(text: str, model: str | None = None) -> list[float]:
    """获取文本的向量嵌入（embedding）。

    把一段文字变成一串数字（向量），相似的文字在向量空间中距离更近。
    这是语义搜索的基础。

    Args:
        text:  要向量化的文本
        model: embedding 模型名

    Returns:
        浮点数列表，如 [0.023, -0.451, ...]

    注意:
        DeepSeek 不提供 Embedding API，所以我们后面会改用本地模型。
        这里先保留接口，Task 6 时替换实现。
    """
    client = get_client()
    response = client.embeddings.create(
        model=model or settings.llm.embedding_model,
        input=text,
    )
    return response.data[0].embedding

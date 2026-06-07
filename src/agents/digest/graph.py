"""Digest Agent 的 LangGraph 工作流组装。

工作流:
  URL路由 → 内容解析 → 观点提取 → 持久化 → 卡片回复

        ┌─────────────┐
        │  url_router │  识别平台 + 去重检查
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │    parse    │  调用解析器抓取内容
        └──────┬──────┘
               │
          ┌────┴────┐
          │ 失败？   │── 是 ──▶ format_card（显示错误）
          │ 成功？   │── 是 ──▶ extract_viewpoint
          └─────────┘
               │
               ▼
        ┌─────────────────┐
        │ extract_viewpoint│  LLM Function Calling
        └────────┬────────┘
                 │
            ┌────┴────┐
            │ 失败？   │── 是 ──▶ END（返回错误卡片）
            │ 成功？   │── 是 ──▶ store
            └─────────┘
                 │
                 ▼
        ┌─────────────┐
        │    store    │  SQLite + Chroma 持久化
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │ format_card │  生成 Markdown 知识卡片
        └──────┬──────┘
               │
               ▼
             END
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


def should_after_parse(state: dict) -> str:
    """解析后的条件路由。

    解析失败 → 直接跳到 format_card（显示错误信息）
    解析成功 → 继续 extract_viewpoint
    """
    if state.get("parse_error"):
        return "format_card"
    return "extract_viewpoint"


def should_after_extract(state: dict) -> str:
    """观点提取后的条件路由。

    提取失败（已有 reply_card 错误信息）→ 直接结束
    提取成功 → 继续 store
    """
    if state.get("reply_card"):
        return END
    return "store"


def build_digest_graph() -> StateGraph:
    """构建并编译 Digest Agent 的 LangGraph 工作流。

    Returns:
        编译好的 StateGraph，可调用 .ainvoke({"url": "..."}) 或
        .invoke({"url": "..."}) 执行
    """
    workflow = StateGraph(DigestState)

    # 注册五个节点
    workflow.add_node("url_router", url_router_node)
    workflow.add_node("parse", parse_node)
    workflow.add_node("extract_viewpoint", extract_viewpoint_node)
    workflow.add_node("store", store_node)
    workflow.add_node("format_card", format_card_node)

    # 设置入口
    workflow.set_entry_point("url_router")

    # 构建边
    workflow.add_edge("url_router", "parse")

    # 条件边：解析成功 → 提取观点；解析失败 → 格式化错误卡片
    workflow.add_conditional_edges(
        "parse",
        should_after_parse,
        {
            "extract_viewpoint": "extract_viewpoint",
            "format_card": "format_card",
        },
    )

    # 条件边：观点提取成功 → 存储；失败 → 直接结束
    workflow.add_conditional_edges(
        "extract_viewpoint",
        should_after_extract,
        {
            "store": "store",
            END: END,
        },
    )

    workflow.add_edge("store", "format_card")
    workflow.add_edge("format_card", END)

    return workflow.compile()


# 全局单例 — 模块导入时编译一次，之后复用
digest_agent = build_digest_graph()

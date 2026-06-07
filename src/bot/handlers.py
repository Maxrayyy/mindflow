"""Telegram Bot 消息处理器。

定义四个处理器：
  /start  → 欢迎消息，告诉用户怎么用
  /help   → 使用指南
  /stats  → 知识库统计
  普通消息 → 检测 URL → 启动 Digest Agent → 返回知识卡片
"""
import re
from telegram import Update
from telegram.ext import ContextTypes
from src.agents.digest.graph import digest_agent


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /start 命令"""
    welcome = (
        "🧠 *MindFlow* — 你的第二大脑\n\n"
        "直接发送任意文章链接给我，我会：\n\n"
        "1\\. 🔍 自动识别平台并抓取内容\n"
        "2\\. 🧠 提取核心观点和论证链\n"
        "3\\. 🏷️ 自动分类打标签\n"
        "4\\. 📊 判断文章立场\n"
        "5\\. 💾 存入你的个人知识库\n\n"
        "以后你还可以问我：\n"
        "• \"关于 XXX，我收藏了哪些观点？\"\n"
        "• \"最近有哪些和我观点相反的文章？\"\n\n"
        "发送链接试试吧 👇"
    )
    await update.message.reply_text(welcome, parse_mode="MarkdownV2")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /help 命令"""
    help_text = (
        "📚 *MindFlow 使用指南*\n\n"
        "*核心功能：* 发送任意文章链接，我会自动解析并提取观点。\n\n"
        "*支持平台：*\n"
        "• 知乎 \\(zhihu\\.com\\)\n"
        "• 微信公众号 \\(mp\\.weixin\\.qq\\.com\\)\n"
        "• 微博 \\(weibo\\.com\\)\n"
        "• 小红书 \\(xiaohongshu\\.com\\)\n"
        "• B站专栏 \\(bilibili\\.com/read\\)\n"
        "• 其他网页（通用解析）\n\n"
        "*命令列表：*\n"
        "/start \\- 开始使用\n"
        "/help \\- 查看帮助\n"
        "/stats \\- 查看知识库统计\n\n"
        "*使用技巧：*\n"
        "• 直接转发或粘贴链接\n"
        "• 解析通常在 10\\-30 秒内完成\n"
        "• 重复的链接会自动跳过"
    )
    await update.message.reply_text(help_text, parse_mode="MarkdownV2")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /stats 命令 — 查看知识库统计"""
    from src.memory.metadata_store import MetadataStore
    from src.memory.vector_store import VectorStore

    meta = MetadataStore()
    vec = VectorStore()

    article_count = meta.get_article_count()
    vector_count = vec.count()

    stats_text = (
        "📊 *MindFlow 知识库统计*\n\n"
        f"📄 已收藏文章：{article_count} 篇\n"
        f"🧮 向量索引数：{vector_count} 条\n\n"
        "继续发送链接来丰富你的知识库！"
    )
    await update.message.reply_text(stats_text, parse_mode="MarkdownV2")


# URL 匹配正则 — 从消息文本中提取第一个 http/https 链接
URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理普通消息 — 检测 URL 并启动 Digest Agent。

    工作流程:
      1. 提取消息中的 URL
      2. 发送「正在解析...」提示
      3. 调用 digest_agent.ainvoke() 运行完整工作流
      4. 返回知识卡片（或错误信息）
    """
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
        # 运行 Digest Agent — 这会把五个节点串起来执行
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

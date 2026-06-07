"""MindFlow Telegram Bot 入口。

启动方式:
    python -m src.bot.main

前置条件:
    1. .env 中已配置 TELEGRAM_BOT_TOKEN（从 @BotFather 获取）
    2. .env 中已配置 OPENAI_API_KEY 和 OPENAI_BASE_URL
    3. 网络能访问 Telegram API（国内需要代理）：
       - TELEGRAM_PROXY=auto           → 自动检测 WSL2 网关 + Clash 端口
       - TELEGRAM_PROXY=http://x:x:7890 → 手动指定代理地址
"""
import os
import subprocess
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram.request import HTTPXRequest
from src.config import settings
from src.bot.handlers import (
    start_command,
    help_command,
    stats_command,
    handle_message,
)


def _detect_proxy() -> str | None:
    """自动检测代理地址。

    支持两种模式：
      1. TELEGRAM_PROXY=auto — WSL2 下自动找 Windows 主机的 Clash 代理
      2. TELEGRAM_PROXY=http://... — 直接用指定的地址
    """
    proxy_url = os.getenv("TELEGRAM_PROXY", "")
    if not proxy_url:
        return None

    if proxy_url == "auto":
        # WSL2: 通过 ip route 获取 Windows 主机 IP
        try:
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True, timeout=5,
            )
            for part in result.stdout.split():
                if part.count(".") == 3 and part.split(".")[0] in ("172", "192", "10"):
                    gateway = part
                    url = f"http://{gateway}:7897"
                    print(f"🔗 自动检测代理: {url}")
                    return url
        except Exception:
            pass
    else:
        print(f"🔗 使用代理: {proxy_url}")
        return proxy_url

    return None


def main() -> None:
    """启动 Telegram Bot"""
    print("🤖 MindFlow Bot 正在启动...")
    print(f"📊 LLM 模型: {settings.llm.model}")
    print(f"💾 数据目录: {settings.storage.data_dir}")

    if not settings.bot.token:
        print("❌ 未配置 TELEGRAM_BOT_TOKEN，请在 .env 中设置后重试")
        return

    # 代理配置 — 国内访问 Telegram API 需要
    proxy_url = _detect_proxy()
    request = HTTPXRequest(proxy=proxy_url) if proxy_url else None

    # Application 是 Bot 的核心 — 管理事件循环和处理器分发
    app = Application.builder().token(settings.bot.token).request(request).build()

    # 注册命令处理器 — 精确匹配 /start、/help、/stats
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))

    # 注册消息处理器 — 匹配所有非命令的文本消息
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot 已启动，等待消息...")
    # run_polling 开始轮询 Telegram 服务器，收到消息自动分发给对应处理器
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

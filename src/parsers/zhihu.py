"""
知乎解析器 — 支持知乎问答页面和专栏文章。

技术要点：
  1. httpx.AsyncClient — 异步发 HTTP 请求
  2. BeautifulSoup — 解析 HTML，用 CSS 选择器定位内容区域
  3. User-Agent 伪装 — 告诉知乎"我是浏览器"，否则会被拒绝访问
  4. 走 Clash 代理 — WSL2 网络被 TUN 劫持，走代理才能用 Clash 的直连规则
"""
import os
import subprocess
import httpx
from bs4 import BeautifulSoup
from src.parsers.base import BaseParser, ParsedArticle, ParseError


def _get_proxy() -> str | None:
    """解析 TELEGRAM_PROXY 环境变量，支持 auto 模式自动检测 WSL2 网关。"""
    proxy_url = os.getenv("TELEGRAM_PROXY", "")
    if not proxy_url:
        return None
    if proxy_url == "auto":
        try:
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True, timeout=5,
            )
            for part in result.stdout.split():
                if part.count(".") == 3:
                    return f"http://{part}:7897"
        except Exception:
            pass
    return proxy_url


class ZhihuParser(BaseParser):
    platform_name = "zhihu"

    def can_parse(self, url: str) -> bool:
        """只要 URL 里包含 zhihu.com 就认为能处理"""
        return "zhihu.com" in url

    async def parse(self, url: str) -> ParsedArticle:
        try:
            # 走 Clash 代理 — Clash 里配 zhihu.com → DIRECT 就能用国内 IP 访问
            async with httpx.AsyncClient(timeout=15.0, proxy=_get_proxy()) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                        "Accept": (
                            "text/html,application/xhtml+xml,application/xml;"
                            "q=0.9,image/webp,*/*;q=0.8"
                        ),
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Cache-Control": "no-cache",
                        "Sec-Ch-Ua": (
                            '"Not_A Brand";v="8", "Chromium";v="120", '
                            '"Google Chrome";v="120"'
                        ),
                        "Sec-Ch-Ua-Mobile": "?0",
                        "Sec-Ch-Ua-Platform": '"Windows"',
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "none",
                        "Sec-Fetch-User": "?1",
                    },
                    follow_redirects=True,  # 如果链接跳转，自动跟随
                )
                response.raise_for_status()  # 如果返回 404/500 等错误，直接抛异常

            # 第 2 步：用 BeautifulSoup 解析 HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # 第 3 步：提取标题
            # CSS 选择器说明：
            #   h1.QuestionHeader-title  → <h1 class="QuestionHeader-title">
            #   h1.Post-Title            → <h1 class="Post-Title">
            #   selector1, selector2     → 逗号 = "或者"
            title = ""
            title_tag = (
                soup.select_one("h1.QuestionHeader-title")
                or soup.select_one("h1.Post-Title")
                or soup.select_one("h1")  # 最终兜底：随便找一个 h1
            )
            if title_tag:
                title = title_tag.get_text(strip=True)

            # 第 4 步：提取正文
            # 知乎的内容放在这些 CSS class 里：
            #   .RichText           — 回答的正文
            #   .Post-RichText      — 专栏文章正文
            #   .RichContent-inner  — 其他可能的位置
            content_parts = []
            selectors = [".RichText", ".Post-RichText", ".RichContent-inner"]

            for selector in selectors:
                blocks = soup.select(selector)
                for block in blocks:
                    text = block.get_text(separator="\n", strip=True)
                    if text:
                        content_parts.append(text)

            content = "\n\n".join(content_parts)

            if not content:
                raise ParseError("未能提取到知乎页面正文内容", platform="zhihu", url=url)

            return ParsedArticle(
                title=title or "知乎文章",
                author="知乎用户",
                content=content[:5000],  # 限制 5000 字，避免超过 LLM token 限制
                platform="zhihu",
                parse_method="beautifulsoup",
                metadata={"url": url},
            )

        except httpx.HTTPError as e:
            raise ParseError(f"知乎请求失败（可能是网络问题）: {e}", platform="zhihu", url=url)
        except ParseError:
            raise  # 直接往上抛，不要包一层
        except Exception as e:
            raise ParseError(f"知乎解析未知错误: {e}", platform="zhihu", url=url)

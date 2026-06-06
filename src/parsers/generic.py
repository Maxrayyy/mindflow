"""
通用网页解析器 — 两层兜底策略。

策略 1: Jina Reader（优先）
  把 URL 前面加上 r.jina.ai 即可，免费，返回干净 Markdown。
  例如: https://r.jina.ai/https://example.com/article

策略 2: readability-lxml（本地兜底）
  类似 Safari 的「阅读模式」，自动识别网页中的正文区域，
  过滤掉导航栏、广告、侧边栏。

两个都失败 → 抛出 ParseError，最终用户看到友好的错误提示。
"""
import httpx
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDoc
from src.parsers.base import BaseParser, ParsedArticle, ParseError


class GenericParser(BaseParser):
    platform_name = "generic"

    def can_parse(self, url: str) -> bool:
        # 通用解析器接受任何 URL（作为最后一道防线）
        return True

    async def parse(self, url: str) -> ParsedArticle:
        # 先试 Jina Reader
        article = await self._try_jina_reader(url)
        if article:
            return article

        # Jina 失败 → 试本地 readability
        article = await self._try_readability(url)
        if article:
            return article

        raise ParseError("无法解析该网页：Jina Reader 和本地解析均失败", platform="generic", url=url)

    async def _try_jina_reader(self, url: str) -> ParsedArticle | None:
        """Jina Reader: 在 URL 前加 r.jina.ai。返回干净 Markdown。"""
        try:
            jina_url = f"https://r.jina.ai/{url}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    jina_url,
                    headers={"Accept": "text/markdown"},
                    follow_redirects=True,
                )
                if response.status_code != 200:
                    return None

                text = response.text.strip()
                if not text or len(text) < 50:
                    return None

                # Jina 返回 Markdown，第一行通常是标题
                lines = text.split("\n")
                title = lines[0].lstrip("# ").strip() if lines else "未命名"
                content = "\n".join(lines[1:]) if len(lines) > 1 else text

                return ParsedArticle(
                    title=title[:200],
                    author="",
                    content=content[:5000],
                    platform="generic",
                    parse_method="jina_reader",
                    metadata={"url": url},
                )
        except Exception:
            return None

    async def _try_readability(self, url: str) -> ParsedArticle | None:
        """本地 readability 解析：自动识别正文区域。"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                    },
                    follow_redirects=True,
                )
                response.raise_for_status()

            doc = ReadabilityDoc(response.text)
            title = doc.title() or ""
            html_content = doc.summary()

            soup = BeautifulSoup(html_content, "html.parser")
            content = soup.get_text(separator="\n", strip=True)

            if not content or len(content) < 100:
                # readability 失败 → 回退到 body 全文
                soup = BeautifulSoup(response.text, "html.parser")
                body = soup.find("body")
                if body:
                    for tag in body(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()
                    content = body.get_text(separator="\n", strip=True)

            if not content or len(content) < 50:
                return None

            return ParsedArticle(
                title=title[:200] or "未命名文章",
                author="",
                content=content[:5000],
                platform="generic",
                parse_method="readability",
                metadata={"url": url},
            )

        except Exception:
            return None

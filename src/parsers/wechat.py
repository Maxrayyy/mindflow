"""
微信公众号文章解析器。

与知乎解析器结构几乎一样，只有两个区别：
  1. HTML 结构不同 → CSS 选择器不同
  2. 公众号文章结构更规范 → 提取更容易

公众号文章的固定结构：
  <h1 id="activity-name">   ← 标题
  <div id="js_content">     ← 正文
  <div id="js_name">        ← 公众号名称
"""
import httpx
from bs4 import BeautifulSoup
from src.parsers.base import BaseParser, ParsedArticle, ParseError


class WechatParser(BaseParser):
    platform_name = "wechat"

    def can_parse(self, url: str) -> bool:
        return "mp.weixin.qq.com" in url

    async def parse(self, url: str) -> ParsedArticle:
        try:
            # 第 1 步：发请求
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

            soup = BeautifulSoup(response.text, "html.parser")

            # 第 2 步：提取标题 — #activity-name 是公众号文章标题的固定 ID
            title = "微信公众号文章"
            title_tag = soup.select_one("#activity-name")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # 第 3 步：提取作者 — #js_name 是公众号名称的固定 ID
            author = "未知公众号"
            author_tag = soup.select_one("#js_name")
            if author_tag:
                author = author_tag.get_text(strip=True)

            # 第 4 步：提取正文 — #js_content 是正文的固定 ID
            content_tag = soup.select_one("#js_content")
            if not content_tag:
                raise ParseError("未找到公众号正文内容", platform="wechat", url=url)

            # 移除隐藏元素（公众号文章里有一些不可见的占位符）
            for hidden in content_tag.select('[style*="visibility: hidden"]'):
                hidden.decompose()

            content = content_tag.get_text(separator="\n", strip=True)

            if not content:
                raise ParseError("公众号文章正文为空", platform="wechat", url=url)

            return ParsedArticle(
                title=title,
                author=author,
                content=content[:5000],
                platform="wechat",
                parse_method="beautifulsoup",
                metadata={"url": url},
            )

        except httpx.HTTPError as e:
            raise ParseError(f"公众号请求失败: {e}", platform="wechat", url=url)
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"公众号解析未知错误: {e}", platform="wechat", url=url)

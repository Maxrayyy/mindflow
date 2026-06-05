"""
解析器抽象基类 — 定义所有平台解析器的统一接口。

为什么需要抽象基类？
  不同平台（知乎/公众号/B站）的 HTML 结构完全不同，
  但最终都要返回同样的 ParsedArticle 结构。
  基类强制每个解析器实现 can_parse() 和 parse()，
  保证接口一致，换平台就像换电池。

类比：USB 接口 — 不管你插的是 U盘、鼠标还是键盘，
      只要符合 USB 规范（我们的 BaseParser），就能用。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ParsedArticle:
    """解析后的统一文章结构。

    不管原始来源是知乎还是公众号，
    最终都变成这个格式 — 这就是「统一接口」的价值。
    """
    title: str                              # 文章标题
    author: str = "未知"                      # 作者
    content: str = ""                        # 清洗后的正文
    platform: str = "generic"                # 平台标识: zhihu/wechat/weibo/...
    metadata: dict = field(default_factory=dict)  # 平台特有信息，如原始 URL
    parse_method: str = "unknown"            # 用哪种方式解析的（方便调试）

    def to_text(self) -> str:
        """转为 LLM 可读的文本。

        后续观点提取时，我们直接把这个文本喂给 LLM。
        """
        return (
            f"标题：{self.title}\n"
            f"作者：{self.author}\n"
            f"平台：{self.platform}\n"
            f"\n正文：\n{self.content}"
        )


class BaseParser(ABC):
    """解析器基类 — 所有平台解析器必须继承它。

    子类必须实现：
      - can_parse(url): 判断能不能处理这个 URL
      - parse(url):     实际抓取并返回 ParsedArticle

    子类必须设置：
      - platform_name:  平台标识字符串
    """

    # 子类必须覆盖这个属性
    platform_name: str = "generic"

    @abstractmethod
    def can_parse(self, url: str) -> bool:
        """判断该解析器能否处理这个 URL。

        例如 ZhihuParser 只对 zhihu.com 的链接返回 True。
        """
        ...

    @abstractmethod
    async def parse(self, url: str) -> ParsedArticle:
        """异步解析 URL，返回统一的 ParsedArticle。

        为什么是 async？
          网络请求（抓网页）是 IO 密集型操作，async 让程序在等待
          网页返回时可以去干别的事，不浪费时间。

        Raises:
            ParseError: 解析失败时抛出
        """
        ...


class ParseError(Exception):
    """解析异常 — 带平台和 URL 信息，方便定位问题"""
    def __init__(self, message: str, platform: str = "", url: str = ""):
        self.platform = platform
        self.url = url
        super().__init__(message)

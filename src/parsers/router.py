"""
URL 路由器 — 根据 URL 模式自动识别平台，分发到对应的解析器。

核心逻辑非常简单：
  1. 用正则表达式匹配 URL 中的域名和路径
  2. 匹配成功 → 返回平台标识
  3. 都不匹配 → 返回 "generic"（通用解析器兜底）

为什么不用 AI 来判断平台？
  不需要！URL 格式是固定规则，正则比 AI 快 1000 倍、100% 准确、零成本。
  AI 应该用在「理解内容」这种规则搞不定的事情上。
"""
import re
from src.parsers.base import BaseParser, ParseError

# URL → 平台 的映射表
# 格式: (平台标识, 正则表达式)
# 顺序重要！越具体的规则越靠前
PLATFORM_RULES: list[tuple[str, str]] = [
    # 知乎 — 问答和专栏两种格式
    ("zhihu", r"zhihu\.com/(question/\d+|answer/\d+|p/\w+)"),
    ("zhihu", r"zhuanlan\.zhihu\.com/p/\w+"),

    # 微信公众号 — 文章都在 /s/ 路径下
    ("wechat", r"mp\.weixin\.qq\.com/s/"),

    # 微博 — PC 和移动端两个域名
    ("weibo", r"(weibo\.com/|m\.weibo\.cn/)"),

    # 小红书 — 主站和短链
    ("xiaohongshu", r"(xhslink\.com/|xiaohongshu\.com/)"),

    # B站 — 专栏文章和视频简介
    ("bilibili", r"bilibili\.com/(read/|video/)"),
]


def detect_platform(url: str) -> str:
    """识别 URL 所属平台。

    纯函数，无副作用 — 同样的 URL 永远返回同样的结果。

    Args:
        url: 任意网址

    Returns:
        平台标识: zhihu / wechat / weibo / xiaohongshu / bilibili / generic

    Examples:
        >>> detect_platform("https://zhuanlan.zhihu.com/p/12345")
        "zhihu"
        >>> detect_platform("https://example.com/blog")
        "generic"
    """
    if not url:
        return "generic"

    for platform_name, pattern in PLATFORM_RULES:
        if re.search(pattern, url, re.IGNORECASE):
            return platform_name

    return "generic"


class URLRouter:
    """URL 路由器 — 持有所有解析器，负责 URL → 解析器的分发。

    用法:
        router = URLRouter()
        router.register(ZhihuParser())
        router.register(WechatParser())

        platform = router.route("https://zhuanlan.zhihu.com/p/123")
        parser = router.get_parser(platform)
        article = await parser.parse(url)
    """

    def __init__(self):
        # 解析器注册表: {"zhihu": ZhihuParser(), "wechat": WechatParser(), ...}
        self._parsers: dict[str, BaseParser] = {}

    def register(self, parser: BaseParser) -> None:
        """注册一个解析器。同一个平台后注册的会覆盖先注册的。"""
        self._parsers[parser.platform_name] = parser

    def get_parser(self, platform: str) -> BaseParser | None:
        """根据平台标识获取解析器。未注册返回 None。"""
        return self._parsers.get(platform)

    def route(self, url: str) -> str:
        """识别 URL 平台，保证至少返回 'generic'"""
        return detect_platform(url)

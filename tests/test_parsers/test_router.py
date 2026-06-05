"""URL 路由测试 — 验证每个平台的 URL 能正确识别

测试哲学：
  每个测试只测一件事（一个 URL → 一个平台），
  出问题一眼就知道哪个规则挂了。
"""
from src.parsers.router import detect_platform, URLRouter


class TestDetectPlatform:
    """测试 detect_platform() 函数 — 纯规则匹配，不需要网络"""

    def test_zhihu_question(self):
        """知乎问答链接"""
        assert detect_platform("https://www.zhihu.com/question/12345678") == "zhihu"

    def test_zhihu_zhuanlan(self):
        """知乎专栏链接"""
        assert detect_platform("https://zhuanlan.zhihu.com/p/abc123xyz") == "zhihu"

    def test_wechat_article(self):
        """微信公众号文章"""
        assert detect_platform("https://mp.weixin.qq.com/s/abc123def456") == "wechat"

    def test_weibo_pc(self):
        """微博 PC 端"""
        assert detect_platform("https://weibo.com/1234567890/AbCdEf") == "weibo"

    def test_weibo_mobile(self):
        """微博移动端"""
        assert detect_platform("https://m.weibo.cn/detail/1234567890") == "weibo"

    def test_xiaohongshu(self):
        """小红书主站"""
        assert detect_platform("https://www.xiaohongshu.com/explore/abc123") == "xiaohongshu"

    def test_bilibili_article(self):
        """B站专栏"""
        assert detect_platform("https://www.bilibili.com/read/cv12345678") == "bilibili"

    def test_bilibili_video(self):
        """B站视频"""
        assert detect_platform("https://www.bilibili.com/video/BV1xx411c7mD") == "bilibili"

    def test_generic_url(self):
        """不在已知平台列表中的 URL"""
        assert detect_platform("https://www.example.com/blog/hello-world") == "generic"

    def test_empty_url(self):
        """空字符串也应有兜底"""
        assert detect_platform("") == "generic"


class TestURLRouter:
    """测试 URLRouter 类"""

    def test_route_works_without_parsers(self):
        """即使没注册任何解析器，route() 也能正常工作"""
        router = URLRouter()
        assert router.route("https://mp.weixin.qq.com/s/abc") == "wechat"

    def test_get_parser_returns_none_when_not_registered(self):
        """未注册的平台返回 None"""
        router = URLRouter()
        assert router.get_parser("zhihu") is None

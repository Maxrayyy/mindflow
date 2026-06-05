# 正则表达式入门 — 用 MindFlow 的 URL 路由来学

> 看完这篇，你不仅能读懂 `router.py` 里的正则，还能自己写。

---

## 1. 正则是什么？

一句话：**用符号描述"字符串长什么样"的语言。**

```
正常语言:  "匹配所有以 zhihu.com 开头的网址"
正则语言:  r"zhihu\.com"
```

正则就像一个筛子——你描述一个模式，它帮你从一堆文字中筛出匹配的。

---

## 2. 必须记住的 6 个符号

| 符号 | 含义 | 示例 | 能匹配什么？ |
|------|------|------|------------|
| `.` | 任意**一个**字符 | `zhihu.com` | `zhihuXcom`、`zhihu.com`（注意：点号本身也能匹配） |
| `\.` | 真正的「点」 | `zhihu\.com` | 只匹配 `zhihu.com` |
| `\d` | 一个数字 0-9 | `\d+` | `123`、`45678` |
| `\w` | 一个字母/数字/下划线 | `\w+` | `abc`、`abc123`、`hello_world` |
| `+` | 前面的东西出现 1 次或多次 | `\d+` | `1`、`99`、`123456` |
| `*` | 前面的东西出现 0 次或多次 | `.*` | `""`（空）、`abc`、`任意内容` |

---

## 3. 逐行拆解 MindFlow 的 URL 规则

### 规则 1: 知乎

```python
("zhihu", r"zhihu\.com/(question/\d+|answer/\d+|p/\w+)")
```

逐段拆解：

```
zhihu\.com/                    # 匹配 "zhihu.com/"（\. 是真点号）
           (                   # 开始分组（里面是三选一）
            question/\d+       #   选项A: "question/" + 一到多个数字
            |                  #   | = "或者"
            answer/\d+         #   选项B: "answer/" + 一到多个数字
            |                  #   | = "或者"
            p/\w+              #   选项C: "p/" + 一到多个字母/数字
           )                   # 分组结束
```

**能匹配的 URL：**
```
https://www.zhihu.com/question/12345678    ✅ question/ + 数字
https://www.zhihu.com/answer/999           ✅ answer/ + 数字
https://www.zhihu.com/p/hello123           ✅ p/ + 字母数字组合
https://www.zhihu.com/people/max           ❌ people 不在规则中
```

### 规则 2: 知乎专栏

```python
("zhihu", r"zhuanlan\.zhihu\.com/p/\w+")
```

```
zhuanlan\.zhihu\.com/p/        # 匹配 "zhuanlan.zhihu.com/p/"
                      \w+      # 一到多个字母/数字
```

**能匹配的 URL：**
```
https://zhuanlan.zhihu.com/p/abc123xyz   ✅
https://zhuanlan.zhihu.com/p/12345678    ✅
```

### 规则 3: 微信公众号

```python
("wechat", r"mp\.weixin\.qq\.com/s/")
```

```
mp\.weixin\.qq\.com/s/         # 就是匹配这个固定字符串
```

**能匹配的 URL：**
```
https://mp.weixin.qq.com/s/abc123def456   ✅
https://mp.weixin.qq.com/s/anything       ✅
```

> 这个最简单——公众号文章都在 `/s/` 路径下，直接匹配固定字符串就行。不需要 `+` `*` 这些符号。

### 规则 4: 微博

```python
("weibo", r"(weibo\.com/|m\.weibo\.cn/)")
```

```
(                     # 开始分组（二选一）
 weibo\.com/          #   选项A: "weibo.com/"
 |                    #   "或者"
 m\.weibo\.cn/        #   选项B: "m.weibo.cn/"
)                     # 结束分组
```

**能匹配的 URL：**
```
https://weibo.com/1234567890/AbCdEf     ✅ PC端
https://m.weibo.cn/detail/1234567890    ✅ 移动端
```

### 规则 5: 小红书

```python
("xiaohongshu", r"(xhslink\.com/|xiaohongshu\.com/)")
```

```
xhs = 小红书首字母缩写
(xhslink\.com/|xiaohongshu\.com/)    # 短链域名 或 主站域名
```

### 规则 6: B站

```python
("bilibili", r"bilibili\.com/(read/|video/)")
```

```
bilibili\.com/                  # B站主域名
              (read/|video/)    # 专栏 或 视频
```

---

## 4. 为什么有些字符要加 `\`？

```
.com  → 正则里的 . 表示"任意字符"，能匹配 .com 也能匹配 Xcom
\.com → 加 \ 转义后只匹配真正的点号
```

**需要转义的字符：** `.` `*` `+` `?` `(` `)` `[` `]` `{` `}` `\` `|` `^` `$`

> 记不住？没关系。凡是"标点符号"，在正则里大概率有特殊含义。要匹配它本身，前面加 `\`。

---

## 5. `r"..."` 前面的 `r` 是什么意思？

```python
"zhihu\\.com"    # 普通字符串，\\ 才表示一个反斜杠
r"zhihu\.com"    # 原始字符串，\. 就是 \. 不需要额外转义
```

`r` = raw string（原始字符串），让 `\` 保持原样，**写正则必备**。否则你得写 `\\d` 才能匹配数字，太容易出错了。

---

## 6. 速查小抄

| 你要匹配... | 用这个 | 示例 |
|------------|--------|------|
| 固定文字 | 直接写 | `mp.weixin.qq.com` |
| 数字 | `\d+` | 匹配 `123`、`45678` |
| 字母数字混合 | `\w+` | 匹配 `abc123`、`hello` |
| 任意字符 | `.*` | 匹配任意内容 |
| 真正的点号 | `\.` | 匹配 `.` |
| 二选一 | `(A|B)` | `(weibo\.com|m\.weibo\.cn)` |
| 出现 1 次以上 | `+` | `\d+` 至少 1 个数字 |
| 出现 0 次以上 | `*` | `.*` 可以什么都没有 |

---

## 7. 练手：如果新增一个平台，你怎么写规则？

假设要加**即刻**（域名是 `okjk.com`）：

```python
("jike", r"okjk\.com/")
```

假设要加**豆瓣**的日记页面（`douban.com/note/`）：

```python
("douban", r"douban\.com/note/\d+")
```

---

就这么简单。正则不需要背，用的时候查这张表就够了。

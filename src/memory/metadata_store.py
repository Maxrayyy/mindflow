"""SQLite 元数据存储 — 精确查询文章信息和用户偏好。

Chroma 管语义搜索，SQLite 管精确查询：
  - 这篇文章存过没？（查 url）
  - 总共收藏了多少篇？
  - 用户偏好标签是什么？
"""
import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from src.config import settings


class MetadataStore:
    """SQLite 元数据管理器"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = str(db_path or settings.storage.db_path)
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 让查询结果可用 row["field"] 访问
        return conn

    def _ensure_tables(self) -> None:
        """创建表（如果不存在）"""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    platform TEXT NOT NULL,
                    title TEXT NOT NULL,
                    author TEXT DEFAULT '未知',
                    clean_content TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    stance TEXT DEFAULT 'neutral',
                    extracted_at TEXT NOT NULL,
                    source_metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS viewpoints (
                    id TEXT PRIMARY KEY,
                    article_id TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    evidence TEXT DEFAULT '[]',
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (article_id) REFERENCES articles(id)
                );

                CREATE TABLE IF NOT EXISTS article_tags (
                    article_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (article_id, tag),
                    FOREIGN KEY (article_id) REFERENCES articles(id)
                );

                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

    def insert_article(
        self, url: str, platform: str, title: str, clean_content: str,
        author: str = "未知", summary: str = "", stance: str = "neutral",
        source_metadata: dict | None = None,
    ) -> str:
        """插入文章，返回 article_id。URL 重复时会更新。"""
        article_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO articles
                   (id, url, platform, title, author, clean_content,
                    summary, stance, extracted_at, source_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (article_id, url, platform, title, author, clean_content,
                 summary, stance, now, json.dumps(source_metadata or {}, ensure_ascii=False)),
            )
        return article_id

    def insert_viewpoints(self, article_id: str, viewpoints: list[dict]) -> None:
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            for vp in viewpoints:
                conn.execute(
                    """INSERT INTO viewpoints (id, article_id, claim, reasoning, evidence, confidence, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (str(uuid.uuid4()), article_id,
                     vp.get("claim", ""), vp.get("reasoning", ""),
                     json.dumps(vp.get("evidence", []), ensure_ascii=False),
                     vp.get("confidence", 1.0), now),
                )

    def insert_tags(self, article_id: str, tags: list[str]) -> None:
        with self._get_conn() as conn:
            for tag in tags:
                conn.execute(
                    "INSERT OR IGNORE INTO article_tags (article_id, tag) VALUES (?, ?)",
                    (article_id, tag),
                )

    def get_article(self, article_id: str) -> dict | None:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
            return dict(row) if row else None

    def article_exists(self, url: str) -> str | None:
        """检查 URL 是否已存在，返回 article_id 或 None"""
        with self._get_conn() as conn:
            row = conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()
            return row["id"] if row else None

    def set_preference(self, key: str, value: str) -> None:
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now),
            )

    def get_preference(self, key: str, default: str = "") -> str:
        with self._get_conn() as conn:
            row = conn.execute("SELECT value FROM user_preferences WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def get_article_count(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM articles").fetchone()
            return row["cnt"] if row else 0

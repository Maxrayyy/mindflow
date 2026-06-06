"""Chroma 向量存储 — 语义搜索的核心。

工作原理：
  存入: 文章文本 → get_embedding() → 512维向量 → Chroma
  搜索: 查询文本 → get_embedding() → 512维向量 → Chroma 找最接近的 N 个

Chroma 自动处理向量索引和相似度计算，我们用 cosine 距离。
"""
import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.config import settings
from src.llm.client import get_embedding


class VectorStore:
    """Chroma 向量存储管理器"""

    def __init__(self, collection_name: str = "articles"):
        # PersistentClient = 数据存硬盘，重启不丢
        self.client = chromadb.PersistentClient(
            path=str(settings.storage.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # 集合 = 类似 SQL 的 table。cosine = 用余弦距离衡量相似度
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_article(
        self,
        article_id: str,
        title: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """向量化文章并存入 Chroma"""
        text = f"{title}\n\n{content}"[:8000]  # 限制长度
        embedding = get_embedding(text)

        safe_meta = {"source": "mindflow"}
        if metadata:
            safe_meta.update(metadata)

        self.collection.add(
            ids=[article_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[safe_meta],
        )

    def search_similar(self, query: str, n_results: int = 5) -> list[dict]:
        """语义搜索 — 返回最相似的 N 篇文章"""
        query_embedding = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                items.append({
                    "article_id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    # cosine 距离 → 相似度（1=完全相同，0=完全无关）
                    "score": 1 - results["distances"][0][i],
                })
        return items

    def count(self) -> int:
        return self.collection.count()

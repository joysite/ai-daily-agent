from __future__ import annotations
import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import requests
import chromadb

logger = logging.getLogger(__name__)

def _document(item: dict[str, Any]) -> str:
    highlights = item.get("highlights") or []
    if isinstance(highlights, str):
        highlights = [highlights]
    return "\n".join([
        "标题：" + str(item.get("title") or ""),
        "摘要：" + str(item.get("summary") or item.get("raw_summary") or ""),
        "核心信息：" + "；".join(str(x) for x in highlights[:5]),
    ]).strip()

def _stable_id(item: dict[str, Any]) -> str:
    url = str(item.get("url") or item.get("link") or "").strip().rstrip("/")
    seed = url or (str(item.get("title") or "") + "|" + str(item.get("published_at") or ""))
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()

class SemanticDeduplicator:
    def __init__(self, path: str, collection_name: str, ollama_base_url: str,
                 embedding_model: str, dedup_days: int = 7,
                 cosine_distance_threshold: float = 0.18, top_k: int = 5,
                 document_prefix: str = "search_document: ", query_prefix: str = "search_query: ",
                 timeout: int = 60):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})
        self.base = ollama_base_url.rstrip("/")
        self.model = embedding_model
        self.days = max(1, int(dedup_days))
        self.threshold = float(cosine_distance_threshold)
        self.top_k = max(1, int(top_k))
        self.doc_prefix, self.query_prefix = document_prefix, query_prefix
        self.timeout = int(timeout)
        self.session = requests.Session()

    def _embed(self, text: str, query: bool = False) -> list[float]:
        prefix = self.query_prefix if query else self.doc_prefix
        try:
            response = self.session.post(f"{self.base}/api/embed",
                json={"model": self.model, "input": prefix + text}, timeout=self.timeout)
            response.raise_for_status()
            vectors = response.json().get("embeddings")
            if vectors and vectors[0]:
                return [float(v) for v in vectors[0]]
        except requests.RequestException:
            logger.debug("新版 Ollama /api/embed 不可用，尝试兼容接口")
        response = self.session.post(f"{self.base}/api/embeddings",
            json={"model": self.model, "prompt": prefix + text}, timeout=self.timeout)
        response.raise_for_status()
        vector = response.json().get("embedding")
        if not vector:
            raise RuntimeError("Ollama 没有返回 embedding")
        return [float(v) for v in vector]

    def check_and_add(self, item: dict[str, Any]) -> dict[str, Any]:
        text = _document(item)
        item_id = _stable_id(item)
        if not text:
            return {"is_duplicate": False, "duplicate_of": "", "distance": None}
        existing = self.collection.get(ids=[item_id], include=["metadatas"])
        if existing.get("ids"):
            return {"is_duplicate": False, "duplicate_of": "", "distance": 0.0}

        vector = self._embed(text, query=True)
        count = self.collection.count()
        if count:
            result = self.collection.query(
                query_embeddings=[vector], n_results=min(self.top_k, count),
                where={"created_epoch": {"$gte": (datetime.now(timezone.utc)-timedelta(days=self.days)).timestamp()}},
                include=["distances", "metadatas"])
            ids = (result.get("ids") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            for other_id, distance, meta in zip(ids, distances, metas):
                if distance is not None and float(distance) <= self.threshold:
                    meta = meta or {}
                    return {"is_duplicate": True,
                            "duplicate_of": str(meta.get("url") or meta.get("title") or other_id),
                            "distance": float(distance)}
        metadata = {
            "url": str(item.get("url") or item.get("link") or "")[:2000],
            "title": str(item.get("title") or "")[:1000],
            "source": str(item.get("source") or "")[:300],
            "created_epoch": time.time(),
        }
        self.collection.add(ids=[item_id], embeddings=[vector], documents=[text], metadatas=[metadata])
        return {"is_duplicate": False, "duplicate_of": "", "distance": None}

    def process_many(self, items: list[dict[str, Any]], fail_open: bool = True):
        unique, duplicates = [], []
        for item in items:
            try:
                result = self.check_and_add(item)
                item["_dedup_distance"] = result.get("distance")
                item["_dedup_duplicate_of"] = result.get("duplicate_of", "")
                item["_dedup_is_duplicate"] = result["is_duplicate"]
                (duplicates if result["is_duplicate"] else unique).append(item)
            except Exception as exc:
                logger.exception("语义去重失败：%s", item.get("title", "（无标题）"))
                if not fail_open:
                    raise
                item["_dedup_error"] = str(exc)
                item["_dedup_is_duplicate"] = False
                unique.append(item)
        return unique, duplicates
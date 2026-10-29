from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time

import chromadb
import requests
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config

logger = logging.getLogger("ai_daily_agent.mcp")
mcp = FastMCP("AI Daily Agent")

_CONFIG = load_config()
_DB_PATH = Path(_CONFIG["database"]["path"]).resolve()
_DB_TIMEOUT = float(_CONFIG.get("database", {}).get("timeout", 30))

_VECTOR_CONFIG = _CONFIG["vector_store"]
_OLLAMA_CONFIG = _CONFIG["ollama"]
_CHROMA_PATH = str(Path(_VECTOR_CONFIG["path"]).resolve())
_COLLECTION_NAME = _VECTOR_CONFIG["collection_name"]
_QUERY_PREFIX = _VECTOR_CONFIG.get("query_prefix", "search_query: ")
_OLLAMA_URL = _OLLAMA_CONFIG["base_url"].rstrip("/")
_EMBEDDING_MODEL = _OLLAMA_CONFIG["embedding_model"]


def _read_sql(
    sql: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    """使用只读连接执行查询，不创建或修改数据库。"""
    if not _DB_PATH.is_file():
        raise FileNotFoundError(f"SQLite 数据库文件不存在：{_DB_PATH}")

    # 使用 SQLite URI 的只读模式，避免意外写入。
    uri = _DB_PATH.as_uri() + "?mode=ro"
    conn = sqlite3.connect(
        uri,
        uri=True,
        timeout=_DB_TIMEOUT,
    )
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


@mcp.tool()
def query_news_stats() -> dict[str, Any]:
    """查询 AI 新闻总量、分类分布、重复新闻数量和最近任务运行情况。"""
    try:
        news_summary = _read_sql(
            """
            SELECT
                COUNT(*) AS total_news,
                COALESCE(SUM(is_duplicate), 0) AS duplicate_news,
                COALESCE(AVG(score), 0) AS average_score
            FROM news
            """
        )[0]

        category_stats = _read_sql(
            """
            SELECT
                category,
                COUNT(*) AS count
            FROM news
            GROUP BY category
            ORDER BY count DESC, category ASC
            """
        )

        run_summary = _read_sql(
            """
            SELECT
                COUNT(*) AS total_runs,
                COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0)
                    AS successful_runs,
                COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0)
                    AS failed_runs,
                COALESCE(SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END), 0)
                    AS running_runs
            FROM runs
            """
        )[0]

        recent_runs = _read_sql(
            """
            SELECT
                id,
                started_at,
                finished_at,
                status,
                collected_count,
                processed_count,
                unique_count,
                duplicate_count,
                report_path,
                error_message
            FROM runs
            ORDER BY id DESC
            LIMIT 10
            """
        )

        return {
            "success": True,
            "database": str(_DB_PATH),
            "news": news_summary,
            "categories": category_stats,
            "runs": run_summary,
            "recent_runs": recent_runs,
        }
    except (sqlite3.Error, OSError, ValueError) as exc:
        logger.exception("查询新闻统计失败")
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _embed_query(query: str) -> list[float]:
    """调用本地 Ollama，将查询文本转换为向量。"""
    text = _QUERY_PREFIX + query
    response = requests.post(
        f"{_OLLAMA_URL}/api/embed",
        json={"model": _EMBEDDING_MODEL, "input": text},
        timeout=60,
    )

    if response.ok:
        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings and embeddings[0]:
            return embeddings[0]

    # 兼容旧版 Ollama 的 embedding 接口。
    response = requests.post(
        f"{_OLLAMA_URL}/api/embeddings",
        json={"model": _EMBEDDING_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    vector = response.json().get("embedding")
    if not vector:
        raise ValueError("Ollama 未返回有效的查询向量")
    return vector


@mcp.tool()
def search_news_semantic(query: str, days: int = 7) -> dict[str, Any]:
    """按语义检索近期 AI 新闻。days 为 1 至 365 天，最多返回 10 条。"""
    query = query.strip() if isinstance(query, str) else ""
    if not query:
        return {"success": False, "error": "查询内容不能为空"}

    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 365:
        return {"success": False, "error": "days 必须是 1 到 365 之间的整数"}

    try:
        client = chromadb.PersistentClient(path=_CHROMA_PATH)
        collection = client.get_collection(_COLLECTION_NAME)
        total = collection.count()

        if total == 0:
            return {"success": True, "query": query, "days": days, "count": 0, "results": []}

        vector = _embed_query(query)
        cutoff = time.time() - days * 24 * 60 * 60

        result = collection.query(
            query_embeddings=[vector],
            n_results=min(10, total),
            where={"created_epoch": {"$gte": cutoff}},
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        results = []
        for item_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = metadata or {}
            results.append({
                "id": item_id,
                "title": metadata.get("title", ""),
                "url": metadata.get("url", ""),
                "source": metadata.get("source", ""),
                "created_epoch": metadata.get("created_epoch"),
                "document": document or "",
                "distance": distance,
            })

        return {
            "success": True,
            "query": query,
            "days": days,
            "count": len(results),
            "results": results,
        }
    except (ValueError, OSError, sqlite3.Error, requests.RequestException) as exc:
        logger.exception("语义检索新闻失败")
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        logger.exception("语义检索发生异常")
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s：%(message)s",
    )
    mcp.run(transport="stdio")

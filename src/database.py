from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("ai_daily_agent.database")


class NewsDatabase:
    """负责 AI 新闻与任务运行记录的 SQLite 持久化。"""

    def __init__(self, db_path: str, timeout: int = 30) -> None:
        self.db_path = Path(db_path).expanduser()
        self.timeout = timeout

        # 创建数据库文件的父目录
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据表和索引
        self.init_db()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """提供自动提交、异常回滚和关闭连接的上下文管理器。"""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout,
        )
        conn.row_factory = sqlite3.Row

        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """初始化新闻表、运行记录表和必要索引。"""
        with self.connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL DEFAULT '其他',
                    score REAL NOT NULL DEFAULT 0,
                    published_at TEXT NOT NULL DEFAULT '',
                    collected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    is_duplicate INTEGER NOT NULL DEFAULT 0,
                    duplicate_of TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}'
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_published_at
                ON news(published_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_category
                ON news(category)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    collected_count INTEGER NOT NULL DEFAULT 0,
                    processed_count INTEGER NOT NULL DEFAULT 0,
                    unique_count INTEGER NOT NULL DEFAULT 0,
                    duplicate_count INTEGER NOT NULL DEFAULT 0,
                    report_path TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT ''
                )
            """)

        logger.info("SQLite 数据库初始化完成：%s", self.db_path)

    def start_run(self) -> int:
        """创建一次运行记录并返回运行 ID。"""
        with self.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO runs (status) VALUES (?)",
                ("running",),
            )
            run_id = int(cursor.lastrowid)

        logger.info("已创建运行记录，run_id=%d", run_id)
        return run_id

    def finish_run(
        self,
        run_id: int,
        *,
        collected_count: int = 0,
        processed_count: int = 0,
        unique_count: int = 0,
        duplicate_count: int = 0,
        report_path: str = "",
        status: str = "success",
        error_message: str = "",
    ) -> None:
        """更新运行结果和统计数据。"""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE runs
                SET finished_at = CURRENT_TIMESTAMP,
                    status = ?,
                    collected_count = ?,
                    processed_count = ?,
                    unique_count = ?,
                    duplicate_count = ?,
                    report_path = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (
                    status,
                    collected_count,
                    processed_count,
                    unique_count,
                    duplicate_count,
                    report_path,
                    error_message,
                    run_id,
                ),
            )

            if cursor.rowcount == 0:
                raise ValueError(f"运行记录不存在：run_id={run_id}")

        logger.info(
            "运行记录已更新：run_id=%d，status=%s",
            run_id,
            status,
        )

    def save_news(
        self,
        item: dict[str, Any],
        *,
        is_duplicate: bool = False,
        duplicate_of: str = "",
    ) -> int:
        """保存或更新新闻；同一 URL 只保留一条记录。"""
        url = str(item.get("url") or item.get("link") or "").strip()
        if not url:
            raise ValueError("新闻缺少 url 或 link，无法保存")

        title = str(item.get("title") or "")
        summary = str(
            item.get("summary")
            or item.get("ai_summary")
            or item.get("llm_summary")
            or ""
        )
        content = str(item.get("content") or item.get("text") or "")
        source = str(item.get("source") or item.get("source_name") or "")
        category = str(item.get("category") or "其他")
        published_at = str(
            item.get("published_at")
            or item.get("published")
            or ""
        )

        try:
            score = float(item.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0

        payload_json = json.dumps(
            item,
            ensure_ascii=False,
            default=str,
        )

        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO news (
                    url, title, summary, content, source, category,
                    score, published_at, is_duplicate, duplicate_of,
                    payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    summary = excluded.summary,
                    content = excluded.content,
                    source = excluded.source,
                    category = excluded.category,
                    score = excluded.score,
                    published_at = excluded.published_at,
                    is_duplicate = excluded.is_duplicate,
                    duplicate_of = excluded.duplicate_of,
                    payload_json = excluded.payload_json
                """,
                (
                    url,
                    title,
                    summary,
                    content,
                    source,
                    category,
                    score,
                    published_at,
                    int(is_duplicate),
                    duplicate_of,
                    payload_json,
                ),
            )

            row = conn.execute(
                "SELECT id FROM news WHERE url = ?",
                (url,),
            ).fetchone()

        news_id = int(row["id"])
        logger.debug("新闻已保存或更新：news_id=%d，title=%s", news_id, title)
        return news_id

    def get_news(self, url: str) -> dict[str, Any] | None:
        """按 URL 查询新闻，便于测试和后续业务使用。"""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM news WHERE url = ?",
                (url,),
            ).fetchone()

        return dict(row) if row else None

    def count_news(self) -> int:
        """返回数据库中的新闻总数。"""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM news"
            ).fetchone()

        return int(row["total"])

    def close(self) -> None:
        """兼容需要显式关闭数据库的调用方。

        当前实现采用短连接模式，每个方法执行完毕后都会关闭连接，
        因此这里不需要额外释放长期连接。
        """
        logger.debug("SQLite 使用短连接模式，无需额外关闭操作")
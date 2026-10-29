from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def isolated_agent(tmp_path, monkeypatch):
    """使用临时目录和模拟组件，隔离测试主流程。"""
    main_module = importlib.import_module("src.main")

    db_path = tmp_path / "data" / "test.db"
    report_path = tmp_path / "output" / "test-report.md"
    log_path = tmp_path / "logs"

    config = {
        "app": {
            "name": "AI Daily Agent 测试",
            "log_dir": str(log_path),
            "log_level": "INFO",
        },
        "database": {
            "path": str(db_path),
            "timeout": 5,
        },
        "ollama": {
            "base_url": "http://127.0.0.1:11434",
            "embedding_model": "test-model",
            "timeout": 5,
        },
        "vector_store": {
            "enabled": True,
            "path": str(tmp_path / "chroma"),
            "collection_name": "test_news",
            "dedup_days": 7,
            "cosine_distance_threshold": 0.15,
            "top_k": 5,
            "fail_open": True,
        },
    }

    events = []

    collected_news = [
        {
            "url": "https://example.com/news-1",
            "title": "人工智能新闻一",
            "content": "测试新闻正文一",
            "source": "测试源",
        },
        {
            "url": "https://example.com/news-2",
            "title": "人工智能新闻二",
            "content": "测试新闻正文二",
            "source": "测试源",
        },
    ]

    processed_news = [
        {
            **collected_news[0],
            "summary": "这是第一条测试摘要",
            "category": "模型",
            "score": 8,
        },
        {
            **collected_news[1],
            "summary": "这是第二条测试摘要",
            "category": "论文",
            "score": 7,
        },
    ]

    class FakeCollector:
        def __init__(self, received_config):
            assert received_config is config

        def collect(self):
            events.append("collect")
            return [dict(item) for item in collected_news]

    class FakeProcessor:
        def __init__(self, received_config):
            assert received_config is config

        def process(self, items):
            events.append("process")
            assert len(items) == 2
            return [dict(item) for item in processed_news]

    class FakeDeduplicator:
        def __init__(self, **kwargs):
            assert kwargs["path"] == str(tmp_path / "chroma")

        def process_many(self, items, fail_open=True):
            events.append("deduplicate")
            assert fail_open is True

            unique = [dict(items[0])]
            duplicate = dict(items[1])
            duplicate["_dedup_duplicate_of"] = items[0]["url"]

            return unique, [duplicate]

    class FakeReportGenerator:
        def __init__(self, received_config):
            assert received_config is config

        def generate(self, items, stats):
            events.append("report")
            assert len(items) == 1
            assert stats == {
                "collected_count": 2,
                "processed_count": 2,
                "unique_count": 1,
                "duplicate_count": 1,
            }

            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                "# 测试日报\n\n人工智能新闻一\n",
                encoding="utf-8",
            )
            return str(report_path)

    monkeypatch.setattr(main_module, "load_config", lambda: config)
    monkeypatch.setattr(main_module, "setup_logging", lambda *args: None)
    monkeypatch.setattr(main_module, "NewsCollector", FakeCollector)
    monkeypatch.setattr(main_module, "NewsProcessor", FakeProcessor)
    monkeypatch.setattr(main_module, "SemanticDeduplicator", FakeDeduplicator)
    monkeypatch.setattr(main_module, "ReportGenerator", FakeReportGenerator)

    return {
        "main_module": main_module,
        "db_path": db_path,
        "report_path": report_path,
        "events": events,
    }


def test_main_success_saves_news_and_run_record(isolated_agent):
    """验证主流程成功时的调用顺序、新闻入库和运行统计。"""
    context = isolated_agent

    result = context["main_module"].main()

    assert result == 0
    assert context["events"] == [
        "collect",
        "process",
        "deduplicate",
        "report",
    ]
    assert context["report_path"].exists()

    with sqlite3.connect(context["db_path"]) as conn:
        conn.row_factory = sqlite3.Row

        news_rows = conn.execute(
            """
            SELECT url, is_duplicate, duplicate_of
            FROM news
            ORDER BY url
            """
        ).fetchall()

        assert len(news_rows) == 2

        news_by_url = {row["url"]: dict(row) for row in news_rows}

        assert news_by_url["https://example.com/news-1"]["is_duplicate"] == 0

        duplicate_row = news_by_url["https://example.com/news-2"]
        assert duplicate_row["is_duplicate"] == 1
        assert duplicate_row["duplicate_of"] == "https://example.com/news-1"

        run_rows = conn.execute(
            """
            SELECT status, collected_count, processed_count,
                   unique_count, duplicate_count, report_path,
                   error_message
            FROM runs
            """
        ).fetchall()

        assert len(run_rows) == 1
        run = dict(run_rows[0])

        assert run["status"] == "success"
        assert run["collected_count"] == 2
        assert run["processed_count"] == 2
        assert run["unique_count"] == 1
        assert run["duplicate_count"] == 1
        assert run["report_path"] == str(context["report_path"])
        assert run["error_message"] == ""


def test_main_when_collection_fails_records_failed_run(
    isolated_agent, monkeypatch
):
    """验证采集阶段异常时，运行记录会标记为失败。"""
    context = isolated_agent
    main_module = context["main_module"]

    class BrokenCollector:
        def __init__(self, config):
            pass

        def collect(self):
            raise RuntimeError("模拟采集失败")

    monkeypatch.setattr(main_module, "NewsCollector", BrokenCollector)

    result = main_module.main()

    assert result == 1

    with sqlite3.connect(context["db_path"]) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT status, collected_count, processed_count,
                   error_message
            FROM runs
            """
        ).fetchall()

        assert len(rows) == 1
        run = dict(rows[0])

        assert run["status"] == "failed"
        assert run["collected_count"] == 0
        assert run["processed_count"] == 0
        assert run["error_message"] == "模拟采集失败"


def test_main_when_failure_recording_also_fails(
    isolated_agent, monkeypatch, caplog
):
    """验证失败记录写入异常不会覆盖原始错误或改变返回值。"""
    import logging

    context = isolated_agent
    main_module = context["main_module"]

    class BrokenCollector:
        def __init__(self, config):
            pass

        def collect(self):
            raise RuntimeError("原始采集错误")

    monkeypatch.setattr(main_module, "NewsCollector", BrokenCollector)

    original_finish_run = main_module.NewsDatabase.finish_run

    def broken_finish_run(self, run_id, **kwargs):
        if kwargs.get("status") == "failed":
            raise RuntimeError("失败记录写入错误")
        return original_finish_run(self, run_id, **kwargs)

    monkeypatch.setattr(
        main_module.NewsDatabase,
        "finish_run",
        broken_finish_run,
    )

    with caplog.at_level(logging.ERROR):
        result = main_module.main()

    assert result == 1
    assert "原始采集错误" in caplog.text
    assert "失败记录写入错误" in caplog.text


def test_main_when_report_generation_fails(
    isolated_agent, monkeypatch
):
    """验证日报生成失败后，新闻数据保留且运行状态标记为失败。"""
    context = isolated_agent
    main_module = context["main_module"]

    class BrokenReportGenerator:
        def __init__(self, config):
            pass

        def generate(self, items, stats):
            raise RuntimeError("模拟日报生成失败")

    monkeypatch.setattr(
        main_module,
        "ReportGenerator",
        BrokenReportGenerator,
    )

    result = main_module.main()

    assert result == 1

    with sqlite3.connect(context["db_path"]) as conn:
        conn.row_factory = sqlite3.Row

        news_count = conn.execute(
            "SELECT COUNT(*) AS total FROM news"
        ).fetchone()["total"]

        run = conn.execute(
            """
            SELECT status, unique_count, duplicate_count, error_message
            FROM runs
            """
        ).fetchone()

    assert news_count == 2
    assert run["status"] == "failed"
    assert run["unique_count"] == 1
    assert run["duplicate_count"] == 1
    assert run["error_message"] == "模拟日报生成失败"


from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保项目根目录位于 Python 模块搜索路径中
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.logger import setup_logging
from src.collector import NewsCollector
from src.llm_processor import NewsProcessor
from src.database import NewsDatabase
from src.deduplicator import SemanticDeduplicator
from src.report_generator import ReportGenerator


logger = logging.getLogger("ai_daily_agent")


def main() -> int:
    """执行每日 AI 新闻采集与处理流程。"""

    # 读取配置并初始化日志
    config = load_config()
    setup_logging(
        config["app"]["log_dir"],
        config["app"].get("log_level", "INFO"),
    )

    logger.info(
        "========== %s 开始运行 ==========",
        config.get("app", {}).get("name", "AI Daily Agent"),
    )

    # 初始化 SQLite 数据库并创建本次运行记录
    db = NewsDatabase(
        config["database"]["path"],
        config["database"].get("timeout", 30),
    )
    run_id = db.start_run()

    collected_count = 0
    processed_count = 0
    duplicate_count = 0
    unique_count = 0
    report_path = ""

    try:
        # 第一阶段：采集新闻
        collected = NewsCollector(config).collect()
        collected_count = len(collected)

        logger.info("新闻采集阶段完成：%d 条", collected_count)

        # 第二阶段：使用本地 Ollama 筛选新闻并生成摘要
        # 注意：当前 llm_processor.py 中定义的类名为 NewsProcessor
        processed = NewsProcessor(config).process(collected)
        processed_count = len(processed)

        logger.info("LLM 新闻处理阶段完成：%d 条", processed_count)

        # 第三阶段：Chroma 语义去重
        vector_cfg = config.get("vector_store", {})

        if vector_cfg.get("enabled", True):
            deduplicator = SemanticDeduplicator(
                path=vector_cfg["path"],
                collection_name=vector_cfg.get(
                    "collection_name",
                    "ai_news",
                ),
                ollama_base_url=config["ollama"]["base_url"],
                embedding_model=config["ollama"]["embedding_model"],
                dedup_days=vector_cfg.get("dedup_days", 7),
                cosine_distance_threshold=vector_cfg.get(
                    "cosine_distance_threshold",
                    0.18,
                ),
                top_k=vector_cfg.get("top_k", 5),
                document_prefix=vector_cfg.get(
                    "document_prefix",
                    "search_document: ",
                ),
                query_prefix=vector_cfg.get(
                    "query_prefix",
                    "search_query: ",
                ),
                timeout=config["ollama"].get("timeout", 60),
            )

            unique, duplicates = deduplicator.process_many(
                processed,
                fail_open=vector_cfg.get("fail_open", True),
            )
        else:
            logger.info("语义去重功能已关闭")
            unique = processed
            duplicates = []

        unique_count = len(unique)
        duplicate_count = len(duplicates)

        logger.info(
            "语义去重阶段完成：唯一 %d 条，重复 %d 条",
            unique_count,
            duplicate_count,
        )

        # 第四阶段：保存新闻到 SQLite
        for item in unique:
            db.save_news(item, is_duplicate=False)

        for item in duplicates:
            db.save_news(
                item,
                is_duplicate=True,
                duplicate_of=str(
                    item.get("_dedup_duplicate_of") or ""
                ),
            )

        logger.info(
            "新闻持久化完成：唯一 %d 条，重复记录 %d 条",
            unique_count,
            duplicate_count,
        )

        # 第五阶段：生成 Markdown 日报
        report_path = ReportGenerator(config).generate(
            unique,
            stats={
                "collected_count": collected_count,
                "processed_count": processed_count,
                "unique_count": unique_count,
                "duplicate_count": duplicate_count,
            },
        )

        # 第六阶段：记录本次运行结果
        db.finish_run(
            run_id,
            collected_count=collected_count,
            processed_count=processed_count,
            unique_count=unique_count,
            duplicate_count=duplicate_count,
            report_path=report_path,
            status="success",
        )

        logger.info(
            "运行成功。日报路径：%s",
            report_path or "未生成",
        )

        return 0

    except Exception as exc:
        # 记录主流程的原始异常，便于排查真正的故障原因
        logger.exception("Agent 运行失败")

        try:
            # 尽可能保存失败状态和已经完成的阶段统计
            db.finish_run(
                run_id,
                collected_count=collected_count,
                processed_count=processed_count,
                unique_count=unique_count,
                duplicate_count=duplicate_count,
                report_path=report_path,
                status="failed",
                error_message=str(exc),
            )
        except Exception:
            # 失败记录写入异常不能覆盖主流程的原始错误
            logger.exception("写入运行失败记录时再次发生异常")

        return 1


if __name__ == "__main__":
    raise SystemExit(main())

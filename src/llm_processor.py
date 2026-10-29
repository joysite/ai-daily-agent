
"""
LLM 新闻筛选、分类与摘要模块。

输入：collector.py 输出的新闻字典列表。
输出：经过筛选、摘要和字段校验的新闻列表。
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any

from src.ollama_client import OllamaClient


logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = {
    "模型",
    "论文",
    "产品",
    "融资",
    "行业",
}


class NewsProcessor:
    """新闻价值筛选和摘要处理器。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        self.filter_config = config.get("filter", {})
        self.summary_config = config.get("summary", {})
        self.ollama_config = config.get("ollama", {})

        self.min_score = int(
            self.filter_config.get("min_score", 6)
        )

        self.max_input_chars = int(
            self.ollama_config.get("max_input_chars", 10000)
        )

        self.max_highlights = int(
            self.summary_config.get("max_highlights", 3)
        )

        self.min_summary_chars = int(
            self.summary_config.get("min_chars", 100)
        )

        self.max_summary_chars = int(
            self.summary_config.get("max_chars", 250)
        )

        self.client = OllamaClient(config)

        # 通过模块文件的位置定位项目根目录，避免依赖当前工作目录。
        project_root = Path(__file__).resolve().parent.parent
        prompt_dir = project_root / "prompts"

        self.filter_prompt = (
            prompt_dir / "filter_news.txt"
        ).read_text(encoding="utf-8")

        self.summary_prompt = (
            prompt_dir / "summarize_news.txt"
        ).read_text(encoding="utf-8")

    
    @staticmethod
    def _render_news(news: dict[str, Any]) -> dict[str, Any]:
        """将单条新闻转换为结构化数据，保持 JSON 字段完整。"""
        return {
            "title": news.get("title", ""),
            "source": news.get("source", ""),
            "published_at": news.get("published_at", ""),
            "url": news.get("url", ""),
            "summary": news.get("summary", ""),
            "content": news.get("content", ""),
        }

    def _build_filter_prompt(
        self,
        batch: list[dict[str, Any]],
    ) -> str:
        """构建批量筛选提示词，避免截断 JSON 字符串。"""
        items = []

        for index, news in enumerate(batch):
            news_data = self._render_news(news)

            # 只截断新闻正文，不截断序列化后的 JSON。
            content = news_data.get("content", "")
            if isinstance(content, str):
                # 为其他字段预留空间，避免单条新闻过长。
                content_limit = max(0, self.max_input_chars - 2000)
                if len(content) > content_limit:
                    news_data["content"] = (
                        content[:content_limit] + "……[正文已截断]"
                    )

            # 直接序列化结构化数据，确保 JSON 格式完整。
            items.append({
                "index": index,
                "news": news_data,
            })

        input_json = json.dumps(items, ensure_ascii=False)

        return (
            self.filter_prompt
            + "\n\n请分析以下输入新闻：\n"
            + input_json
        )


    def _filter_batch(
        self,
        batch: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """筛选一批新闻，并严格校验模型返回字段。"""
        prompt = self._build_filter_prompt(batch)

        result = self.client.chat_json(
            prompt,
            task_name=f"新闻筛选（{len(batch)}条）",
        )

        items = result.get("items")

        if not isinstance(items, list):
            logger.error(
                "筛选结果结构异常：顶层字段=%s，模型返回内容=%s",
                list(result.keys()),
                json.dumps(result, ensure_ascii=False)[:1500],
            )
            raise ValueError("筛选结果缺少 items 数组")

        # 以 index 建立映射，不直接相信模型的返回顺序。
        indexed: dict[int, dict[str, Any]] = {}

        for item in items:
            if not isinstance(item, dict):
                continue

            index = item.get("index")

            # bool 是 int 的子类，因此显式排除。
            if isinstance(index, bool) or not isinstance(index, int):
                continue

            if 0 <= index < len(batch):
                if index in indexed:
                    raise ValueError(
                        f"模型重复返回新闻 index：{index}"
                    )

                indexed[index] = item

        results = []

        for index, news in enumerate(batch):
            item = indexed.get(index)

            # 缺少结果时保守过滤，避免把未审核新闻直接收录。
            if item is None:
                logger.warning(
                    "筛选结果缺失，保守跳过：%s",
                    news.get("title", ""),
                )
                continue

            keep = item.get("keep")
            score = item.get("score")
            category = item.get("category")
            reason = item.get("reason")

            # 记录模型对每条新闻的完整筛选判断，便于排查评分和分类问题。
            logger.info(
                "模型筛选详情：标题=%s，keep=%s，评分=%s，分类=%s，理由=%s",
                news.get("title", ""),
                keep,
                score,
                category,
                reason,
            )

            if not isinstance(keep, bool):
                logger.warning("筛选字段 keep 无效，跳过：%s",
                               news.get("title", ""))
                continue

            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or not 1 <= score <= 10
            ):
                logger.warning("筛选评分无效，跳过：%s",
                               news.get("title", ""))
                continue

            if category not in ALLOWED_CATEGORIES:
                logger.warning("新闻分类无效，跳过：%s",
                               news.get("title", ""))
                continue

            if not isinstance(reason, str):
                reason = ""

            # Python 端再次执行最低分数判断。
            if not keep or score < self.min_score:
                logger.info(
                    "过滤新闻：评分=%d，标题=%s，原因=%s",
                    score,
                    news.get("title", ""),
                    reason,
                )
                continue

            accepted = dict(news)
            accepted["score"] = score
            accepted["category"] = category
            accepted["filter_reason"] = reason.strip()

            results.append(accepted)

        return results

    
    def _build_summary_prompt(
        self,
        news: dict[str, Any],
    ) -> str:
        """构建单条新闻摘要提示词，避免截断 JSON 字符串。"""
        news_data = self._render_news(news)

        # 只截断正文，避免破坏 JSON 结构。
        content = news_data.get("content", "")
        if isinstance(content, str):
            content_limit = max(0, self.max_input_chars - 2000)
            if len(content) > content_limit:
                news_data["content"] = (
                    content[:content_limit] + "……[正文已截断]"
                )

        serialized = json.dumps(
            news_data,
            ensure_ascii=False,
        )

        return (
            self.summary_prompt
            + "\n\n请为以下新闻生成摘要：\n"
            + serialized
        )


    def _summarize(
        self,
        news: dict[str, Any],
    ) -> dict[str, Any] | None:
        """生成并校验单条新闻摘要。"""
        result = self.client.chat_json(
            self._build_summary_prompt(news),
            task_name=f"新闻摘要：{news.get('title', '')[:40]}",
        )

        summary = result.get("summary")
        highlights = result.get("highlights", [])
        key_facts = result.get("key_facts", [])

        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("摘要为空或类型错误")

        summary = summary.strip()

        # 避免摘要超出日报配置长度。
        if len(summary) > self.max_summary_chars:
            summary = summary[:self.max_summary_chars].rstrip()

        # 对过短摘要不伪造内容，直接标记为无效。
        # 摘要过短时拒绝该结果，避免低质量摘要进入日报。
        if len(summary) < self.min_summary_chars:
            logger.warning(
                "摘要过短，丢弃该条新闻：标题=%s，长度=%d，最低要求=%d",
                news.get("title", ""),
                len(summary),
                self.min_summary_chars,
            )
            return None

        if not isinstance(highlights, list):
            highlights = []

        highlights = [
            item.strip()
            for item in highlights
            if isinstance(item, str) and item.strip()
        ][:self.max_highlights]

        if not isinstance(key_facts, list):
            key_facts = []

        validated_facts = []

        for fact in key_facts[:5]:
            if not isinstance(fact, dict):
                continue

            fact_text = fact.get("fact")
            evidence = fact.get("evidence")

            if (
                isinstance(fact_text, str)
                and fact_text.strip()
                and isinstance(evidence, str)
                and evidence.strip()
            ):
                validated_facts.append(
                    {
                        "fact": fact_text.strip(),
                        "evidence": evidence.strip(),
                    }
                )

        enriched = dict(news)
        enriched["summary"] = summary
        enriched["highlights"] = highlights
        enriched["key_facts"] = validated_facts

        return enriched

    def process(
        self,
        news_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """执行批量筛选和逐条摘要，返回最终新闻列表。"""
        if not news_list:
            logger.info("没有待处理新闻，跳过 LLM 阶段")
            return []

        logger.info(
            "开始 LLM 新闻处理，输入新闻数：%d",
            len(news_list),
        )

        # 逐条筛选，减少多条新闻之间的语义干扰。
        batch_size = 1
        accepted_news = []

        for start in range(0, len(news_list), batch_size):
            batch = news_list[start:start + batch_size]

            try:
                accepted_news.extend(
                    self._filter_batch(batch)
                )

            except Exception:
                # 批次筛选失败时不把未经审核的新闻直接放行。
                logger.exception(
                    "新闻批次筛选失败，起始位置：%d",
                    start,
                )

        logger.info(
            "新闻筛选结束：输入 %d 条，通过 %d 条",
            len(news_list),
            len(accepted_news),
        )

        summarized_news = []

        for index, news in enumerate(accepted_news, start=1):
            try:
                result = self._summarize(news)

                if result is not None:
                    summarized_news.append(result)

                logger.info(
                    "摘要处理进度：%d/%d",
                    index,
                    len(accepted_news),
                )

            except Exception:
                logger.exception(
                    "单条新闻摘要失败，跳过：%s",
                    news.get("title", ""),
                )

        summarized_news.sort(
            key=lambda item: (
                item.get("score", 0),
                item.get("published_at", ""),
            ),
            reverse=True,
        )

        max_news = int(
            self.config.get("app", {}).get(
                "max_news_per_report",
                20,
            )
        )

        summarized_news = summarized_news[:max_news]

        logger.info(
            "LLM 新闻处理完成：最终输出 %d 条",
            len(summarized_news),
        )

        return summarized_news


def process_news(
    news_list: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """供主程序调用的简化接口。"""
    processor = NewsProcessor(config)
    return processor.process(news_list)

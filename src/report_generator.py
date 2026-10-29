from __future__ import annotations
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, config: dict[str, Any]):
        self.app = config.get("app", {})
        self.cfg = config.get("report", {})
        self.output = Path(self.app.get("output_dir", "./output"))
        self.output.mkdir(parents=True, exist_ok=True)
        self.timezone = self.app.get("timezone", "Asia/Shanghai")
        self.limit = int(self.app.get("max_news_per_report", 20))

    def generate(self, items: list[dict[str, Any]], stats: dict[str, int] | None = None) -> str:
        tz = ZoneInfo(self.timezone)
        now = datetime.now(tz)
        filename = now.strftime(self.cfg.get("filename_format", "%Y-%m-%d.md"))
        path = self.output / filename
        selected = sorted(items, key=lambda x: (int(x.get("score") or 0), str(x.get("published_at") or "")),
                          reverse=True)[:self.limit]
        if not selected and not self.cfg.get("generate_empty_report", True):
            return ""
        lines = [f"# AI 每日热点新闻 — {now:%Y-%m-%d}", "",
                 f"> 生成时间：{now:%Y-%m-%d %H:%M:%S %Z}  ",
                 f"> 本次报告收录：{len(selected)} 条", ""]
        if self.cfg.get("include_statistics", True):
            stat = stats or {}
            lines += ["## 今日统计", "",
                f"- RSS 采集：{stat.get('collected_count', 0)} 条",
                f"- LLM 筛选通过：{stat.get('processed_count', len(items))} 条",
                f"- 语义去重后：{stat.get('unique_count', len(items))} 条",
                f"- 语义重复：{stat.get('duplicate_count', 0)} 条",
                "- 分类分布：" + "、".join(f"{k} {v} 条" for k,v in Counter(str(x.get('category') or '行业') for x in selected).most_common()),
                ""]
        if not selected:
            lines += ["## 今日暂无符合条件的新闻", "", "请检查 RSS 源、时间窗口、模型和筛选阈值。", ""]
        else:
            categories: dict[str, list[dict[str, Any]]] = {}
            for item in selected:
                category = str(item.get("category") or "行业")
                categories.setdefault(category, []).append(item)
            for category in sorted(categories) if self.cfg.get("group_by_category", True) else ["热点新闻"]:
                lines += [f"## {category}", ""]
                group = categories.get(category, selected)
                for index, item in enumerate(group, 1):
                    title = str(item.get("title") or "（无标题）")
                    url = str(item.get("url") or item.get("link") or "")
                    if self.cfg.get("show_source_url", True) and url:
                        lines += [f"### {index}. [{title}]({url})", ""]
                    else:
                        lines += [f"### {index}. {title}", ""]
                    meta = []
                    if item.get("source"):
                        meta.append("来源：" + str(item["source"]))
                    if self.cfg.get("show_publish_time", True):
                        meta.append("发布时间：" + str(item.get("published_at") or "未知"))
                    if self.cfg.get("show_score", True):
                        meta.append("评分：" + str(item.get("score", "—")))
                    if meta:
                        lines += ["> " + " | ".join(meta), ""]
                    lines += [str(item.get("summary") or item.get("raw_summary") or "暂无摘要"), ""]
                    if self.cfg.get("show_highlights", True) and item.get("highlights"):
                        lines += ["**核心亮点**", *[f"- {x}" for x in item["highlights"][:5]], ""]
        lines += ["---", "", "本报告由本地 AI Daily Agent 自动生成；摘要请以原始来源为准。", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("日报已生成：%s", path.resolve())
        return str(path.resolve())
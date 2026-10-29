
"""
新闻采集模块。

职责：
1. 串行采集配置中的 RSS 源。
2. 将发布时间统一转换为带时区的 datetime。
3. 过滤 24 小时以外的新闻。
4. 必要时通过 trafilatura 提取网页正文。
5. 输出统一格式的新闻字典。

所有注释和日志均使用中文。
"""

from __future__ import annotations

import hashlib
import logging
import re
import time

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)

import feedparser
import requests
import trafilatura

from dateutil import parser as date_parser


logger = logging.getLogger(__name__)

# 这些查询参数通常属于统计追踪参数，可以在生成去重标识时移除。
TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
}

# 仅接受 HTTP/HTTPS 链接。
ALLOWED_SCHEMES = {"http", "https"}


def clean_text(value: Any) -> str:
    """清理 HTML 残留、连续空白和无效文本。"""
    if value is None:
        return ""

    text = str(value)

    # 移除常见的 HTML 标签残留。
    text = re.sub(r"<[^>]+>", " ", text)

    # 规范空白字符。
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_url(url: str) -> str:
    """
    规范 URL。

    保留文章必要的查询参数，只移除常见追踪参数。
    不尝试自行猜测或拼接文章地址。
    """
    url = (url or "").strip()

    if not url:
        return ""

    try:
        parts = urlsplit(url)

        if parts.scheme.lower() not in ALLOWED_SCHEMES:
            return ""

        if not parts.hostname:
            return ""

        # 移除 URL 片段和常见追踪参数。
        query_items = [
            (key, value)
            for key, value in parse_qsl(
                parts.query,
                keep_blank_values=True,
            )
            if key.lower() not in TRACKING_PARAMS
            and not key.lower().startswith("utm_")
        ]

        normalized = urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                parts.path,
                urlencode(query_items, doseq=True),
                "",
            )
        )

        return normalized

    except (ValueError, TypeError) as exc:
        logger.warning("URL 规范化失败：%s，原因：%s", url, exc)
        return ""


def parse_publish_time(entry: Any) -> datetime | None:
    """
    尽可能解析 RSS 发布时间。

    优先使用 published，其次使用 updated。
    返回 UTC 时区的 datetime；解析失败返回 None。
    不使用当前时间伪造新闻发布时间。
    """
    candidates = [
        entry.get("published"),
        entry.get("updated"),
        entry.get("created"),
    ]

    for value in candidates:
        if not value:
            continue

        try:
            # RFC 2822 格式及带时区的标准日期。
            parsed = parsedate_to_datetime(value)

            if parsed is not None:
                if parsed.tzinfo is None:
                    logger.warning(
                        "RSS 发布时间没有时区信息，暂按 UTC 处理：%s",
                        value,
                    )
                    parsed = parsed.replace(tzinfo=timezone.utc)

                return parsed.astimezone(timezone.utc)

        except (TypeError, ValueError, OverflowError):
            pass

        try:
            # 兼容 ISO 8601 和其他常见日期格式。
            parsed = date_parser.parse(value)

            if parsed.tzinfo is None:
                logger.warning(
                    "RSS 发布时间没有时区信息，暂按 UTC 处理：%s",
                    value,
                )
                parsed = parsed.replace(tzinfo=timezone.utc)

            return parsed.astimezone(timezone.utc)

        except (ValueError, TypeError, OverflowError):
            logger.debug("无法解析候选发布时间：%s", value)

    # 兼容 feedparser 提供的结构化时间。
    for field in ("published_parsed", "updated_parsed"):
        parsed_tuple = entry.get(field)

        if parsed_tuple:
            try:
                import calendar

                timestamp = calendar.timegm(parsed_tuple)
                return datetime.fromtimestamp(
                    timestamp,
                    tz=timezone.utc,
                )
            except (ValueError, TypeError, OverflowError, OSError):
                continue

    return None


def make_content_hash(title: str, url: str) -> str:
    """生成稳定的基础去重标识。"""
    raw = f"{title.strip().lower()}|{normalize_url(url)}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


class NewsCollector:
    """RSS 与网页补充采集器。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.collector_config = config.get("collector", {})

        self.lookback_hours = max(
            1,
            int(self.collector_config.get("lookback_hours", 24)),
        )

        self.request_delay = max(
            0.0,
            float(
                self.collector_config.get(
                    "request_delay_seconds",
                    2.0,
                )
            ),
        )

        self.request_timeout = max(
            1,
            int(
                self.collector_config.get(
                    "request_timeout",
                    15,
                )
            ),
        )

        self.max_items_per_feed = max(
            1,
            int(
                self.collector_config.get(
                    "max_items_per_feed",
                    50,
                )
            ),
        )

        self.max_candidates = max(
            1,
            int(
                self.collector_config.get(
                    "max_candidates",
                    200,
                )
            ),
        )

        self.enable_web_fallback = bool(
            self.collector_config.get(
                "enable_web_fallback",
                True,
            )
        )

        self.user_agent = self.collector_config.get(
            "user_agent",
            "AIDailyAgent/1.0",
        )

        # 一个 Session 复用连接，但所有请求仍然按顺序执行。
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": (
                    "application/rss+xml, application/atom+xml, "
                    "application/xml, text/xml, text/html;q=0.9, "
                    "*/*;q=0.8"
                ),
            }
        )

        # 记录本轮已见过的 URL，防止多个源重复输出同一文章。
        self.seen_urls: set[str] = set()

        # 统一使用 UTC，避免服务器时区影响时间过滤。
        self.now = datetime.now(timezone.utc)
        self.cutoff = self.now - timedelta(
            hours=self.lookback_hours
        )

    def _request(self, url: str) -> requests.Response | None:
        """执行单次 HTTP GET 请求。"""
        try:
            response = self.session.get(
                url,
                timeout=self.request_timeout,
                allow_redirects=True,
            )
            response.raise_for_status()
            return response

        except requests.RequestException as exc:
            logger.warning(
                "请求失败：%s，原因：%s",
                url,
                exc,
            )
            return None

    def _fetch_feed(self, feed_url: str) -> Any | None:
        """下载并解析一个 RSS/Atom 源。"""
        response = self._request(feed_url)

        if response is None:
            return None

        try:
            parsed = feedparser.parse(response.content)

            # bozo 表示解析器发现格式问题，不一定意味着完全不可用。
            if getattr(parsed, "bozo", False):
                logger.warning(
                    "RSS 存在格式问题：%s，原因：%s",
                    feed_url,
                    getattr(parsed, "bozo_exception", "未知"),
                )

            if not parsed.entries:
                logger.warning("RSS 没有可用条目：%s", feed_url)

            return parsed

        except Exception:
            logger.exception("RSS 解析异常：%s", feed_url)
            return None

    def _extract_web_content(self, url: str) -> str:
        """
        获取网页并提取正文。

        不调用 trafilatura 自带下载器，避免绕过本模块的请求超时、
        User-Agent 和抓取延时控制。
        """
        if not self.enable_web_fallback:
            return ""

        if not url:
            return ""

        # 遵守采集延时，不进行并发请求。
        if self.request_delay:
            time.sleep(self.request_delay)

        response = self._request(url)

        if response is None:
            return ""

        # 防止把 PDF、图片等非 HTML 内容交给正文提取器。
        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        if (
            "text/html" not in content_type
            and "application/xhtml+xml" not in content_type
            and "text/plain" not in content_type
        ):
            logger.debug(
                "网页不是可处理的 HTML 文本，跳过正文提取：%s",
                url,
            )
            return ""

        try:
            content = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )

            return clean_text(content)

        except Exception:
            logger.exception("网页正文提取失败：%s", url)
            return ""

    def _entry_to_news(
        self,
        entry: Any,
        source_name: str,
        category_hint: str,
    ) -> dict[str, Any] | None:
        """将 RSS 条目转换为统一新闻对象。"""
        title = clean_text(entry.get("title"))
        url = normalize_url(entry.get("link", ""))

        if not title:
            logger.debug("跳过无标题条目，来源：%s", source_name)
            return None

        if not url:
            logger.debug(
                "跳过无有效链接的条目：%s",
                title,
            )
            return None

        published_at = parse_publish_time(entry)

        # 严格过滤：没有可信发布时间时，不将其作为最新新闻。
        if published_at is None:
            logger.warning(
                "新闻缺少有效发布时间，跳过：%s",
                title,
            )
            return None

        if published_at < self.cutoff:
            logger.debug(
                "新闻已超过时间窗口，跳过：%s，发布时间：%s",
                title,
                published_at.isoformat(),
            )
            return None

        # 不接收未来时间的异常记录。
        if published_at > self.now + timedelta(minutes=10):
            logger.warning(
                "新闻发布时间异常（未来时间），跳过：%s",
                title,
            )
            return None

        if url in self.seen_urls:
            logger.debug("重复链接，跳过：%s", url)
            return None

        self.seen_urls.add(url)

        summary = clean_text(entry.get("summary", ""))

        # 某些 Atom/RSS 条目将正文放在 content 字段中。
        if not summary:
            contents = entry.get("content", [])

            if contents:
                summary = clean_text(
                    contents[0].get("value", "")
                )

        content = summary

        # 摘要较短时，尝试提取网页正文。
        if len(content) < 200 and self.enable_web_fallback:
            web_content = self._extract_web_content(url)

            if len(web_content) > len(content):
                content = web_content

        return {
            "title": title,
            "url": url,
            "source": source_name,
            "published_at": published_at.isoformat(),
            "summary": summary,
            "content": content,
            "collected_at": self.now.isoformat(),
            "category_hint": category_hint,
            "content_hash": make_content_hash(title, url),
            "metadata": {
                "rss_id": clean_text(entry.get("id", "")),
                "author": clean_text(entry.get("author", "")),
                "language": "",
            },
        }

    def collect(self) -> list[dict[str, Any]]:
        """
        采集所有启用的 RSS 源。

        返回按发布时间从新到旧排序的新闻列表。
        """
        feeds = self.collector_config.get("feeds", [])
        results: list[dict[str, Any]] = []

        enabled_feeds = [
            feed for feed in feeds
            if feed.get("enabled", True)
        ]

        logger.info(
            "开始采集新闻，启用 RSS 源数量：%d，时间窗口：最近 %d 小时",
            len(enabled_feeds),
            self.lookback_hours,
        )

        for feed_index, feed in enumerate(enabled_feeds):
            if len(results) >= self.max_candidates:
                logger.info("达到候选新闻数量上限，停止采集")
                break

            name = feed.get("name", "未命名来源")
            feed_url = feed.get("url", "")
            category_hint = feed.get("category_hint", "行业")

            if not feed_url:
                logger.warning("RSS 源缺少 URL，跳过：%s", name)
                continue

            logger.info("开始采集 RSS：%s", name)

            parsed = self._fetch_feed(feed_url)

            if parsed is None:
                logger.warning("RSS 源采集失败，继续下一个来源：%s", name)
                continue

            entries = parsed.entries[:self.max_items_per_feed]

            logger.info(
                "RSS 解析完成：%s，原始条目数：%d",
                name,
                len(parsed.entries),
            )

            for entry in entries:
                if len(results) >= self.max_candidates:
                    break

                try:
                    news = self._entry_to_news(
                        entry,
                        source_name=name,
                        category_hint=category_hint,
                    )

                    if news is not None:
                        results.append(news)

                except Exception:
                    logger.exception(
                        "新闻条目处理失败，来源：%s，标题：%s",
                        name,
                        clean_text(entry.get("title", "")),
                    )

            # RSS 源之间同样保持间隔，不并发访问。
            if (
                self.request_delay
                and feed_index < len(enabled_feeds) - 1
            ):
                time.sleep(self.request_delay)

        results.sort(
            key=lambda item: item["published_at"],
            reverse=True,
        )

        logger.info(
            "新闻采集结束：有效新闻 %d 条",
            len(results),
        )

        return results

    def close(self) -> None:
        """释放 HTTP 连接。"""
        self.session.close()


def collect_news(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """供主程序调用的简化接口。"""
    collector = NewsCollector(config)

    try:
        return collector.collect()
    finally:
        collector.close()

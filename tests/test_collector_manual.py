
import logging
from pathlib import Path

import yaml

from src.collector import collect_news


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# 从项目根目录加载配置。
config_path = Path(__file__).resolve().parents[1] / "settings.yaml"

with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# 执行真实 RSS 采集。
news_list = collect_news(config)

print(f"\n本次共采集到 {len(news_list)} 条新闻\n")

for index, news in enumerate(news_list[:10], start=1):
    print(f"{index}. {news['title']}")
    print(f"   来源：{news['source']}")
    print(f"   发布时间：{news['published_at']}")
    print(f"   链接：{news['url']}")
    print(f"   正文长度：{len(news['content'])} 字符")
    print("-" * 60)


"""LLM 新闻筛选与摘要的本地集成测试。"""

import logging
from pathlib import Path

import yaml

from src.llm_processor import process_news


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

ROOT = Path(__file__).resolve().parents[1]

with (ROOT / "settings.yaml").open(
    "r",
    encoding="utf-8",
) as file:
    config = yaml.safe_load(file)

# 使用一条测试新闻，验证本地模型的完整处理链路。
test_news = [
    {
        "title": "示例：某 AI 团队发布新模型及评测结果",
        "url": "https://example.com/ai-model-release",
        "source": "示例科技媒体",
        "published_at": "2026-10-26T10:00:00",
        "content": (
            "该团队发布了 Example-AI-7B 模型，并公开了模型技术报告。"
            "报告列出了多个公开基准测试的评测结果、测试条件和对比模型。"
            "研究人员可以通过技术报告进一步核验相关数据。"
        ),
    }
]

results = process_news(test_news, config)

print(f"\n最终保留新闻：{len(results)} 条")

for news in results:
    print("\n标题：", news["title"])
    print("分类：", news["category"])
    print("评分：", news["score"])
    print("筛选理由：", news["filter_reason"])
    print("摘要：", news["summary"])
    print("核心亮点：", news["highlights"])
    print("关键事实：", news["key_facts"])

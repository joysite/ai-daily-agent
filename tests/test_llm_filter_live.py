"""真实 Ollama 新闻筛选小样本测试，不采集新闻、不写数据库。"""

import json
import logging
from pathlib import Path

import yaml

from src.llm_processor import NewsProcessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

ROOT = Path(__file__).resolve().parents[1]

with (ROOT / "settings.yaml").open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# 使用主题差异明显的人工样本，检查分类、理由和实体是否串题。
test_news = [
    {
        "title": "示例研究：提出面向小目标检测的新算法",
        "url": "https://example.com/paper-small-object",
        "source": "示例学术媒体",
        "published_at": "2026-10-28",
        "summary": "研究团队提出一种面向小目标检测的新算法。",
        "content": (
            "这篇研究论文介绍了 TinyScope 算法，"
            "重点讨论小目标检测方法。"
            "本文没有提供具体数据集、准确率或性能提升数值。"
        ),
    },
    {
        "title": "示例公司发布 Nova-3 大语言模型",
        "url": "https://example.com/model-nova3",
        "source": "示例科技媒体",
        "published_at": "2026-10-28",
        "summary": "示例公司公布 Nova-3 模型及其技术说明。",
        "content": (
            "示例公司正式发布 Nova-3 大语言模型，"
            "并提供模型使用说明。"
            "新闻没有披露基准测试分数或与其他模型的对比结果。"
        ),
    },
    {
        "title": "示例软件公司推出 AI 文档整理工具",
        "url": "https://example.com/product-docai",
        "source": "示例产品资讯",
        "published_at": "2026-10-28",
        "summary": "一款用于整理文档的 AI 软件正式上线。",
        "content": (
            "示例软件公司推出 DocPilot AI 文档整理工具，"
            "支持对用户上传的文档进行分类和摘要。"
            "该新闻没有涉及基础模型训练或学术论文。"
        ),
    },
    {
        "title": "示例 AI 初创公司完成 3000 万美元融资",
        "url": "https://example.com/funding",
        "source": "示例商业媒体",
        "published_at": "2026-10-28",
        "summary": "一家 AI 初创公司宣布完成新一轮融资。",
        "content": (
            "示例 AI 初创公司 Northstar AI 宣布完成"
            "3000 万美元融资，资金将用于扩大研发团队。"
            "新闻未披露新模型发布或论文研究成果。"
        ),
    },
    {
        "title": "示例城市发布人工智能公共服务使用指引",
        "url": "https://example.com/policy",
        "source": "示例政务资讯",
        "published_at": "2026-10-28",
        "summary": "当地政府发布人工智能公共服务使用指引。",
        "content": (
            "示例城市公布人工智能公共服务使用指引，"
            "内容涉及数据保护、人工审核和服务责任。"
            "该消息属于公共政策，不是模型发布、产品发布或融资事件。"
        ),
    },
]

processor = NewsProcessor(config)

# 直接调用筛选阶段，不调用 process()，因此不会触发摘要生成。
results = processor._filter_batch(test_news)

print("\n========== 真实模型筛选结果 ==========")
print(f"输入新闻：{len(test_news)} 条")
print(f"通过筛选：{len(results)} 条")
print("\n--- 筛选阶段保留结果 ---")

for item in results:
    print(json.dumps(
        {
            "title": item.get("title"),
            "category": item.get("category"),
            "score": item.get("score"),
            "filter_reason": item.get("filter_reason"),
        },
        ensure_ascii=False,
        indent=2,
    ))

print("\n--- 人工核对清单 ---")
for index, news in enumerate(test_news):
    print(f"{index}: {news['title']}")

print("\n请检查模型理由是否提及当前新闻正文中不存在的实体、算法、数据或金额。")
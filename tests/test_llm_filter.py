
"""新闻筛选逻辑的离线自动化测试，不调用真实 Ollama。"""

import logging

import pytest

from src.llm_processor import NewsProcessor


@pytest.fixture
def processor():
    """创建处理器，并替换模型客户端为离线模拟对象。"""
    instance = NewsProcessor.__new__(NewsProcessor)
    instance.config = {}
    instance.filter_config = {"min_score": 6}
    instance.summary_config = {}
    instance.ollama_config = {}
    instance.min_score = 6
    instance.max_input_chars = 10000
    instance.max_highlights = 3
    instance.min_summary_chars = 100
    instance.max_summary_chars = 250
    instance.filter_prompt = ""
    instance.summary_prompt = ""
    instance.client = None
    return instance


class FakeClient:
    """按预设内容返回模型结果的模拟客户端。"""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def chat_json(self, prompt, task_name=""):
        self.calls.append((prompt, task_name))
        return self.result


def make_news(title="新 AI 模型发布"):
    """生成一条最小测试新闻。"""
    return {
        "title": title,
        "url": "https://example.com/news",
        "source": "测试来源",
        "published_at": "2026-10-28",
        "content": "测试新闻正文",
    }


def make_item(
    index=0,
    keep=True,
    score=8,
    category="模型",
    reason="当前新闻介绍了新模型的发布。",
):
    """生成一条模拟筛选结果。"""
    return {
        "index": index,
        "keep": keep,
        "score": score,
        "category": category,
        "reason": reason,
    }


def test_filter_accepts_valid_news(processor):
    """合法且达到最低分数的新闻应被保留。"""
    processor.client = FakeClient({
        "items": [make_item()],
    })
    news = make_news()

    result = processor._filter_batch([news])

    assert len(result) == 1
    assert result[0]["title"] == news["title"]
    assert result[0]["score"] == 8
    assert result[0]["category"] == "模型"
    assert result[0]["filter_reason"] == "当前新闻介绍了新模型的发布。"


def test_filter_rejects_score_below_minimum(processor):
    """低于最低分数的新闻应被过滤。"""
    processor.client = FakeClient({
        "items": [make_item(keep=True, score=3)],
    })

    assert processor._filter_batch([make_news()]) == []


def test_filter_accepts_score_at_minimum(processor):
    """评分恰好达到最低分数时应被保留。"""
    processor.client = FakeClient({
        "items": [make_item(keep=True, score=6)],
    })

    result = processor._filter_batch([make_news()])

    assert len(result) == 1
    assert result[0]["score"] == 6


def test_filter_rejects_keep_false(processor):
    """模型明确判定不收录时应过滤。"""
    processor.client = FakeClient({
        "items": [
            make_item(keep=False, score=8, reason="缺乏明确的新进展。")
        ],
    })

    assert processor._filter_batch([make_news()]) == []


@pytest.mark.parametrize(
    "invalid_item",
    [
        make_item(keep="true"),
        make_item(score=True),
        make_item(score=11),
        make_item(category="未知分类"),
    ],
)
def test_filter_skips_invalid_fields(processor, invalid_item):
    """非法字段应被拒绝，不得进入结果。"""
    processor.client = FakeClient({
        "items": [invalid_item],
    })

    assert processor._filter_batch([make_news()]) == []


def test_filter_raises_on_duplicate_index(processor):
    """重复 index 必须报错。"""
    processor.client = FakeClient({
        "items": [
            make_item(index=0),
            make_item(index=0, category="论文"),
        ],
    })

    with pytest.raises(ValueError, match="重复返回新闻 index"):
        processor._filter_batch([make_news()])


def test_filter_skips_missing_index_result(processor, caplog):
    """缺失结果应保守跳过，并记录警告。"""
    processor.client = FakeClient({
        "items": [],
    })

    with caplog.at_level(logging.WARNING):
        result = processor._filter_batch([make_news()])

    assert result == []
    assert "筛选结果缺失，保守跳过" in caplog.text


def test_filter_rejects_missing_items(processor):
    """缺少 items 数组时应报错。"""
    processor.client = FakeClient({})

    with pytest.raises(ValueError, match="缺少 items 数组"):
        processor._filter_batch([make_news()])


def test_filter_prompt_contains_isolation_rules(processor):
    """提示词应明确要求逐条隔离并禁止跨新闻混用信息。"""
    from pathlib import Path

    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "filter_news.txt"
    )
    prompt = prompt_path.read_text(encoding="utf-8")

    assert "每条新闻必须独立判断" in prompt
    assert "禁止在不同新闻之间交换、混用或复制事实" in prompt
    assert "理由必须直接对应当前新闻的标题、摘要或正文" in prompt

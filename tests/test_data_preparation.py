"""Basic unit tests for the data formatting logic (no GPU/model needed)."""

from src.data_preparation import format_example


def test_format_example_basic():
    cfg = {"data": {"instruction_field": "instruction", "response_field": "output"}}
    example = {"instruction": "What is LoRA?", "output": "A parameter-efficient fine-tuning method."}

    result = format_example(example, cfg)

    assert "### Instruction:" in result["text"]
    assert "### Response:" in result["text"]
    assert "What is LoRA?" in result["text"]
    assert "parameter-efficient" in result["text"]


def test_format_example_strips_whitespace():
    cfg = {"data": {"instruction_field": "instruction", "response_field": "output"}}
    example = {"instruction": "  hello  ", "output": "  world  "}

    result = format_example(example, cfg)

    assert "hello" in result["text"]
    assert "world" in result["text"]
    assert "  hello  " not in result["text"]

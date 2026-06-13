# -*- coding: utf-8 -*-
"""LLM module tests — message assembly logic (no real API calls)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.llm import QwenVisionLLM


def test_constructor_defaults():
    """QwenVisionLLM initializes with default model name."""
    llm = QwenVisionLLM()
    assert llm.model_name == "qwen3.7-plus"
    assert llm._client is None  # lazy init
    print("[PASS] Test 1/6: Constructor defaults")


def test_constructor_custom_model():
    """QwenVisionLLM accepts custom model name."""
    llm = QwenVisionLLM(model="qwen-plus")
    assert llm.model_name == "qwen-plus"
    print("[PASS] Test 2/6: Custom model name")


def test_build_messages_no_system_prompt():
    """_build_messages without system prompt produces correct OpenAI format."""
    llm = QwenVisionLLM()
    frames = ["base64img1", "base64img2"]
    history: list[dict] = []
    msgs = llm._build_messages(
        user_text="你好",
        frames=frames,
        history=history,
        system_prompt="",
    )
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "你好\n以上画面按时间顺序排列，第一张是最新的。"}
    assert content[1]["type"] == "image_url"
    assert "base64img1" in content[1]["image_url"]["url"]
    assert content[2]["type"] == "image_url"
    assert "base64img2" in content[2]["image_url"]["url"]
    print("[PASS] Test 3/6: Build messages without system prompt")


def test_build_messages_with_system_prompt():
    """_build_messages includes system prompt as first message."""
    llm = QwenVisionLLM()
    msgs = llm._build_messages(
        user_text="测试",
        frames=["img"],
        history=[],
        system_prompt="你是助手",
    )
    assert len(msgs) == 2  # system + user
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "你是助手"
    assert msgs[1]["role"] == "user"
    print("[PASS] Test 4/6: Build messages with system prompt")


def test_build_messages_with_history():
    """_build_messages includes dialog history before current utterance."""
    llm = QwenVisionLLM()
    history = [
        {"role": "user", "content": "这是什么？"},
        {"role": "assistant", "content": "这是一个杯子。"},
    ]
    msgs = llm._build_messages(
        user_text="它是什么颜色？",
        frames=["img"],
        history=history,
        system_prompt="",
    )
    assert len(msgs) == 3  # user(history) + assistant(history) + user(current)
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "这是什么？"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "这是一个杯子。"
    assert msgs[2]["role"] == "user"
    assert isinstance(msgs[2]["content"], list)
    print("[PASS] Test 5/6: Build messages with history")


def test_frame_truncation():
    """More than 5 frames should be truncated to 5."""
    llm = QwenVisionLLM()
    frames = [f"img{i}" for i in range(10)]
    msgs = llm._build_messages(
        user_text="测试",
        frames=frames,
        history=[],
        system_prompt="",
    )
    content = msgs[0]["content"]
    image_parts = [p for p in content if p["type"] == "image_url"]
    assert len(image_parts) <= 5, f"Expected <= 5, got {len(image_parts)}"
    print("[PASS] Test 6/6: Frame truncation at 5 images")


if __name__ == "__main__":
    test_constructor_defaults()
    test_constructor_custom_model()
    test_build_messages_no_system_prompt()
    test_build_messages_with_system_prompt()
    test_build_messages_with_history()
    test_frame_truncation()
    print("\nAll 6 tests passed!")

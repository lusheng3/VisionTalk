# -*- coding: utf-8 -*-
"""Config module tests — verifies lean settings after refactor."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import Settings


def test_defaults():
    """Default values match expected post-refactor settings."""
    s = Settings()
    # Retained
    assert s.default_model == "qwen-max"
    assert s.port == 8765
    assert s.vad_threshold == 0.5
    assert s.frame_max_width == 768
    assert s.max_history_turns == 10
    # New
    assert s.frame_interval_speech == 0.5
    assert s.frame_interval_silence == 3.0
    assert s.max_frames_per_llm_call == 5
    print("[PASS] Test 1/3: Default values")


def test_removed_fields():
    """Removed config fields should not exist."""
    s = Settings()
    removed = [
        "zhipu_api_key", "ernie_api_key", "ernie_secret_key",
        "ssim_threshold", "scene_change_threshold",
        "debounce_seconds", "silence_timeout_seconds",
    ]
    for field in removed:
        assert not hasattr(s, field), f"Field '{field}' should be removed"
    print("[PASS] Test 2/3: Removed fields gone")


def test_api_key_empty_default():
    """dashscope_api_key defaults to empty string (set via .env)."""
    s = Settings()
    assert s.dashscope_api_key == ""
    print("[PASS] Test 3/3: API key default")


if __name__ == "__main__":
    test_defaults()
    test_removed_fields()
    test_api_key_empty_default()
    print("\nAll 3 tests passed!")

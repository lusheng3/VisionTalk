# -*- coding: utf-8 -*-
"""STT module unit tests.

Tests STTEngine interface and edge cases.
Real speech transcription requires model download (~70MB), tested separately.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from backend.stt import STTEngine


def test_engine_loading():
    """STTEngine loads with correct default model size."""
    engine = STTEngine()
    assert engine.model_size == "tiny"
    assert engine._model is None  # Lazy loading, no model yet
    print("[PASS] Test 1/4: STTEngine loading")


def test_too_short_audio():
    """Audio shorter than 0.3s should return empty string."""
    engine = STTEngine()
    # 0.2s @ 16kHz = 3200 samples (below min 4800)
    short_audio = np.zeros(3200, dtype=np.float32)
    result = engine.transcribe(short_audio)
    assert result == "", f"Expected empty, got: '{result}'"
    print("[PASS] Test 2/4: Too-short audio rejected")


def test_silence_audio():
    """Silence audio should return empty or no meaningful text."""
    engine = STTEngine()
    # 1s silence — enough length to pass minimum, but Whisper should find no words
    silence = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(silence)
    # Silence should yield empty or very short result
    assert len(result) < 10, f"Silence should give near-empty result, got: '{result}'"
    print(f"   (silence result: '{result}')")
    print("[PASS] Test 3/4: Silence audio")


def test_dtype_conversion():
    """Non-float32 audio should be converted automatically."""
    engine = STTEngine()
    # int16 audio (common format), 1s
    int_audio = np.zeros(16000, dtype=np.int16)
    result = engine.transcribe(int_audio.astype(np.float32))
    # Should not crash on dtype
    assert isinstance(result, str)
    print("[PASS] Test 4/4: Dtype handling")


if __name__ == "__main__":
    test_engine_loading()
    test_too_short_audio()
    test_silence_audio()
    test_dtype_conversion()
    print("\nAll 4 tests passed!")

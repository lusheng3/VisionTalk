# -*- coding: utf-8 -*-
"""VAD module tests — VADDetector only (SpeechSegmenter removed)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from backend.vad import VADDetector


def test_vad_detector_loading():
    """VADDetector loads with correct defaults."""
    d = VADDetector()
    assert d.threshold == 0.5, f"Expected threshold 0.5, got: {d.threshold}"
    assert d._sample_rate == 16000
    print("[PASS] Test 1/4: VADDetector loading")


def test_vad_silence_detection():
    """All-zero audio should be classified as silence."""
    d = VADDetector()
    silence = np.zeros(16000, dtype=np.float32)  # 1s @ 16kHz
    result = d.is_speech(silence)
    assert result is False, f"Silence should return False, got: {result}"
    print("[PASS] Test 2/4: Silence detection")


def test_vad_confidence_range():
    """Confidence score should be within [0, 1]."""
    d = VADDetector()
    silence = np.zeros(8000, dtype=np.float32)  # 0.5s
    confidence = d.get_speech_confidence(silence)
    assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"
    print(f"   (silence confidence: {confidence:.3f})")
    print("[PASS] Test 3/4: Confidence range")


def test_vad_short_chunk():
    """Short audio chunk (< 512 samples) should be handled via zero-padding."""
    d = VADDetector()
    short = np.zeros(200, dtype=np.float32)  # < 512 window
    result = d.is_speech(short)
    assert isinstance(result, bool)
    assert result is False
    print("[PASS] Test 4/4: Short chunk handling")


if __name__ == "__main__":
    test_vad_detector_loading()
    test_vad_silence_detection()
    test_vad_confidence_range()
    test_vad_short_chunk()
    print("\nAll 4 tests passed!")

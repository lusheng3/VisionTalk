# -*- coding: utf-8 -*-
"""VAD module unit tests.

Tests VADDetector and SpeechSegmenter core functionality.
Runs independently with no external service dependencies.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from backend.vad import VADDetector, SpeechSegmenter


def test_vad_detector_loading():
    """VADDetector loads correctly with config defaults."""
    d = VADDetector()
    assert d.threshold == 0.5, f"Expected threshold 0.5, got: {d.threshold}"
    assert d._sample_rate == 16000
    print("[PASS] Test 1/5: VADDetector loading")


def test_vad_silence_detection():
    """All-zero audio should be classified as silence."""
    d = VADDetector()
    silence = np.zeros(16000, dtype=np.float32)  # 1s @ 16kHz
    result = d.is_speech(silence)
    assert result is False, f"Silence should return False, got: {result}"
    print("[PASS] Test 2/5: Silence detection")


def test_vad_confidence_range():
    """Confidence score should be within [0, 1]."""
    d = VADDetector()
    silence = np.zeros(8000, dtype=np.float32)  # 0.5s
    confidence = d.get_speech_confidence(silence)
    assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"
    print(f"   (silence confidence: {confidence:.3f})")
    print("[PASS] Test 3/5: Confidence range")


def test_segmenter_loading():
    """SpeechSegmenter loads with correct default parameters."""
    s = SpeechSegmenter()
    # 2s * 16000Hz = 32000 samples
    assert s.silence_samples == 32000, f"Expected 32000, got: {s.silence_samples}"
    assert s.sample_rate == 16000
    assert not s._is_speaking
    print("[PASS] Test 4/5: SpeechSegmenter loading")


def test_segmenter_silence_passthrough():
    """Feeding pure silence should produce zero speech segments."""
    s = SpeechSegmenter()
    silence = np.zeros(8000, dtype=np.float32)  # 0.5s silence
    segments = s.add_chunk(silence)
    assert len(segments) == 0, f"Silence should yield 0 segments, got: {len(segments)}"
    assert not s._is_speaking, "Should not be in speaking state"
    print("[PASS] Test 5/5: Silence passthrough")


if __name__ == "__main__":
    test_vad_detector_loading()
    test_vad_silence_detection()
    test_vad_confidence_range()
    test_segmenter_loading()
    test_segmenter_silence_passthrough()
    print("\nAll 5 tests passed!")

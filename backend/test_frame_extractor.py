# -*- coding: utf-8 -*-
"""FrameExtractor module unit tests.

Tests key frame extraction, SSIM dedup, resolution compression,
and scene change detection.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from backend.frame_extractor import FrameExtractor, CapturedFrame


def _make_frame(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    """Generate a random BGR frame for testing."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (height, width, 3), dtype=np.uint8)


def test_frame_extractor_loading():
    """FrameExtractor loads with correct config defaults."""
    fe = FrameExtractor()
    assert fe.max_width == 768
    assert fe.jpeg_quality == 80
    assert fe.ssim_threshold == 0.95
    assert fe._last_grayscale is None
    print("[PASS] Test 1/5: FrameExtractor loading")


def test_extract_first_frame():
    """First frame extraction should always return a CapturedFrame."""
    fe = FrameExtractor()
    frame = _make_frame()
    result = fe.extract(frame)
    assert result is not None, "First frame should not be skipped"
    assert isinstance(result, CapturedFrame)
    assert result.width <= 768, f"Width should be <= 768, got: {result.width}"
    assert len(result.image_bytes) > 0, "JPEG bytes should not be empty"
    assert len(result.base64) > 0, "base64 string should not be empty"
    assert result.timestamp > 0
    print(f"   (output: {result.width}x{result.height}, {len(result.image_bytes)} bytes)")
    print("[PASS] Test 2/5: First frame extraction")


def test_dedup_same_frame():
    """Identical frame should be skipped (SSIM ~1.0 > 0.95)."""
    fe = FrameExtractor()
    frame = _make_frame(seed=42)
    # First extraction should succeed
    first = fe.extract(frame)
    assert first is not None, "First extraction should succeed"
    # Second extraction of same frame should be skipped
    second = fe.extract(frame)
    assert second is None, "Identical frame should be deduplicated"
    print("[PASS] Test 3/5: Duplicate frame dedup")


def test_extract_different_frame():
    """A visually different frame should be extracted."""
    fe = FrameExtractor()
    # Extract frame A
    frame_a = _make_frame(seed=1)
    first = fe.extract(frame_a)
    assert first is not None
    # Extract frame B (different seed = different image)
    frame_b = _make_frame(seed=999)
    second = fe.extract(frame_b)
    assert second is not None, "Different frame should not be skipped"
    print("[PASS] Test 4/5: Different frame extraction")


def test_reset():
    """reset() should clear internal state."""
    fe = FrameExtractor()
    frame = _make_frame(seed=7)
    fe.extract(frame)
    assert fe._last_grayscale is not None
    fe.reset()
    assert fe._last_grayscale is None
    print("[PASS] Test 5/5: reset() clears state")


if __name__ == "__main__":
    test_frame_extractor_loading()
    test_extract_first_frame()
    test_dedup_same_frame()
    test_extract_different_frame()
    test_reset()
    print("\nAll 5 tests passed!")

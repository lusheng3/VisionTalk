# Smart Frame Strategy Design Spec

**Date:** 2026-06-14  
**Branch:** feat/smart-frames

---

## Problem

Current frame pipeline has three weaknesses:

1. **Uniform sampling** — head-to-tail equal-interval pick; irrelevant frames from before user raised an object waste LLM tokens
2. **Naive dedup** — compares middle 200 chars of base64 string, which is fragile and inaccurate
3. **Keyword-based visual detection** — hardcoded list misses edge cases ("帮我看看这个", "这本书讲什么")

## Goal

Make frames sent to LLM **information-dense**: fewer frames but each frame matters. Non-visual questions send zero frames. Target: 3-5 key frames per visual query, LLM first token under 6-8s.

## Architecture

```
Camera → 384px JPEG → pHash dedup → frame_diff action detection
                                            │
                                    15s time window
                                            │
                                     PTT ends
                                            │
                              ┌─ Qwen-Turbo router ─┐
                              │   "Needs visual?"    │
                              └───── YES/NO ─────────┘
                                NO              YES
                                 │               │
                             0 frames     Keyframe Selector:
                                         1. Action frames (max 3)
                                         2. End frame (always)
                                         3. Context frames (fill to 3-5)
                                            │
                                         3-5 frames → Qwen-VL
```

## Task Breakdown

### Task 1: Resolution 384px

Change `FRAME_MAX_WIDTH` from 512 to 384, `FRAME_JPEG_QUALITY` from 0.6 to 0.55.

**Why first:** one-line change, trivially testable, immediate token reduction.

### Task 2: 15s Time Window

Replace fixed-size `deque(30)` with timestamp-based sliding window of 15 seconds. Each frame gets `{base64, timestamp, hash, isAction}` instead of raw string.

**Why second:** structural foundation for all subsequent tasks.

### Task 3: pHash Dedup

Implement perceptual hashing in pure JS:
- 32×32 grayscale downsample → average hash → 64-bit integer
- Hamming distance between hashes determines similarity
- Replace base64-string dedup with hash comparison (threshold: distance < 8 = duplicate)

**Why third:** core algorithm, no external dependencies, testable in isolation.

### Task 4: Action Detection

Mark frames where hamming distance from previous frame exceeds threshold (12) as `isAction: true`. These frames indicate user movement — raising an object, pointing, turning.

**Why fourth:** depends on pHash from Task 3.

### Task 5: Keyframe Selector

Replace `getFramesForLLM()` uniform sampling with priority-based selection:
1. All action frames (max 3, most recent first)
2. Last frame (end state, always included)
3. Context frames from recent window (fill to 5 max)

**Why fifth:** depends on action flags from Task 4.

### Task 6: Qwen-Turbo Visual Router

New `backend/router.py`:
- Prompt: "Does this question require viewing the camera? Answer YES or NO."
- Model: `qwen-turbo` (fast, cheap)
- Latency target: 200-400ms
- Replace hardcoded keyword list in `server.py`

**Why last:** backend change, API call with async handling, needs integration test.

## Success Metrics

| Metric | Before | Target |
|------|:--:|:--:|
| Frames per visual query | 3-10 (uniform) | 3-5 (key frames only) |
| Non-visual frames | 0-1 (keyword miss) | 0 (router) |
| Image token per query | ~3000-10000 | ~1500-5000 |
| LLM first token (visual) | 8-11s | 6-8s |
| LLM first token (non-visual) | 5-7s | 2-4s |
| Visual detection accuracy | ~80% (keywords) | ~98% (router) |

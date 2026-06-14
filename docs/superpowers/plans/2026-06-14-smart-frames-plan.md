# Smart Frame Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace naive uniform frame sampling with perceptual-hash dedup, motion-based keyframe selection, time-window buffering, and a Qwen-Turbo visual router.

**Architecture:** All frame processing stays in the frontend (pHash, diff detection, time window, keyframe selector). A lightweight backend router uses Qwen-Turbo to decide whether user query needs images.

---

## Task 1: pHash Dedup (frontend)

**Goal:** Replace fragile base64-middle-string comparison with perceptual hashing.

**Files:**
- Modify: `frontend/index.html`

**Implementation:**

Add a pure-JS pHash function (no dependencies):
```javascript
function pHash(imageData, size = 32) {
    // Resize to size×size grayscale, compute DCT, hash top-left 8×8
    const pixels = [];
    // Step 1: downsample to size×size, extract grayscale
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const sx = Math.floor(x * imageData.width / size);
            const sy = Math.floor(y * imageData.height / size);
            const i = (sy * imageData.width + sx) * 4;
            pixels.push(0.299 * imageData.data[i] + 0.587 * imageData.data[i+1] + 0.114 * imageData.data[i+2]);
        }
    }
    // Step 2: DCT on 32×32 (simplified: average hash — compare to mean)
    const mean = pixels.reduce((a, b) => a + b, 0) / pixels.length;
    let hash = 0n;
    for (let i = 0; i < 64; i++) {
        if (pixels[Math.floor(i * pixels.length / 64)] > mean) {
            hash |= (1n << BigInt(i));
        }
    }
    return hash;
}

function hammingDistance(a, b) {
    let xor = a ^ b;
    let count = 0;
    while (xor) { count++; xor &= (xor - 1n); }
    return count;
}
```

Replace the dedup in `captureFrame()`:
```javascript
// Before: let hash = base64.substring(base64.length >> 1, ...);
// After: compare pHash
const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
const hash = pHash(imgData);
if (lastFrameHash !== null && hammingDistance(hash, lastFrameHash) < 8) return;
lastFrameHash = hash;
```

Also update `getFramesForLLM` secondary dedup to use pHash comparison on the canvas data.

- [ ] **Step 1: Write pHash + hammingDistance utility functions**
- [ ] **Step 2: Replace captureFrame dedup with pHash comparison**
- [ ] **Step 3: Replace getFramesForLLM dedup with pHash comparison**
- [ ] **Step 4: Verify — camera should still capture, dedup now based on visual similarity**
- [ ] **Step 5: Commit**

---

## Task 2: Time Window Buffer (frontend)

**Goal:** Replace fixed 30-frame deque with a 15-second time window.

**Files:**
- Modify: `frontend/index.html`

**Implementation:**

Change the frame buffer structure:
```javascript
// Before:
let frameBuffer = [];  // array, push base64 strings

// After:
let frameBuffer = [];  // array of {base64, timestamp, hash, isAction}
const FRAME_WINDOW_SECONDS = 15;

function addFrame(frameData) {
    frameBuffer.push(frameData);
    // Purge frames older than window
    const cutoff = Date.now() - FRAME_WINDOW_SECONDS * 1000;
    frameBuffer = frameBuffer.filter(f => f.timestamp > cutoff);
}
```

- [ ] **Step 1: Change frameBuffer from string[] to object[] with timestamp**
- [ ] **Step 2: Implement addFrame with time-based purge**
- [ ] **Step 3: Update getFramesForLLM to use timestamp-based selection**
- [ ] **Step 4: Adjust FRAME_INTERVAL_IDLE to 2s (shorter window, more frequent but still low-cost)**
- [ ] **Step 5: Commit**

---

## Task 3: Action Detection (frontend)

**Goal:** Detect frames with significant change (user moved, raised object) and flag them.

**Files:**
- Modify: `frontend/index.html`

**Implementation:**

```javascript
const ACTION_DIFF_THRESHOLD = 12;  // hamming distance threshold

function captureFrame() {
    // ... existing capture logic ...
    const hash = pHash(imgData);
    const isAction = lastFrameHash !== null && hammingDistance(hash, lastFrameHash) >= ACTION_DIFF_THRESHOLD;

    addFrame({
        base64: base64,
        timestamp: Date.now(),
        hash: hash,
        isAction: isAction,
    });

    lastFrameHash = hash;
}
```

- [ ] **Step 1: Add isAction flag to frame data**
- [ ] **Step 2: Compute diff from previous frame hash in captureFrame**
- [ ] **Step 3: Mark isAction = true when hamming distance >= threshold**
- [ ] **Step 4: Commit**

---

## Task 4: Keyframe Selector (frontend)

**Goal:** Replace uniform sampling with prioritized selection: action frames → end frame → context frames.

**Files:**
- Modify: `frontend/index.html`

**Implementation:**

```javascript
function getFramesForLLM(durationSeconds) {
    const MAX_FRAMES = 5;

    // 1. Collect action frames (max 3, most recent first)
    const actions = frameBuffer.filter(f => f.isAction).slice(-3);

    // 2. Always include the last frame
    const lastFrame = frameBuffer[frameBuffer.length - 1];

    // 3. Fill remaining slots with context frames
    const selected = new Set();
    for (const f of actions) selected.add(f);
    if (lastFrame) selected.add(lastFrame);

    // 4. If still under MAX_FRAMES, add weighted recent frames
    if (selected.size < MAX_FRAMES) {
        // Sort by recency, skip already selected
        for (let i = frameBuffer.length - 1; i >= 0 && selected.size < MAX_FRAMES; i--) {
            selected.add(frameBuffer[i]);
        }
    }

    // Convert to array, sort by timestamp, return base64
    return [...selected]
        .sort((a, b) => a.timestamp - b.timestamp)
        .map(f => f.base64);
}
```

- [ ] **Step 1: Implement keyframe selector with action/end/context priority**
- [ ] **Step 2: Cap at MAX_FRAMES=5**
- [ ] **Step 3: Verify frame selection logic with mock data**
- [ ] **Step 4: Commit**

---

## Task 5: Visual Router (backend)

**Goal:** Replace hardcoded keyword check with Qwen-Turbo YES/NO judgment.

**Files:**
- Create: `backend/router.py`
- Modify: `backend/server.py`

**Implementation:**

```python
# backend/router.py
"""Visual router — uses Qwen-Turbo to decide if user query needs camera frames."""
import time, logging
from openai import AsyncOpenAI
from backend.config import settings

log = logging.getLogger("VisionTalk")

ROUTER_PROMPT = """判断用户的问题是否需要查看摄像头画面才能回答。
- 需要画面: 回复 YES
- 不需要画面: 回复 NO
只回复 YES 或 NO，不要其他内容。"""

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _client

async def needs_visual(user_text: str) -> bool:
    t0 = time.time()
    client = _get_client()
    resp = await client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": user_text},
        ],
        max_tokens=3,
        temperature=0,
    )
    result = resp.choices[0].message.content.strip().upper()
    elapsed = time.time() - t0
    is_visual = "YES" in result
    log.info(f"[Router] 🧠 Qwen-Turbo → {result} | 耗时 {elapsed:.2f}s")
    return is_visual
```

In `server.py`, replace the keyword check:
```python
# Before:
VISUAL_KEYWORDS = [...]
is_visual = any(kw in text_lower for kw in VISUAL_KEYWORDS)

# After:
from backend.router import needs_visual
is_visual = await needs_visual(text.strip())
```

- [ ] **Step 1: Create backend/router.py with needs_visual()**
- [ ] **Step 2: Replace keyword list in server.py with router call**
- [ ] **Step 3: Test — visual question routes to YES, non-visual to NO**
- [ ] **Step 4: Commit**

---

## Task 6: Resolution Downscale (frontend)

**Goal:** Reduce frame size from 512px to 384px for faster LLM processing.

**Files:**
- Modify: `frontend/index.html`

**Implementation:**

```javascript
// Before: const FRAME_MAX_WIDTH = 512;
// After:  const FRAME_MAX_WIDTH = 384;
```

Also increase JPEG compression slightly:
```javascript
// Before: const FRAME_JPEG_QUALITY = 0.6;
// After:  const FRAME_JPEG_QUALITY = 0.55;
```

- [ ] **Step 1: Change FRAME_MAX_WIDTH to 384**
- [ ] **Step 2: Change FRAME_JPEG_QUALITY to 0.55**
- [ ] **Step 3: Commit**

---

## Task 7: Integration & Commit Summary

- [ ] **Step 1: Full integration test — camera → PTT → STT → Router → Keyframes → LLM**
- [ ] **Step 2: Verify logs show frame count, action count, router decision**
- [ ] **Step 3: Final commit**

---

## PR Split

Since this is a single feature (smart frames), submit as 1 PR with coherent commit history.

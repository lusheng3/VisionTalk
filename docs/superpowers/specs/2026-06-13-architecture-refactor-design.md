# Architecture Refactor Design

**Date:** 2026-06-13
**Branch:** feat/llm-adapters
**Decision:** 方案一 · 精简管道

---

## Requirements Summary

| Dimension | Decision |
|-----------|----------|
| Interaction | PTT (Push-to-Talk), press to speak / release to stop |
| Dialog mode | Multi-turn continuous, full conversation history |
| Frame strategy | Mixed: 0.5s interval during speech, 3s during silence |
| Visual context | Current frame + history frames (max 5 per LLM call) |
| Model | Alibaba Qwen only, single model |
| TTS | Deferred — `print()` text output for now |

---

## Module Breakdown

```
backend/
  frame_grabber.py    # Timer-based frame capture + JPEG compression
  vad.py              # VADDetector only (SpeechSegmenter removed)
  stt.py              # Unchanged
  llm.py              # Single-file QwenVisionLLM, no abstract base
  dialog.py           # Dialog manager: orchestrates pipeline, maintains history
  config.py           # Lean config: removed zhipu/ernie/ssim/scene_change
```

| Module | Responsibility | Public API |
|--------|---------------|------------|
| `frame_grabber.py` | Timer-driven frame capture, resize, JPEG encode, sliding buffer | `start()`, `stop()`, `set_mode(speech/silence)`, `get_current()`, `get_recent(n)` |
| `vad.py` | Per-chunk speech detection via Silero VAD | `is_speech(chunk) → bool` |
| `stt.py` | Speech-to-text via Faster-Whisper (unchanged) | `transcribe(audio) → str` |
| `llm.py` | Qwen-VL multimodal call, text + images → response | `chat(user_text, frames, history) → str` |
| `dialog.py` | Holds all 4 modules, PTT flow orchestration, dialog history | `press()`, `release()`, `get_history()` |

### Deleted

- `models/base.py` — abstract base class, unnecessary for single model
- `models/qwen.py` — merged into `llm.py`
- `vad.py` `SpeechSegmenter` — silence-based segmentation not needed for PTT
- `backend/models/` — entire directory removed

---

## PTT Interaction Flow

```
User presses key (press)
  │
  ├─▶ FrameGrabber.set_mode("speech")    # high-frequency capture (0.5s)
  ├─▶ Start recording (pyaudio)          # accumulate audio chunks
  │
  │   ... user holding key ...
  │   FrameGrabber captures every 0.5s → push to buffer
  │   Audio keeps appending
  │
User releases key (release)
  │
  ├─▶ Stop recording → full audio (float32, 16kHz)
  ├─▶ FrameGrabber.set_mode("silence")   # low-frequency capture (3s)
  │
  ├─▶ STT: audio → text
  │    If text empty → return silently (false trigger)
  │
  ├─▶ Assemble LLM input:
  │     - system_prompt
  │     - dialog history (last N turns)
  │     - current speech text
  │     - current frame + recent history frames (max 5)
  │
  ├─▶ LLM.chat() → response text
  │
  ├─▶ Store in dialog history (user Q + AI A + frame references)
  │
  └─▶ Output response (print / future TTS)
```

**False trigger protection:** if STT returns empty text, skip LLM call, no history mutation.

**Silence frame capture:** continues at 3s interval even when user is not speaking, so `get_recent(n)` always has "pre-speech" frames available.

---

## Data Structures

```python
# dialog.py internal structures

@dataclass
class Turn:
    """One conversation round."""
    user_text: str
    assistant_text: str
    frames: list[CapturedFrame]     # frames captured during this turn
    timestamp: float

@dataclass
class CapturedFrame:
    """Single frame produced by FrameGrabber."""
    base64: str                     # JPEG base64 for direct LLM consumption
    width: int
    height: int
    timestamp: float
```

### History Strategy

- **Upper limit:** N turns (`config.max_history_turns`, default 10)
- **Per turn:** stores user text, AI response, and frame references
- **LLM context assembly:**
  1. system_prompt
  2. Recent N dialog turns (user + assistant messages)
  3. Current frame + speech-period frames (max 5)
  4. Current user utterance

### Frame Buffer Strategy

`FrameGrabber` maintains:
- `current_frame: CapturedFrame | None` — latest captured frame
- `frame_buffer: deque[CapturedFrame]` — sliding window of last 30 frames

`Dialog` queries:
- `get_current()` → latest frame
- `get_recent(n)` → uniformly sampled n frames

---

## LLM Module (llm.py)

```python
class QwenVisionLLM:
    """Qwen vision-language model via DashScope OpenAI-compatible API."""

    def __init__(self, model: str = "qwen-max"):
        self._model = model
        self._client: AsyncOpenAI | None = None

    async def chat(
        self,
        user_text: str,
        frames: list[str],           # base64 images (current + history, max 5)
        history: list[dict],         # past turns
        system_prompt: str = "",
    ) -> tuple[str, int, int]:       # (text, input_tokens, output_tokens)
        ...
```

### Message Assembly Order

1. system_prompt (if present)
2. Dialog history (user/assistant alternating)
3. Current frame + speech frames as multimodal content
   - Text part: current user utterance
   - Image parts: current frame first, history frames follow
   - Hint appended: "以上画面按时间顺序排列，第一张是最新的"

### Image Count Control

- Max 5 frames per LLM call (token budget)
- Priority: current frame → speech-period frames → history frames

### Removed (vs. current `models/`)

| Removed | Reason |
|---------|--------|
| `base.py` / `BaseVisionLLM` | Single model, no abstraction needed |
| `VisionMessage` dataclass | Use dict directly |
| `LLMResponse` dataclass | Return tuple |
| `models/__init__.py` | Entire directory removed |

---

## FrameGrabber

```python
class FrameGrabber:
    """Timer-driven frame capture from camera, JPEG encode, sliding buffer.

    Two modes:
      - speech: 0.5s interval
      - silence: 3s interval
    """

    def __init__(
        self,
        camera_id: int = 0,
        max_width: int = 768,
        jpeg_quality: int = 75,
        buffer_size: int = 30,
    ):
        ...

    def start(self): ...
    def stop(self): ...
    def set_mode(self, mode: str): ...    # "speech" | "silence"
    def get_current(self) -> CapturedFrame | None: ...
    def get_recent(self, n: int) -> list[CapturedFrame]: ...
```

**Internal:** background thread loop: read camera → resize → JPEG encode → push to `collections.deque(maxlen=buffer_size)`

### Removed (vs. current `frame_extractor.py`)

| Removed | Reason |
|---------|--------|
| SSIM dedup (`extract()` similarity check) | Timer-driven, no dedup needed |
| `is_scene_change()` | No scene change detection |
| `reset()` / `_last_grayscale` | Stateless |
| `skimage.metrics` dependency | No longer needed |
| `CapturedFrame.image_bytes` | Only keep base64 |

---

## VAD Changes

Keep `VADDetector`, remove `SpeechSegmenter`:

```python
class VADDetector:
    """Voice activity detection via Silero VAD."""
    def __init__(self, threshold: float = 0.5): ...
    def is_speech(self, audio_chunk: np.ndarray) -> bool: ...
```

PTT recording logic lives in `dialog.py` as a private method (~30 lines of pyaudio loop).

---

## Config Changes

```python
# Removed
zhipu_api_key, ernie_api_key, ernie_secret_key
ssim_threshold, scene_change_threshold
debounce_seconds, silence_timeout_seconds

# Retained
dashscope_api_key, default_model, port
vad_threshold, frame_max_width, max_history_turns

# Added
frame_interval_speech: float = 0.5
frame_interval_silence: float = 3.0
max_frames_per_llm_call: int = 5
```

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `config.py` | ✏️ Trim | Remove 7 config items, add 3 |
| `vad.py` | ✏️ Trim | Remove `SpeechSegmenter`, keep `VADDetector` |
| `frame_extractor.py` → `frame_grabber.py` | 🔄 Rewrite | Timer-based capture, no SSIM |
| `stt.py` | ✅ Keep | Interface sufficient as-is |
| `models/` directory | ❌ Delete | Entire directory |
| `llm.py` | 🆕 New | Single-file Qwen, no abstraction |
| `dialog.py` | 🆕 New | Core orchestration + PTT flow + history |
| Tests | ✏️ Update | Adapt to new interfaces, remove models tests |

---

## Dependencies

```diff
# requirements.txt
- scikit-image        # SSIM no longer needed
  openai              # Retained (DashScope compatible)
  faster-whisper       # Retained
  silero-vad          # Retained
  torch               # Retained
+ pyaudio             # New: PTT microphone capture
```

---

## Test Strategy

| Module | What to test |
|--------|-------------|
| `frame_grabber` | Mode switch changes interval, buffer cap, `get_recent` uniform sampling |
| `llm` | Message assembly structure (system/user/assistant order, data URI format), >5 frame truncation |
| `dialog` | Mocked deps: press starts record → release calls STT → empty text skips LLM, history cap trimming, correct frame references |
| `vad` | `is_speech()` returns bool |
| `stt` | Unchanged |

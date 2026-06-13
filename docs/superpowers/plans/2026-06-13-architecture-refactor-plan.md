# Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor backend from multi-provider abstract LLM adapters + SSIM frame extraction to a lean single-model PTT pipeline with timer-based frame capture and dialog management.

**Architecture:** Five focused files — `config.py` (lean settings), `frame_grabber.py` (timer-based capture), `vad.py` (voice detection), `llm.py` (single Qwen call), `dialog.py` (PTT orchestration + multi-turn history). DialogManager holds all modules, press()/release() drive the full pipeline.

**Tech Stack:** Python 3.10+, openai (DashScope compatible), faster-whisper, silero-vad, opencv-python, pyaudio, pillow

---

### Task 1: Simplify Config

**Files:**
- Modify: `backend/config.py`
- Test: `backend/test_config.py` (new)

- [ ] **Step 1: Write failing test for new config**

Create `backend/test_config.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python backend/test_config.py`
Expected: FAIL — `frame_interval_speech` etc. not yet defined, and old fields (like `zhipu_api_key`) still exist.

- [ ] **Step 3: Rewrite config.py**

Replace `backend/config.py` with:
```python
"""应用配置管理。从 .env 文件和环境变量加载配置。"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """全局配置，自动从 .env 和环境变量加载。"""

    # API Key
    dashscope_api_key: str = ""

    # 默认模型
    default_model: str = "qwen-max"

    # 服务端口
    port: int = 8765

    # VAD 灵敏度 (0-1, 越大越激进地判定为语音)
    vad_threshold: float = 0.5

    # 对话历史保留轮数
    max_history_turns: int = 10

    # 截帧最大宽度 (像素)
    frame_max_width: int = 768

    # 说话时抓帧间隔 (秒)
    frame_interval_speech: float = 0.5

    # 静音时抓帧间隔 (秒)
    frame_interval_silence: float = 3.0

    # 每次发给 LLM 的最大帧数
    max_frames_per_llm_call: int = 5

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python backend/test_config.py`
Expected: 3/3 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/config.py backend/test_config.py
git commit -m "feat: simplify config — remove unused providers, add frame timing settings

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Create FrameGrabber (timer-based capture)

**Files:**
- Create: `backend/frame_grabber.py`
- Test: `backend/test_frame_grabber.py` (new)

- [ ] **Step 1: Write failing test for FrameGrabber**

Create `backend/test_frame_grabber.py`:
```python
# -*- coding: utf-8 -*-
"""FrameGrabber tests — timer-based capture with mode switching."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from backend.frame_grabber import FrameGrabber, CapturedFrame


def test_constructor_defaults():
    """FrameGrabber initializes with correct defaults."""
    fg = FrameGrabber()
    assert fg.max_width == 768
    assert fg.jpeg_quality == 75
    assert fg.buffer.maxlen == 30
    assert fg.current_frame is None
    # Default mode is silence (3s)
    assert fg._interval == 3.0
    print("[PASS] Test 1/6: Constructor defaults")


def test_captured_frame_dataclass():
    """CapturedFrame stores frame data correctly."""
    cf = CapturedFrame(
        base64="ZmFrZQ==",
        width=640,
        height=480,
        timestamp=1234567890.0,
    )
    assert cf.base64 == "ZmFrZQ=="
    assert cf.width == 640
    assert cf.height == 480
    assert cf.timestamp == 1234567890.0
    print("[PASS] Test 2/6: CapturedFrame dataclass")


def test_set_mode_speech():
    """set_mode('speech') switches to 0.5s interval."""
    fg = FrameGrabber()
    fg.set_mode("speech")
    assert fg._interval == 0.5
    print("[PASS] Test 3/6: Mode switch to speech")


def test_set_mode_silence():
    """set_mode('silence') switches to 3.0s interval."""
    fg = FrameGrabber()
    fg.set_mode("speech")  # change first
    fg.set_mode("silence")
    assert fg._interval == 3.0
    print("[PASS] Test 4/6: Mode switch to silence")


def test_set_mode_invalid():
    """Invalid mode raises ValueError."""
    fg = FrameGrabber()
    try:
        fg.set_mode("invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] Test 5/6: Invalid mode raises")


def test_get_recent_returns_list():
    """get_recent returns a list (empty when no frames captured)."""
    fg = FrameGrabber()
    result = fg.get_recent(3)
    assert isinstance(result, list)
    assert len(result) == 0  # no frames yet
    print("[PASS] Test 6/6: get_recent returns list")


if __name__ == "__main__":
    test_constructor_defaults()
    test_captured_frame_dataclass()
    test_set_mode_speech()
    test_set_mode_silence()
    test_set_mode_invalid()
    test_get_recent_returns_list()
    print("\nAll 6 tests passed!")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python backend/test_frame_grabber.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.frame_grabber'`

- [ ] **Step 3: Implement FrameGrabber**

Create `backend/frame_grabber.py`:
```python
"""定时抓帧模块。

从摄像头定时抓取帧，JPEG 编码，维护滑动窗口缓冲区。
两种模式: speech (0.5s 高频) / silence (3s 低频)。
"""
import threading
import time
import base64
import io
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from backend.config import settings


@dataclass
class CapturedFrame:
    """单帧数据，base64 编码可直接发给 LLM。"""
    base64: str
    width: int
    height: int
    timestamp: float


class FrameGrabber:
    """定时抓帧器，后台线程循环读取摄像头。

    Usage:
        fg = FrameGrabber()
        fg.start()
        fg.set_mode("speech")    # high frequency
        frame = fg.current_frame
        fg.stop()
    """

    def __init__(
        self,
        camera_id: int = 0,
        max_width: int | None = None,
        jpeg_quality: int = 75,
        buffer_size: int = 30,
    ):
        self.max_width = max_width or settings.frame_max_width
        self.jpeg_quality = jpeg_quality
        self.buffer: deque[CapturedFrame] = deque(maxlen=buffer_size)
        self._camera_id = camera_id
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._interval = settings.frame_interval_silence  # default: silence mode
        self._lock = threading.Lock()

    # ── Properties ──

    @property
    def current_frame(self) -> CapturedFrame | None:
        """最新抓到的帧。"""
        with self._lock:
            if self.buffer:
                return self.buffer[-1]
            return None

    # ── Lifecycle ──

    def start(self) -> None:
        """启动后台抓帧线程。"""
        if self._running:
            return
        self._cap = cv2.VideoCapture(self._camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._camera_id}")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止抓帧线程并释放摄像头。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Mode ──

    def set_mode(self, mode: str) -> None:
        """切换抓帧频率。

        Args:
            mode: "speech" → 高频 (frame_interval_speech 秒)
                  "silence" → 低频 (frame_interval_silence 秒)
        """
        if mode == "speech":
            self._interval = settings.frame_interval_speech
        elif mode == "silence":
            self._interval = settings.frame_interval_silence
        else:
            raise ValueError(f"Unknown mode: {mode!r}, expected 'speech' or 'silence'")

    # ── Frame access ──

    def get_current(self) -> CapturedFrame | None:
        """获取最新帧（同 current_frame 属性）。"""
        return self.current_frame

    def get_recent(self, n: int) -> list[CapturedFrame]:
        """获取最近 n 帧，均匀采样。

        Args:
            n: 要返回的帧数量。

        Returns:
            按时间排序的帧列表（旧→新），数量 <= n。
        """
        with self._lock:
            frames = list(self.buffer)
        if len(frames) <= n:
            return frames
        # 均匀采样 n 帧，始终包含最新帧
        step = (len(frames) - 1) / (n - 1)
        indices = [int(i * step) for i in range(n - 1)] + [len(frames) - 1]
        return [frames[i] for i in sorted(set(indices))]

    # ── Internal ──

    def _capture_loop(self) -> None:
        """后台线程：循环读取摄像头帧。"""
        last_capture = 0.0
        while self._running:
            now = time.time()
            if now - last_capture < self._interval:
                time.sleep(0.05)  # 避免忙等
                continue

            ret, frame_bgr = self._cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            try:
                cf = self._encode(frame_bgr, now)
                with self._lock:
                    self.buffer.append(cf)
                last_capture = now
            except Exception:
                pass  # 单帧失败不中断循环

    def _encode(self, frame_bgr: np.ndarray, timestamp: float) -> CapturedFrame:
        """BGR 帧 → CapturedFrame（缩放 + JPEG + base64）。"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        h, w = frame_rgb.shape[:2]
        if w > self.max_width:
            ratio = self.max_width / w
            new_w = self.max_width
            new_h = int(h * ratio)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

        pil_img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=self.jpeg_quality)
        image_bytes = buf.getvalue()

        return CapturedFrame(
            base64=base64.b64encode(image_bytes).decode("ascii"),
            width=frame_rgb.shape[1],
            height=frame_rgb.shape[0],
            timestamp=timestamp,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python backend/test_frame_grabber.py`
Expected: 6/6 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/frame_grabber.py backend/test_frame_grabber.py
git commit -m "feat: add FrameGrabber — timer-based capture with speech/silence modes

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Trim VAD (remove SpeechSegmenter)

**Files:**
- Modify: `backend/vad.py`
- Modify: `backend/test_vad.py`

- [ ] **Step 1: Update VAD test — remove SpeechSegmenter tests**

Replace `backend/test_vad.py` with:
```python
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
    # Should not crash; short silence returns False
    assert result is False
    print("[PASS] Test 4/4: Short chunk handling")


if __name__ == "__main__":
    test_vad_detector_loading()
    test_vad_silence_detection()
    test_vad_confidence_range()
    test_vad_short_chunk()
    print("\nAll 4 tests passed!")
```

- [ ] **Step 2: Verify pre-refactor state**

Run: `python backend/test_vad.py`
Expected: 4/4 PASS (tests only exercise VADDetector — passes on both old and new code)

Run: `python -c "from backend.vad import SpeechSegmenter; print('exists')"`
Expected: `exists` (confirm SpeechSegmenter still present before removing)

- [ ] **Step 3: Remove SpeechSegmenter from vad.py**

Replace `backend/vad.py` with:
```python
"""语音活动检测 (Voice Activity Detection)，基于 Silero VAD。

检测音频流中是否包含语音，用于 PTT 模式下辅助判断。
"""
import numpy as np
import torch
from backend.config import settings


class VADDetector:
    """封装 Silero VAD 模型，提供语音检测。"""

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold or settings.vad_threshold
        self._model = None
        self._sample_rate = 16000

    def _load_model(self):
        """延迟加载 VAD 模型（首次调用时自动下载）。"""
        if self._model is None:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()
        return self._model

    def _to_tensor(self, audio: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor (required by silero-vad v6+)."""
        if isinstance(audio, torch.Tensor):
            return audio
        return torch.from_numpy(audio)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """判断单个音频块是否包含语音。

        Silero VAD 要求 512 样本的固定窗口，内部自动分段取均值。

        Args:
            audio_chunk: float32 数组，形状 (n_samples,)，16kHz 采样率。

        Returns:
            True 表示检测到语音，False 表示静音。
        """
        model = self._load_model()
        window = 512
        samples = len(audio_chunk)

        if samples < window:
            padded = np.zeros(window, dtype=np.float32)
            padded[:samples] = audio_chunk
            tensor = self._to_tensor(padded)
            confidence = model(tensor, self._sample_rate).item()
        else:
            confidences = []
            for i in range(0, samples - window + 1, window):
                segment = audio_chunk[i:i + window]
                tensor = self._to_tensor(segment)
                confidences.append(model(tensor, self._sample_rate).item())
            confidence = sum(confidences) / len(confidences)

        return confidence > self.threshold

    def get_speech_confidence(self, audio_chunk: np.ndarray) -> float:
        """获取语音置信度 (0-1)，越大越可能是语音。"""
        model = self._load_model()
        window = 512
        samples = len(audio_chunk)

        if samples < window:
            padded = np.zeros(window, dtype=np.float32)
            padded[:samples] = audio_chunk
            tensor = self._to_tensor(padded)
            return model(tensor, self._sample_rate).item()
        else:
            confidences = []
            for i in range(0, samples - window + 1, window):
                segment = audio_chunk[i:i + window]
                tensor = self._to_tensor(segment)
                confidences.append(model(tensor, self._sample_rate).item())
            return sum(confidences) / len(confidences)
```

- [ ] **Step 4: Run tests to verify**

Run: `python backend/test_vad.py`
Expected: 4/4 PASS

Run: `python -c "from backend.vad import SpeechSegmenter"`
Expected: `ImportError: cannot import name 'SpeechSegmenter'` — confirms SpeechSegmenter is gone

- [ ] **Step 5: Commit**

```bash
git add backend/vad.py backend/test_vad.py
git commit -m "refactor: remove SpeechSegmenter from VAD — PTT mode doesn't need silence segmentation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Create LLM Module (single-file Qwen)

**Files:**
- Create: `backend/llm.py`
- Test: `backend/test_llm.py` (new)

- [ ] **Step 1: Write failing test for LLM module**

Create `backend/test_llm.py`:
```python
# -*- coding: utf-8 -*-
"""LLM module tests — message assembly and image count control."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.llm import QwenVisionLLM


def test_constructor_defaults():
    """QwenVisionLLM initializes with default model name."""
    llm = QwenVisionLLM()
    assert llm.model_name == "qwen-max"
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
    # Should have 1 user message
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    # Content is a list: [text, image, image]
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "你好"}
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
    # system(0) + user(history) + assistant(history) + user(current)
    assert len(msgs) == 4
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "这是什么？"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "这是一个杯子。"
    assert msgs[2]["role"] == "user"
    # Current message content is multimodal list
    assert isinstance(msgs[2]["content"], list)
    print("[PASS] Test 5/6: Build messages with history")


def test_frame_truncation():
    """More than 5 frames should be truncated to 5."""
    llm = QwenVisionLLM()
    frames = [f"img{i}" for i in range(10)]  # 10 frames
    msgs = llm._build_messages(
        user_text="测试",
        frames=frames,
        history=[],
        system_prompt="",
    )
    content = msgs[0]["content"]
    # 1 text + max 5 images = 6 content parts
    image_parts = [p for p in content if p["type"] == "image_url"]
    assert len(image_parts) <= 5, f"Expected <= 5 images, got {len(image_parts)}"
    print("[PASS] Test 6/6: Frame truncation at 5 images")


if __name__ == "__main__":
    test_constructor_defaults()
    test_constructor_custom_model()
    test_build_messages_no_system_prompt()
    test_build_messages_with_system_prompt()
    test_build_messages_with_history()
    test_frame_truncation()
    print("\nAll 6 tests passed!")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python backend/test_llm.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.llm'`

- [ ] **Step 3: Implement QwenVisionLLM**

Create `backend/llm.py`:
```python
"""通义千问视觉语言模型调用。

通过 DashScope OpenAI 兼容接口，单模型无抽象层。
"""
from openai import AsyncOpenAI

from backend.config import settings


class QwenVisionLLM:
    """通义千问多模态模型适配器。

    通过 OpenAI 兼容接口调用 DashScope。
    支持模型: qwen-max, qwen-plus
    """

    def __init__(self, model: str | None = None):
        self._model = model or settings.default_model
        self._client: AsyncOpenAI | None = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> AsyncOpenAI:
        """Lazy init OpenAI client. Only called when making API requests."""
        if self._client is None:
            api_key = settings.dashscope_api_key
            if not api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY not set. "
                    "Configure it in .env or set the environment variable."
                )
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        return self._client

    async def chat(
        self,
        user_text: str,
        frames: list[str],
        history: list[dict] | None = None,
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> tuple[str, int, int]:
        """调用通义千问 API。

        Args:
            user_text: 用户当前说的话。
            frames: base64 编码的图片列表（当前帧 + 历史帧，内部截断至 5 张）。
            history: 历史对话 [{"role":"user","content":"..."}, ...]。
            system_prompt: 系统提示词。
            max_tokens: 最大输出 token 数。

        Returns:
            (回复文本, input_tokens, output_tokens)
        """
        history = history or []
        messages = self._build_messages(user_text, frames, history, system_prompt)
        client = self._get_client()

        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return (
            choice.message.content or "",
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )

    def _build_messages(
        self,
        user_text: str,
        frames: list[str],
        history: list[dict],
        system_prompt: str,
    ) -> list[dict]:
        """组装 OpenAI 兼容的多模态消息列表。

        顺序: system_prompt → history → current (text + images)
        """
        messages: list[dict] = []

        # 1. System prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 2. Dialog history (text-only turns)
        for turn in history:
            messages.append(turn)

        # 3. Current user message: text + images
        content_parts: list[dict] = []

        # Text
        text = user_text
        if frames:
            text += "\n以上画面按时间顺序排列，第一张是最新的。"
        content_parts.append({"type": "text", "text": text})

        # Images (max 5)
        max_frames = settings.max_frames_per_llm_call
        for b64 in frames[:max_frames]:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        messages.append({"role": "user", "content": content_parts})

        return messages
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python backend/test_llm.py`
Expected: 6/6 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/llm.py backend/test_llm.py
git commit -m "feat: add QwenVisionLLM — single-file Qwen adapter, no abstract base

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Create Dialog Manager

**Files:**
- Create: `backend/dialog.py`
- Test: `backend/test_dialog.py` (new)

- [ ] **Step 1: Write failing test for DialogManager**

Create `backend/test_dialog.py`:
```python
# -*- coding: utf-8 -*-
"""DialogManager tests — PTT flow and history management."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import AsyncMock, MagicMock, patch
from backend.dialog import DialogManager, Turn


def test_turn_dataclass():
    """Turn stores one conversation round."""
    t = Turn(
        user_text="你好",
        assistant_text="你好！有什么可以帮你？",
        frames=[],
        timestamp=1234567890.0,
    )
    assert t.user_text == "你好"
    assert t.assistant_text == "你好！有什么可以帮你？"
    assert t.frames == []
    assert t.timestamp == 1234567890.0
    print("[PASS] Test 1/8: Turn dataclass")


def test_constructor():
    """DialogManager initializes with all sub-modules."""
    dm = DialogManager()
    assert dm._history == []
    assert dm._system_prompt == ""
    assert dm._recording is False
    print("[PASS] Test 2/8: Constructor")


def test_set_system_prompt():
    """set_system_prompt stores the prompt."""
    dm = DialogManager()
    dm.set_system_prompt("你是一个有用的助手")
    assert dm._system_prompt == "你是一个有用的助手"
    print("[PASS] Test 3/8: Set system prompt")


def test_get_history_empty():
    """get_history returns empty list initially."""
    dm = DialogManager()
    history = dm.get_history()
    assert history == []
    print("[PASS] Test 4/8: Empty history")


def test_get_history_after_turns():
    """get_history returns formatted history after adding turns."""
    dm = DialogManager()
    dm._history = [
        Turn(user_text="Q1", assistant_text="A1", frames=[], timestamp=1.0),
        Turn(user_text="Q2", assistant_text="A2", frames=[], timestamp=2.0),
    ]
    formatted = dm.get_history()
    assert len(formatted) == 2
    assert formatted[0] == {"role": "user", "content": "Q1"}
    assert formatted[1] == {"role": "assistant", "content": "A1"}
    print("[PASS] Test 5/8: History formatting")


def test_history_truncation():
    """get_history respects max_history_turns."""
    dm = DialogManager()
    # Create 15 turns, config says max 10
    for i in range(15):
        dm._history.append(Turn(
            user_text=f"Q{i}",
            assistant_text=f"A{i}",
            frames=[],
            timestamp=float(i),
        ))
    formatted = dm.get_history()
    # 10 turns = 20 messages (user + assistant)
    assert len(formatted) == 20
    # Should contain the LATEST turns (Q5-A5 through Q14-A14)
    assert formatted[0]["content"] == "Q5"
    assert formatted[-1]["content"] == "A14"
    print("[PASS] Test 6/8: History truncation")


def test_release_empty_stt_skips_llm():
    """_process_turn with empty STT result should skip LLM call."""
    import numpy as np
    dm = DialogManager()

    # Mock STT to return empty
    dm._stt = MagicMock()
    dm._stt.transcribe.return_value = ""

    # Mock LLM to track calls
    dm._llm = MagicMock()

    audio = np.zeros(8000, dtype=np.float32)
    import asyncio
    result = asyncio.run(dm._process_turn(audio))
    assert result == ""  # empty text → no LLM call → empty response
    assert dm._llm.chat.call_count == 0
    print("[PASS] Test 7/8: Empty STT skips LLM")


def test_process_turn_stores_history():
    """Successful turn stores history."""
    import numpy as np
    dm = DialogManager()

    dm._stt = MagicMock()
    dm._stt.transcribe.return_value = "这是什么？"

    dm._llm = AsyncMock()
    dm._llm.chat.return_value = ("这是一个苹果。", 100, 20)

    dm._frame_grabber = MagicMock()
    dm._frame_grabber.get_current.return_value = None
    dm._frame_grabber.get_recent.return_value = []

    audio = np.zeros(8000, dtype=np.float32)
    import asyncio
    result = asyncio.run(dm._process_turn(audio))

    assert result == "这是一个苹果。"
    assert len(dm._history) == 1
    assert dm._history[0].user_text == "这是什么？"
    assert dm._history[0].assistant_text == "这是一个苹果。"
    print("[PASS] Test 8/8: Successful turn stores history")


if __name__ == "__main__":
    test_turn_dataclass()
    test_constructor()
    test_set_system_prompt()
    test_get_history_empty()
    test_get_history_after_turns()
    test_history_truncation()
    test_release_empty_stt_skips_llm()
    test_process_turn_stores_history()
    print("\nAll 8 tests passed!")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python backend/test_dialog.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.dialog'`

- [ ] **Step 3: Implement DialogManager**

Create `backend/dialog.py`:
```python
"""对话管理器。

编排 PTT 交互流程：录音 → STT → LLM → 回复。
维护多轮对话历史和画面引用。
"""
import time
import threading
from dataclasses import dataclass
from collections import deque

import numpy as np

from backend.config import settings
from backend.vad import VADDetector
from backend.stt import STTEngine
from backend.llm import QwenVisionLLM
from backend.frame_grabber import FrameGrabber, CapturedFrame


@dataclass
class Turn:
    """一轮对话记录。"""
    user_text: str
    assistant_text: str
    frames: list[CapturedFrame]
    timestamp: float


class DialogManager:
    """PTT 对话管理器。

    持有 VAD、STT、LLM、FrameGrabber 四个模块，
    通过 press()/release() 驱动完整交互流程。

    Usage:
        dm = DialogManager()
        dm.set_system_prompt("你是一个有用的助手")
        dm.start()                    # 启动抓帧器
        dm.press()                    # 用户按下 PTT
        ... user speaking ...
        dm.release()                  # 用户松开 → 转写 → LLM → 输出
        dm.stop()
    """

    def __init__(
        self,
        system_prompt: str = "",
        vad_threshold: float | None = None,
    ):
        self._system_prompt = system_prompt
        self._history: list[Turn] = []

        # Sub-modules (lazy init)
        self._vad: VADDetector | None = None
        self._stt: STTEngine | None = None
        self._llm: QwenVisionLLM | None = None
        self._frame_grabber: FrameGrabber | None = None

        # PTT state
        self._recording = False
        self._audio_chunks: list[np.ndarray] = []
        self._audio_stream = None
        self._ptt_thread: threading.Thread | None = None
        self._vad_threshold = vad_threshold or settings.vad_threshold

        # Audio config
        self._sample_rate = 16000
        self._chunk_size = 1024  # samples per read

    # ── Lifecycle ──

    def start(self) -> None:
        """启动后台模块（抓帧器）。"""
        if self._frame_grabber is None:
            self._frame_grabber = FrameGrabber()
        self._frame_grabber.start()

    def stop(self) -> None:
        """停止所有后台模块。"""
        if self._frame_grabber is not None:
            self._frame_grabber.stop()

    # ── System prompt ──

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词。"""
        self._system_prompt = prompt

    # ── History ──

    def get_history(self) -> list[dict]:
        """获取格式化的对话历史（最近 max_history_turns 轮）。

        Returns:
            OpenAI 兼容的消息列表 [{"role":"user","content":"..."}, ...]。
        """
        max_turns = settings.max_history_turns
        recent = self._history[-max_turns:] if len(self._history) > max_turns else self._history

        formatted: list[dict] = []
        for turn in recent:
            formatted.append({"role": "user", "content": turn.user_text})
            formatted.append({"role": "assistant", "content": turn.assistant_text})
        return formatted

    def clear_history(self) -> None:
        """清空对话历史。"""
        self._history.clear()

    # ── PTT ──

    def press(self) -> None:
        """PTT 按下：切换到语音模式，开始录音。"""
        if self._recording:
            return
        self._recording = True
        self._audio_chunks = []

        # 切换抓帧模式
        if self._frame_grabber is not None:
            self._frame_grabber.set_mode("speech")

        # 在后台线程中录音（模拟异步）
        self._ptt_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._ptt_thread.start()

    def release(self) -> str:
        """PTT 松开：停止录音，处理完整流程。

        Returns:
            AI 回复文本。空字符串表示无回复（误触或处理失败）。
        """
        if not self._recording:
            return ""
        self._recording = False

        # 等待录音线程结束
        if self._ptt_thread is not None:
            self._ptt_thread.join(timeout=3.0)
            self._ptt_thread = None

        # 切换回静音模式
        if self._frame_grabber is not None:
            self._frame_grabber.set_mode("silence")

        # 拼接音频
        if not self._audio_chunks:
            return ""
        audio = np.concatenate(self._audio_chunks)
        self._audio_chunks = []

        # 同步执行处理
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._process_turn(audio))

    # ── Internal ──

    def _record_loop(self) -> None:
        """后台录音循环。"""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self._sample_rate,
                input=True,
                frames_per_buffer=self._chunk_size,
            )
            while self._recording:
                try:
                    data = stream.read(self._chunk_size, exception_on_overflow=False)
                    chunk = np.frombuffer(data, dtype=np.float32)
                    self._audio_chunks.append(chunk)
                except Exception:
                    break
            stream.stop_stream()
            stream.close()
            p.terminate()
        except ImportError:
            # pyaudio not available — generate silence for testing
            import time as _time
            while self._recording:
                self._audio_chunks.append(np.zeros(self._chunk_size, dtype=np.float32))
                _time.sleep(self._chunk_size / self._sample_rate)

    async def _process_turn(self, audio: np.ndarray) -> str:
        """处理完整一轮对话：STT → LLM → 存储历史。

        Args:
            audio: 完整录音数据 (float32, 16kHz)。

        Returns:
            AI 回复文本，空字符串表示跳过。
        """
        # 1. STT
        if self._stt is None:
            self._stt = STTEngine()
        text = self._stt.transcribe(audio, self._sample_rate)
        if not text.strip():
            return ""

        # 2. 收集帧
        frames: list[str] = []
        if self._frame_grabber is not None:
            # 当前帧
            current = self._frame_grabber.get_current()
            if current is not None:
                frames.append(current.base64)
            # 最近历史帧
            recent = self._frame_grabber.get_recent(settings.max_frames_per_llm_call - len(frames))
            for f in recent:
                if f.base64 not in frames:
                    frames.append(f.base64)

        # 本轮帧引用（用于历史记录）
        captured: list[CapturedFrame] = []
        if self._frame_grabber is not None:
            cf = self._frame_grabber.get_current()
            if cf is not None:
                captured.append(cf)

        # 3. LLM
        if self._llm is None:
            self._llm = QwenVisionLLM()
        history = self.get_history()
        reply, _, _ = await self._llm.chat(
            user_text=text,
            frames=frames,
            history=history,
            system_prompt=self._system_prompt,
        )

        # 4. 存储历史
        turn = Turn(
            user_text=text,
            assistant_text=reply,
            frames=captured,
            timestamp=time.time(),
        )
        self._history.append(turn)

        return reply
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python backend/test_dialog.py`
Expected: 8/8 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/dialog.py backend/test_dialog.py
git commit -m "feat: add DialogManager — PTT orchestration with multi-turn history

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Cleanup — Delete Old Files

**Files:**
- Delete: `backend/models/` (entire directory: `__init__.py`, `base.py`, `qwen.py`, `test_adapters.py`)
- Delete: `backend/frame_extractor.py`
- Delete: `backend/test_frame_extractor.py`

- [ ] **Step 1: Remove old files**

```bash
git rm -r backend/models/
git rm backend/frame_extractor.py
git rm backend/test_frame_extractor.py
```

- [ ] **Step 2: Verify no remaining references**

Run: `python -c "import backend.models" 2>&1`
Expected: `ModuleNotFoundError: No module named 'backend.models'`

Run: `cd backend && python -c "from frame_grabber import FrameGrabber; print('frame_grabber OK')"`
Expected: `frame_grabber OK`

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: remove old models/ directory and frame_extractor.py

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Update Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Replace `requirements.txt` with:
```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
websockets>=12.0
python-multipart>=0.0.9
pydantic-settings>=2.1.0
openai>=1.12.0
numpy>=1.26.0
opencv-python>=4.9.0
silero-vad>=5.1
faster-whisper>=1.0.0
pillow>=10.2.0
pyaudio>=0.2.14
aiofiles>=23.2.0
```

Changes:
- Removed: `scikit-image` (SSIM no longer needed), `edge-tts` (TTS deferred)
- Added: `pyaudio` (PTT microphone capture)

- [ ] **Step 2: Verify all core imports work**

```bash
cd backend && python -c "
from config import settings
from vad import VADDetector
from stt import STTEngine
from llm import QwenVisionLLM
from frame_grabber import FrameGrabber, CapturedFrame
from dialog import DialogManager, Turn
print('All imports OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: update dependencies — remove scikit-image/edge-tts, add pyaudio

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Run Full Test Suite

- [ ] **Step 1: Run all backend tests**

```bash
python backend/test_config.py
python backend/test_vad.py
python backend/test_stt.py
python backend/test_frame_grabber.py
python backend/test_llm.py
python backend/test_dialog.py
```

Expected: all tests pass.

- [ ] **Step 2: Verify final file structure**

Expected structure after refactor:
```
backend/
  __init__.py
  config.py
  test_config.py
  vad.py
  test_vad.py
  stt.py
  test_stt.py
  frame_grabber.py
  test_frame_grabber.py
  llm.py
  test_llm.py
  dialog.py
  test_dialog.py
```

- [ ] **Step 3: Final commit (if any cleanup needed)**

```bash
git status
# If clean, done. If not, commit remaining changes.
```

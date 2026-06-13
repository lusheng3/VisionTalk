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

"""关键帧提取模块。

从摄像头视频流中截取关键帧，提供去重、分辨率压缩、场景变化检测。
"""
import base64
import io
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from backend.config import settings


@dataclass
class CapturedFrame:
    """单张截帧，包含图像数据和元信息。"""
    image_bytes: bytes       # JPEG 编码的字节
    base64: str              # base64 字符串，可直接发给 LLM
    width: int
    height: int
    timestamp: float         # 捕获时间 (time.time())


class FrameExtractor:
    """从视频帧流中提取关键帧，处理去重和压缩。"""

    def __init__(
        self,
        max_width: int | None = None,
        jpeg_quality: int = 80,
        ssim_threshold: float | None = None,
    ):
        self.max_width = max_width or settings.frame_max_width
        self.jpeg_quality = jpeg_quality
        self.ssim_threshold = ssim_threshold or settings.ssim_threshold
        self._last_grayscale: np.ndarray | None = None

    def extract(self, frame_bgr: np.ndarray) -> CapturedFrame | None:
        """从 BGR 帧提取关键帧。

        如果帧与上一帧几乎相同（SSIM > 阈值），返回 None 跳过此帧。

        Args:
            frame_bgr: OpenCV BGR 格式的帧 (H, W, 3)，uint8。

        Returns:
            CapturedFrame 如果是新画面，否则 None。
        """
        # 转 RGB 用于后续处理
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 计算与上一帧的相似度
        current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._last_grayscale is not None:
            h1, w1 = current_gray.shape
            h2, w2 = self._last_grayscale.shape
            if h1 != h2 or w1 != w2:
                resized = cv2.resize(current_gray, (w2, h2))
            else:
                resized = current_gray
            similarity = ssim(self._last_grayscale, resized, data_range=255)
            if similarity > self.ssim_threshold:
                return None  # 画面几乎相同，跳过

        self._last_grayscale = current_gray

        # 缩放分辨率
        h, w = frame_rgb.shape[:2]
        if w > self.max_width:
            ratio = self.max_width / w
            new_w = self.max_width
            new_h = int(h * ratio)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            h, w = new_h, new_w

        # JPEG 编码
        pil_img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=self.jpeg_quality)
        image_bytes = buf.getvalue()

        return CapturedFrame(
            image_bytes=image_bytes,
            base64=base64.b64encode(image_bytes).decode("ascii"),
            width=w,
            height=h,
            timestamp=time.time(),
        )

    def is_scene_change(self, frame_bgr: np.ndarray) -> bool:
        """检测画面是否发生剧烈变化（场景切换）。

        与上一次截帧的画面比较，SSIM 低于 scene_change_threshold 视为剧变。
        """
        if self._last_grayscale is None:
            return True
        current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h1, w1 = current_gray.shape
        h2, w2 = self._last_grayscale.shape
        if h1 != h2 or w1 != w2:
            resized = cv2.resize(current_gray, (w2, h2))
        else:
            resized = current_gray
        similarity = ssim(self._last_grayscale, resized, data_range=255)
        return similarity < settings.scene_change_threshold

    def reset(self):
        """重置状态，清除上一帧记录。"""
        self._last_grayscale = None

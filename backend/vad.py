"""语音活动检测 (Voice Activity Detection)，基于 Silero VAD。

检测音频流中的语音段起止位置，过滤静音和噪声。
"""
import numpy as np
import torch
from backend.config import settings


class VADDetector:
    """封装 Silero VAD 模型，提供语音段检测。"""

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
            # 不够一个窗口，补零
            padded = np.zeros(window, dtype=np.float32)
            padded[:samples] = audio_chunk
            tensor = self._to_tensor(padded)
            confidence = model(tensor, self._sample_rate).item()
        else:
            # 滑动窗口取均值
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



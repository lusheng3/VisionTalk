"""语音转文字 (Speech-to-Text)，基于 Faster-Whisper 本地推理。

零 API 成本，中文识别质量好。
"""
import numpy as np


class STTEngine:
    """封装 Faster-Whisper 模型，提供语音转文字。

    首次使用时会自动下载模型到本地缓存 (~70MB for tiny)。
    """

    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        """延迟加载 Whisper 模型。"""
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )
        return self._model

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is float32 and normalized to [-1, 1].

        Handles both native float32 (from browser AudioContext) and
        int16 (from WAV files) by detecting the value range.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # If values exceed [-1, 1], assume int16 range and normalize
        abs_max = np.max(np.abs(audio))
        if abs_max > 1.0:
            audio = audio / 32768.0

        return audio

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """将音频数据转写为文字。

        Args:
            audio: 单声道音频数据，支持 float32 [-1,1] 或 int16。
                  内部自动归一化到 [-1, 1]。
            sample_rate: 采样率，默认 16000。

        Returns:
            转写后的文字字符串。转写失败返回空字符串。
        """
        if len(audio) < sample_rate * 0.3:  # 少于 0.3 秒
            return ""

        try:
            model = self._load_model()
            audio = self._normalize_audio(audio)

            segments, _ = model.transcribe(
                audio,
                language="zh",
                beam_size=5,
                vad_filter=True,
            )
            text_parts = [seg.text.strip() for seg in segments]
            return "".join(text_parts)
        except Exception as e:
            print(f"[STT] Transcribe failed: {e}")
            return ""

"""语音转文字 (Speech-to-Text)，基于 DashScope Paraformer-v2 云端 API。

中文识别率业界顶尖，延迟 ~0.5-2s。与 LLM 共用同一个 DASHSCOPE_API_KEY。
"""
import io
import wave
import time
import logging

import httpx
import numpy as np

from backend.config import settings

log = logging.getLogger("VisionTalk")


class STTEngine:
    """DashScope Paraformer-v2 云端语音识别。"""

    def __init__(self):
        self._api_key = settings.dashscope_api_key
        self._url = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """将音频数据转写为文字。

        Args:
            audio: float32 单声道音频，范围 [-1, 1]。
            sample_rate: 采样率，默认 16000。

        Returns:
            转写后的文字字符串。失败返回空字符串。
        """
        if len(audio) < sample_rate * 0.3:
            return ""

        try:
            # float32 → int16 WAV bytes
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm.tobytes())
            wav_bytes = buf.getvalue()

            # Call DashScope Paraformer v2
            t0 = time.time()
            response = httpx.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={
                    "model": "paraformer-v2",
                    "format": "wav",
                    "sample_rate": str(sample_rate),
                },
                timeout=30,
            )
            elapsed = time.time() - t0

            if response.status_code != 200:
                log.error(f"[STT] API {response.status_code}: {response.text[:200]}")
                return ""

            result = response.json()
            output = result.get("output", {})
            if output.get("task_status") != "SUCCEEDED":
                log.error(f"[STT] Task failed: {output}")
                return ""

            results = output.get("results", [])
            text = results[0].get("text", "") if results else ""
            log.info(f"[STT] ☁️ Paraformer: 「{text}」 耗时 {elapsed:.1f}s")
            return text

        except Exception as e:
            log.error(f"[STT] Transcribe failed: {e}")
            return ""

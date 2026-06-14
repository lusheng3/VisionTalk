"""语音转文字 (Speech-to-Text)，基于 DashScope Paraformer-v2 云端 API。

中文识别率业界顶尖，延迟 ~0.5-2s。与 LLM 共用同一个 DASHSCOPE_API_KEY。
"""
import io
import wave
import time
import logging
import os

import dashscope
import numpy as np

from backend.config import settings

log = logging.getLogger("VisionTalk")


class STTEngine:
    """DashScope Paraformer-v2 云端语音识别。"""

    def __init__(self):
        dashscope.api_key = settings.dashscope_api_key

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """将音频数据转写为文字。"""
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
            buf.seek(0)

            t0 = time.time()
            response = dashscope.audio.asr.Transcription.call(
                model="paraformer-v2",
                format="wav",
                sample_rate=sample_rate,
                file=buf,
            )
            elapsed = time.time() - t0

            if response.status_code != 200:
                log.error(f"[STT] API {response.status_code}: {response.message}")
                return ""

            results = response.output.get("results", [])
            text = results[0].get("text", "") if results else ""
            log.info(f"[STT] ☁️ Paraformer: 「{text}」 耗时 {elapsed:.1f}s")
            return text

        except Exception as e:
            log.error(f"[STT] Transcribe failed: {e}")
            return ""

"""语音转文字 (Speech-to-Text)，基于 DashScope Paraformer-v2 云端 API。

中文识别率业界顶尖，延迟 ~1-3s（异步模式）。与 LLM 共用同一个 DASHSCOPE_API_KEY。
"""
import io
import wave
import base64
import time
import logging

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
            # float32 → int16 WAV bytes → base64 data URL
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm.tobytes())
            wav_b64 = base64.b64encode(buf.getvalue()).decode()
            data_url = f"data:audio/wav;base64,{wav_b64}"

            t0 = time.time()

            # Submit async task
            task = dashscope.audio.asr.Transcription.async_call(
                model="paraformer-v2",
                file_urls=[data_url],
            )
            if task.status_code != 200:
                log.error(f"[STT] Submit failed: {task.message}")
                return ""

            tid = task.output.get("task_id", "")
            if not tid:
                log.error(f"[STT] No task_id")
                return ""

            # Poll for result
            for _ in range(60):
                time.sleep(0.5)
                result = dashscope.audio.asr.Transcription.fetch(task=tid)
                status = result.output.get("task_status", "")
                if status == "SUCCEEDED":
                    elapsed = time.time() - t0
                    # Debug: dump full output structure
                    log.info(f"[STT] Raw output: {result.output}")
                    # Parse: output.results[0] -> {file_url, results: [{text}], subtask_status}
                    outer_results = result.output.get("results", [])
                    if outer_results:
                        inner = outer_results[0].get("results", [])
                        if inner:
                            text = inner[0].get("text", "")
                            log.info(f"[STT] ☁️ Paraformer: 「{text}」 耗时 {elapsed:.1f}s")
                            return text
                    log.warning(f"[STT] No text in results")
                    return ""
                elif status == "FAILED":
                    log.error(f"[STT] Task FAILED: {result.output.get('message')}")
                    return ""

            log.error(f"[STT] Timeout polling {tid}")
            return ""

        except Exception as e:
            log.error(f"[STT] Transcribe failed: {e}")
            return ""

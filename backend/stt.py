"""语音转文字 (Speech-to-Text)，基于 DashScope Paraformer-v2 云端 API。"""
import io, wave, base64, time, logging

import dashscope
import httpx
import numpy as np

from backend.config import settings

log = logging.getLogger("VisionTalk")


class STTEngine:
    """DashScope Paraformer-v2 云端语音识别。"""

    def __init__(self):
        dashscope.api_key = settings.dashscope_api_key

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if len(audio) < sample_rate * 0.3:
            return ""

        try:
            # Encode WAV → data URL
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
                wf.writeframes(pcm.tobytes())
            wav_b64 = base64.b64encode(buf.getvalue()).decode()

            t0 = time.time()

            # Submit async transcription
            task = dashscope.audio.asr.Transcription.async_call(
                model="paraformer-v2",
                file_urls=[f"data:audio/wav;base64,{wav_b64}"],
                language_hints=["zh"],
            )
            if task.status_code != 200:
                log.error(f"[STT] submit error: {task.message}")
                return ""
            tid = task.output.get("task_id", "")
            if not tid:
                return ""

            # Poll until SUCCEEDED, then grab transcription_url
            transcript_url = None
            for _ in range(60):
                time.sleep(0.5)
                r = dashscope.audio.asr.Transcription.fetch(task=tid)
                s = r.output.get("task_status", "")
                if s == "SUCCEEDED":
                    # Walk nested dict to find transcription_url
                    def find_url(d, depth=0):
                        if depth > 6:
                            return None
                        if isinstance(d, dict):
                            if "transcription_url" in d:
                                return d["transcription_url"]
                            for v in d.values():
                                u = find_url(v, depth + 1)
                                if u:
                                    return u
                        elif isinstance(d, list):
                            for item in d:
                                u = find_url(item, depth + 1)
                                if u:
                                    return u
                        return None

                    transcript_url = find_url(r.output)
                    break
                elif s == "FAILED":
                    log.error("[STT] task FAILED")
                    return ""

            if not transcript_url:
                log.error("[STT] no transcription_url found")
                return ""

            # Download the actual result JSON
            resp = httpx.get(transcript_url, timeout=15)
            if resp.status_code != 200:
                log.error(f"[STT] download result: {resp.status_code}")
                return ""

            data = resp.json()
            text = data.get("transcripts", [{}])[0].get("text", "")
            elapsed = time.time() - t0
            log.info(f"[STT] ☁️ Paraformer: 「{text}」 耗时 {elapsed:.1f}s")
            return text

        except Exception as e:
            log.error(f"[STT] {e}")
            return ""

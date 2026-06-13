"""VisionTalk Web 服务 — FastAPI + WebSocket 视觉对话后端。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import wave
import base64

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.stt import STTEngine
from backend.llm import QwenVisionLLM

app = FastAPI(title="VisionTalk")

# Static files (frontend)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Lazy-init engines
stt_engine: STTEngine | None = None
llm_engine: QwenVisionLLM | None = None


def decode_wav_base64(b64: str) -> np.ndarray:
    """Decode base64 WAV (int16 PCM, 16kHz mono) to float32 numpy array."""
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), 'rb') as wf:
        n_frames = wf.getnframes()
        pcm = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global stt_engine
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")
            if msg_type == "ptt_start":
                print("[WS] PTT start")
                await ws.send_json({"type": "status", "text": "录音中..."})
            elif msg_type == "ptt_stop":
                print(f"[WS] PTT stop, frames: {len(data.get('frames', []))}")
                await ws.send_json({"type": "status", "text": "处理中..."})
            elif msg_type == "audio":
                print(f"[WS] audio received, {len(data.get('data', ''))} chars base64")
                if stt_engine is None:
                    stt_engine = STTEngine()
                audio = decode_wav_base64(data["data"])
                text = stt_engine.transcribe(audio, 16000)
                if text.strip():
                    await ws.send_json({"type": "transcript", "text": text.strip()})
                    # LLM call
                    global llm_engine
                    if llm_engine is None:
                        llm_engine = QwenVisionLLM()
                    frames = data.get("frames", [])
                    reply, _, _ = await llm_engine.chat(
                        user_text=text.strip(),
                        frames=frames,
                        system_prompt="你是一个有用的视觉助手。只在用户明确询问画面内容时才描述画面。回答简洁，不超过三句话。",
                    )
                    await ws.send_json({"type": "response", "text": reply})
                else:
                    await ws.send_json({"type": "error", "text": "未识别到语音，请重试"})
            else:
                await ws.send_json({"type": "error", "text": f"Unknown type: {msg_type}"})
    except WebSocketDisconnect:
        print("[WS] Client disconnected")


app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)

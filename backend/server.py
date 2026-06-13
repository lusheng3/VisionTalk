"""VisionTalk Web 服务 — FastAPI + WebSocket 视觉对话后端。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import wave
import base64
import time
import logging
import traceback
from datetime import datetime

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ── Logging: 同时输出到控制台和文件 ──
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"visiontalk-{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("VisionTalk")

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
    global stt_engine, llm_engine
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
    await ws.accept()
    log.info(f"[连接] {client} 已连接")

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")
            t_start = time.time()

            if msg_type == "ptt_start":
                log.info(f"[PTT] 🎤 用户开始录音")
                await ws.send_json({"type": "status", "text": "录音中..."})

            elif msg_type == "ptt_stop":
                n_frames = len(data.get("frames", []))
                log.info(f"[PTT] ⏹ 录音结束，前端传来 {n_frames} 帧")
                await ws.send_json({"type": "status", "text": "转写中..."})

            elif msg_type == "audio":
                audio_b64 = data.get("data", "")
                log.info(f"[1/3] 📦 收到音频数据: {len(audio_b64)//1024}KB")

                # WAV 解码
                try:
                    audio = decode_wav_base64(audio_b64)
                    duration = len(audio) / 16000
                    log.info(f"[1/3] ✅ 解码成功: {duration:.1f}秒, {len(audio)}采样点")
                except Exception as e:
                    log.error(f"[1/3] ❌ WAV解码失败: {e}")
                    await ws.send_json({"type": "error", "text": "音频解码失败"})
                    continue

                # STT 转写
                try:
                    if stt_engine is None:
                        log.info("[2/3] 🔧 首次加载 STT 模型 (faster-whisper tiny)...")
                        stt_engine = STTEngine()
                    text = stt_engine.transcribe(audio, 16000)
                    stt_cost = time.time() - t_start
                    if text.strip():
                        log.info(f"[2/3] ✅ STT 转写结果: 「{text.strip()}」 耗时 {stt_cost:.1f}s")
                    else:
                        log.warning(f"[2/3] ⚠️  STT 返回空文本 (可能是静音或语音不清晰) 耗时 {stt_cost:.1f}s")
                except Exception as e:
                    log.error(f"[2/3] ❌ STT 异常: {e}\n{traceback.format_exc()}")
                    await ws.send_json({"type": "error", "text": "语音识别失败"})
                    continue

                if not text.strip():
                    await ws.send_json({"type": "error", "text": "未识别到语音，请重试"})
                    log.info(f"[本轮结束] 无文本，跳过 LLM")
                    continue

                await ws.send_json({"type": "transcript", "text": text.strip()})

                # LLM 调用
                try:
                    if llm_engine is None:
                        log.info("[3/3] 🔧 首次加载 LLM...")
                        llm_engine = QwenVisionLLM()
                    frames = data.get("frames", [])
                    log.info(f"[3/3] 🤖 调用 LLM | 输入文本=「{text.strip()}」 | 帧数={len(frames)}")
                    reply, in_tok, out_tok = await llm_engine.chat(
                        user_text=text.strip(),
                        frames=frames,
                        system_prompt="你是一个有用的视觉助手。只在用户明确询问画面内容时才描述画面。回答简洁，不超过三句话。",
                    )
                    llm_cost = time.time() - t_start
                    log.info(f"[3/3] ✅ LLM 回复: 「{reply}」")
                    log.info(f"[3/3]    Token: {in_tok}入 / {out_tok}出 | LLM耗时 {llm_cost:.1f}s")
                    await ws.send_json({"type": "response", "text": reply})
                except Exception as e:
                    log.error(f"[3/3] ❌ LLM 调用失败: {e}\n{traceback.format_exc()}")
                    await ws.send_json({"type": "error", "text": "AI 暂时不可用，稍后重试"})

                total = time.time() - t_start
                log.info(f"[本轮结束] 总耗时 {total:.1f}s")

            else:
                log.warning(f"[未知] 收到未知消息类型: {msg_type}")
                await ws.send_json({"type": "error", "text": f"未知消息: {msg_type}"})

    except WebSocketDisconnect:
        log.info(f"[断开] {client} 正常断开")
    except Exception as e:
        log.error(f"[异常] {client} 未捕获错误: {e}\n{traceback.format_exc()}")


app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    log.info(f"🚀 VisionTalk 启动 | 日志文件: {LOG_FILE}")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")

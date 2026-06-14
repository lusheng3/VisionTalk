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
frontend_dir = Path(__file__).parent.parent / "frontend"

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

    pending_frames: list[str] = []

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")
            t_start = time.time()

            if msg_type == "ptt_start":
                log.info(f"[PTT] 🎤 用户开始录音")

            elif msg_type == "ptt_stop":
                pending_frames = data.get("frames", [])
                log.info(f"[PTT] ⏹ 录音结束，前端传来 {len(pending_frames)} 帧")
                await ws.send_json({"type": "status", "text": "🔊 语音转文字中..."})

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
                        log.info("[2/3] ☁️ 调用云端 STT (Paraformer-v2)...")
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

                stt_time = time.time() - t_start
                await ws.send_json({"type": "transcript", "text": text.strip()})
                await ws.send_json({"type": "status", "text": "🤖 AI 思考中..."})

                # LLM 调用（流式）
                try:
                    if llm_engine is None:
                        log.info("[3/3] 🔧 首次加载 LLM...")
                        llm_engine = QwenVisionLLM()
                    frames = pending_frames
                    pending_frames = []

                    # 非视觉问题跳过图片（省 token + 加速）
                    VISUAL_KEYWORDS = [
                        "看", "这", "那", "什么", "画面", "图片", "照片", "颜色",
                        "哪个", "哪里", "里面", "手上", "手里", "桌面", "屏幕",
                        "这是", "那是", "介绍", "描述", "识别", "镜头", "摄像头",
                        "你看到", "看看", "帮我", "瞧瞧",
                    ]
                    text_lower = text.strip()
                    is_visual = any(kw in text_lower for kw in VISUAL_KEYWORDS)
                    if not is_visual and frames:
                        log.info(f"[3/3] 🎯 非视觉问题，跳过 {len(frames)} 帧")
                        frames = []
                    elif frames:
                        log.info(f"[3/3] 🤖 流式 LLM | 输入文本=「{text.strip()}」 | 帧数={len(frames)}")
                    else:
                        log.info(f"[3/3] 🤖 流式 LLM | 输入文本=「{text.strip()}」 | 无帧")

                    reply_parts = []
                    first_token_time = None
                    async for token in llm_engine.chat_stream(
                        user_text=text.strip(),
                        frames=frames,
                        system_prompt="你是一个有用的视觉助手。只在用户明确询问画面内容时才描述画面。回答简洁，不超过三句话。",
                    ):
                        if first_token_time is None:
                            first_token_time = time.time() - t_start
                            log.info(f"[3/3] ⚡ 首 token 延迟: {first_token_time:.1f}s")
                        reply_parts.append(token)
                        await ws.send_json({"type": "stream", "text": token})

                    reply = "".join(reply_parts)
                    llm_time = time.time() - t_start - stt_time
                    total = time.time() - t_start
                    log.info(f"[3/3] ✅ LLM 回复: 「{reply[:80]}{'...' if len(reply) > 80 else ''}」")
                    log.info(f"[3/3]    首token={first_token_time:.1f}s | LLM纯耗时={llm_time:.1f}s | 总耗时={total:.1f}s")
                    await ws.send_json({"type": "response", "text": reply})
                    await ws.send_json({
                        "type": "timing",
                        "stt": f"{stt_time:.1f}s",
                        "llm": f"{llm_time:.1f}s",
                        "total": f"{total:.1f}s",
                        "frames": len(frames),
                    })
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


app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    log.info(f"🚀 VisionTalk 启动 | 日志文件: {LOG_FILE}")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")

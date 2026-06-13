"""VisionTalk Web 服务 — FastAPI + WebSocket 视觉对话后端。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="VisionTalk")

# Static files (frontend)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
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
            else:
                await ws.send_json({"type": "error", "text": f"Unknown type: {msg_type}"})
    except WebSocketDisconnect:
        print("[WS] Client disconnected")


app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)

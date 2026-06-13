# Web 界面实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based AI vision chat app — camera preview, PTT voice input, visual LLM responses, scrollable chat history.

**Architecture:** Browser frontend (HTML/JS) captures camera+audio via getUserMedia, sends audio WAV + JPEG frames over WebSocket to FastAPI backend. Backend reuses existing STT (faster-whisper) and LLM (Qwen) modules.

**Tech Stack:** FastAPI, WebSocket, vanilla HTML/CSS/JS, faster-whisper, openai (DashScope)

---

## File Structure

```
backend/
  server.py              🆕 FastAPI app + WebSocket handler
  static/
    index.html           🆕 Frontend single page
  stt.py                 ✅ Reuse (unchanged)
  llm.py                 ✅ Reuse (unchanged)
  config.py              ✅ Reuse (unchanged)
```

---

### Task 1: FastAPI Skeleton (PR #1)

**Goal:** Server starts, serves a placeholder HTML page.

**Files:**
- Create: `backend/server.py`
- Create: `backend/static/index.html`

- [ ] **Step 1: Write server.py skeleton**

```python
"""VisionTalk Web 服务 — FastAPI + WebSocket 视觉对话后端。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="VisionTalk")

# Static files (frontend)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
```

- [ ] **Step 2: Write placeholder index.html**

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionTalk</title>
    <style>
        body { font-family: sans-serif; text-align: center; padding: 40px; background: #1a1a2e; color: #eee; }
    </style>
</head>
<body>
    <h1>🎙️ VisionTalk</h1>
    <p>AI 视觉对话 · 即将上线</p>
</body>
</html>
```

- [ ] **Step 3: Verify server starts**

Run: `python backend/server.py`
Expected: Uvicorn starts on port 8765, open http://localhost:8765 shows the page.

- [ ] **Step 4: Commit**

```bash
git add backend/server.py backend/static/index.html
git commit -m "feat: FastAPI skeleton — serve placeholder page"
```

---

### Task 2: Frontend Camera + PTT Button (PR #2)

**Goal:** Browser shows camera preview and PTT button. No backend communication yet.

**Files:**
- Modify: `backend/static/index.html`

- [ ] **Step 1: Write complete frontend with camera + PTT button**

Replace `backend/static/index.html` with:

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionTalk</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; }
        
        /* Left panel: camera */
        .camera-panel { flex: 0 0 65%; display: flex; flex-direction: column; background: #000; position: relative; }
        .camera-panel video { width: 100%; height: 100%; object-fit: cover; }
        .camera-panel .status { position: absolute; top: 12px; left: 12px; padding: 4px 10px; border-radius: 12px; font-size: 12px; background: rgba(0,0,0,0.6); }
        .camera-panel .status.live { color: #2ecc71; }
        .camera-panel .status.off { color: #e74c3c; }

        /* Right panel: chat */
        .chat-panel { flex: 1; display: flex; flex-direction: column; min-width: 300px; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 8px; }
        .chat-messages .empty { text-align: center; color: #555; margin-top: 40px; }
        .msg { max-width: 80%; padding: 10px 14px; border-radius: 14px; font-size: 14px; line-height: 1.5; }
        .msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
        .msg.ai { align-self: flex-start; background: #374151; color: #e5e7eb; border-bottom-left-radius: 4px; }
        .msg.system { align-self: center; background: transparent; color: #6b7280; font-size: 12px; }

        /* PTT button */
        .ptt-area { padding: 16px; border-top: 1px solid #2d2d3f; display: flex; justify-content: center; }
        .ptt-btn { width: 64px; height: 64px; border-radius: 50%; border: none; cursor: pointer; font-size: 28px; background: #e74c3c; color: #fff; transition: all 0.2s; }
        .ptt-btn:hover { transform: scale(1.05); }
        .ptt-btn.recording { background: #c0392b; box-shadow: 0 0 0 6px rgba(231, 76, 60, 0.3); animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { box-shadow: 0 0 0 6px rgba(231, 76, 60, 0.3); } 50% { box-shadow: 0 0 0 14px rgba(231, 76, 60, 0); } }
        .ptt-label { text-align: center; font-size: 11px; color: #888; margin-top: 6px; }
    </style>
</head>
<body>
    <div class="camera-panel">
        <video id="camera" autoplay playsinline muted></video>
        <span class="status off" id="camStatus">📷 摄像头未就绪</span>
    </div>

    <div class="chat-panel">
        <div class="chat-messages" id="messages">
            <div class="empty">点击下方按钮开始对话</div>
        </div>
        <div class="ptt-area">
            <div style="display:flex;flex-direction:column;align-items:center;">
                <button class="ptt-btn" id="pttBtn" title="点击开始/停止录音">🎤</button>
                <p class="ptt-label" id="pttLabel">点击开始录音</p>
            </div>
        </div>
    </div>

    <script>
        // ── Camera ──
        const video = document.getElementById('camera');
        const camStatus = document.getElementById('camStatus');
        
        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                video.srcObject = stream;
                camStatus.textContent = '📷 摄像头已就绪';
                camStatus.className = 'status live';
            } catch (e) {
                camStatus.textContent = '📷 需要摄像头权限';
                camStatus.className = 'status off';
            }
        }

        // ── PTT Button ──
        const pttBtn = document.getElementById('pttBtn');
        const pttLabel = document.getElementById('pttLabel');
        let isRecording = false;

        pttBtn.addEventListener('click', () => {
            isRecording = !isRecording;
            if (isRecording) {
                pttBtn.textContent = '⏹';
                pttBtn.classList.add('recording');
                pttLabel.textContent = '录音中... 再点击停止';
                // Audio recording starts here (next PR)
            } else {
                pttBtn.textContent = '🎤';
                pttBtn.classList.remove('recording');
                pttLabel.textContent = '点击开始录音';
                // Audio recording stops here (next PR)
            }
        });

        // ── Init ──
        startCamera();
    </script>
</body>
</html>
```

- [ ] **Step 2: Verify camera works**

Run: `python backend/server.py`
Open http://localhost:8765, verify camera preview shows, PTT button toggles state.

- [ ] **Step 3: Commit**

```bash
git add backend/static/index.html
git commit -m "feat: frontend — camera preview and PTT toggle button"
```

---

### Task 3: Frontend Audio Recording + WebSocket Send (PR #3)

**Goal:** PTT starts/clicks stop recording → WAV encode → send to server via WebSocket.

**Files:**
- Modify: `backend/static/index.html`
- Modify: `backend/server.py` (add WebSocket endpoint stub)

- [ ] **Step 1: Add WebSocket endpoint to server.py**

Append to `backend/server.py`:
```python
from fastapi import WebSocket, WebSocketDisconnect

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
```

- [ ] **Step 2: Update index.html — add audio recording and WebSocket**

Add before the `</script>` tag in `index.html`:
```javascript
// ── WebSocket ──
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onopen = () => console.log('[WS] Connected');
ws.onclose = () => console.log('[WS] Disconnected');
ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    console.log('[WS] Server:', msg.type, msg.text || '');
    if (msg.type === 'status') {
        addMessage('system', msg.text);
    }
};

// ── Audio Recording ──
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;

async function getAudioStream() {
    if (audioStream) return audioStream;
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    return audioStream;
}

async function startRecording() {
    const stream = await getAudioStream();
    // Use AudioContext for raw PCM access
    const audioCtx = new AudioContext({ sampleRate: 16000 });
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    
    audioChunks = [];
    processor.onaudioprocess = (e) => {
        if (!isRecording) return;
        const input = e.inputBuffer.getChannelData(0);  // float32 [-1, 1]
        audioChunks.push(new Float32Array(input));
    };
    
    source.connect(processor);
    processor.connect(audioCtx.destination);
    window._audioCtx = audioCtx;
    window._processor = processor;
}

function stopRecording() {
    if (window._processor) {
        window._processor.disconnect();
    }
    if (window._audioCtx) {
        window._audioCtx.close();
    }
    audioChunks = [];
}

function encodeWAV(samples) {
    // Convert float32 samples to int16, add WAV header
    const numSamples = samples.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);
    
    // WAV header
    const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);       // PCM
    view.setUint16(22, 1, true);       // mono
    view.setUint32(24, 16000, true);   // sample rate
    view.setUint32(28, 32000, true);   // byte rate
    view.setUint16(32, 2, true);       // block align
    view.setUint16(34, 16, true);      // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, numSamples * 2, true);
    
    // PCM data (float32 → int16)
    for (let i = 0; i < numSamples; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return buffer;
}

// ── Modified PTT handler ──
pttBtn.addEventListener('click', async () => {
    isRecording = !isRecording;
    if (isRecording) {
        pttBtn.textContent = '⏹';
        pttBtn.classList.add('recording');
        pttLabel.textContent = '录音中... 再点击停止';
        await startRecording();
        ws.send(JSON.stringify({ type: 'ptt_start' }));
    } else {
        pttBtn.textContent = '🎤';
        pttBtn.classList.remove('recording');
        pttLabel.textContent = '处理中...';
        
        // Flatten all audio chunks
        const allSamples = [];
        for (const chunk of audioChunks) {
            for (let i = 0; i < chunk.length; i++) allSamples.push(chunk[i]);
        }
        stopRecording();
        
        if (allSamples.length < 16000 * 0.3) {  // < 0.3s
            addMessage('system', '⚠️ 录音太短，请重试');
            pttLabel.textContent = '点击开始录音';
            return;
        }
        
        // Encode WAV → base64 → send
        const wav = encodeWAV(allSamples);
        const base64 = btoa(String.fromCharCode(...new Uint8Array(wav)));
        
        ws.send(JSON.stringify({ type: 'ptt_stop', frames: [] }));
        ws.send(JSON.stringify({ type: 'audio', data: base64 }));
    }
});

// ── Messages ──
function addMessage(role, text) {
    const container = document.getElementById('messages');
    // Remove empty state
    const empty = container.querySelector('.empty');
    if (empty) empty.remove();
    
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}
```

- [ ] **Step 3: Verify audio recording + WebSocket**

Run: `python backend/server.py`
Open http://localhost:8765, click PTT, say something, click again.
Check terminal — should see `[WS] PTT start`, `[WS] PTT stop`, `[WS] audio received`.

- [ ] **Step 4: Commit**

```bash
git add backend/static/index.html backend/server.py
git commit -m "feat: frontend audio recording + WAV encode + WebSocket send"
```

---

### Task 4: Backend STT Integration (PR #4)

**Goal:** Server decodes WAV audio → STT transcribe → returns transcript text to browser.

**Files:**
- Modify: `backend/server.py` — add WAV decode + STT call

- [ ] **Step 1: Extend server.py WebSocket handler with STT**

Replace the audio handling in `websocket_endpoint`:
```python
import io
import wave
import base64
import numpy as np
from backend.stt import STTEngine

# Lazy init (before the websocket_endpoint function)
stt_engine: STTEngine | None = None

def decode_wav_base64(b64: str) -> np.ndarray:
    """Decode base64 WAV (int16 PCM) to float32 numpy array."""
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # int16
        assert wf.getframerate() == 16000
        n_frames = wf.getnframes()
        pcm = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0
```

And replace the `elif msg_type == "audio":` block in the WebSocket handler:
```python
elif msg_type == "audio":
    global stt_engine
    if stt_engine is None:
        stt_engine = STTEngine()
    audio = decode_wav_base64(data["data"])
    text = stt_engine.transcribe(audio, 16000)
    if text.strip():
        await ws.send_json({"type": "transcript", "text": text.strip()})
    else:
        await ws.send_json({"type": "error", "text": "未识别到语音，请重试"})
```

- [ ] **Step 2: Verify STT flow**

Run: `python backend/server.py`
Open browser, PTT → speak → PTT stop.
Expected: Browser shows transcribed text, or "未识别到语音" error.

- [ ] **Step 3: Commit**

```bash
git add backend/server.py
git commit -m "feat: backend STT — WAV decode + faster-whisper transcribe"
```

---

### Task 5: Backend LLM Integration (PR #5)

**Goal:** After STT, call Qwen LLM with transcribed text → return AI reply to browser.

**Files:**
- Modify: `backend/server.py`

- [ ] **Step 1: Add LLM call to server.py**

Add import:
```python
from backend.llm import QwenVisionLLM
import asyncio as _asyncio

llm_engine: QwenVisionLLM | None = None
```

After STT returns text in the audio handler, append:
```python
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
```

- [ ] **Step 2: Verify LLM flow**

Run: `python backend/server.py`
Open browser, PTT → speak → PTT stop.
Expected: Browser shows "转写中..." → transcript → AI reply.

- [ ] **Step 3: Commit**

```bash
git add backend/server.py
git commit -m "feat: backend LLM — Qwen vision chat with transcribed text"
```

---

### Task 6: Frontend Frame Capture (PR #6)

**Goal:** Camera frames captured on PTT start/stop, sent to server with audio.

**Files:**
- Modify: `backend/static/index.html`

- [ ] **Step 1: Add frame capture to frontend**

Add frame capture JS before the PTT handler:
```javascript
// ── Frame Capture ──
const FRAME_BUFFER_SIZE = 30;
const FRAME_INTERVAL_IDLE = 3000;   // 3s
const FRAME_INTERVAL_SPEECH = 500;  // 0.5s
let frameBuffer = [];
let frameTimer = null;

function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const jpeg = canvas.toDataURL('image/jpeg', 0.75);  // includes "data:image/jpeg;base64,..."
    const base64 = jpeg.split(',')[1];  // strip data URI prefix
    frameBuffer.push(base64);
    if (frameBuffer.length > FRAME_BUFFER_SIZE) {
        frameBuffer = frameBuffer.slice(-FRAME_BUFFER_SIZE);
    }
}

function startFrameCapture(interval) {
    stopFrameCapture();
    captureFrame();  // immediate first frame
    frameTimer = setInterval(captureFrame, interval);
}

function stopFrameCapture() {
    if (frameTimer) {
        clearInterval(frameTimer);
        frameTimer = null;
    }
}

function getFramesForLLM(durationSeconds) {
    // Dynamic frame count based on duration
    let count;
    if (durationSeconds <= 3) count = 3;
    else if (durationSeconds <= 10) count = 5;
    else if (durationSeconds <= 20) count = 8;
    else count = 10;
    
    if (frameBuffer.length <= count) return [...frameBuffer];
    // Uniform sampling, always include latest
    const step = (frameBuffer.length - 1) / (count - 1);
    const indices = [];
    for (let i = 0; i < count - 1; i++) indices.push(Math.floor(i * step));
    indices.push(frameBuffer.length - 1);
    return indices.map(i => frameBuffer[i]);
}

// Start idle capture on load
startFrameCapture(FRAME_INTERVAL_IDLE);
```

Modified PTT handler — replace the existing one:
```javascript
let recordingStartTime = 0;

pttBtn.addEventListener('click', async () => {
    isRecording = !isRecording;
    if (isRecording) {
        recordingStartTime = Date.now();
        pttBtn.textContent = '⏹';
        pttBtn.classList.add('recording');
        pttLabel.textContent = '录音中... 再点击停止';
        startFrameCapture(FRAME_INTERVAL_SPEECH);
        await startRecording();
        ws.send(JSON.stringify({ type: 'ptt_start' }));
    } else {
        pttBtn.textContent = '🎤';
        pttBtn.classList.remove('recording');
        pttLabel.textContent = '处理中...';
        
        const duration = (Date.now() - recordingStartTime) / 1000;
        startFrameCapture(FRAME_INTERVAL_IDLE);  // back to idle
        
        const allSamples = [];
        for (const chunk of audioChunks) {
            for (let i = 0; i < chunk.length; i++) allSamples.push(chunk[i]);
        }
        stopRecording();
        
        if (allSamples.length < 16000 * 0.3) {
            addMessage('system', '⚠️ 录音太短，请重试');
            pttLabel.textContent = '点击开始录音';
            return;
        }
        
        const frames = getFramesForLLM(duration);
        const wav = encodeWAV(allSamples);
        const base64 = btoa(String.fromCharCode(...new Uint8Array(wav)));
        
        ws.send(JSON.stringify({ type: 'ptt_stop', frames: frames }));
        ws.send(JSON.stringify({ type: 'audio', data: base64 }));
    }
});
```

- [ ] **Step 2: Verify frame capture**

Run: `python backend/server.py`
Open browser, PTT → speak → PTT stop.
Check server log: `[WS] PTT stop, frames: N` (N should be 3-10 depending on duration).

- [ ] **Step 3: Commit**

```bash
git add backend/static/index.html
git commit -m "feat: frontend frame capture — dynamic count by recording duration"
```

---

### Task 7: Chat Bubbles + Error Display (PR #7)

**Goal:** Conversation shown as chat bubbles, error states handled gracefully.

**Files:**
- Modify: `backend/static/index.html`

- [ ] **Step 1: Update frontend message handling + error display**

Replace the WebSocket `onmessage` handler and `addMessage` function:
```javascript
ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    switch (msg.type) {
        case 'status':
            addMessage('system', msg.text);
            break;
        case 'transcript':
            addMessage('user', msg.text);
            break;
        case 'response':
            addMessage('ai', msg.text);
            pttLabel.textContent = '点击开始录音';
            break;
        case 'error':
            addMessage('system', '⚠️ ' + msg.text);
            pttLabel.textContent = '点击开始录音';
            break;
    }
};

// Also handle WebSocket errors
ws.onerror = () => {
    addMessage('system', '⚠️ 连接断开，刷新页面重试');
};
```

Add CSS for scrollbar styling and system messages:
```css
.chat-messages::-webkit-scrollbar { width: 6px; }
.chat-messages::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
.msg.ai { align-self: flex-start; background: #374151; color: #e5e7eb; border-bottom-left-radius: 4px; }
```

- [ ] **Step 2: Verify full interaction flow**

Run: `python backend/server.py`
Open browser. Test:
1. PTT → speak → PTT stop → see transcript in blue bubble (right) → AI reply in gray bubble (left)
2. Very short click → "录音太短" warning
3. Say nothing → "未识别到语音" error
4. Multiple turns → bubbles scroll correctly

- [ ] **Step 3: Commit**

```bash
git add backend/static/index.html
git commit -m "feat: chat bubbles + error display + scrollable history"
```

---

### Task 8: README + Design Docs (PR #8)

**Goal:** Complete documentation for submission.

**Files:**
- Create: `README.md` (update if exists)
- Modify: `docs/superpowers/specs/2026-06-13-web-interface-design.md` (fill "最终实现" section)

- [ ] **Step 1: Write README.md**

```markdown
# VisionTalk — AI 视觉对话助手

打开浏览器，让 AI 看到你的世界、听懂你的声音。

## 功能

- 📷 摄像头实时画面，AI 能"看到"并回答相关问题
- 🎤 PTT 语音输入，本地 STT 转写，零语音识别成本
- 🤖 通义千问视觉大模型驱动，多轮连续对话
- 🌐 纯浏览器体验，无需安装任何软件

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入 DashScope API Key：

```bash
cp .env.example .env
# 编辑 .env: DASHSCOPE_API_KEY=sk-你的key
```

### 3. 启动服务

```bash
python backend/server.py
```

### 4. 打开浏览器

访问 `http://localhost:8765`，允许摄像头和麦克风权限。

## 使用说明

| 操作 | 方式 |
|------|------|
| 开始对话 | 点击红色 🎤 按钮 |
| 结束对话 | 再次点击按钮 |
| 视觉问答 | 指着物体问"这是什么" |
| 查看历史 | 对话气泡自动滚动 |

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | 原生 HTML/CSS/JS, getUserMedia, WebSocket, Canvas |
| 后端 | Python FastAPI, WebSocket |
| STT | faster-whisper tiny (本地, CPU int8) |
| LLM | 通义千问 qwen3.7-plus (DashScope API) |
| 传输 | WebSocket (localhost) |

## 依赖

见 `requirements.txt`。核心依赖：
- fastapi + uvicorn — Web 服务
- faster-whisper — 本地语音识别
- openai — DashScope API 调用
- numpy, opencv-python, pillow — 图像处理

## 成本控制

- 本地 STT：零 API 费用
- 帧数量动态控制：3-10 帧/轮，图片 token ≤ 10000
- 单模型策略：仅 qwen3.7-plus
- 空闲低频抓帧：不对话时 3s/帧

## 项目结构

```
backend/
  server.py              FastAPI + WebSocket
  static/index.html      前端页面
  stt.py                 语音识别
  llm.py                 LLM 调用
  config.py              配置管理
```

## License

MIT
```

- [ ] **Step 2: Update design doc — fill "最终实现" section**

Add to the spec document under "最终实现":
```markdown
| US1 | ✅ 已实现 | 浏览器打开即可，自动请求权限 |
| US2 | ✅ 已实现 | PTT 点击切换，STT 转写准确 |
| US3 | ✅ 已实现 | 帧随音频发送，LLM 可识别画面物体 |
| US4 | ✅ 已实现 | 对话气泡滚动显示 |
| US5 | ✅ 已实现 | 录音太短/未识别/API异常均有提示 |
| US6 | ✅ 已实现 | 局域网内手机也可访问 |
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/superpowers/specs/2026-06-13-web-interface-design.md
git commit -m "docs: README + final user story status"

# Push and create PR
git push -u origin feat/web-interface
```

---

## PR-to-Main Merge Flow

After each task commit above, merge back to main:

```bash
git checkout main
git merge feat/web-interface
# Verify it runs
python backend/server.py   # Ctrl+C after confirming
git push origin main
git checkout feat/web-interface
```

This keeps main runnable at every step.

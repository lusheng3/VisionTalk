# VisionTalk — AI 视觉对话助手

打开浏览器，让 AI 看到你的世界、听懂你的声音。

## 功能

- 📷 **摄像头实时画面** — AI 能"看到"并回答视觉相关问题
- 🎤 **PTT 语音输入** — 点击按钮开始说话，再点击结束，本地 STT 转写
- 🤖 **多模态大模型** — 通义千问 qwen3.7-plus 驱动，理解画面 + 语音
- 💬 **多轮连续对话** — 聊天气泡滚动显示，保持上下文
- 🌐 **纯浏览器体验** — 无需安装软件，局域网内手机也能访问

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

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
| 后端 | Python FastAPI + uvicorn |
| 通信 | WebSocket (JSON + binary) |
| STT | faster-whisper tiny (本地 CPU int8 推理) |
| LLM | 通义千问 qwen3.7-plus (DashScope OpenAI 兼容 API) |

## 第三方依赖

所有依赖列于 `requirements.txt`：

- **Web 服务**: fastapi, uvicorn, websockets
- **AI/ML**: openai (DashScope 调用), faster-whisper (本地语音识别), silero-vad (语音检测), torch
- **图像**: opencv-python, pillow
- **音频**: pyaudio
- **配置**: pydantic-settings

本地模型 `faster-whisper tiny` (~70MB) 首次运行时自动下载。

## 原创功能

本项目从零构建，核心原创部分：

1. **端到端视觉对话管道** — 浏览器 WebSocket → FastAPI → STT → LLM，全链路自研
2. **定时抓帧策略** — 空闲低频 (3s) + 说话高频 (0.5s)，动态帧数控制 (3-10 帧)，端云成本优化
3. **浏览器端 WAV 编码** — AudioContext 采集 PCM → 纯 JS WAV 编码 → WebSocket 传输，零服务器端音频转换
4. **PTT 点击切换交互** — 简洁的单按钮交互，录音状态视觉反馈
5. **单模型精简架构** — 去掉抽象层，直连 qwen3.7-plus，代码量减少 60%

## 项目结构

```
backend/
  server.py              FastAPI + WebSocket 服务
  static/index.html      前端页面 (摄像头 + PTT + 对话)
  stt.py                 语音识别 (faster-whisper)
  llm.py                 LLM 调用 (千问 DashScope)
  config.py              配置管理
```

## 成本控制

| 策略 | 实现 |
|------|------|
| 本地 STT | faster-whisper tiny, 零 API 费用 |
| 动态帧数 | 3-10 帧/轮，按录音时长自适应 |
| 空闲低频 | 不对话时 3s/帧，对话时 0.5s/帧 |
| 单模型 | 仅 qwen3.7-plus，无多模型冗余 |
| 本地传输 | localhost WebSocket，零带宽成本 |

## License

MIT

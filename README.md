# AI VisionTalk — AI 视觉对话助手

基于 Web 的多模态 AI 对话应用。打开摄像头和麦克风，AI 能看见你、听懂你、回应你。

## 技术栈

| 层 | 技术 |
|---|------|
| 后端框架 | FastAPI + uvicorn (Python) |
| 语音检测 | Silero VAD (ONNX) |
| 语音识别 | Faster-Whisper (本地) |
| 图像处理 | OpenCV + scikit-image |
| 语音合成 | Edge TTS (免费) |
| AI 模型 | 通义千问 Qwen-VL / 智谱 GLM-4V |
| 前端 | 原生 HTML/JS + Tailwind CSS (CDN) |

## 第三方依赖

所有依赖列于 `requirements.txt`，均为开源或免费服务：

- **Python 包**: fastapi, uvicorn, openai, numpy, opencv-python, scikit-image, silero-vad, faster-whisper, edge-tts, pillow, pydantic-settings, aiofiles
- **CDN**: Tailwind CSS (通过 `<script>` 标签加载)
- **AI API**: 阿里云 DashScope / 智谱 Open API (需自行申请 API Key)
- **本地模型**: Silero VAD (ONNX)、Faster-Whisper (自动下载)

## 原创功能

本项目从零构建，核心原创部分包括：

1. **端云协同编排器** (`backend/orchestrator.py`) — VAD→STT→截帧→LLM→TTS 全流程串联
2. **语音触发截帧策略** (`backend/frame_extractor.py`) — 头中尾三帧 + SSIM 去重 + 场景变化检测
3. **多模态对话会话管理** (`backend/session.py`) — 上下文记忆 + token 统计 + 历史压缩
4. **双模式交互** (`frontend/app.js` + `backend/main.py`) — PTT 按键 + 持续对话无缝切换
5. **多 LLM 后端适配器** (`backend/models/`) — 统一接口，一键切换国内 AI 模型

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DashScope API Key

# 3. 启动服务
python -m backend.main

# 4. 打开浏览器
# http://localhost:8765
```

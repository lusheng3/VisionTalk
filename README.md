# VisionTalk — AI 视觉对话助手

打开浏览器，让 AI 看到你的世界、听懂你的声音。

## Demo 演示

📺 **[观看演示视频](https://www.bilibili.com/video/BV1T4JK6mEti/?spm_id_from=333.1387.upload.video_card.click)**

## 功能

- 📷 **摄像头实时画面** — AI 能"看到"并回答视觉相关问题
- 🎤 **PTT 语音输入** — 点击按钮开始说话，再点击结束，云端高精度转写
- 🧠 **智能视觉路由** — 自动判断是否需要画面，非视觉问题零帧秒回
- 🤖 **多模态大模型** — 通义千问 qwen3.7-plus 驱动，理解画面 + 语音
- 💬 **多轮连续对话** — 聊天气泡滚动显示，保持上下文
- ⚡ **流式输出** — LLM 边生成边返回，感知延迟大幅降低
- 🌐 **纯浏览器体验** — 无需安装软件，局域网内手机也能访问

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入两个 API Key：
#   DASHSCOPE_API_KEY=sk-你的key   (通义千问 + Paraformer STT)
#   DEEPSEEK_API_KEY=sk-你的key    (DeepSeek 视觉路由器)
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
| STT | DashScope Paraformer-v2 (云端异步) |
| 视觉路由 | DeepSeek-chat |
| LLM | 通义千问 qwen3.7-plus (DashScope) |

## 第三方依赖

所有依赖列于 `requirements.txt`：

- **Web 服务**: fastapi, uvicorn, websockets
- **AI/ML**: openai (DashScope + DeepSeek 调用), dashscope (Paraformer SDK)
- **图像**: opencv-python, pillow
- **音频**: pyaudio
- **配置**: pydantic-settings

## 原创功能

本项目从零构建，核心原创部分：

1. **端到端视觉对话管道** — 浏览器 WebSocket → FastAPI → 云端 STT → 视觉路由 → 流式 LLM，全链路自研
2. **智能视觉路由器** — DeepSeek-chat 判断问题是否需要画面，非视觉问题自动跳过图片
3. **PTT 窗口帧采样** — 只取说话期间的帧，均匀采样 3-5 帧，384px 压缩，端云成本平衡
4. **浏览器端 WAV 编码** — AudioContext 采集 PCM → 纯 JS WAV 编码 → WebSocket 传输
5. **结构化日志** — [1/3] 音频 → [2/3] STT → [3/3] LLM，每步耗时记录

## 项目结构

```
backend/
  server.py              FastAPI + WebSocket 服务
  stt.py                 语音识别 (Paraformer-v2)
  llm.py                 LLM 调用 (千问 DashScope)
  router.py              视觉路由器 (DeepSeek-chat)
  config.py              配置管理
frontend/
  index.html              前端页面 (摄像头 + PTT + 对话)
```

## 成本控制

| 策略 | 实现 |
|------|------|
| 云端 STT | DashScope Paraformer-v2，按量计费，识别率 >95% |
| 视觉路由器 | 非视觉问题 0 帧，避免不必要 token 消耗 |
| PTT 窗口隔离 | 只发说话期间帧，384px 压缩，图片 token ≤5000/轮 |
| 流式输出 | 首 token 到达即显示，减少感知等待 |
| 单模型 | 仅 qwen3.7-plus，无多模型冗余 |

详见 [DESIGN.md](DESIGN.md)。

## License

MIT

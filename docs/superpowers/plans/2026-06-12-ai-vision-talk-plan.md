# AI 视觉对话助手 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 Web 端 AI 视觉对话助手，用户打开摄像头和麦克风与 AI 自然对话，AI 能看见画面并语音回复。

**Architecture:** Python FastAPI 后端作为核心引擎（VAD/STT/截帧/LLM编排/TTS），浏览器前端作为 I/O 层（摄像头采集/音频播放/WebSocket通信）。

**Tech Stack:** FastAPI + uvicorn, Silero VAD, Whisper, OpenCV, Edge TTS, 通义千问 API, 原生 HTML/JS + Tailwind CSS

---

## Commit & PR 规范

### 分支策略

```
main          ← 始终可运行，每个 PR 合入后保持可用
  ├── feat/project-setup     (PR1)
  ├── feat/vad               (PR2)
  ├── feat/frame-extractor   (PR3)
  ├── feat/stt               (PR4)
  ├── feat/llm-adapters      (PR5)
  ├── feat/tts               (PR6)
  ├── feat/session           (PR7)
  ├── feat/orchestrator      (PR8)
  ├── feat/websocket-server  (PR9)
  ├── feat/frontend          (PR10)
  ├── feat/scene-detection   (PR11)
  ├── feat/context-memory    (PR12)
  ├── feat/multi-model       (PR13)
  └── feat/image-upload      (PR14)
```

### 原则

1. **每个 Task 一个分支、一个 PR** — 只做一件事，粒度尽可能小
2. **每个 commit 一个模块文件** — 出现问题可精确回退到具体模块
3. **主分支始终可运行** — 任意时刻 `main` 分支代码应为可复现状态
4. **全程持续提交** — 不堆到最后一天，开发周期内保持 commit 时间戳均匀分布
5. **合入后再开下一分支** — 避免冲突，保证依赖顺序

### PR 模板

每个 PR 必须包含以下四部分：

```markdown
## ① 标题
feat: [一句话说明本 PR 新增/修改了什么]

## ② 功能描述
[说明该功能的作用与使用方式]

## ③ 实现思路
[技术选型或核心实现逻辑]

## ④ 测试方式
[如何验证该功能正常运行，含具体命令和预期结果]
```

### Task → PR 映射

| PR | 分支 | 模块 | 依赖 | 预计 commits |
|----|------|------|------|-------------|
| PR1 | feat/project-setup | 项目骨架 + config | - | 3 |
| PR2 | feat/vad | VAD 语音检测 | PR1 | 1-2 |
| PR3 | feat/frame-extractor | 关键帧截取+去重 | PR1 | 1-2 |
| PR4 | feat/stt | Whisper 语音识别 | PR1 | 1-2 |
| PR5 | feat/llm-adapters | LLM 适配器基类+千问 | PR1 | 2 |
| PR6 | feat/tts | Edge TTS 合成 | PR1 | 1 |
| PR7 | feat/session | 会话管理 | PR1 | 1 |
| PR8 | feat/orchestrator | 全流程编排器 | PR2-7 | 1 |
| PR9 | feat/websocket-server | FastAPI + WebSocket | PR8 | 1 |
| PR10 | feat/frontend | 前端页面+交互 | PR9 | 2-3 |
| PR11 | feat/scene-detection | 画面变化感知 | PR9 | 1 |
| PR12 | feat/context-memory | 多轮记忆增强 | PR9 | 1 |
| PR13 | feat/multi-model | 多模型切换+GLM适配 | PR9 | 2 |
| PR14 | feat/image-upload | 图片上传功能 | PR9 | 2 |

> 注：PR2-7 互不依赖，可并行开发。PR8 须等 PR2-7 全部合入后再开始。

---

## 文件结构总览

```
ai_VisionTalk/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口 + WebSocket 路由
│   ├── config.py            # pydantic-settings 配置
│   ├── vad.py               # Silero VAD 封装
│   ├── stt.py               # Whisper 语音识别
│   ├── frame_extractor.py   # 关键帧截取 + 去重 + 压缩
│   ├── orchestrator.py      # LLM 编排 + 上下文组装
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py          # LLM 适配器抽象基类
│   │   ├── qwen.py          # 通义千问适配器
│   │   ├── glm.py           # 智谱 GLM 适配器
│   │   └── ernie.py         # 文心一言适配器
│   ├── tts.py               # Edge TTS 封装
│   └── session.py           # 对话会话管理
├── frontend/
│   ├── index.html           # 主页面
│   ├── style.css            # 样式
│   └── app.js               # 前端逻辑
├── docs/
│   └── superpowers/
│       ├── specs/           # 设计文档
│       └── plans/           # 本计划
├── requirements.txt
└── .env.example
```

---

## Phase 0: 项目初始化

### Task 0.1: 创建项目目录结构和依赖文件

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `backend/__init__.py`
- Create: `backend/models/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```bash
cat > requirements.txt << 'REQS'
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
websockets>=12.0
python-multipart>=0.0.9
pydantic-settings>=2.1.0
openai>=1.12.0
numpy>=1.26.0
opencv-python>=4.9.0
scikit-image>=0.22.0
silero-vad>=5.1
faster-whisper>=1.0.0
edge-tts>=6.1.0
pillow>=10.2.0
aiofiles>=23.2.0
REQS
echo "requirements.txt created"
```

- [ ] **Step 2: 创建 .env.example**

```bash
cat > .env.example << 'ENV'
# 通义千问 (DashScope)
DASHSCOPE_API_KEY=sk-your-key-here

# 智谱 GLM
ZHIPU_API_KEY=your-zhipu-api-key

# 百度文心
ERNIE_API_KEY=your-ernie-api-key
ERNIE_SECRET_KEY=your-ernie-secret-key

# 默认使用的模型: qwen-max / qwen-plus / glm-4v / ernie-vl
DEFAULT_MODEL=qwen-max

# 服务端口
PORT=8765
ENV
echo ".env.example created"
```

- [ ] **Step 3: 初始化 Git 仓库**

```bash
git init
git checkout -b main
echo ".env" >> .gitignore
echo ".superpowers/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

- [ ] **Step 4: 创建空的 __init__.py 文件**

```bash
touch backend/__init__.py
touch backend/models/__init__.py
echo "Package init files created"
```

- [ ] **Step 5: 创建 README.md 骨架**

```bash
cat > README.md << 'README'
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

所有依赖列于 requirements.txt，均为开源或免费服务：
- **Python 包**: fastapi, uvicorn, openai, numpy, opencv-python, scikit-image, silero-vad, faster-whisper, edge-tts, pillow, pydantic-settings, aiofiles
- **CDN**: Tailwind CSS (通过 `<script>` 标签加载)
- **AI API**: 阿里云 DashScope / 智谱 Open API (需自行申请 API Key)
- **本地模型**: Silero VAD (ONNX)、Faster-Whisper (自动下载)

## 原创功能

本项目从零构建，核心原创部分包括：
1. **端云协同编排器** (backend/orchestrator.py) — VAD→STT→截帧→LLM→TTS 全流程串联
2. **语音触发截帧策略** (backend/frame_extractor.py) — 头中尾三帧 + SSIM 去重 + 场景变化检测
3. **多模态对话会话管理** (backend/session.py) — 上下文记忆 + token 统计 + 历史压缩
4. **双模式交互** (frontend/app.js + backend/main.py) — PTT 按键 + 持续对话无缝切换
5. **多 LLM 后端适配器** (backend/models/) — 统一接口，一键切换国内 AI 模型

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
README
echo "README.md created"
```

- [ ] **Step 6: 验证目录结构**

Run: `ls -R backend/ && cat README.md | head -5`  
Expected: 看到 `__init__.py`, `models/` 及 README 内容

---

### Task 0.2: 配置管理模块

**Files:**
- Create: `backend/config.py`

- [ ] **Step 1: 编写 config.py**

```python
"""应用配置管理。从 .env 文件和环境变量加载配置。"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """全局配置，自动从 .env 和环境变量加载。"""

    # 阿里云 DashScope (通义千问)
    dashscope_api_key: str = ""

    # 智谱 GLM
    zhipu_api_key: str = ""

    # 百度文心
    ernie_api_key: str = ""
    ernie_secret_key: str = ""

    # 默认模型: qwen-max | qwen-plus | glm-4v | ernie-vl
    default_model: str = "qwen-max"

    # 服务端口
    port: int = 8765

    # VAD 灵敏度 (0-1, 越大越激进地判定为语音)
    vad_threshold: float = 0.5

    # 对话历史保留轮数
    max_history_turns: int = 10

    # 截帧最大宽度 (像素)
    frame_max_width: int = 768

    # SSIM 去重阈值 (低于此值视为不同帧)
    ssim_threshold: float = 0.95

    # 画面剧变 SSIM 阈值 (低于此值视为场景切换)
    scene_change_threshold: float = 0.7

    # API 去抖间隔 (秒)
    debounce_seconds: float = 1.5

    # 持续对话模式静音超时 (秒)
    silence_timeout_seconds: float = 2.0

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
```

- [ ] **Step 2: 验证配置模块能正确加载**

Run: `python -c "from backend.config import settings; print(f'Port: {settings.port}, Model: {settings.default_model}')"`

Expected: `Port: 8765, Model: qwen-max`

---

## Phase 1: P0 — 后端核心模块

### Task 1.1: VAD 语音活动检测模块

**Files:**
- Create: `backend/vad.py`

- [ ] **Step 1: 编写 vad.py**

```python
"""语音活动检测 (Voice Activity Detection)，基于 Silero VAD。

检测音频流中的语音段起止位置，过滤静音和噪声。
"""
import numpy as np
from backend.config import settings


class VADDetector:
    """封装 Silero VAD 模型，提供语音段检测。"""

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold or settings.vad_threshold
        self._model = None
        self._sample_rate = 16000  # Silero VAD 要求 16kHz

    def _load_model(self):
        """延迟加载 VAD 模型（首次调用时自动下载）。"""
        if self._model is None:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()
        return self._model

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """判断单个音频块是否包含语音。

        Args:
            audio_chunk: float32 数组，形状 (n_samples,)，16kHz 采样率。

        Returns:
            True 表示检测到语音，False 表示静音。
        """
        model = self._load_model()
        confidence = model(audio_chunk, self._sample_rate).item()
        return confidence > self.threshold

    def get_speech_confidence(self, audio_chunk: np.ndarray) -> float:
        """获取语音置信度 (0-1)，越大越可能是语音。"""
        model = self._load_model()
        return model(audio_chunk, self._sample_rate).item()


class SpeechSegmenter:
    """将连续音频流切分为语音段。

    跟踪语音活动状态，当检测到语音开始时标记起始位置，
    检测到静音持续一定时长后标记结束位置。
    """

    def __init__(
        self,
        vad: VADDetector | None = None,
        silence_samples: int | None = None,
        sample_rate: int = 16000,
    ):
        self.vad = vad or VADDetector()
        self.sample_rate = sample_rate
        # 持续这么多采样点无语音则判定语音段结束
        self.silence_samples = silence_samples or int(
            settings.silence_timeout_seconds * sample_rate
        )
        self._buffer: list[np.ndarray] = []
        self._silence_counter: int = 0
        self._is_speaking: bool = False

    def add_chunk(self, chunk: np.ndarray) -> list[np.ndarray]:
        """添加音频块，返回已完成的语音段列表。

        每个返回的语音段是一个完整的 numpy 数组 (拼接后的单声道 16kHz float32)。
        如果当前没有完成的语音段，返回空列表。
        """
        completed_segments: list[np.ndarray] = []
        is_speech = self.vad.is_speech(chunk)

        if is_speech:
            self._buffer.append(chunk)
            self._silence_counter = 0
            self._is_speaking = True
        elif self._is_speaking:
            # 语音后的静音，累加计数器
            self._silence_counter += len(chunk)
            self._buffer.append(chunk)

            if self._silence_counter >= self.silence_samples:
                # 静音超时，语音段结束
                segment = np.concatenate(self._buffer)
                completed_segments.append(segment)
                self._buffer.clear()
                self._silence_counter = 0
                self._is_speaking = False

        return completed_segments

    def flush(self) -> np.ndarray | None:
        """强制结束当前语音段（如 PTT 模式松开按钮时调用）。"""
        if self._buffer:
            segment = np.concatenate(self._buffer)
            self._buffer.clear()
            self._silence_counter = 0
            self._is_speaking = False
            return segment
        return None

    def reset(self):
        """重置状态。"""
        self._buffer.clear()
        self._silence_counter = 0
        self._is_speaking = False
```

- [ ] **Step 2: 验证 VAD 模块可导入**

Run: `python -c "from backend.vad import VADDetector, SpeechSegmenter; print('VAD module OK')"`

Expected: `VAD module OK`

---

### Task 1.2: 关键帧提取模块

**Files:**
- Create: `backend/frame_extractor.py`

- [ ] **Step 1: 编写 frame_extractor.py**

```python
"""关键帧提取模块。

从摄像头视频流中截取关键帧，提供去重、分辨率压缩、场景变化检测。
"""
import base64
import io
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from backend.config import settings


@dataclass
class CapturedFrame:
    """单张截帧，包含图像数据和元信息。"""
    image_bytes: bytes       # JPEG 编码的字节
    base64: str              # base64 字符串，可直接发给 LLM
    width: int
    height: int
    timestamp: float         # 捕获时间 (time.time())


class FrameExtractor:
    """从视频帧流中提取关键帧，处理去重和压缩。"""

    def __init__(
        self,
        max_width: int | None = None,
        jpeg_quality: int = 80,
        ssim_threshold: float | None = None,
    ):
        self.max_width = max_width or settings.frame_max_width
        self.jpeg_quality = jpeg_quality
        self.ssim_threshold = ssim_threshold or settings.ssim_threshold
        self._last_grayscale: np.ndarray | None = None

    def extract(self, frame_bgr: np.ndarray) -> CapturedFrame | None:
        """从 BGR 帧提取关键帧。

        如果帧与上一帧几乎相同（SSIM > 阈值），返回 None 跳过此帧。

        Args:
            frame_bgr: OpenCV BGR 格式的帧 (H, W, 3)，uint8。

        Returns:
            CapturedFrame 如果是新画面，否则 None。
        """
        # 转 RGB 用于后续处理
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 计算与上一帧的相似度
        current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._last_grayscale is not None:
            # 缩放到相同尺寸再比较
            h1, w1 = current_gray.shape
            h2, w2 = self._last_grayscale.shape
            if h1 != h2 or w1 != w2:
                resized = cv2.resize(current_gray, (w2, h2))
            else:
                resized = current_gray
            similarity = ssim(self._last_grayscale, resized, data_range=255)
            if similarity > self.ssim_threshold:
                return None  # 画面几乎相同，跳过

        self._last_grayscale = current_gray

        # 缩放分辨率
        h, w = frame_rgb.shape[:2]
        if w > self.max_width:
            ratio = self.max_width / w
            new_w = self.max_width
            new_h = int(h * ratio)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            h, w = new_h, new_w

        # JPEG 编码
        pil_img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=self.jpeg_quality)
        image_bytes = buf.getvalue()

        return CapturedFrame(
            image_bytes=image_bytes,
            base64=base64.b64encode(image_bytes).decode("ascii"),
            width=w,
            height=h,
            timestamp=time.time(),
        )

    def is_scene_change(self, frame_bgr: np.ndarray) -> bool:
        """检测画面是否发生剧烈变化（场景切换）。

        与上一次截帧的画面比较，SSIM 低于 scene_change_threshold 视为剧变。
        """
        if self._last_grayscale is None:
            return True
        current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h1, w1 = current_gray.shape
        h2, w2 = self._last_grayscale.shape
        if h1 != h2 or w1 != w2:
            resized = cv2.resize(current_gray, (w2, h2))
        else:
            resized = current_gray
        similarity = ssim(self._last_grayscale, resized, data_range=255)
        return similarity < settings.scene_change_threshold

    def reset(self):
        """重置状态，清除上一帧记录。"""
        self._last_grayscale = None
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from backend.frame_extractor import FrameExtractor, CapturedFrame; print('FrameExtractor module OK')"`

Expected: `FrameExtractor module OK`

---

### Task 1.3: STT 语音识别模块

**Files:**
- Create: `backend/stt.py`

- [ ] **Step 1: 编写 stt.py**

```python
"""语音转文字 (Speech-to-Text)，基于 Faster-Whisper 本地推理。

零 API 成本，中文识别质量好。
"""
import numpy as np
from backend.config import settings


class STTEngine:
    """封装 Faster-Whisper 模型，提供语音转文字。

    首次使用时会自动下载模型到本地缓存。
    """

    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        """延迟加载 Whisper 模型。"""
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",  # CPU 上 int8 量化，速度和内存平衡
            )
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """将音频数据转写为文字。

        Args:
            audio: float32 数组，单声道。
            sample_rate: 采样率，默认 16000。

        Returns:
            转写后的文字字符串。转写失败返回空字符串。
        """
        if len(audio) < sample_rate * 0.3:  # 少于 0.3 秒，太短
            return ""

        try:
            model = self._load_model()
            # 确保是 float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            segments, _ = model.transcribe(
                audio,
                language="zh",
                beam_size=5,
                vad_filter=True,  # Whisper 内置 VAD，双重保障
            )
            text_parts = [seg.text.strip() for seg in segments]
            return "".join(text_parts)
        except Exception as e:
            print(f"[STT] 转写失败: {e}")
            return ""
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from backend.stt import STTEngine; print('STT module OK')"`

Expected: `STT module OK`

---

### Task 1.4: LLM 适配器抽象基类 + 通义千问适配器

**Files:**
- Create: `backend/models/base.py`
- Create: `backend/models/qwen.py`

- [ ] **Step 1: 编写抽象基类 base.py**

```python
"""LLM 适配器抽象基类。所有模型适配器继承此类，实现统一接口。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VisionMessage:
    """发给视觉 LLM 的多模态消息。"""
    text: str
    image_base64_list: list[str]  # base64 编码的 JPEG 图片列表


@dataclass
class LLMResponse:
    """LLM 调用结果。"""
    text: str
    model_name: str
    input_tokens: int
    output_tokens: int


class BaseVisionLLM(ABC):
    """视觉语言模型适配器基类。

    子类必须实现 chat() 方法，返回 LLMResponse。
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回模型名称标识（如 'qwen-max'）。"""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[VisionMessage],
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """发送多模态消息给 LLM，返回回复。

        Args:
            messages: 对话历史，含文本和图片。
            system_prompt: 系统提示词。
            max_tokens: 最大输出 token 数。

        Returns:
            LLMResponse 包含回复文本和 token 用量。
        """
        ...
```

- [ ] **Step 2: 编写通义千问适配器 qwen.py**

```python
"""通义千问 (Qwen-VL) 视觉语言模型适配器。

通过 DashScope API 调用，支持 qwen-max 和 qwen-plus。
"""
from openai import AsyncOpenAI

from backend.config import settings
from backend.models.base import BaseVisionLLM, LLMResponse, VisionMessage


class QwenVisionLLM(BaseVisionLLM):
    """通义千问多模态模型适配器。

    通过 OpenAI 兼容接口调用 DashScope。
    """

    def __init__(self, model: str = "qwen-max"):
        self._model = model
        self._client = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[VisionMessage],
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """调用通义千问 API。"""
        formatted = self._format_messages(messages, system_prompt)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=formatted,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model_name=self._model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def _format_messages(
        self,
        messages: list[VisionMessage],
        system_prompt: str,
    ) -> list[dict]:
        """将内部消息格式转换为 DashScope 兼容的 OpenAI 多模态格式。

        通义千问支持 content 为数组的多模态格式：
        [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
        """
        formatted = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        for msg in messages:
            content_parts: list[dict] = []
            content_parts.append({"type": "text", "text": msg.text})
            for b64 in msg.image_base64_list:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            formatted.append({"role": "user", "content": content_parts})

        return formatted
```

- [ ] **Step 3: 验证模块可导入**

Run: `python -c "from backend.models.base import BaseVisionLLM, VisionMessage, LLMResponse; from backend.models.qwen import QwenVisionLLM; print('LLM models OK')"`

Expected: `LLM models OK`

---

### Task 1.5: TTS 语音合成模块

**Files:**
- Create: `backend/tts.py`

- [ ] **Step 1: 编写 tts.py**

```python
"""文字转语音 (Text-to-Speech)，基于 Edge TTS。

免费使用，中文自然度高，无需 API key。
"""
import asyncio
import io
from typing import AsyncIterator


class TTSEngine:
    """Edge TTS 封装，将文字转为 MP3 音频流。"""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        self.voice = voice

    async def synthesize(self, text: str) -> bytes:
        """将文字合成为 MP3 音频字节。

        Args:
            text: 要合成的文字。

        Returns:
            MP3 编码的音频字节，失败返回空 bytes。
        """
        if not text.strip():
            return b""

        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self.voice)
            audio_chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            return b"".join(audio_chunks)
        except Exception as e:
            print(f"[TTS] 合成失败: {e}")
            return b""

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """流式合成，边生成边返回音频块。

        适合长文本，降低首播延迟。
        """
        if not text.strip():
            return

        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self.voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as e:
            print(f"[TTS] 流式合成失败: {e}")
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from backend.tts import TTSEngine; print('TTS module OK')"`

Expected: `TTS module OK`

---

### Task 1.6: 对话会话管理模块

**Files:**
- Create: `backend/session.py`

- [ ] **Step 1: 编写 session.py**

```python
"""对话会话管理。

维护每个 WebSocket 连接的对话历史和上下文状态。
"""
from dataclasses import dataclass, field

from backend.config import settings
from backend.frame_extractor import CapturedFrame


@dataclass
class ConversationTurn:
    """一轮对话记录。"""
    user_text: str
    ai_text: str
    frames: list[CapturedFrame] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class Session:
    """单个对话会话，维护历史和 token 统计。"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: list[ConversationTurn] = []
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        # 最近一次截帧（用于"上一帧"参照）
        self.last_frame: CapturedFrame | None = None

    def add_turn(self, turn: ConversationTurn):
        """添加一轮对话，控制历史长度。"""
        self.history.append(turn)
        self._total_input_tokens += turn.input_tokens
        self._total_output_tokens += turn.output_tokens

        # 保留最近 N 轮完整对话，超出部分做截断
        if len(self.history) > settings.max_history_turns:
            self.history = self.history[-settings.max_history_turns:]

    def get_recent_turns(self, n: int = 3) -> list[ConversationTurn]:
        """获取最近 n 轮对话。"""
        return self.history[-n:]

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def turn_count(self) -> int:
        return len(self.history)
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from backend.session import Session, ConversationTurn; print('Session module OK')"`

Expected: `Session module OK`

---

### Task 1.7: Orchestrator 编排器

**Files:**
- Create: `backend/orchestrator.py`

- [ ] **Step 1: 编写 orchestrator.py**

```python
"""编排器：串联 VAD → 截帧 → STT → LLM → TTS 全流程。

负责端云协同的核心逻辑：端侧预处理后按需调用云端 API。
"""
import time
from typing import AsyncIterator

import numpy as np

from backend.config import settings
from backend.frame_extractor import CapturedFrame, FrameExtractor
from backend.models.base import BaseVisionLLM, VisionMessage
from backend.models.qwen import QwenVisionLLM
from backend.session import ConversationTurn, Session
from backend.stt import STTEngine
from backend.tts import TTSEngine


SYSTEM_PROMPT = """你是一个视觉对话助手，能够通过摄像头看到用户和周围环境。
你可以看到画面中的物体、人物、场景，也能听到用户说的话。
请像一个友好的朋友一样自然对话。回答简洁、有帮助、口语化。
如果用户指着画面中的东西问问题，直接描述你看到的内容。"""


class Orchestrator:
    """对话编排器，负责整个对话管线的协调。

    对外暴露单个方法 process_turn()，接收音频和帧，
    返回文字回复和 TTS 音频。
    """

    def __init__(self):
        self.stt = STTEngine(model_size="tiny")
        self.frame_extractor = FrameExtractor()
        self.tts = TTSEngine()
        self._llm: BaseVisionLLM | None = None
        self._last_call_time: float = 0.0

    def _get_llm(self) -> BaseVisionLLM:
        """获取或创建 LLM 实例（延迟初始化）。"""
        if self._llm is None:
            self._llm = self._create_llm(settings.default_model)
        return self._llm

    def _create_llm(self, model_name: str) -> BaseVisionLLM:
        """根据模型名创建对应的适配器。"""
        if model_name in ("qwen-max", "qwen-plus"):
            return QwenVisionLLM(model=model_name)
        # 后续 P2 扩展其他模型
        raise ValueError(f"不支持的模型: {model_name}")

    def switch_model(self, model_name: str):
        """切换 LLM 后端（P2 功能）。"""
        self._llm = self._create_llm(model_name)

    async def process_turn(
        self,
        session: Session,
        audio_segment: np.ndarray,
        frames: list[np.ndarray],  # BGR 原始帧列表
    ) -> dict:
        """处理一轮对话：语音转文字、截帧、调用 LLM、合成 TTS。

        Args:
            session: 当前对话会话。
            audio_segment: 用户语音数据 (float32, 16kHz mono)。
            frames: 说话期间采集的原始视频帧列表。

        Returns:
            {
                "text": "AI 的文字回复",
                "audio_mp3": b"...",  # MP3 字节
                "input_tokens": 100,
                "output_tokens": 200,
            }
        """
        # --- 端侧处理 ---

        # 1. STT: 语音转文字
        user_text = self.stt.transcribe(audio_segment)
        if not user_text:
            return {"text": "", "audio_mp3": b"", "input_tokens": 0, "output_tokens": 0}

        # 2. 截帧: 从原始帧中提取关键帧（去重 + 压缩）
        captured_frames: list[CapturedFrame] = []
        frame_count = len(frames)
        if frame_count > 0:
            # 取头、中、尾三帧
            indices = [0]
            if frame_count >= 3:
                indices.append(frame_count // 2)
            indices.append(frame_count - 1)
            # 去重
            seen_indices = set()
            for i in indices:
                if i in seen_indices:
                    continue
                seen_indices.add(i)
                cf = self.frame_extractor.extract(frames[i])
                if cf:
                    captured_frames.append(cf)

        # --- 去抖检查 ---
        now = time.time()
        if now - self._last_call_time < settings.debounce_seconds and session.turn_count > 0:
            # 间隔太短，可能重复触发，跳过
            pass
        self._last_call_time = now

        # --- 云端处理 ---

        # 3. 组装多模态消息
        image_b64_list = [cf.base64 for cf in captured_frames]
        # 将上轮 AI 回复加入历史（如果存在）
        history_turns = session.get_recent_turns(n=3)
        messages: list[VisionMessage] = []
        for turn in history_turns:
            turn_b64 = [cf.base64 for cf in turn.frames]
            messages.append(VisionMessage(text=turn.user_text, image_base64_list=turn_b64))

        # 当前轮
        current_msg = VisionMessage(text=user_text, image_base64_list=image_b64_list)
        messages.append(current_msg)

        # 4. 调用 LLM
        llm = self._get_llm()
        response = await llm.chat(messages, system_prompt=SYSTEM_PROMPT)

        # 5. TTS 合成
        audio_mp3 = await self.tts.synthesize(response.text)

        # 6. 记录到会话
        turn = ConversationTurn(
            user_text=user_text,
            ai_text=response.text,
            frames=captured_frames,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        session.add_turn(turn)

        return {
            "text": response.text,
            "audio_mp3": audio_mp3,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from backend.orchestrator import Orchestrator, SYSTEM_PROMPT; print('Orchestrator module OK')"`

Expected: `Orchestrator module OK`

---

### Task 1.8: FastAPI WebSocket 服务入口 + 静态文件服务

**Files:**
- Create: `backend/main.py`

- [ ] **Step 1: 编写 main.py**

```python
"""FastAPI 应用入口 + WebSocket 路由。

提供 WebSocket 端点和前端静态文件服务。
"""
import json
import struct

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import settings
from backend.frame_extractor import FrameExtractor
from backend.orchestrator import Orchestrator
from backend.session import Session
from backend.vad import SpeechSegmenter, VADDetector

# --- 应用初始化 ---
app = FastAPI(title="AI Vision Talk")

# 全局服务实例 (所有连接共享)
orchestrator = Orchestrator()
vad_detector = VADDetector()

# 每个 WebSocket 连接有独立的 Session 和 SpeechSegmenter 和 FrameExtractor
# 通过 session_id 关联


@app.get("/")
async def root():
    """返回前端页面。"""
    return FileResponse("frontend/index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket 端点，处理实时音视频对话。

    协议（JSON 控制消息 + 二进制帧数据）:

    客户端 → 服务端:
      - JSON: {"type": "ptt_start"}      # PTT 开始录音
      - JSON: {"type": "ptt_stop"}       # PTT 结束录音，触发处理
      - JSON: {"type": "mode", "mode": "ptt"|"continuous"}
      - 二进制: 视频帧 JPEG 字节 (持续发送，用于服务端分析)
      - 二进制: 音频 PCM 块 (16kHz, mono, float32, 4096 samples/chunk)

    服务端 → 客户端:
      - JSON: {"type": "status", "status": "listening"|"processing"|"responding"|"idle"}
      - JSON: {"type": "response", "text": "...", "tokens": {...}}
      - 二进制: TTS 音频 (MP3)
    """
    await ws.accept()
    session = Session(session_id=str(id(ws)))
    segmenter = SpeechSegmenter(vad=vad_detector)
    frame_extractor = FrameExtractor()

    # 模式: "ptt" 或 "continuous"
    mode = "ptt"
    # PTT 模式下的音频缓冲区
    ptt_buffer: list[np.ndarray] = []
    is_recording = False
    # 持续对话模式下的截帧缓冲区
    continuous_frames: list[np.ndarray] = []

    async def send_status(status: str):
        await ws.send_json({"type": "status", "status": status})

    async def send_response(text: str, audio_mp3: bytes, input_tokens: int, output_tokens: int):
        await ws.send_json({
            "type": "response",
            "text": text,
            "tokens": {"input": input_tokens, "output": output_tokens},
        })
        if audio_mp3:
            # 发送音频长度 + 音频数据
            await ws.send_bytes(struct.pack("!I", len(audio_mp3)) + audio_mp3)

    try:
        while True:
            data = await ws.receive()

            if "text" in data:
                # JSON 控制消息
                msg = json.loads(data["text"])
                msg_type = msg.get("type", "")

                if msg_type == "ptt_start":
                    is_recording = True
                    ptt_buffer.clear()
                    continuous_frames.clear()
                    await send_status("listening")

                elif msg_type == "ptt_stop":
                    is_recording = False
                    if not ptt_buffer:
                        continue
                    await send_status("processing")

                    audio_segment = np.concatenate(ptt_buffer)
                    ptt_buffer.clear()

                    # 调用编排器处理
                    result = await orchestrator.process_turn(
                        session=session,
                        audio_segment=audio_segment,
                        frames=continuous_frames,
                    )
                    continuous_frames.clear()

                    if result["text"]:
                        await send_status("responding")
                        await send_response(
                            result["text"],
                            result["audio_mp3"],
                            result["input_tokens"],
                            result["output_tokens"],
                        )
                    await send_status("idle")

                elif msg_type == "mode":
                    mode = msg.get("mode", "ptt")
                    segmenter.reset()
                    frame_extractor.reset()

            elif "bytes" in data:
                raw = data["bytes"]
                # 区分音频和视频帧：视频帧以 JPEG magic bytes 开头 (0xFF 0xD8)
                if len(raw) > 2 and raw[:2] == b'\xff\xd8':
                    # JPEG 视频帧
                    import cv2
                    nparr = np.frombuffer(raw, np.uint8)
                    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame_bgr is not None:
                        if is_recording:
                            continuous_frames.append(frame_bgr)
                        # 持续对话模式下也收集帧
                        if mode == "continuous":
                            continuous_frames.append(frame_bgr)
                else:
                    # PCM 音频数据 (float32, 16kHz mono)
                    audio_chunk = np.frombuffer(raw, dtype=np.float32)

                    if mode == "ptt" and is_recording:
                        ptt_buffer.append(audio_chunk)
                    elif mode == "continuous":
                        segments = segmenter.add_chunk(audio_chunk)
                        for seg in segments:
                            await send_status("processing")
                            # 使用收集的帧
                            frames = list(continuous_frames)
                            continuous_frames.clear()
                            result = await orchestrator.process_turn(
                                session=session,
                                audio_segment=seg,
                                frames=frames,
                            )
                            if result["text"]:
                                await send_status("responding")
                                await send_response(
                                    result["text"],
                                    result["audio_mp3"],
                                    result["input_tokens"],
                                    result["output_tokens"],
                                )
                            await send_status("idle")

    except WebSocketDisconnect:
        print(f"[WS] 客户端断开: {session.session_id}")


# 挂载静态文件
app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
```

- [ ] **Step 2: 验证应用能启动**

Run: `python -m backend.main`  
Expected: 看到 uvicorn 启动日志，端口 8765

---

## Phase 2: P0 — 前端

### Task 2.1: 主页面 HTML + CSS

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/style.css`

- [ ] **Step 1: 编写 index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 视觉对话助手</title>
    <link rel="stylesheet" href="/static/style.css">
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-950 text-white min-h-screen flex flex-col">
    <!-- 顶部状态栏 -->
    <header class="flex items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-800">
        <h1 class="text-lg font-semibold">👁️ AI VisionTalk</h1>
        <div class="flex items-center gap-3">
            <span id="statusBadge" class="px-3 py-1 rounded-full text-xs font-medium bg-gray-700 text-gray-300">
                🟢 就绪
            </span>
            <span id="tokenInfo" class="text-xs text-gray-500 hidden">
                Tokens: <span id="tokenCount">0</span>
            </span>
        </div>
    </header>

    <!-- 主区域：摄像头 + 对话 -->
    <main class="flex-1 flex flex-col md:flex-row gap-0 overflow-hidden">
        <!-- 左侧：摄像头预览 -->
        <section class="relative flex-1 min-h-[300px] bg-black flex items-center justify-center">
            <video id="camera" autoplay playsinline muted
                   class="w-full h-full object-cover mirror-mode"></video>
            <!-- 摄像头未开启时的占位 -->
            <div id="cameraPlaceholder" class="absolute inset-0 flex flex-col items-center justify-center text-gray-500">
                <svg class="w-16 h-16 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                          d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <p class="text-sm">点击下方按钮开启摄像头</p>
            </div>
            <!-- 处理中遮罩 -->
            <div id="processingOverlay" class="absolute inset-0 bg-black/60 flex items-center justify-center hidden">
                <div class="flex flex-col items-center gap-3">
                    <div class="animate-spin w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                    <span id="processingText" class="text-sm text-blue-300">思考中...</span>
                </div>
            </div>
        </section>

        <!-- 右侧：对话文字面板 -->
        <section class="w-full md:w-80 lg:w-96 bg-gray-900 flex flex-col border-l border-gray-800">
            <div class="px-4 py-3 border-b border-gray-800">
                <h2 class="text-sm font-medium text-gray-400">💬 对话记录</h2>
            </div>
            <div id="chatMessages" class="flex-1 overflow-y-auto p-4 space-y-3">
                <div class="text-center text-gray-600 text-sm py-8">
                    开始说话，AI 会在这里显示回复
                </div>
            </div>
        </section>
    </main>

    <!-- 底部控制栏 -->
    <footer class="bg-gray-900 border-t border-gray-800 px-4 py-3">
        <div class="max-w-lg mx-auto flex items-center gap-3">
            <!-- 模式切换 -->
            <button id="modeToggle" class="flex-shrink-0 px-3 py-2 rounded-lg text-xs font-medium
                                           bg-gray-800 text-gray-400 hover:bg-gray-700 transition"
                    title="切换对话模式">
                <span id="modeIcon">🎙️</span> <span id="modeLabel">PTT</span>
            </button>

            <!-- 主按钮 (PTT / 唤醒) -->
            <button id="talkButton"
                    class="flex-1 py-4 rounded-2xl text-lg font-bold transition-all duration-200
                           bg-blue-600 hover:bg-blue-500 active:bg-red-500 active:scale-95
                           shadow-lg shadow-blue-600/30">
                按住说话
            </button>

            <!-- 摄像头开关 -->
            <button id="cameraToggle" class="flex-shrink-0 px-3 py-2 rounded-lg text-xs font-medium
                                            bg-gray-800 text-gray-400 hover:bg-gray-700 transition"
                    title="开关摄像头">
                📷
            </button>
        </div>
    </footer>

    <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 编写 style.css**

```css
/* 摄像头镜像模式 (前置摄像头通常需要镜像) */
.mirror-mode {
    transform: scaleX(-1);
}

/* 对话消息动画 */
@keyframes slideIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.chat-message {
    animation: slideIn 0.3s ease-out;
}

/* 滚动条美化 */
#chatMessages::-webkit-scrollbar {
    width: 4px;
}
#chatMessages::-webkit-scrollbar-track {
    background: transparent;
}
#chatMessages::-webkit-scrollbar-thumb {
    background: #374151;
    border-radius: 2px;
}

/* 按钮过渡 */
#talkButton {
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;
}
```

- [ ] **Step 3: 验证页面可访问**

Run: `python -m backend.main` (如果还没运行)  
然后浏览器访问 `http://localhost:8765` 查看页面是否加载。

---

### Task 2.2: 前端 JS 核心逻辑

**Files:**
- Create: `frontend/app.js`

- [ ] **Step 1: 编写 app.js**

```javascript
/**
 * AI VisionTalk 前端逻辑
 *
 * 职责：
 * 1. 摄像头和麦克风采集 (getUserMedia)
 * 2. WebSocket 通信 (发送音频/视频帧，接收 AI 响应)
 * 3. UI 状态管理 (PTT/持续对话模式切换，状态指示)
 */

// ==================== DOM 元素 ====================
const camera = document.getElementById('camera');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const processingOverlay = document.getElementById('processingOverlay');
const processingText = document.getElementById('processingText');
const talkButton = document.getElementById('talkButton');
const modeToggle = document.getElementById('modeToggle');
const modeIcon = document.getElementById('modeIcon');
const modeLabel = document.getElementById('modeLabel');
const cameraToggle = document.getElementById('cameraToggle');
const statusBadge = document.getElementById('statusBadge');
const chatMessages = document.getElementById('chatMessages');
const tokenInfo = document.getElementById('tokenInfo');
const tokenCount = document.getElementById('tokenCount');

// ==================== 全局状态 ====================
let ws = null;                    // WebSocket 连接
let mediaStream = null;           // 摄像头+麦克风流
let audioContext = null;          // AudioContext
let processor = null;             // ScriptProcessorNode
let isPTTRecording = false;       // PTT 是否正在录音
let mode = 'ptt';                 // 'ptt' | 'continuous'
let cameraOn = false;
let totalTokens = 0;

// ==================== WebSocket ====================
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        console.log('[WS] 已连接');
        setStatus('idle', '🟢 就绪');
    };

    ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
            const msg = JSON.parse(event.data);
            handleMessage(msg);
        } else {
            // 二进制数据 = TTS 音频
            handleAudioData(new Uint8Array(event.data));
        }
    };

    ws.onclose = () => {
        console.log('[WS] 断开，3秒后重连...');
        setStatus('idle', '🔴 断开');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error('[WS] 错误:', err);
    };
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'status':
            updateStatusBadge(msg.status);
            break;
        case 'response':
            handleResponse(msg);
            break;
    }
}

function handleResponse(msg) {
    // 显示文字
    addChatMessage('ai', msg.text);
    // 更新 token 计数
    if (msg.tokens) {
        totalTokens += msg.tokens.input + msg.tokens.output;
        tokenInfo.classList.remove('hidden');
        tokenCount.textContent = totalTokens.toLocaleString();
    }
}

async function handleAudioData(data) {
    // 解析: [4 字节长度] + [MP3 数据]
    if (data.length < 4) return;
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const mp3Length = view.getUint32(0, false);
    const mp3Data = data.slice(4, 4 + mp3Length);
    await playAudio(mp3Data);
}

async function playAudio(mp3Data) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    try {
        const audioBuffer = await audioContext.decodeAudioData(mp3Data.buffer.slice(
            mp3Data.byteOffset, mp3Data.byteOffset + mp3Data.byteLength
        ));
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start();
    } catch (e) {
        console.error('[Audio] 播放失败:', e);
    }
}

function sendJSON(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(obj));
    }
}

function sendAudioChunk(float32Array) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(float32Array.buffer);
    }
}

function sendVideoFrame(jpegBytes) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(jpegBytes);
    }
}

// ==================== 摄像头和麦克风 ====================
async function startCamera() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user',
            },
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
            },
        });

        // 显示视频
        camera.srcObject = mediaStream;
        cameraPlaceholder.classList.add('hidden');
        cameraOn = true;
        cameraToggle.textContent = '📷✓';

        // 启动音频处理管线
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(mediaStream);

        // 将音频下采样到 16kHz mono
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = (e) => {
            if (mode === 'continuous' || isPTTRecording) {
                const input = e.inputBuffer.getChannelData(0);
                const copy = new Float32Array(input.length);
                copy.set(input);
                sendAudioChunk(copy);
            }
        };
        source.connect(processor);
        processor.connect(audioContext.destination);

        // 启动视频帧捕获（用于 webSocket 持续分析）
        startFrameCapture();

        console.log('[Camera] 摄像头和麦克风已启动');
    } catch (err) {
        console.error('[Camera] 启动失败:', err);
        alert('无法访问摄像头或麦克风，请检查权限设置。');
    }
}

function stopCamera() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (processor) {
        processor.disconnect();
        processor = null;
    }
    camera.srcObject = null;
    cameraPlaceholder.classList.remove('hidden');
    cameraOn = false;
    cameraToggle.textContent = '📷';
    console.log('[Camera] 已关闭');
}

// ==================== 视频帧捕获 ====================
let frameCaptureInterval = null;

function startFrameCapture() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    frameCaptureInterval = setInterval(() => {
        if (!cameraOn || !camera.srcObject) return;
        // 画到 canvas 上
        canvas.width = camera.videoWidth || 640;
        canvas.height = camera.videoHeight || 480;
        if (canvas.width === 0) return;
        ctx.drawImage(camera, 0, 0);
        // 导出 JPEG
        canvas.toBlob((blob) => {
            if (blob) {
                blob.arrayBuffer().then(buf => {
                    sendVideoFrame(new Uint8Array(buf));
                });
            }
        }, 'image/jpeg', 0.7);
    }, 1000);  // 每秒一帧，用于服务端分析（不是发送给 LLM 的帧）
}

// ==================== UI 交互 ====================
talkButton.addEventListener('pointerdown', () => {
    if (mode !== 'ptt') return;
    if (!cameraOn) {
        startCamera().then(() => beginPTT());
    } else {
        beginPTT();
    }
});

talkButton.addEventListener('pointerup', () => {
    if (mode !== 'ptt') return;
    endPTT();
});

talkButton.addEventListener('pointerleave', () => {
    if (mode !== 'ptt' || !isPTTRecording) return;
    endPTT();
});

function beginPTT() {
    isPTTRecording = true;
    talkButton.textContent = '松开发送';
    talkButton.classList.remove('bg-blue-600', 'hover:bg-blue-500');
    talkButton.classList.add('bg-red-600', 'hover:bg-red-500');
    setStatus('listening', '🎙️ 聆听中...');
    sendJSON({ type: 'ptt_start' });
}

function endPTT() {
    if (!isPTTRecording) return;
    isPTTRecording = false;
    talkButton.textContent = '按住说话';
    talkButton.classList.remove('bg-red-600', 'hover:bg-red-500');
    talkButton.classList.add('bg-blue-600', 'hover:bg-blue-500');
    setStatus('processing', '⏳ 处理中...');
    sendJSON({ type: 'ptt_stop' });
}

modeToggle.addEventListener('click', () => {
    if (mode === 'ptt') {
        // 切换到持续对话模式
        mode = 'continuous';
        modeIcon.textContent = '🌊';
        modeLabel.textContent = '持续';
        talkButton.textContent = '点击停止';
        talkButton.onpointerdown = null;
        talkButton.onpointerup = null;
        talkButton.onpointerleave = null;
        talkButton.onclick = () => {
            // 停止持续对话
            mode = 'ptt';
            modeIcon.textContent = '🎙️';
            modeLabel.textContent = 'PTT';
            talkButton.textContent = '按住说话';
            talkButton.onclick = null;
            setupPTTListeners();
            setStatus('idle', '🟢 就绪');
            sendJSON({ type: 'mode', mode: 'ptt' });
        };
        setStatus('idle', '🌊 持续聆听中...');
    } else {
        // 切回 PTT
        mode = 'ptt';
        modeIcon.textContent = '🎙️';
        modeLabel.textContent = 'PTT';
        talkButton.textContent = '按住说话';
        talkButton.onclick = null;
        setupPTTListeners();
        setStatus('idle', '🟢 就绪');
    }
    sendJSON({ type: 'mode', mode });
});

function setupPTTListeners() {
    talkButton.addEventListener('pointerdown', () => {
        if (!cameraOn) startCamera().then(() => beginPTT());
        else beginPTT();
    });
    talkButton.addEventListener('pointerup', endPTT);
    talkButton.addEventListener('pointerleave', () => {
        if (isPTTRecording) endPTT();
    });
}

cameraToggle.addEventListener('click', () => {
    if (cameraOn) {
        stopCamera();
    } else {
        startCamera();
    }
});

// ==================== 状态和 UI 辅助 ====================
function setStatus(status, text) {
    processingText.textContent = text;
    if (status === 'processing') {
        processingOverlay.classList.remove('hidden');
    } else {
        processingOverlay.classList.add('hidden');
    }
}

function updateStatusBadge(status) {
    const statusMap = {
        'idle': '🟢 就绪',
        'listening': '🎙️ 聆听中',
        'processing': '⏳ 思考中',
        'responding': '🔊 回复中',
    };
    statusBadge.textContent = statusMap[status] || status;
}

function addChatMessage(role, text) {
    const div = document.createElement('div');
    div.className = 'chat-message px-3 py-2 rounded-lg text-sm ' +
        (role === 'user'
            ? 'bg-blue-900/50 text-blue-100 ml-8'
            : 'bg-gray-800 text-gray-200 mr-8');
    div.innerHTML = `<span class="text-xs text-gray-500 block mb-1">${role === 'user' ? '🧑 你' : '🤖 AI'}</span>${escapeHtml(text)}`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // 清除占位文字
    const placeholder = chatMessages.querySelector('.text-center');
    if (placeholder) placeholder.remove();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==================== 初始化 ====================
function init() {
    setupPTTListeners();
    connectWebSocket();
}

init();
```

- [ ] **Step 2: 验证前端文件存在**

Run: `ls -la frontend/`  
Expected: 看到 `index.html`, `style.css`, `app.js`

---

## Phase 3: P1 — 增强功能

### Task 3.1: 画面变化感知 (Scene Change Detection)

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: 在 WebSocket 循环中添加场景变化检测**

在 `backend/main.py` 的视频帧处理部分（binary 分支中），在解码帧之后添加场景变化检测逻辑。找到处理 JPEG 帧的代码块，增加：

```python
# 在后台线程中定期检查场景变化
# 在 is_recording 或 continuous 模式下，当检测到场景剧变时记录
if frame_extractor.is_scene_change(frame_bgr):
    print(f"[Frame] 检测到场景变化 (session={session.session_id})")
    # 将突变帧加入 continuous_frames，在下次处理时一起发送
    if not is_recording and mode == "continuous":
        continuous_frames.append(frame_bgr)
```

**完整代码**：这部分需要在 main.py 的 bytes 分支中，找到 `continuous_frames.append(frame_bgr)` 的地方，在其附近添加场景变化检测。

---

### Task 3.2: 多轮对话上下文增强

**Files:**
- Modify: `backend/orchestrator.py`

- [ ] **Step 1: 优化 SYSTEM_PROMPT，增加指代消解能力**

```python
SYSTEM_PROMPT = """你是一个视觉对话助手，能够通过摄像头看到用户和周围环境。
你可以看到画面中的物体、人物、场景，也能听到用户说的话。

重要原则：
1. 像友好的朋友一样自然对话，用口语化的中文，不要用书面语。
2. 回答简洁、有帮助，一般 2-4 句话即可。
3. 记住对话历史：如果用户说"刚才那个"、"前面提到的"，你要结合历史上下文回答。
4. 如果用户指着画面问"这是什么"，直接描述你看到的。
5. 如果看不到用户所指的物体，诚实说"我看不太清楚，能靠近一点吗？"
6. 对话历史中包括了之前的画面，如果当前画面看不清，可以结合之前看到的画面判断。
"""
```

- [ ] **Step 2: 在 orchestrator.process_turn() 中增加短期记忆摘要**

在 `orchestrator.py` 的 `process_turn()` 方法中，在组装 messages 之前添加：

```python
# 如果历史超过 5 轮，在第一轮之前插入摘要上下文
if len(history_turns) > 5:
    summary_parts = []
    for t in history_turns[:-3]:  # 旧的轮次做摘要
        summary_parts.append(f"用户: {t.user_text} → AI: {t.ai_text[:50]}...")
    summary_text = "之前的对话摘要:\n" + "\n".join(summary_parts)
    # 将摘要作为第一条消息的 prefix
    if messages:
        messages[0].text = summary_text + "\n\n---\n当前对话:\n" + messages[0].text
```

---

## Phase 4: P2 — 扩展功能

### Task 4.1: 智谱 GLM-4V 适配器

**Files:**
- Create: `backend/models/glm.py`

- [ ] **Step 1: 编写 glm.py**

```python
"""智谱 GLM-4V 视觉语言模型适配器。"""
from openai import AsyncOpenAI

from backend.config import settings
from backend.models.base import BaseVisionLLM, LLMResponse, VisionMessage


class GLMVisionLLM(BaseVisionLLM):
    """智谱 GLM-4V 多模态模型适配器。

    通过 OpenAI 兼容接口调用智谱 API。
    """

    def __init__(self, model: str = "glm-4v"):
        self._model = model
        self._client = AsyncOpenAI(
            api_key=settings.zhipu_api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[VisionMessage],
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> LLMResponse:
        formatted = self._format_messages(messages, system_prompt)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=formatted,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model_name=self._model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def _format_messages(
        self,
        messages: list[VisionMessage],
        system_prompt: str,
    ) -> list[dict]:
        """智谱 GLM-4V 也支持 OpenAI 兼容的多模态格式。"""
        formatted = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        for msg in messages:
            content_parts: list[dict] = []
            content_parts.append({"type": "text", "text": msg.text})
            for b64 in msg.image_base64_list:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            formatted.append({"role": "user", "content": content_parts})

        return formatted
```

---

### Task 4.2: 多模型切换支持

**Files:**
- Modify: `backend/orchestrator.py`

- [ ] **Step 1: 更新 _create_llm() 方法**

在 `orchestrator.py` 中扩展 `_create_llm()` 方法：

```python
def _create_llm(self, model_name: str) -> BaseVisionLLM:
    """根据模型名创建对应的适配器。"""
    if model_name in ("qwen-max", "qwen-plus"):
        return QwenVisionLLM(model=model_name)
    elif model_name.startswith("glm"):
        from backend.models.glm import GLMVisionLLM
        return GLMVisionLLM(model=model_name)
    raise ValueError(f"不支持的模型: {model_name}")
```

---

### Task 4.3: 前端图片上传功能

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/app.js`
- Modify: `backend/main.py`

- [ ] **Step 1: 在 index.html 中添加上传按钮**

在底部控制栏的 cameraToggle 旁边添加：

```html
<!-- 图片上传按钮 -->
<label for="imageUpload" class="flex-shrink-0 px-3 py-2 rounded-lg text-xs font-medium
                                 bg-gray-800 text-gray-400 hover:bg-gray-700 transition cursor-pointer"
       title="上传图片">
    🖼️
</label>
<input id="imageUpload" type="file" accept="image/*" class="hidden">
```

- [ ] **Step 2: 在 app.js 中添加上传逻辑**

```javascript
const imageUpload = document.getElementById('imageUpload');

imageUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // 读取为 base64
    const reader = new FileReader();
    reader.onload = () => {
        const base64 = reader.result.split(',')[1];  // 去掉 data:...;base64, 前缀
        sendJSON({ type: 'image_upload', image_base64: base64 });
        addChatMessage('user', '📷 [上传了一张图片]');
        setStatus('processing', '⏳ 分析图片中...');
    };
    reader.readAsDataURL(file);
    imageUpload.value = '';  // 重置，允许重复上传同一文件
});
```

- [ ] **Step 3: 在 main.py 中处理 image_upload 消息**

在 `websocket_endpoint` 的 JSON 消息处理部分添加：

```python
elif msg_type == "image_upload":
    import base64 as b64
    import cv2
    img_b64 = msg.get("image_base64", "")
    if img_b64:
        img_bytes = b64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is not None:
            result = await orchestrator.process_turn(
                session=session,
                audio_segment=np.array([], dtype=np.float32),
                frames=[frame_bgr],
            )
            if result["text"]:
                await send_status("responding")
                await send_response(
                    result["text"],
                    result["audio_mp3"],
                    result["input_tokens"],
                    result["output_tokens"],
                )
                await send_status("idle")
```

---

## Phase 5: 集成测试 & 收尾

### Task 5.1: 端到端连通性测试

- [ ] **Step 1: 启动后端**

```bash
python -m backend.main
```

- [ ] **Step 2: 浏览器打开 http://localhost:8765**

验证：
1. 页面正常加载
2. 点击 📷 按钮 → 浏览器弹出摄像头/麦克风权限请求
3. 允许权限 → 摄像头预览显示
4. 按住"按住说话"按钮 → 说话 → 松开
5. 查看终端日志是否有正常的处理流程输出

---

### Task 5.2: 更新设计文档的最终状态

- [ ] **Step 1: 回填设计文档中用户故事的"最终"列**

将 `docs/superpowers/specs/2026-06-12-ai-vision-talk-design.md` 中用户故事表的 ⬜ 根据实际实现状态更新为 ✅ 或 ❌。

- [ ] **Step 2: 回填成本控制策略的"实际采用"列**

同样更新成本控制策略表。

---

## 实现顺序总结 (PR 编排)

```
Phase 0: 项目初始化
  PR1 (T0.1+T0.2): requirements.txt + .env + config.py + 目录结构
    ↓
Phase 1: P0 后端 — 全部串行 (模块化开发，逐个验证)
  PR2 (T1.1): VAD
    ↓
  PR3 (T1.2): Frame
    ↓
  PR4 (T1.3): STT
    ↓
  PR5 (T1.4): LLM 适配器
    ↓
  PR6 (T1.5): TTS
    ↓
  PR7 (T1.6): Session
    ↓
  PR8 (T1.7): Orchestrator (串联 PR2-7 所有模块)
    ↓
  PR9 (T1.8): FastAPI + WebSocket 入口
    ↓
Phase 2: P0 前端
  PR10 (T2.1+T2.2): HTML + CSS + app.js
    ↓  ← 🎯 到此 P0 MVP 可运行
Phase 3: P1 增强
  PR11 (T3.1): 画面变化感知
  PR12 (T3.2): 多轮记忆增强
    ↓
Phase 4: P2 扩展
  PR13 (T4.1+T4.2): GLM 适配器 + 多模型切换
  PR14 (T4.3): 图片上传
    ↓
Phase 5: 收尾
  PR15 (T5.1+T5.2): README + 端到端测试 + 设计文档回填
```

> **关键里程碑**: PR10 合入后，main 分支即具备完整的 P0 MVP 能力（摄像头+语音→AI视觉对话+语音回复），评委可随时 clone 并运行演示。

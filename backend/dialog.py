"""对话管理器。

编排 PTT 交互流程：录音 → STT → LLM → 回复。
维护多轮对话历史和画面引用。
"""
import time
import threading
from dataclasses import dataclass

import numpy as np

from backend.config import settings
from backend.vad import VADDetector
from backend.stt import STTEngine
from backend.llm import QwenVisionLLM
from backend.frame_grabber import FrameGrabber, CapturedFrame


@dataclass
class Turn:
    """一轮对话记录。"""
    user_text: str
    assistant_text: str
    frames: list[CapturedFrame]
    timestamp: float


class DialogManager:
    """PTT 对话管理器。

    持有 VAD、STT、LLM、FrameGrabber 四个模块，
    通过 press()/release() 驱动完整交互流程。

    Usage:
        dm = DialogManager()
        dm.set_system_prompt("你是一个有用的助手")
        dm.start()                    # 启动抓帧器
        dm.press()                    # 用户按下 PTT
        ... user speaking ...
        response = dm.release()       # 用户松开 → 转写 → LLM → 输出
        dm.stop()
    """

    def __init__(
        self,
        system_prompt: str = "",
        vad_threshold: float | None = None,
    ):
        self._system_prompt = system_prompt
        self._history: list[Turn] = []

        # Sub-modules (lazy init)
        self._vad: VADDetector | None = None
        self._stt: STTEngine | None = None
        self._llm: QwenVisionLLM | None = None
        self._frame_grabber: FrameGrabber | None = None

        # PTT state
        self._recording = False
        self._audio_chunks: list[np.ndarray] = []
        self._ptt_thread: threading.Thread | None = None

        # Audio config
        self._sample_rate = 16000
        self._chunk_size = 1024  # samples per read

    # ── Lifecycle ──

    def start(self) -> None:
        """启动后台模块（抓帧器）。"""
        if self._frame_grabber is None:
            self._frame_grabber = FrameGrabber()
        self._frame_grabber.start()

    def stop(self) -> None:
        """停止所有后台模块。"""
        if self._frame_grabber is not None:
            self._frame_grabber.stop()

    # ── System prompt ──

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词。"""
        self._system_prompt = prompt

    # ── History ──

    def get_history(self) -> list[dict]:
        """获取格式化的对话历史（最近 max_history_turns 轮）。

        Returns:
            OpenAI 兼容的消息列表 [{"role":"user","content":"..."}, ...]。
        """
        max_turns = settings.max_history_turns
        recent = self._history[-max_turns:] if len(self._history) > max_turns else self._history

        formatted: list[dict] = []
        for turn in recent:
            formatted.append({"role": "user", "content": turn.user_text})
            formatted.append({"role": "assistant", "content": turn.assistant_text})
        return formatted

    def clear_history(self) -> None:
        """清空对话历史。"""
        self._history.clear()

    # ── PTT ──

    def press(self) -> None:
        """PTT 按下：切换到语音模式，开始录音。"""
        if self._recording:
            return
        self._recording = True
        self._audio_chunks = []

        # 切换抓帧模式
        if self._frame_grabber is not None:
            self._frame_grabber.set_mode("speech")

        # 在后台线程中录音
        self._ptt_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._ptt_thread.start()

    def release(self) -> str:
        """PTT 松开：停止录音，处理完整流程。

        Returns:
            AI 回复文本。空字符串表示无回复（误触或处理失败）。
        """
        if not self._recording:
            return ""
        self._recording = False

        # 等待录音线程结束
        if self._ptt_thread is not None:
            self._ptt_thread.join(timeout=3.0)
            self._ptt_thread = None

        # 切换回静音模式
        if self._frame_grabber is not None:
            self._frame_grabber.set_mode("silence")

        # 拼接音频
        if not self._audio_chunks:
            return ""
        audio = np.concatenate(self._audio_chunks)
        self._audio_chunks = []

        # 同步执行异步处理
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._process_turn(audio))

    # ── Internal ──

    def _record_loop(self) -> None:
        """后台录音循环。"""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self._sample_rate,
                input=True,
                frames_per_buffer=self._chunk_size,
            )
            while self._recording:
                try:
                    data = stream.read(self._chunk_size, exception_on_overflow=False)
                    chunk = np.frombuffer(data, dtype=np.float32)
                    self._audio_chunks.append(chunk)
                except Exception:
                    break
            stream.stop_stream()
            stream.close()
            p.terminate()
        except ImportError:
            # pyaudio not available — simulate silence
            while self._recording:
                self._audio_chunks.append(np.zeros(self._chunk_size, dtype=np.float32))
                time.sleep(self._chunk_size / self._sample_rate)

    async def _process_turn(self, audio: np.ndarray) -> str:
        """处理完整一轮对话：STT → LLM → 存储历史。

        Args:
            audio: 完整录音数据 (float32, 16kHz)。

        Returns:
            AI 回复文本，空字符串表示跳过。
        """
        # 1. STT
        if self._stt is None:
            self._stt = STTEngine()
        text = self._stt.transcribe(audio, self._sample_rate)
        if not text.strip():
            return ""

        # 2. 收集帧
        frames: list[str] = []
        if self._frame_grabber is not None:
            current = self._frame_grabber.get_current()
            if current is not None:
                frames.append(current.base64)
            max_frames = settings.max_frames_per_llm_call
            recent = self._frame_grabber.get_recent(max_frames - len(frames))
            for f in recent:
                if f.base64 not in frames:
                    frames.append(f.base64)

        # 本轮帧引用
        captured: list[CapturedFrame] = []
        if self._frame_grabber is not None:
            cf = self._frame_grabber.get_current()
            if cf is not None:
                captured.append(cf)

        # 3. LLM
        if self._llm is None:
            self._llm = QwenVisionLLM()
        history = self.get_history()
        reply, _, _ = await self._llm.chat(
            user_text=text,
            frames=frames,
            history=history,
            system_prompt=self._system_prompt,
        )

        # 4. 存储历史
        turn = Turn(
            user_text=text,
            assistant_text=reply,
            frames=captured,
            timestamp=time.time(),
        )
        self._history.append(turn)

        return reply

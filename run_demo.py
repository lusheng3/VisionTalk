# -*- coding: utf-8 -*-
"""VisionTalk 演示脚本。

用法:
    python run_demo.py

操作:
    r     → 开始录音 (最长60秒，按 Enter 提前结束)
    s     → 跳过录音，手动输入文字
    q     → 退出
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import msvcrt
import numpy as np
import pyaudio
from backend.llm import QwenVisionLLM
from backend.stt import STTEngine
from backend.frame_grabber import FrameGrabber

SAMPLE_RATE = 16000
CHUNK = 1024
MAX_RECORD_SECONDS = 60


def record_audio(max_seconds: int = MAX_RECORD_SECONDS) -> np.ndarray | None:
    """录音，最长 max_seconds 秒。按 Enter 停止。

    返回 float32 音频数组，用户取消返回 None。
    """
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        is_int16 = True
    except Exception:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        is_int16 = False

    chunks = []
    max_chunks = int(SAMPLE_RATE / CHUNK * max_seconds)
    print(f"  🎤 录音中... 按 Enter 停止 (最长 {max_seconds}s)", end="", flush=True)

    try:
        for i in range(max_chunks):
            # 检查是否有按键
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b"\r":  # Enter
                    print("\r  录音停止 (手动)", " " * 20)
                    break

            data = stream.read(CHUNK, exception_on_overflow=False)
            chunks.append(data)

            # 每秒更新进度
            if i % int(SAMPLE_RATE / CHUNK) == 0:
                elapsed = i * CHUNK / SAMPLE_RATE
                print(f"\r  🎤 录音中... {elapsed:.0f}s 按 Enter 停止", end="", flush=True)
        else:
            print(f"\r  录音完成 ({max_seconds}s 上限)", " " * 10)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if not chunks:
        return None

    if is_int16:
        audio = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(b"".join(chunks), dtype=np.float32)

    return audio


def main():
    print("=" * 50)
    print("VisionTalk Demo")
    print("=" * 50)

    # 启动摄像头
    print("\n启动摄像头...")
    fg = FrameGrabber()
    try:
        fg.start()
        print("  摄像头已就绪")
    except Exception as e:
        print(f"  摄像头启动失败: {e}，将以无画面模式运行")
        fg = None

    # 初始化模块
    stt = STTEngine()
    llm = QwenVisionLLM()

    print("准备就绪！")
    print("-" * 50)
    print("  r → 开始录音 (按 Enter 停止)")
    print("  s → 手动输入文字")
    print("  q → 退出")
    print("-" * 50)

    retry_count = 0

    try:
        while True:
            cmd = input("\n>>> ").strip().lower()

            if cmd == "q":
                break

            # ── 获取用户输入 ──
            if cmd == "s":
                text = input("  输入文字: ").strip()
                if not text:
                    print("  输入为空，跳过")
                    continue
            elif cmd == "r":
                # 录音
                if fg:
                    fg.set_mode("speech")
                audio = record_audio()
                if fg:
                    fg.set_mode("silence")

                if audio is None or len(audio) < SAMPLE_RATE * 0.3:
                    print("  录音太短，请重试")
                    continue

                print(f"  音频: {len(audio)/SAMPLE_RATE:.1f}s, 转写中...")

                # STT
                text = stt.transcribe(audio, SAMPLE_RATE)
                if not text.strip():
                    retry_count += 1
                    if retry_count >= 3:
                        print(f"  ⚠️  连续 3 次未识别到语音，请检查麦克风。可按 s 手动输入")
                        retry_count = 0
                    else:
                        print(f"  ⚠️  未识别到语音 ({retry_count}/3)，请重试")
                    continue
                retry_count = 0
                print(f"  转写: {text}")
            else:
                print("  请输入 r (录音) / s (打字) / q (退出)")
                continue

            # ── 收集帧 ──
            frames: list[str] = []
            if fg:
                cf = fg.get_current()
                if cf:
                    frames.append(cf.base64)
                for f in fg.get_recent(4):
                    if f.base64 not in frames:
                        frames.append(f.base64)

            # ── LLM ──
            import asyncio
            print("  🤖 思考中...")

            async def call_llm():
                return await llm.chat(
                    user_text=text,
                    frames=frames,
                    system_prompt="你是一个有用的视觉助手。你可以看到摄像头画面，但只在用户明确询问画面内容时才描述（如'你看到了什么'）。其他情况直接回答问题，不主动描述画面。回答简洁，不超过三句话。",
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            reply, in_tok, out_tok = loop.run_until_complete(call_llm())

            print(f"\n  🤖 AI: {reply}")
            if frames:
                print(f"  (发送 {len(frames)} 帧, tokens: {in_tok} in / {out_tok} out)")

    except KeyboardInterrupt:
        print("\n\n退出...")
    finally:
        if fg:
            fg.stop()
        print("再见！")


if __name__ == "__main__":
    main()

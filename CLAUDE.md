# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

VisionTalk is a browser-based AI vision chat app. User opens a web page, camera and microphone are captured by the browser, audio is transcribed locally (faster-whisper), frames + text are sent to Qwen vision LLM via DashScope API.

## Commands

```bash
# Web app (main entry point)
python backend/server.py                # Start at http://localhost:8765

# CLI demo (keyboard PTT, no browser)
python run_demo.py                      # r=record, s=type text, q=quit

# Run all tests
python backend/test_config.py           # Config defaults
python backend/test_vad.py              # VAD silence detection
python backend/test_stt.py              # STT edge cases (loads whisper model)
python backend/test_llm.py              # LLM message assembly (no API call)

# Audio diagnostics
python debug_audio.py                   # Test mic + STT independently
```

## Architecture

Two modes share the same STT/LLM backend:

```
Web mode (primary):                    CLI mode (legacy):
  browser (getUserMedia)                 terminal (pyaudio + opencv)
    → WebSocket → server.py                → dialog.py
      → stt.py (faster-whisper)              → stt.py
      → llm.py (Qwen DashScope)              → llm.py
```

**Web pipeline** (`backend/server.py` + `frontend/index.html`):
- Frontend: vanilla HTML/JS, getUserMedia for camera+mic, Canvas → JPEG frames, AudioContext → PCM → WAV encode → WebSocket JSON
- Backend: single `/ws` WebSocket endpoint, WAV decode via `wave` stdlib, STT → LLM, structured logging with timing to both console and `logs/visiontalk-YYYYMMDD.log`
- Protocol: `ptt_start` → `audio`{wav base64} + `ptt_stop`{frames[]} → `transcript` → `response`

**CLI pipeline** (`run_demo.py`, `backend/dialog.py`):
- `msvcrt`-based non-blocking recording, PTT via `r` key, Enter to stop, max 60s
- `dialog.py` holds all modules but has known threading issues with pyaudio — prefer `run_demo.py` direct approach

## Key modules

| File | Role | Notes |
|------|------|------|
| `backend/server.py` | FastAPI + WebSocket | Main entry for web app |
| `frontend/index.html` | Entire frontend | Single file, no framework |
| `backend/stt.py` | STTEngine | faster-whisper tiny, CPU int8, downloads ~70MB on first run |
| `backend/llm.py` | QwenVisionLLM | DashScope OpenAI-compatible API, single model no abstraction |
| `backend/config.py` | Settings | Loads from `.env`, `pydantic-settings` |
| `backend/frame_grabber.py` | Timer-based camera capture | speech/silence dual mode, unused by web mode |
| `backend/vad.py` | VADDetector only | Silero VAD, SpeechSegmenter removed (PTT replaces it) |
| `backend/dialog.py` | PTT orchestrator | Used by CLI demo only, has pyaudio threading bugs |

## Configuration

Copy `.env.example` to `.env` and set `DASHSCOPE_API_KEY=sk-xxx`. Default model is `qwen3.7-plus`. `.env` is gitignored.

## Design docs

- `docs/superpowers/specs/2026-06-13-architecture-refactor-design.md` — Lean pipeline refactor
- `docs/superpowers/specs/2026-06-13-web-interface-design.md` — Web interface design
- `docs/superpowers/plans/` — Implementation plans

## Notes

- `cli.py` and old `models/` directory were removed during architecture refactor
- `frame_grabber.py`, `vad.py`, `dialog.py` are CLI-mode only; web mode doesn't use them
- The `@` prefix in commit messages was a bash here-string artifact — fixed via `git filter-branch`
- Logs go to `logs/visiontalk-YYYYMMDD.log` (gitignored), showing [1/3] audio → [2/3] STT → [3/3] LLM

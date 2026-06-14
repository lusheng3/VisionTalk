"""视觉路由器 — 用 DeepSeek-V4-Flash 判断用户问题是否需要摄像头画面。"""
import time
import logging
from openai import AsyncOpenAI
from backend.config import settings

log = logging.getLogger("VisionTalk")

_client: AsyncOpenAI | None = None

ROUTER_PROMPT = """判断用户的问题是否需要查看摄像头画面才能回答。
- 需要画面（用户提到了画面中的物体、场景、动作等）→ 回复 YES
- 不需要画面（纯文字问题，如自我介绍、闲聊、知识问答等）→ 回复 NO
只回复 YES 或 NO，不要其他内容。"""


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
    return _client


async def needs_visual(user_text: str) -> bool:
    """判断用户问题是否需要摄像头画面。"""
    t0 = time.time()
    try:
        client = _get_client()
        resp = await client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": user_text},
            ],
            max_tokens=10,
        )
        raw = resp.choices[0].message.content or ""
        result = raw.strip().upper()
        elapsed = time.time() - t0
        # Debug: log full response on empty
        if not result:
            log.warning(f"[Router] ⚠️ empty response → fallback YES")
            log.warning(f"[Router] debug: finish_reason={resp.choices[0].finish_reason}, model={resp.model}")
            log.warning(f"[Router] debug: full message={resp.choices[0].message}")
            return True
        is_visual = "YES" in result
        log.info(f"[Router] 🧠 DeepSeek → {result} | 耗时 {elapsed:.2f}s | text='{user_text[:30]}'")
        return is_visual
    except Exception as e:
        log.error(f"[Router] failed: {e}")
        # Fallback: assume visual (safe side — send frames)
        return True

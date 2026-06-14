"""通义千问视觉语言模型调用。

通过 DashScope OpenAI 兼容接口，单模型无抽象层。
"""
from openai import AsyncOpenAI

from backend.config import settings


class QwenVisionLLM:
    """通义千问多模态模型适配器。

    通过 OpenAI 兼容接口调用 DashScope。
    支持模型: qwen-max, qwen-plus
    """

    def __init__(self, model: str | None = None):
        self._model = model or settings.default_model
        self._client: AsyncOpenAI | None = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> AsyncOpenAI:
        """Lazy init OpenAI client. Only called when making API requests."""
        if self._client is None:
            api_key = settings.dashscope_api_key
            if not api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY not set. "
                    "Configure it in .env or set the environment variable."
                )
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        return self._client

    async def chat(
        self,
        user_text: str,
        frames: list[str],
        history: list[dict] | None = None,
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> tuple[str, int, int]:
        """调用通义千问 API。

        Args:
            user_text: 用户当前说的话。
            frames: base64 编码的图片列表（当前帧 + 历史帧，内部截断至 5 张）。
            history: 历史对话 [{"role":"user","content":"..."}, ...]。
            system_prompt: 系统提示词。
            max_tokens: 最大输出 token 数。

        Returns:
            (回复文本, input_tokens, output_tokens)
        """
        history = history or []
        messages = self._build_messages(user_text, frames, history, system_prompt)
        client = self._get_client()

        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return (
            choice.message.content or "",
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )

    def _build_messages(
        self,
        user_text: str,
        frames: list[str],
        history: list[dict],
        system_prompt: str,
    ) -> list[dict]:
        """组装 OpenAI 兼容的多模态消息列表。

        顺序: system_prompt → history → current (text + images)
        """
        messages: list[dict] = []

        # 1. System prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 2. Dialog history (text-only turns)
        for turn in history:
            messages.append(turn)

        # 3. Current user message: text + images
        content_parts: list[dict] = []

        # Text
        text = user_text
        if frames:
            text += "\n以上画面按时间顺序排列，第一张是最新的。"
        content_parts.append({"type": "text", "text": text})

        # Images (max 5)
        max_frames = settings.max_frames_per_llm_call
        for b64 in frames[:max_frames]:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        messages.append({"role": "user", "content": content_parts})

        return messages

    async def chat_stream(
        self,
        user_text: str,
        frames: list[str],
        history: list[dict] | None = None,
        system_prompt: str = "",
        max_tokens: int = 1024,
    ):
        """流式调用通义千问 API，逐 token 返回。

        Yields:
            str: 每个文本增量 token。
        """
        history = history or []
        messages = self._build_messages(user_text, frames, history, system_prompt)
        client = self._get_client()

        stream = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

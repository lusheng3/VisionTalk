"""应用配置管理。从 .env 文件和环境变量加载配置。"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """全局配置，自动从 .env 和环境变量加载。"""

    # API Keys
    dashscope_api_key: str = ""
    deepseek_api_key: str = ""

    # 默认模型
    default_model: str = "qwen3.7-plus"

    # 服务端口
    port: int = 8765

    # VAD 灵敏度 (0-1, 越大越激进地判定为语音)
    vad_threshold: float = 0.5

    # 对话历史保留轮数
    max_history_turns: int = 10

    # 截帧最大宽度 (像素)
    frame_max_width: int = 768

    # 说话时抓帧间隔 (秒)
    frame_interval_speech: float = 0.5

    # 静音时抓帧间隔 (秒)
    frame_interval_silence: float = 3.0

    # 每次发给 LLM 的最大帧数
    max_frames_per_llm_call: int = 5

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()

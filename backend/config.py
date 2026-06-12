"""应用配置管理。从 .env 文件和环境变量加载配置。"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """全局配置，自动从 .env 和环境变量加载。"""

    # API Keys
    dashscope_api_key: str = ""
    zhipu_api_key: str = ""
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

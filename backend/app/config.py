from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    app_name: str = "BetaBuddy"
    app_version: str = "1.0.0"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite+aiosqlite:///./betabuddy.db"

    # File storage
    upload_dir: Path = Path("uploads")
    max_upload_size: int = 500 * 1024 * 1024  # 500MB

    # Video processing
    supported_video_formats: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    default_fps: int = 30

    # Pose estimation
    pose_model_complexity: int = 1  # 0, 1, or 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    llm_timeout: int = 60

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

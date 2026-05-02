"""Centralised configuration loaded from environment variables.

All other modules should import ``settings`` from here rather than reading
``os.environ`` directly. This keeps secrets handling and defaults in one place.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from .env at process start."""

    # --- LLM ---
    gemini_api_key: str = Field(default="", description="Google AI Studio key")

    # --- Optional news APIs (we'll see in Week 1 if we need these) ---
    finnhub_api_key: str = Field(default="")
    marketaux_api_key: str = Field(default="")

    # --- HuggingFace (optional, helps with rate limits on model downloads) ---
    huggingface_hub_token: str = Field(default="", alias="HUGGINGFACE_HUB_TOKEN")

    # --- Storage ---
    duckdb_path: Path = Field(default=Path("data/morningedge.duckdb"))

    # --- App ---
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

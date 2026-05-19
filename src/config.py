"""Pydantic settings for DocMind RAG."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chroma_persist_dir: str = "./chroma_db"
    embed_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama3.2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)


settings = Settings()

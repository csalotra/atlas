from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "Atlas"
    environment: str = "development"
    log_level: str = "INFO"

    database_url: str

    litellm_base_url: str = "http://localhost:4000/v1"
    litellm_master_key: str
    litellm_model: str = "atlas-default"
    llm_timeout: float = 60.0
    llm_max_retries: int = 2

    embedding_model: str = "atlas-embedding"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "atlas_chunks"
    embedding_dimensions: int = 1536

    document_storage_dir: Path = Path("data/documents")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

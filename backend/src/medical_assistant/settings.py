from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PFL_",
        extra="ignore",
        env_file=Path(__file__).resolve().parents[3] / "tmp/work/runtime/.env",
        env_file_encoding="utf-8",
    )

    database_url: str = "postgresql+psycopg://assistant:assistant@127.0.0.1:5439/assistant"
    read_database_url: str = ""
    data_dir: Path = Path("tmp/work/runtime/app-data")
    codex_home: Path = Path("tmp/work/runtime/codex-home")
    codex_binary: Path = Path("/opt/homebrew/bin/codex")
    provider: str = "codex"
    model: str = "gpt-6-sol"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    reranker_threads: int = 4
    app_origin: str = "http://127.0.0.1:8080"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "http://127.0.0.1:3045"
    langfuse_public_url: str = "http://127.0.0.1:3045"
    launch_token: str = ""
    service_token: str = ""
    bridge_url: str = ""
    bridge_token: str = ""
    provider_timeout_seconds: int = 300
    run_timeout_seconds: int = 650
    max_active_runs: int = Field(default=4, ge=1, le=32)
    max_upload_bytes: int = 26214400
    max_pdf_pages: int = 200
    parser_timeout_seconds: int = 120
    embedding_threads: int = 4
    frontend_dir: Path = Path("frontend/dist")


def get_settings() -> Settings:
    return Settings()

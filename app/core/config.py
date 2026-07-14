from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM / gateway
    llm_provider: str = "mock"  # mock|openai|anthropic|gemini|ollama|groq
    api_key: str = "dev-workbench-key"
    rate_limit_per_minute: int = 120

    corpus_dir: Path = ROOT / "data" / "corpus"
    prompts_dir: Path = ROOT / "prompts"
    db_path: Path = ROOT / "data" / "runtime" / "workbench.db"
    golden_path: Path = ROOT / "data" / "golden" / "rag_cases.jsonl"
    dataset_path: Path = ROOT / "data" / "datasets" / "churn_demo.csv"
    experiments_dir: Path = ROOT / "experiments"
    registry_dir: Path = ROOT / "experiments" / "registry"
    catalog_path: Path = ROOT / "data" / "catalog.json"
    lineage_path: Path = ROOT / "data" / "lineage.json"

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    ollama_base_url: str = "http://127.0.0.1:11434"

    rag_top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 80
    agent_max_steps: int = 6
    require_human_approval: bool = False
    vector_backend: str = "memory"
    parent_child_chunking: bool = False
    rag_use_rrf: bool = True
    api_roles_json: str = '{"dev-workbench-key":"admin","viewer-key":"viewer","operator-key":"operator"}'
    langchain_api_key: str | None = None
    reports_dir: Path = ROOT / "reports"


@lru_cache
def get_settings() -> Settings:
    import os

    s = Settings()
    # Env aliases for convenience
    if os.getenv("AIWB_GATEWAY") or os.getenv("AIWB_LLM_PROVIDER"):
        s.llm_provider = os.getenv("AIWB_LLM_PROVIDER") or os.environ.get("AIWB_GATEWAY", s.llm_provider)
    if os.getenv("AIWB_API_KEY"):
        s.api_key = os.environ["AIWB_API_KEY"]
    if os.getenv("OPENAI_API_KEY"):
        s.openai_api_key = os.environ["OPENAI_API_KEY"]
    if os.getenv("OPENAI_BASE_URL"):
        s.openai_base_url = os.environ["OPENAI_BASE_URL"]
    if os.getenv("OPENAI_MODEL"):
        s.openai_model = os.environ["OPENAI_MODEL"]
    if os.getenv("ANTHROPIC_API_KEY"):
        s.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
    if os.getenv("GEMINI_API_KEY"):
        s.gemini_api_key = os.environ["GEMINI_API_KEY"]
    if os.getenv("GROQ_API_KEY"):
        s.groq_api_key = os.environ["GROQ_API_KEY"]
    if os.getenv("OLLAMA_BASE_URL"):
        s.ollama_base_url = os.environ["OLLAMA_BASE_URL"]
    if os.getenv("AIWB_CORPUS_DIR"):
        s.corpus_dir = Path(os.environ["AIWB_CORPUS_DIR"])
    if os.getenv("AIWB_DB_PATH"):
        s.db_path = Path(os.environ["AIWB_DB_PATH"])
    if os.getenv("AIWB_PROMPTS_DIR"):
        s.prompts_dir = Path(os.environ["AIWB_PROMPTS_DIR"])
    if os.getenv("AIWB_GOLDEN_PATH"):
        s.golden_path = Path(os.environ["AIWB_GOLDEN_PATH"])
    if os.getenv("AIWB_REQUIRE_APPROVAL", "").lower() in ("1", "true", "yes"):
        s.require_human_approval = True
    if os.getenv("AIWB_VECTOR_BACKEND"):
        s.vector_backend = os.getenv("AIWB_VECTOR_BACKEND", s.vector_backend)
    if os.getenv("AIWB_PARENT_CHILD", "").lower() in ("1", "true", "yes"):
        s.parent_child_chunking = True
    if os.getenv("LANGCHAIN_API_KEY") and not s.langchain_api_key:
        s.langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    if os.getenv("AIWB_API_ROLES_JSON"):
        s.api_roles_json = os.getenv("AIWB_API_ROLES_JSON")
    return s

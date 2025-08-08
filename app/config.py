from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    openai_api_key: str = Field(alias="OPENAI_API_KEY")

    model_router: str = Field(default="gpt-5-nano", alias="MODEL_ROUTER")
    model_planner: str = Field(default="gpt-5", alias="MODEL_PLANNER")
    model_vision: str = Field(default="gpt-5", alias="MODEL_VISION")
    model_vision_fallback: str = Field(default="gpt-4o", alias="MODEL_VISION_FALLBACK")
    model_rag: str = Field(default="gpt-5-mini", alias="MODEL_RAG")
    model_web_synth: str = Field(default="gpt-5-mini", alias="MODEL_WEB_SYNTH")
    model_data: str = Field(default="gpt-5", alias="MODEL_DATA")
    model_critic_first: str = Field(default="gpt-5-mini", alias="MODEL_CRITIC_FIRST")
    model_critic_final: str = Field(default="gpt-5", alias="MODEL_CRITIC_FINAL")
    model_summary: str = Field(default="gpt-5-mini", alias="MODEL_SUMMARY")
    model_local_fallback: str = Field(default="gpt-5-nano", alias="MODEL_LOCAL_FALLBACK")

    embedding_model: str = Field(default="text-embedding-3-large", alias="EMBEDDING_MODEL")

    llama_gguf_path: str = Field(default="", alias="LLAMA_GGUF_PATH")

    # Costos (opcional)
    price_in_gpt5: float = Field(default=10, alias="PRICE_IN_GPT5")
    price_out_gpt5: float = Field(default=20, alias="PRICE_OUT_GPT5")
    price_in_gpt5_mini: float = Field(default=2, alias="PRICE_IN_GPT5_MINI")
    price_out_gpt5_mini: float = Field(default=8, alias="PRICE_OUT_GPT5_MINI")
    price_in_gpt5_nano: float = Field(default=0.3, alias="PRICE_IN_GPT5_NANO")
    price_out_gpt5_nano: float = Field(default=1.2, alias="PRICE_OUT_GPT5_NANO")
    price_in_gpt4o: float = Field(default=5, alias="PRICE_IN_GPT4O")
    price_out_gpt4o: float = Field(default=15, alias="PRICE_OUT_GPT4O")
    price_embeddings: float = Field(default=0.13, alias="PRICE_EMBEDDINGS")

    # Storage
    base_dir: Path = Path(".").resolve()
    cache_dir: Path = base_dir / ".cache"
    vectorstore_dir: Path = base_dir / ".vectorstore"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
settings.cache_dir.mkdir(parents=True, exist_ok=True)
settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)

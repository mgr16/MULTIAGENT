from typing import List, Dict, Optional
from app.config import settings
from app.logging_setup import get_logger

log = get_logger("local_llm")

try:
    from llama_cpp import Llama
except Exception as e:
    Llama = None  # type: ignore

_llm = None

def get_local_llm() -> Optional["Llama"]:
    global _llm
    if _llm:
        return _llm
    if not settings.llama_gguf_path or not Llama:
        log.info("Local Llama not configured or llama-cpp not installed; returning None.")
        return None
    _llm = Llama(model_path=settings.llama_gguf_path, n_ctx=4096, n_threads=4)
    return _llm

def local_generate(prompt: str, max_tokens: int = 512) -> str:
    llm = get_local_llm()
    if not llm:
        raise RuntimeError("Local LLM not available")
    out = llm(prompt=prompt, max_tokens=max_tokens, temperature=0.2)
    return out["choices"][0]["text"]

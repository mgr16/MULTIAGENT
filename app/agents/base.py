from __future__ import annotations
from typing import Optional, Dict, Any, List
from app.blackboard import Blackboard
from app.logging_setup import get_logger
from app.models.openai_llm import call_llm, LLMUsage
from app.caching.llm_cache import LLMCache
from app.config import settings

class BaseAgent:
    name: str = "base"

    def __init__(self, bb: Blackboard, cache: Optional[LLMCache] = None):
        self.bb = bb
        self.log = get_logger(self.name)
        self.cache = cache or LLMCache(settings.base_dir / ".llm_cache.sqlite")

    # --- helpers ---
    async def _record_usage(self, usage: LLMUsage):
        key = "usage_log"
        log = await self.bb.get(key) or []
        log.append(usage.model + f"|in={usage.input_tokens}|out={usage.output_tokens}|${usage.cost_usd:.6f}")
        await self.bb.set(key, log)

    def _cache_get(self, model: str, messages: List[Dict[str, str]], extra: Dict[str, Any] | None = None):
        k = self.cache.key(model, messages, extra)
        return self.cache.get(k)

    def _cache_set(self, model: str, messages: List[Dict[str, str]], value: Any, extra: Dict[str, Any] | None = None, ttl: int = 3600):
        k = self.cache.key(model, messages, extra)
        self.cache.set(k, value, ttl)

    # --- interface ---
    async def act(self) -> None:
        raise NotImplementedError

    # --- policy ---
    def choose_model(self, importance: str = "medium", default: Optional[str] = None) -> str:
        if default:
            return default
        if importance == "high":
            return settings.model_planner  # gpt-5 por defecto
        if importance == "low":
            return settings.model_local_fallback  # gpt-5-nano por defecto
        return settings.model_rag  # gpt-5-mini por defecto

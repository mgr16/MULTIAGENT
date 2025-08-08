from __future__ import annotations
from typing import Dict, List
from pydantic import ValidationError
from app.agents.base import BaseAgent
from app.models.openai_llm import call_llm
from app.guardrails.schemas import RouterOutput

SYS = (
    "You are a routing expert. Classify the user's query into a domain and suggest agents. "
    "Always output a strict JSON object matching the RouterOutput schema."
)

class RouterAgent(BaseAgent):
    name = "router"

    async def act(self):
        user_query = await self.bb.get("input") or ""
        msgs: List[Dict[str, str]] = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"Query:\n{user_query}\n\nOutput fields: domain, confidence, suggested_agents"},
        ]
        model = self.choose_model(importance="low", default="gpt-5-nano")
        cached = self._cache_get(model, msgs)
        if cached:
            await self.bb.set("router_output", cached[0])
            return

        text, usage = call_llm(model, msgs, json_object=True, temperature=0.1, max_tokens=300)
        try:
            out = RouterOutput.model_validate_json(text)
        except ValidationError:
            # Fallback robusto
            out = RouterOutput(domain="general", confidence=0.6, suggested_agents=["planner"])
        await self._record_usage(usage)
        await self.bb.set("router_output", out.model_dump())
        self._cache_set(model, msgs, out.model_dump(), ttl=1800)

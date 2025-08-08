from __future__ import annotations
from app.agents.base import BaseAgent
from app.models.local_llm import local_generate

class LocalTextAgent(BaseAgent):
    name = "local_text"

    async def act(self):
        query = await self.bb.get("input") or ""
        try:
            out = local_generate(f"Answer briefly and helpfully:\n\n{query}\n")
        except Exception:
            out = "Local model not available."
        await self.bb.set("local_text_result", out.strip())

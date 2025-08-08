from __future__ import annotations
from typing import List, Dict, Any
from app.agents.base import BaseAgent
from app.models.openai_llm import call_llm

SYS = (
    "Generate possible hypotheses or sub-questions as a JSON list of strings. "
    "Focus on clarifying unknowns that would improve the final answer."
)

class HypothesisAgent(BaseAgent):
    name = "hypothesis"

    async def act(self):
        query = await self.bb.get("input") or ""
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"Query: {query}\nOutput just a JSON list of short hypotheses."},
        ]
        text, usage = call_llm(self.choose_model("low"), msgs, json_object=True, temperature=0.4, max_tokens=200)
        await self._record_usage(usage)
        try:
            items = eval(text) if text.strip().startswith("[") else []
        except Exception:
            items = []
        await self.bb.set("hypotheses", items)

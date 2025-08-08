from __future__ import annotations
from typing import List, Dict, Any
from pydantic import ValidationError
from app.agents.base import BaseAgent
from app.models.openai_llm import call_llm
from app.guardrails.schemas import CriticVerdict, CriticIssue

SYS = (
    "You are an adversarial critic. Check for: missing citations, contradictions, math errors, and format errors. "
    "Return strict JSON CriticVerdict with: confidence (0-1), conflicts (int), issues: list of {{kind, detail}}."
)

TEMPLATE = """Question:
{query}

Draft answer:
{answer}

Evidence:
- RAG citations: {rag}
- Web snippets (urls only): {web_urls}
"""

class CriticAgent(BaseAgent):
    name = "critic"

    async def act(self):
        query = await self.bb.get("input") or ""
        draft = await self.bb.get("draft_answer") or {"final_answer": ""}
        rag = await self.bb.get("rag_citations") or []
        web = await self.bb.get("web_snippets") or []
        web_urls = [w.get("url","") for w in web]

        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": TEMPLATE.format(query=query, answer=draft, rag=rag, web_urls=web_urls)},
        ]
        model = self.choose_model(importance="medium", default="gpt-5-mini")
        text, usage = call_llm(model, msgs, json_object=True, temperature=0.0, max_tokens=600)
        await self._record_usage(usage)
        try:
            verdict = CriticVerdict.model_validate_json(text)
        except ValidationError:
            verdict = CriticVerdict(confidence=0.5, conflicts=0, issues=[])
        await self.bb.set("critic_verdict", verdict.model_dump())

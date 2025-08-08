from __future__ import annotations
from typing import List, Dict, Any
from app.agents.base import BaseAgent
from app.models.openai_llm import call_llm

SYS = (
    "You are a careful quantitative analyst. If given tables/numbers, compute and check. "
    "Return a JSON object with fields: {'analysis': string, 'key_numbers': [{'name':..., 'value':...}], 'assumptions': [...] }"
)

class DataAnalysisAgent(BaseAgent):
    name = "data"

    async def act(self):
        query = await self.bb.get("input") or ""
        web = await self.bb.get("web_snippets") or []
        rag = await self.bb.get("rag_context") or {}
        vision = await self.bb.get("vision_struct") or {}
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"Question:\n{query}\n\nEvidence:web={web}\nrag={rag}\nvision={vision}\n"}
        ]
        text, usage = call_llm(self.choose_model(importance="high", default="gpt-5"), msgs, json_object=True, temperature=0.0, max_tokens=1000)
        await self._record_usage(usage)
        try:
            obj = eval(text) if text.strip().startswith("{") else {"analysis": text, "key_numbers": [], "assumptions": []}
        except Exception:
            obj = {"analysis": text, "key_numbers": [], "assumptions": []}
        await self.bb.set("analysis_numeric", obj)

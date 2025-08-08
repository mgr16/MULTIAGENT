from __future__ import annotations
from typing import List, Dict, Any
from pydantic import ValidationError
from app.agents.base import BaseAgent
from app.models.openai_llm import call_llm
from app.guardrails.schemas import SummaryOutput
from app.logging_setup import get_logger

log = get_logger("summary")

SYS = (
    "You produce a precise, sourced answer. Output JSON with fields final_answer (string) and citations (list of strings). "
    "If you cannot, output a concise answer anyway."
)

TEMPLATE = """Question:\n{query}\n\nEvidence (trimmed):\nRAG snippets: {rag}\nWeb snippets: {web}\nVision: {vision}\nAnalysis: {analysis}\n\nInstructions:\n- Provide a concise but complete answer.\n- Cite up to 5 supporting sources (filenames or URLs).\n- If evidence insufficient, state limitations.\n"""

class SummaryAgent(BaseAgent):
    name = "summary"

    async def act(self):
        query = await self.bb.get("input") or ""
        rag = await self.bb.get("rag_citations") or []
        # Trim rag: keep first 5 entries concise
        rag_trim: List[str] = []
        for r in rag[:5]:
            rs = str(r)
            if len(rs) > 160:
                rs = rs[:157] + "..."
            rag_trim.append(rs)
        web = await self.bb.get("web_snippets") or []
        web_trim = [{"url": w.get("url",""), "summary": (w.get("summary") or "")[:220]} for w in web[:5]]
        vision = await self.bb.get("vision_struct") or {}
        analysis = await self.bb.get("analysis_numeric") or {}
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": TEMPLATE.format(query=query, rag=rag_trim, web=web_trim, vision=vision, analysis=analysis)},
        ]
        model = self.choose_model(importance="medium", default="gpt-5-mini")
        # Avoid json_object True (was causing empty responses); let model free-form.
        text, usage = call_llm(model, msgs, json_object=False, temperature=0.2, max_tokens=400)
        await self._record_usage(usage)

        parsed: SummaryOutput | None = None
        raw = text.strip()
        if raw:
            if raw.lstrip().startswith("{") and raw.rstrip().endswith("}"):
                try:
                    parsed = SummaryOutput.model_validate_json(raw)
                except ValidationError as e:
                    log.warning(f"JSON validation failed; using raw text. Error: {e}")
        else:
            raw = "No answer generated."

        if parsed is None:
            # Extract citations heuristically (URLs or filenames inside parentheses / after http)
            cands: List[str] = []
            import re
            for m in re.findall(r"https?://\S+", raw):
                cands.append(m.rstrip(').,;'))
            if not cands:
                cands = [w.get("url","") for w in web_trim if w.get("url")] + rag_trim
            cands = [c for c in cands if c][:5]
            parsed = SummaryOutput(final_answer=raw, citations=cands)

        await self.bb.set("draft_answer", parsed.model_dump())
        if parsed.final_answer.strip():
            current_final = await self.bb.get("final_answer")
            if not current_final or not str(current_final).strip():
                await self.bb.set("final_answer", parsed.final_answer.strip())

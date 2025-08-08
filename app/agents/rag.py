from __future__ import annotations
from typing import List, Dict, Tuple
from pydantic import ValidationError
from app.agents.base import BaseAgent
from app.guardrails.schemas import RAGPassage, RAGOutput
from app.rag.vectorstore import SimpleFAISS
from app.rag.chunking import chunk_text
from app.config import settings
from pathlib import Path
import orjson

class RAGAgent(BaseAgent):
    name = "rag"

    def __init__(self, bb, cache=None, corpus_dir: Path | None = None):
        super().__init__(bb, cache)
        self.vs = SimpleFAISS(settings.vectorstore_dir)
        self.corpus_dir = corpus_dir or (settings.base_dir / "corpus")
        # Lazy load if exists
        try:
            self.vs.load("default")
        except Exception:
            pass

    def ingest_directory(self):
        if not self.corpus_dir.exists():
            return
        texts, metas = [], []
        for p in self.corpus_dir.rglob("*"):
            if p.suffix.lower() not in {".txt", ".md"}:
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for ch in chunk_text(txt, target_tokens=1200):
                texts.append(ch)
                metas.append({"source": str(p)})
        if texts:
            self.vs.add_texts(texts, metas)
            self.vs.save("default")

    async def act(self):
        # build if empty
        if self.vs.index is None:
            self.ingest_directory()

        query = await self.bb.get("input") or ""
        results = self.vs.search(query, k=20) if self.vs.index else []
        # Simple rerank by score (already IP), keep top 5
        top = results[:5]
        passages = [
            RAGPassage(source=m.get("source","unknown"), text=t) for (t, m, _d) in top
        ]
        out = RAGOutput(passages=passages)
        await self.bb.set("rag_context", out.model_dump())

        # store citations as list of sources
        citations = list({p.source for p in passages})
        await self.bb.set("rag_citations", citations)

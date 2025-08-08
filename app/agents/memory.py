from __future__ import annotations
from typing import Any, Dict, List
from app.agents.base import BaseAgent
from pathlib import Path
import orjson
from datetime import datetime
from app.config import settings

class MemoryAgent(BaseAgent):
    name = "memory"
    path = settings.base_dir / ".memory.jsonl"

    def append(self, record: Dict[str, Any]):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as f:
            f.write(orjson.dumps(record) + b"\n")

    async def act(self):
        query = await self.bb.get("input") or ""
        answer = await self.bb.get("final_answer") or (await self.bb.get("draft_answer")) or {}
        rec = {"ts": datetime.utcnow().isoformat(), "q": query, "a": answer}
        self.append(rec)
        await self.bb.set("memory_last", rec)

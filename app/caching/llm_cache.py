from pathlib import Path
from typing import Any, Dict, List, Optional
from .sqlite_kv import SQLiteKV
from app.utils.hashing import prompt_key

class LLMCache:
    def __init__(self, path: Path):
        self.db = SQLiteKV(path)

    def key(self, model: str, messages: List[Dict[str, str]], extra: Optional[Dict] = None) -> str:
        return prompt_key(model, messages, extra)

    def get(self, k: str):
        return self.db.get(k)

    def set(self, k: str, v: Any, ttl: int = 3600):
        self.db.set(k, v, ttl)

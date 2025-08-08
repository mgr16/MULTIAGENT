from pathlib import Path
from .sqlite_kv import SQLiteKV

class WebCache:
    def __init__(self, path: Path):
        self.db = SQLiteKV(path)

    def get(self, k: str):
        return self.db.get(k)

    def set(self, k: str, v, ttl: int = 3600):
        self.db.set(k, v, ttl)

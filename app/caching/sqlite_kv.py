import sqlite3
from pathlib import Path
from typing import Optional, Tuple
import time
import orjson

class SQLiteKV:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS kv (
            k TEXT PRIMARY KEY,
            v BLOB NOT NULL,
            ttl INTEGER NOT NULL,
            ts INTEGER NOT NULL
        )
        """)
        self.conn.commit()

    def set(self, k: str, v, ttl: int = 3600):
        cur = self.conn.cursor()
        cur.execute("REPLACE INTO kv (k, v, ttl, ts) VALUES (?, ?, ?, ?)",
                    (k, orjson.dumps(v), ttl, int(time.time())))
        self.conn.commit()

    def get(self, k: str) -> Optional[Tuple[any, int]]:
        cur = self.conn.cursor()
        cur.execute("SELECT v, ttl, ts FROM kv WHERE k= ?", (k,))
        row = cur.fetchone()
        if not row:
            return None
        v, ttl, ts = row
        if int(time.time()) - ts > ttl:
            self.delete(k)
            return None
        return orjson.loads(v), ts

    def delete(self, k: str):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM kv WHERE k= ?", (k,))
        self.conn.commit()

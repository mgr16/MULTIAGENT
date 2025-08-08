from typing import Any, Dict, Optional
import asyncio

class Blackboard:
    """Thread-safe / async-safe store."""
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._store[key] = value

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            return self._store.get(key)

    async def exists(self, key: str) -> bool:
        async with self._lock:
            return key in self._store

    async def dump(self) -> Dict[str, Any]:
        async with self._lock:
            return dict(self._store)

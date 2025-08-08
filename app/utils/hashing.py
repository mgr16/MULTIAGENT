import hashlib
import orjson
from typing import Any, Dict, List

def stable_hash(obj: Any) -> str:
    data = orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(data).hexdigest()

def prompt_key(model: str, messages: List[Dict[str, str]], extra: Dict | None = None) -> str:
    payload = {"model": model, "messages": messages, "extra": extra or {}}
    return stable_hash(payload)

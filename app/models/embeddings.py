from typing import List
from openai import OpenAI
from app.config import settings
from app.caching.embeddings_cache import EmbeddingsCache
from app.utils.hashing import stable_hash

_client = OpenAI(api_key=settings.openai_api_key)
_cache = EmbeddingsCache(settings.base_dir / ".emb_cache.sqlite")

def embed_texts(texts: List[str], model: str | None = None) -> List[List[float]]:
    model = model or settings.embedding_model
    key = stable_hash({"m": model, "t": texts})
    cached = _cache.get(key)
    if cached:
        return cached[0]
    resp = _client.embeddings.create(model=model, input=texts)
    vecs = [d.embedding for d in resp.data]
    _cache.set(key, vecs)
    return vecs

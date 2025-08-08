import aiohttp
from typing import Optional
from app.caching.web_cache import WebCache
from app.config import settings
from app.logging_setup import get_logger

log = get_logger("http_client")
_cache = WebCache(settings.base_dir / ".web_cache.sqlite")

async def fetch_text(url: str, ttl: int = 3600, timeout: float = 10.0) -> Optional[str]:
    cached = _cache.get(url)
    if cached:
        return cached[0]
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"}) as r:
                if r.status != 200:
                    log.warning(f"HTTP {r.status} for {url}")
                    return None
                text = await r.text()
                _cache.set(url, text, ttl)
                return text
    except Exception as e:
        log.warning(f"fetch_text error for {url}: {e}")
        return None

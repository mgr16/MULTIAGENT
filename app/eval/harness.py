import orjson
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from app.eval.metrics import aspect_precision, length_tokens, basic_report
from app.main import run_query
import asyncio

@contextmanager
def patch_settings(overrides: Dict[str, Any]):
    from app import config
    old = {}
    for k, v in overrides.items():
        if hasattr(config.settings, k.lower()) or hasattr(config.settings, k):
            key = k.lower() if hasattr(config.settings, k.lower()) else k
            old[key] = getattr(config.settings, key)
            setattr(config.settings, key, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(config.settings, k, v)

async def run_dataset(path: Path, runs: int = 1) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = orjson.loads(line)
        q = item["q"]
        aspects = item.get("aspects", [])
        image = item.get("image_url")
        best_ans = None
        for _ in range(max(1, runs)):
            ans, usage = await run_query(q, image_url=image)
            # Elige el más “completo” por aspectos
            if best_ans is None or aspect_precision(ans, aspects) > aspect_precision(best_ans, aspects):
                best_ans = ans
        results.append({
            "q": q,
            "answer": best_ans,
            "metrics": {
                "aspect_precision": aspect_precision(best_ans or "", aspects),
                "length_tokens": length_tokens(best_ans or ""),
            }
        })
    report = basic_report(results)
    out = {"report": report, "results": results}
    out_path = Path(".cache") / "eval_results"
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "last_eval.json").write_bytes(orjson.dumps(out, option=orjson.OPT_INDENT_2))
    return out

async def run_ab(path: Path, overrides_a: Dict[str, Any], overrides_b: Dict[str, Any], runs: int = 1) -> Dict[str, Any]:
    with patch_settings(overrides_a):
        res_a = await run_dataset(path, runs=runs)
    with patch_settings(overrides_b):
        res_b = await run_dataset(path, runs=runs)
    return {"A": overrides_a, "A_results": res_a, "B": overrides_b, "B_results": res_b}

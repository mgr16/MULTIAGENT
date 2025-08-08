from typing import List, Dict, Any
import re

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def aspect_precision(answer: str, aspects: List[str]) -> float:
    if not aspects:
        return 0.0
    ans = normalize(answer)
    hits = 0
    for a in aspects:
        if normalize(a) in ans:
            hits += 1
    return hits / len(aspects)

def length_tokens(answer: str) -> int:
    # aproximación grosera
    return len(answer.split())

def basic_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    precs = [r["metrics"]["aspect_precision"] for r in results]
    lens = [r["metrics"]["length_tokens"] for r in results]
    return {
        "n": len(results),
        "aspect_precision_avg": sum(precs)/len(precs),
        "answer_len_avg_tokens": sum(lens)/len(lens),
    }

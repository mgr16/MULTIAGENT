import re
from typing import List

def split_into_sentences(text: str) -> List[str]:
    # Regex simple para frases. En prod usa spacy si quieres.
    parts = re.split(r"(?<=[\.!?])\s+", text.strip())
    return [p for p in parts if p]

def chunk_text(text: str, target_tokens: int = 1200, overlap_sentences: int = 2) -> List[str]:
    sents = split_into_sentences(text)
    chunks, cur = [], []
    cur_len = 0
    for s in sents:
        cur.append(s)
        cur_len += len(s.split())
        if cur_len >= target_tokens:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:]
            cur_len = len(" ".join(cur).split())
    if cur:
        chunks.append(" ".join(cur))
    return chunks

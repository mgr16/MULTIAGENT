from typing import List, Tuple
import faiss
import numpy as np
from pathlib import Path
import orjson
from app.models.embeddings import embed_texts

class SimpleFAISS:
    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.index = None
        self.texts: List[str] = []
        self.meta: List[dict] = []

    def add_texts(self, texts: List[str], metadatas: List[dict]):
        vecs = embed_texts(texts)
        arr = np.array(vecs).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(arr.shape[1])
        self.index.add(arr)
        self.texts.extend(texts)
        self.meta.extend(metadatas)

    def save(self, name: str):
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.persist_dir / f"{name}.faiss"))
        with open(self.persist_dir / f"{name}.json", "wb") as f:
            f.write(orjson.dumps({"texts": self.texts, "meta": self.meta}))

    def load(self, name: str):
        self.index = faiss.read_index(str(self.persist_dir / f"{name}.faiss"))
        data = orjson.loads((self.persist_dir / f"{name}.json").read_bytes())
        self.texts = data["texts"]; self.meta = data["meta"]

    def search(self, query: str, k: int = 20) -> List[Tuple[str, dict, float]]:
        qv = np.array(embed_texts([query])[0]).astype("float32")
        D, I = self.index.search(qv.reshape(1, -1), k)
        out = []
        for d, i in zip(D[0], I[0]):
            if i < 0 or i >= len(self.texts): 
                continue
            out.append((self.texts[i], self.meta[i], float(d)))
        return out

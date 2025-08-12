import time
import hashlib
import logging
from typing import List
import ollama
from langchain.embeddings.base import Embeddings


class OllamaEmbedding(Embeddings):
    def __init__(self, model_name: str, retries: int = 3, delay: float = 1.0):
        self.model_name = model_name
        self.retries = retries
        self.delay = delay
        self.cache = {}

    def _safe_embed(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("Empty text")
        text = text.strip().replace("\n", " ")
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        for attempt in range(self.retries):
            try:
                result = ollama.embed(model=self.model_name, input=text)
                emb = result.get("embeddings", [])
                if emb:
                    self.cache[text_hash] = emb[0]
                    return emb[0]
            except Exception as e:
                logging.warning(f"Embedding retry {attempt+1}: {e}")
                time.sleep(self.delay)
        raise RuntimeError("Failed to embed")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._safe_embed(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._safe_embed(query)
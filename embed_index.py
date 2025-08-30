import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class EmbedIndex:
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 index_path: str = "data/index/index.faiss",
                 meta_path: str = "data/index/meta.pkl"):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metadatas = []

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

    def build(self, chunks: list):
        """
        Build FAISS index from chunks and save metadata.
        """
        self._ensure_dir()

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings.astype('float32'))
        self.metadatas = chunks

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Index or metadata not found; build index first.")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadatas = pickle.load(f)

    def query(self, query_text: str, top_k: int = 5):
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build().")
        q_emb = self.model.encode([query_text], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
        D, I = self.index.search(q_emb.astype('float32'), top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            md = self.metadatas[int(idx)].get("metadata", {})
            results.append({
                "score": float(dist),
                "metadata": {
                    "source": md.get("source", "Unknown source"),
                    "page_id": md.get("page_id", "N/A"),
                    "chunk_id": md.get("chunk_id", "N/A"),
                    "text": self.metadatas[int(idx)].get("text", "")
                }
            })
        return results

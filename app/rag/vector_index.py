"""A small persistent vector index.

Uses FAISS when it is installed and falls back to brute-force cosine in NumPy
otherwise. At the corpus sizes this app produces (hundreds to a few thousand
vectors) the NumPy path is genuinely fast enough - FAISS is an optimisation, not
a requirement, so it must not be a hard dependency.
"""

import json
import os
import threading

import numpy as np

try:  # optional acceleration
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class VectorIndex:
    """Append-only index of (vector, payload) pairs with cosine search."""

    def __init__(self, path, dim):
        self.path = path
        self.dim = dim
        self._lock = threading.Lock()
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._payloads = []
        self._faiss = faiss.IndexFlatIP(dim) if _HAS_FAISS else None
        self._load()

    # --- persistence ------------------------------------------------------

    @property
    def _vec_path(self):
        return f"{self.path}.npy"

    @property
    def _meta_path(self):
        return f"{self.path}.jsonl"

    def _load(self):
        if not (os.path.exists(self._vec_path) and os.path.exists(self._meta_path)):
            return
        try:
            vectors = np.load(self._vec_path)
            with open(self._meta_path, encoding="utf-8") as handle:
                payloads = [json.loads(line) for line in handle if line.strip()]
        except Exception as exc:
            print(f"[RAG] Could not load index at {self.path} ({exc}); starting empty.")
            return

        # A dimension change means the embedder changed; the old vectors are
        # meaningless in the new space, so drop them rather than mix spaces.
        if vectors.ndim != 2 or vectors.shape[1] != self.dim or len(vectors) != len(payloads):
            print(f"[RAG] Index at {self.path} does not match dim={self.dim}; rebuilding.")
            return

        self._vectors = vectors.astype(np.float32)
        self._payloads = payloads
        if self._faiss is not None and len(self._vectors):
            self._faiss.add(self._vectors)

    def _persist(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        np.save(self._vec_path, self._vectors)
        with open(self._meta_path, "w", encoding="utf-8") as handle:
            for payload in self._payloads:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # --- api --------------------------------------------------------------

    def __len__(self):
        return len(self._payloads)

    @property
    def backend(self):
        return "faiss" if self._faiss is not None else "numpy"

    def add(self, vector, payload):
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vector.shape[1]}")
        with self._lock:
            self._vectors = np.vstack([self._vectors, vector])
            self._payloads.append(payload)
            if self._faiss is not None:
                self._faiss.add(vector)
            self._persist()

    def search(self, vector, k=3, where=None, min_score=0.0):
        """Return [(score, payload)] sorted best-first.

        `where` filters payloads before ranking, which is how results stay
        scoped to one dataset.
        """
        with self._lock:
            if not len(self._payloads):
                return []

            query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
            candidates = range(len(self._payloads))
            if where is not None:
                candidates = [i for i in candidates if where(self._payloads[i])]
                if not candidates:
                    return []

            # FAISS cannot express the payload filter, so it is only worth using
            # when every vector is a candidate.
            if self._faiss is not None and len(candidates) == len(self._payloads):
                scores, indices = self._faiss.search(query, min(k, len(self._payloads)))
                pairs = [(float(s), int(i)) for s, i in zip(scores[0], indices[0]) if i >= 0]
            else:
                index_list = list(candidates)
                scores = (self._vectors[index_list] @ query.T).ravel()
                order = np.argsort(-scores)[:k]
                pairs = [(float(scores[j]), index_list[j]) for j in order]

            return [(s, self._payloads[i]) for s, i in pairs if s >= min_score]

    def clear(self):
        with self._lock:
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)
            self._payloads = []
            if self._faiss is not None:
                self._faiss.reset()
            for path in (self._vec_path, self._meta_path):
                if os.path.exists(path):
                    os.remove(path)

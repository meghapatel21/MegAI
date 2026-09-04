"""Pluggable text embeddings.

Three backends, same interface:

  fastembed  - default. ONNX, runs locally, no API key, no torch (~200MB).
  azure      - Azure OpenAI text-embedding-3-small, for when the model should
               not ship inside the container.
  hashing    - dependency-free fallback. Deterministic character n-gram hashing;
               weaker than a real model but keeps tests and offline runs working
               with zero downloads.

All backends return L2-normalised float32, so cosine similarity is a dot product.
"""

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

from app import config


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed(self, texts):
        """Embed a list of strings -> (len(texts), dim) L2-normalised float32."""

    def embed_one(self, text):
        return self.embed([text])[0]


def _normalise(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder(Embedder):
    """Character n-gram hashing with sublinear term weighting.

    No model, no download, no network. Captures lexical overlap well enough to
    retrieve near-duplicate questions, which is most of what query memory needs.
    """

    name = "hashing"

    def __init__(self, dim=512):
        self.dim = dim

    @staticmethod
    def _features(text):
        text = re.sub(r"\s+", " ", (text or "").lower().strip())
        tokens = re.findall(r"[a-z0-9_]+", text)
        features = list(tokens)
        features += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]  # bigrams
        for token in tokens:
            padded = f"#{token}#"
            features += [padded[i:i + 3] for i in range(max(len(padded) - 2, 0))]
        return features

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = {}
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                counts[index] = counts.get(index, 0.0) + sign
            for index, value in counts.items():
                # Opposing signs can cancel exactly; log(0) would poison the whole
                # vector with NaN and make every similarity NaN.
                if value == 0:
                    continue
                # Sublinear scaling: a term repeated 10x is not 10x as important.
                out[row, index] = np.sign(value) * (1.0 + np.log(abs(value)))
        return _normalise(out)


class FastEmbedEmbedder(Embedder):
    """BAAI/bge-small-en-v1.5 through fastembed (ONNX runtime, no torch)."""

    name = "fastembed"

    def __init__(self, model_name=None):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name or config.FASTEMBED_MODEL)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(list(self._model.embed(list(texts))))


class AzureOpenAIEmbedder(Embedder):
    """Azure OpenAI embeddings, for keeping the model out of the container."""

    name = "azure"

    def __init__(self):
        from openai import AzureOpenAI

        if not (config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_API_KEY):
            raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required.")

        self._client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
        self._deployment = config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        self.dim = 1536

    def embed(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        response = self._client.embeddings.create(model=self._deployment, input=list(texts))
        return _normalise([item.embedding for item in response.data])


_embedder = None


def get_embedder():
    """Build the configured embedder once, falling back rather than crashing.

    A missing optional dependency or an unreachable endpoint degrades retrieval
    quality; it must not take the whole pipeline down.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    backend = (config.EMBEDDING_BACKEND or "fastembed").lower()
    try:
        if backend == "azure":
            _embedder = AzureOpenAIEmbedder()
        elif backend == "hashing":
            _embedder = HashingEmbedder()
        else:
            _embedder = FastEmbedEmbedder()
        print(f"[RAG] Embedder: {_embedder.name} (dim={_embedder.dim})")
    except Exception as exc:
        print(f"[RAG] '{backend}' embedder unavailable ({exc}); using hashing fallback.")
        _embedder = HashingEmbedder()
    return _embedder


def reset_embedder():
    """Test hook - drops the cached instance so config changes take effect."""
    global _embedder
    _embedder = None

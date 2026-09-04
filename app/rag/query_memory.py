"""Self-populating few-shot corpus.

Every analysis that runs cleanly is embedded and stored with the code that
produced it. A later question retrieves the nearest past successes and passes
them to the Analysis Agent as worked examples, so the system gets better at a
given workbook the more it is used - without anyone curating a prompt.

Examples from the *same* workbook are the valuable ones (identical column
names), so they are ranked ahead of examples from other files.
"""

import hashlib
import os
import time

from app import config
from app.rag.embedder import get_embedder
from app.rag.vector_index import VectorIndex

_index = None


def schema_fingerprint(profile):
    """Stable id for a column layout, so re-uploads of the same file match."""
    parts = []
    for sheet in (profile or {}).get("sheets", []):
        columns = ",".join(sorted(c["name"] for c in sheet.get("columns", [])))
        parts.append(f"{sheet['name']}:{columns}")
    return hashlib.blake2b("|".join(sorted(parts)).encode("utf-8"), digest_size=8).hexdigest()


def _get_index():
    global _index
    if _index is None:
        embedder = get_embedder()
        _index = VectorIndex(os.path.join(config.RAG_DIR, "query_memory"), embedder.dim)
        print(f"[RAG] Query memory: {len(_index)} examples ({_index.backend})")
    return _index


def remember(question, code, profile, row_count):
    """Store a successful analysis. Never raises - this is a side effect."""
    if not config.RAG_QUERY_MEMORY_ENABLED or not question or not code:
        return
    try:
        index = _get_index()
        vector = get_embedder().embed_one(question)
        index.add(vector, {
            "question": question,
            "code": code,
            "schema": schema_fingerprint(profile),
            "row_count": int(row_count),
            "stored_at": time.time(),
        })
    except Exception as exc:
        print(f"[RAG] Could not store query memory: {exc}")


def recall(question, profile, k=None):
    """Return past successes most similar to `question`, best first."""
    if not config.RAG_QUERY_MEMORY_ENABLED or not question:
        return []
    k = k or config.RAG_TOP_K
    try:
        index = _get_index()
        if not len(index):
            return []

        vector = get_embedder().embed_one(question)
        fingerprint = schema_fingerprint(profile)

        # Over-fetch, then prefer same-schema hits: an example written against
        # these exact columns is worth more than a closer-worded one that is not.
        hits = index.search(vector, k=k * 4, min_score=config.RAG_MIN_SCORE)
        ranked = sorted(
            hits,
            key=lambda hit: (hit[1].get("schema") == fingerprint, hit[0]),
            reverse=True,
        )

        out, seen = [], set()
        for score, payload in ranked:
            if payload["question"].strip().lower() in seen:
                continue
            seen.add(payload["question"].strip().lower())
            out.append({
                "question": payload["question"],
                "code": payload["code"],
                "score": round(score, 3),
                "same_file": payload.get("schema") == fingerprint,
            })
            if len(out) >= k:
                break
        return out
    except Exception as exc:
        print(f"[RAG] Could not search query memory: {exc}")
        return []


def stats():
    try:
        index = _get_index()
        return {"examples": len(index), "backend": index.backend}
    except Exception:
        return {"examples": 0, "backend": "unavailable"}


def reset():
    """Test hook."""
    global _index
    if _index is not None:
        _index.clear()
    _index = None

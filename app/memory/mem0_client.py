"""Long-term memory across sessions.

Falls back to a no-op object when Mem0 cannot start, so a missing key or a
read-only filesystem degrades the feature instead of taking the app down.
"""

import os

from app import config

_mem_config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "path": config.MEMORY_DIR,
            "collection_name": "agentic_bi_memories",
        },
    },
    "llm": {
        "provider": "groq",
        "config": {
            "model": config.GROQ_MODEL,
            "api_key": config.GROQ_API_KEY,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    },
}


class _NullMemory:
    """Used when Mem0 is unavailable. Silent by design - it is not an error path."""

    def add(self, *args, **kwargs):
        return None

    def search(self, *args, **kwargs):
        return []


memory = _NullMemory()

if config.GROQ_API_KEY:
    try:
        from mem0 import Memory

        os.makedirs(config.MEMORY_DIR, exist_ok=True)
        memory = Memory.from_config(_mem_config)
        print("[MEMORY] Mem0 initialised.")
    except Exception as exc:
        print(f"[MEMORY] Mem0 unavailable, continuing without long-term memory: {exc}")
else:
    print("[MEMORY] No GROQ_API_KEY - long-term memory disabled.")

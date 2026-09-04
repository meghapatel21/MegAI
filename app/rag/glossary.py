"""Business glossary retrieval.

A company's definition of "active customer" or "ARR" is a policy decision, not
something a model should guess per question. Users upload a glossary once; the
matching definitions are retrieved and injected so the generated code follows
the organisation's rule rather than the model's assumption.

Accepted formats:
  * Markdown / plain text  - "Term: definition", "Term = definition", "## Term"
  * CSV / Excel            - a term column and a definition column
"""

import io
import os
import re

import pandas as pd

from app import config
from app.rag.embedder import get_embedder
from app.rag.vector_index import VectorIndex

_indexes = {}

_TERM_COLUMNS = ("term", "name", "metric", "concept", "field", "kpi")
_DEF_COLUMNS = ("definition", "description", "meaning", "formula", "logic", "notes")


class GlossaryError(Exception):
    pass


def parse_glossary(raw, filename):
    """Parse an uploaded glossary into [{term, definition}]."""
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".csv", ".xlsx", ".xls", ".xlsm"):
        try:
            frame = (pd.read_csv(io.BytesIO(raw)) if ext == ".csv"
                     else pd.read_excel(io.BytesIO(raw)))
        except Exception as exc:
            raise GlossaryError(f"Could not read the glossary: {exc}") from exc

        lowered = {str(c).strip().lower(): c for c in frame.columns}
        term_col = next((lowered[k] for k in _TERM_COLUMNS if k in lowered), None)
        def_col = next((lowered[k] for k in _DEF_COLUMNS if k in lowered), None)

        # Fall back to positional columns when the headers are not recognised.
        if term_col is None or def_col is None:
            if len(frame.columns) < 2:
                raise GlossaryError(
                    "A tabular glossary needs at least two columns: term and definition."
                )
            term_col, def_col = frame.columns[0], frame.columns[1]

        entries = [
            {"term": str(row[term_col]).strip(), "definition": str(row[def_col]).strip()}
            for _, row in frame.iterrows()
            if str(row[term_col]).strip() and str(row[def_col]).strip()
        ]

    elif ext in (".md", ".txt", ".markdown", ""):
        entries = _parse_text(raw.decode("utf-8", errors="replace"))
    else:
        raise GlossaryError(
            f"Unsupported glossary type '{ext}'. Use .md, .txt, .csv, or .xlsx."
        )

    if not entries:
        raise GlossaryError(
            "No definitions found. Use lines like 'ARR: monthly_price * 12 for active subscriptions'."
        )
    return entries


def _parse_text(text):
    entries, heading, buffer = [], None, []

    def flush():
        if heading and buffer:
            body = " ".join(b.strip() for b in buffer if b.strip())
            if body:
                entries.append({"term": heading, "definition": body})

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        markdown_heading = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if markdown_heading:
            flush()
            heading, buffer = markdown_heading.group(1).strip(), []
            continue

        # "Term: definition" / "Term = definition" / "- Term: definition"
        inline = re.match(r"^[-*•]?\s*([A-Za-z][\w %/()\-\.]{1,60}?)\s*[:=]\s+(.+)$", stripped)
        if inline:
            flush()
            heading, buffer = None, []
            entries.append({"term": inline.group(1).strip(), "definition": inline.group(2).strip()})
            continue

        if heading:
            buffer.append(stripped)

    flush()
    return entries


def _index_for(dataset_id):
    embedder = get_embedder()
    key = (dataset_id, embedder.dim)
    if key not in _indexes:
        path = os.path.join(config.RAG_DIR, "glossary", dataset_id)
        _indexes[key] = VectorIndex(path, embedder.dim)
    return _indexes[key]


def store_glossary(dataset_id, entries):
    """Replace the glossary attached to a dataset. Returns the entry count."""
    index = _index_for(dataset_id)
    index.clear()

    embedder = get_embedder()
    # Embedding "term: definition" beats the term alone - the definition text
    # carries the vocabulary a user's question is likely to overlap with.
    vectors = embedder.embed([f"{e['term']}: {e['definition']}" for e in entries])
    for vector, entry in zip(vectors, entries):
        index.add(vector, entry)
    return len(entries)


def _mentions(question, term):
    """Whole-word (or whole-phrase) mention of a defined term in the question."""
    if not term:
        return False
    return re.search(rf"(?<!\w){re.escape(term.strip())}(?!\w)", question, re.IGNORECASE) is not None


def lookup(dataset_id, question, k=None):
    """Definitions relevant to a question, best first.

    Hybrid retrieval. A glossary term is a defined token, so a literal mention
    is a certainty, not a similarity: "what is our ARR by tier?" must return the
    ARR definition regardless of how the surrounding wording embeds. Semantic
    search then adds the terms the user paraphrased rather than named.
    """
    if not config.RAG_GLOSSARY_ENABLED or not question:
        return []
    k = k or config.RAG_TOP_K

    try:
        index = _index_for(dataset_id)
        if not len(index):
            return []

        results, seen = [], set()

        # 1. Lexical: exact term mentions always win.
        for _, payload in index.search(
            get_embedder().embed_one(question), k=len(index), min_score=-1.0
        ):
            if _mentions(question, payload["term"]) and payload["term"] not in seen:
                seen.add(payload["term"])
                results.append({
                    "term": payload["term"],
                    "definition": payload["definition"],
                    "score": 1.0,
                    "matched": "term mentioned",
                })

        # 2. Semantic: paraphrases of a term the user did not name outright.
        for score, payload in index.search(
            get_embedder().embed_one(question), k=k, min_score=config.RAG_MIN_SCORE
        ):
            if payload["term"] in seen:
                continue
            seen.add(payload["term"])
            results.append({
                "term": payload["term"],
                "definition": payload["definition"],
                "score": round(score, 3),
                "matched": "semantic",
            })

        return results[:k]
    except Exception as exc:
        print(f"[RAG] Glossary lookup failed: {exc}")
        return []


def glossary_size(dataset_id):
    try:
        return len(_index_for(dataset_id))
    except Exception:
        return 0


def reset():
    """Test hook."""
    for index in _indexes.values():
        index.clear()
    _indexes.clear()

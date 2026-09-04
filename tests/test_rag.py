"""Embeddings, vector index, query memory, and glossary retrieval.

Tests force the dependency-free hashing embedder so they never download a model
or call an API.
"""

import numpy as np
import pytest

from app import config
from app.rag import glossary, query_memory
from app.rag.embedder import HashingEmbedder, get_embedder, reset_embedder
from app.rag.vector_index import VectorIndex


@pytest.fixture(autouse=True)
def use_hashing_embedder(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "hashing")
    reset_embedder()
    query_memory.reset()
    glossary.reset()
    yield
    query_memory.reset()
    glossary.reset()
    reset_embedder()


# --- embedder -------------------------------------------------------------

def test_embeddings_are_normalised():
    vectors = HashingEmbedder().embed(["total revenue by region", "hello"])
    assert vectors.shape == (2, 512)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embeddings_are_deterministic():
    a = HashingEmbedder().embed_one("revenue by region")
    b = HashingEmbedder().embed_one("revenue by region")
    assert np.allclose(a, b)


def test_similar_text_scores_higher_than_unrelated():
    embedder = HashingEmbedder()
    query = embedder.embed_one("total revenue by region")
    similar = embedder.embed_one("show total revenue for each region")
    unrelated = embedder.embed_one("average support ticket resolution time")
    assert float(query @ similar) > float(query @ unrelated)


def test_falls_back_when_backend_unavailable(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "azure")  # no credentials set
    reset_embedder()
    assert get_embedder().name == "hashing"


# --- vector index ---------------------------------------------------------

def test_index_persists_across_instances(tmp_path):
    path = str(tmp_path / "idx")
    embedder = HashingEmbedder()

    index = VectorIndex(path, embedder.dim)
    index.add(embedder.embed_one("revenue by region"), {"tag": "a"})
    index.add(embedder.embed_one("tickets by agent"), {"tag": "b"})
    assert len(index) == 2

    reopened = VectorIndex(path, embedder.dim)
    assert len(reopened) == 2
    hits = reopened.search(embedder.embed_one("revenue per region"), k=1)
    assert hits and hits[0][1]["tag"] == "a"


def test_index_rebuilds_when_dimension_changes(tmp_path):
    path = str(tmp_path / "idx")
    VectorIndex(path, 512).add(np.ones(512, dtype=np.float32) / np.sqrt(512), {"tag": "old"})
    # A different embedder means the stored vectors live in another space.
    assert len(VectorIndex(path, 384)) == 0


def test_index_filters_by_payload(tmp_path):
    embedder = HashingEmbedder()
    index = VectorIndex(str(tmp_path / "idx"), embedder.dim)
    index.add(embedder.embed_one("revenue by region"), {"owner": "x"})
    index.add(embedder.embed_one("revenue by region"), {"owner": "y"})

    hits = index.search(embedder.embed_one("revenue by region"), k=5,
                        where=lambda p: p["owner"] == "y")
    assert len(hits) == 1 and hits[0][1]["owner"] == "y"


def test_empty_index_returns_nothing(tmp_path):
    index = VectorIndex(str(tmp_path / "idx"), 512)
    assert index.search(np.zeros(512, dtype=np.float32), k=3) == []


# --- query memory ---------------------------------------------------------

PROFILE = {"sheets": [{"name": "sales", "columns": [{"name": "region"}, {"name": "revenue"}]}]}
OTHER_PROFILE = {"sheets": [{"name": "hr", "columns": [{"name": "employee"}, {"name": "salary"}]}]}


def test_recall_returns_similar_past_analysis():
    query_memory.remember("total revenue by region", "result = df.groupby('region')...", PROFILE, 4)
    query_memory.remember("average ticket time by agent", "result = df.groupby('agent')...", PROFILE, 9)

    hits = query_memory.recall("show revenue for each region", PROFILE)
    assert hits
    assert hits[0]["question"] == "total revenue by region"
    assert hits[0]["same_file"] is True


def test_recall_prefers_same_workbook():
    query_memory.remember("revenue by region", "SAME_FILE_CODE", PROFILE, 4)
    query_memory.remember("revenue by region", "OTHER_FILE_CODE", OTHER_PROFILE, 4)

    hits = query_memory.recall("revenue by region", PROFILE)
    assert hits[0]["code"] == "SAME_FILE_CODE"
    assert hits[0]["same_file"] is True


def test_recall_is_empty_before_anything_is_stored():
    assert query_memory.recall("anything at all", PROFILE) == []


def test_schema_fingerprint_is_order_independent():
    a = {"sheets": [{"name": "s", "columns": [{"name": "x"}, {"name": "y"}]}]}
    b = {"sheets": [{"name": "s", "columns": [{"name": "y"}, {"name": "x"}]}]}
    assert query_memory.schema_fingerprint(a) == query_memory.schema_fingerprint(b)


def test_schema_fingerprint_differs_across_layouts():
    assert query_memory.schema_fingerprint(PROFILE) != query_memory.schema_fingerprint(OTHER_PROFILE)


def test_remember_never_raises(monkeypatch):
    monkeypatch.setattr(query_memory, "_get_index", lambda: (_ for _ in ()).throw(IOError("disk full")))
    query_memory.remember("q", "code", PROFILE, 1)  # must not propagate


# --- glossary -------------------------------------------------------------

def test_parses_markdown_glossary():
    text = b"""# Metrics
ARR: monthly_price * 12 for active subscriptions only
Churn = cancelled customers / active at period start

## Active Customer
A customer with at least one order in the trailing 90 days.
"""
    entries = glossary.parse_glossary(text, "glossary.md")
    terms = {e["term"] for e in entries}
    assert "ARR" in terms and "Churn" in terms and "Active Customer" in terms
    arr = next(e for e in entries if e["term"] == "ARR")
    assert "monthly_price * 12" in arr["definition"]


def test_parses_csv_glossary():
    entries = glossary.parse_glossary(
        b"Term,Definition\nARR,monthly price times twelve\nMRR,monthly recurring revenue\n",
        "glossary.csv",
    )
    assert len(entries) == 2
    assert entries[0]["term"] == "ARR"


def test_parses_csv_without_recognised_headers():
    entries = glossary.parse_glossary(b"a,b\nARR,twelve months\n", "g.csv")
    assert entries[0]["term"] == "ARR"


def test_rejects_unsupported_and_empty_glossary():
    with pytest.raises(glossary.GlossaryError):
        glossary.parse_glossary(b"data", "glossary.pdf")
    with pytest.raises(glossary.GlossaryError):
        glossary.parse_glossary(b"nothing parseable here at all", "glossary.md")


def test_lookup_retrieves_relevant_definition():
    entries = glossary.parse_glossary(
        b"ARR: monthly_price multiplied by twelve for active subscriptions\n"
        b"Warehouse Capacity: total pallet slots across all storage locations\n",
        "glossary.md",
    )
    glossary.store_glossary("ds1", entries)

    hits = glossary.lookup("ds1", "what is our ARR by tier?")
    assert hits and hits[0]["term"] == "ARR"


def test_glossary_is_scoped_per_dataset():
    glossary.store_glossary("ds1", [{"term": "ARR", "definition": "annual recurring revenue"}])
    assert glossary.glossary_size("ds1") == 1
    assert glossary.glossary_size("ds2") == 0
    assert glossary.lookup("ds2", "ARR") == []


def test_storing_glossary_replaces_the_previous_one():
    glossary.store_glossary("ds1", [{"term": "A", "definition": "first"}])
    glossary.store_glossary("ds1", [{"term": "B", "definition": "second"}])
    assert glossary.glossary_size("ds1") == 1


def test_lookup_matches_term_mentioned_literally():
    """A named term must be retrieved regardless of embedder quality."""
    glossary.store_glossary("ds3", [
        {"term": "ARR", "definition": "monthly_price multiplied by twelve for active subs"},
        {"term": "Warehouse Capacity", "definition": "total pallet slots across locations"},
    ])
    hits = glossary.lookup("ds3", "what is our ARR by tier?")
    assert hits[0]["term"] == "ARR"
    assert hits[0]["matched"] == "term mentioned"


def test_lookup_matches_multiword_term():
    glossary.store_glossary("ds4", [
        {"term": "Warehouse Capacity", "definition": "total pallet slots across locations"},
    ])
    hits = glossary.lookup("ds4", "show warehouse capacity by site")
    assert hits and hits[0]["term"] == "Warehouse Capacity"


def test_lookup_does_not_match_substring_of_another_word():
    glossary.store_glossary("ds5", [{"term": "ARR", "definition": "annual recurring revenue"}])
    # "arrears" contains "arr" but is a different word.
    hits = [h for h in glossary.lookup("ds5", "show me arrears by customer")
            if h["matched"] == "term mentioned"]
    assert hits == []

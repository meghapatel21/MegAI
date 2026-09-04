"""RAG Agent - retrieves grounding context before any code is written.

Two retrievers, because they answer different questions:

  glossary      "what does *this company* mean by ARR?"      - curated, authoritative
  query memory  "how did we compute something like this before?" - learned, self-populating

Both are advisory. Retrieval failure degrades the answer's grounding; it never
blocks the pipeline.
"""

from app import config
from app.rag import glossary, query_memory


def _format_examples(examples):
    blocks = []
    for example in examples:
        origin = "same file" if example["same_file"] else "another file"
        blocks.append(
            f"# Q: {example['question']}   (similarity {example['score']}, {origin})\n"
            f"{example['code']}"
        )
    return "\n\n".join(blocks)


def _format_definitions(definitions):
    return "\n".join(f"- **{d['term']}**: {d['definition']}" for d in definitions)


def run(state):
    question = state.get("corrected_question") or state["question"]
    dataset_id = state.get("dataset_id", "")
    profile = state.get("profile", {})

    definitions = glossary.lookup(dataset_id, question) if config.RAG_GLOSSARY_ENABLED else []
    examples = query_memory.recall(question, profile) if config.RAG_QUERY_MEMORY_ENABLED else []

    state["rag_definitions"] = definitions
    state["rag_examples"] = examples

    sections = []
    if definitions:
        sections.append(
            "# CERTIFIED BUSINESS DEFINITIONS\n"
            "These are the organisation's own definitions. When the question uses one of\n"
            "these terms, compute it exactly this way - do not substitute your own.\n"
            + _format_definitions(definitions)
        )
    if examples:
        sections.append(
            "# PREVIOUS ANALYSES THAT SUCCEEDED\n"
            "Working code from earlier questions. Reuse the idioms and column names where\n"
            "they apply; the current question may still need something different.\n"
            + _format_examples(examples)
        )

    state["rag_context"] = "\n\n".join(sections)

    retrieved = []
    if definitions:
        retrieved.append(f"{len(definitions)} definition(s)")
    if examples:
        retrieved.append(f"{len(examples)} past example(s)")
    print(f"[RAG] Retrieved: {', '.join(retrieved) if retrieved else 'nothing relevant'}")

    return state

"""Profile Agent - loads the uploaded workbook and normalises the question.

Replaces the old metadata_agent, which routed questions across a fixed 15-table
SQL schema. There is no fixed schema any more: the shape of the data comes from
whatever the user uploaded, so this agent reads the real columns and uses them
as the vocabulary for correcting typos and expanding shorthand questions.
"""

import difflib
import json
import re

from app.agents.llm import get_llm
from app.memory.mem0_client import memory
from app.storage.dataset_store import store


def _column_vocabulary(profile: dict) -> list:
    names = []
    for sheet in profile.get("sheets", []):
        names.extend(col["name"] for col in sheet.get("columns", []))
    return names


def _local_correct(question: str, vocabulary: list) -> str:
    """Cheap typo repair against real column names, used when no LLM is configured."""
    if not vocabulary:
        return question

    def fix(match):
        word = match.group(0)
        if len(word) < 4 or word.lower() in {c.lower() for c in vocabulary}:
            return word
        hit = difflib.get_close_matches(word.lower(), vocabulary, n=1, cutoff=0.85)
        return hit[0] if hit else word

    return re.sub(r"[A-Za-z_]+", fix, question)


def _schema_digest(profile: dict) -> str:
    """Compact schema rendering for the prompt."""
    lines = []
    for sheet in profile.get("sheets", []):
        lines.append(f"- {sheet['name']} ({sheet['row_count']:,} rows)")
        for col in sheet["columns"]:
            bits = [f"{col['name']}: {col['dtype']}"]
            if "sample_values" in col:
                bits.append("e.g. " + ", ".join(col["sample_values"][:6]))
            elif "min" in col:
                bits.append(f"range {col['min']:.4g}..{col['max']:.4g}")
            lines.append(f"    {' | '.join(bits)}")
    return "\n".join(lines)


def run(state):
    dataset_id = state.get("dataset_id")
    if not dataset_id:
        raise ValueError("No dataset uploaded. Upload a spreadsheet before asking a question.")

    meta = store.get_meta(dataset_id)
    profile = meta["profile"]
    state["profile"] = profile
    state["schema_digest"] = _schema_digest(profile)
    state["filename"] = meta.get("filename", "uploaded file")

    question = state["question"]
    vocabulary = _column_vocabulary(profile)

    try:
        memory.search(question, user_id=state.get("user_id", "anonymous"))
    except Exception as exc:
        print(f"[MEMORY] search skipped: {exc}")

    llm = get_llm()
    if not llm:
        state["corrected_question"] = _local_correct(question, vocabulary)
        return state

    prompt = f"""You are a Query Interpreter for a spreadsheet analytics tool.
Rewrite the user's question so it is unambiguous and uses the actual column names
from the data below. Fix typos ("revnu" -> "revenue"), expand shorthand
("Feb 2026 region wise" -> "Total revenue by region for February 2026"), and
resolve references to earlier turns ("that", "those") using the history.

Do NOT answer the question. Do NOT invent columns that are not listed.

# WORKBOOK: {state['filename']}
{state['schema_digest']}

# HISTORY
{state.get('history', [])}

# QUESTION
{question}

Return ONLY a JSON object: {{"corrected_question": "..."}}"""

    try:
        raw = llm.invoke(prompt).content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "", 1).strip()
        state["corrected_question"] = json.loads(raw).get("corrected_question", question)
    except Exception as exc:
        print(f"[PROFILE] interpretation failed, using raw question: {exc}")
        state["corrected_question"] = _local_correct(question, vocabulary)

    return state

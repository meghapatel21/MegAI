"""Insight Agent - writes the analyst narrative over a verified result.

Runs only after the Critic has accepted the numbers, and is given the actual
rows rather than a summary, so it describes what is there instead of guessing.
The prompt forbids inventing causes: the data shows *what* happened, and this
agent is not entitled to claim *why*.
"""

import pandas as pd

from app import config
from app.agents.llm import get_llm


def _table_markdown(df, limit=25):
    head = df.head(limit)
    lines = ["| " + " | ".join(str(c) for c in head.columns) + " |",
             "|" + "|".join("---" for _ in head.columns) + "|"]
    for _, row in head.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:,.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > limit:
        lines.append(f"_({len(df) - limit} more rows not shown)_")
    return "\n".join(lines)


def run(state):
    state.setdefault("insight", "")

    if not config.INSIGHT_ENABLED or state.get("error"):
        return state

    df = pd.DataFrame(state.get("result", []))
    if df.empty:
        return state

    llm = get_llm()
    if not llm:
        return state

    definitions = state.get("rag_definitions") or []
    definitions_block = ""
    if definitions:
        definitions_block = "\n# DEFINITIONS APPLIED\n" + "\n".join(
            f"- {d['term']}: {d['definition']}" for d in definitions
        )

    plan = state.get("plan") or []
    plan_block = ""
    if len(plan) > 1:
        plan_block = "\n# THIS ANSWER COMBINED\n" + "\n".join(
            f"{i}. {s['step']}" for i, s in enumerate(plan, 1)
        )

    prompt = f"""You are a data analyst writing up a finding for a colleague.

# QUESTION
{state.get('corrected_question') or state['question']}

# RESULT ({len(df)} rows)
{_table_markdown(df)}
{definitions_block}{plan_block}

Write 2-4 sentences covering what stands out: the leaders and laggards, the
spread, any outlier, and the single most useful follow-up question.

Rules:
- Cite specific numbers from the table. No vague claims.
- State WHAT the data shows. Never assert WHY - you cannot see causes, and
  guessing at them is how a BI tool loses trust. Phrase any hypothesis as a
  question worth checking.
- No preamble ("Here is the analysis"), no bullet points, no headings.
- If the result is a single value, say what it is and what would give it context.
"""

    try:
        state["insight"] = llm.invoke(prompt).content.strip()
    except Exception as exc:
        print(f"[INSIGHT] skipped: {exc}")
        state["insight"] = ""

    return state

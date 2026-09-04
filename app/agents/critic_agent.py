"""Critic Agent - checks whether a successful run actually answered the question.

The Execute Agent only catches code that *raises*. Code that runs cleanly and
returns nonsense - an empty frame because the date filter used the wrong year,
a column of NaN because a join missed, one row where a breakdown was asked for -
was previously reported as a success. This agent closes that gap.

It is careful about one thing: an empty result can be the correct answer
("customers who churned in 2027"). Retrying those wastes calls and still returns
nothing, so the deterministic checks flag suspicion and the model decides
whether the shape actually contradicts the question.
"""

import pandas as pd

from app import config
from app.agents.llm import get_llm, invoke_json


def _mechanical_findings(df, question):
    """Cheap structural checks. Returns a list of human-readable concerns."""
    findings = []

    if df.empty:
        findings.append("the result has no rows")
        return findings

    numeric = df.select_dtypes("number")
    for column in numeric.columns:
        if numeric[column].isna().all():
            findings.append(f"every value in '{column}' is NaN")
        elif numeric[column].isna().mean() > 0.5:
            findings.append(f"over half the values in '{column}' are NaN")

    if len(df) == 1 and any(
        word in question.lower()
        for word in ("by ", "each", "per ", "breakdown", "compare", "trend", "across", "top ")
    ):
        findings.append(
            "the question asks for a breakdown but only one row came back"
        )

    if len(df.columns) == 1 and "compare" in question.lower():
        findings.append("the question asks for a comparison but only one column came back")

    return findings


def run(state):
    # Nothing to critique if the run already failed - the reflexion edge owns that.
    if state.get("error"):
        return state

    state["critique"] = {"ok": True}
    state["warning"] = None

    if not config.CRITIC_ENABLED:
        return state

    df = pd.DataFrame(state.get("result", []))
    question = state.get("current_step") or state.get("corrected_question") or state["question"]
    findings = _mechanical_findings(df, question)

    if not findings:
        return state

    llm = get_llm()
    if not llm:
        # Without a model, surface the concern rather than silently retrying.
        state["critique"] = {"ok": False, "reason": "; ".join(findings)}
        state["warning"] = "; ".join(findings)
        return state

    preview = df.head(5).to_dict("records") if not df.empty else []
    prompt = f"""You review the output of a data analysis and decide whether it
answered the question or whether the code is wrong.

An empty or odd-looking result is NOT automatically wrong - "which orders were
cancelled in 2027" correctly returns nothing if there are none. Only call it
wrong when the code plainly contradicts the question or the data.

# QUESTION
{question}

# COLUMNS IN THE SOURCE DATA
{state.get('schema_digest', '')}

# CODE THAT RAN
{state.get('code', '')}

# RESULT: {len(df)} row(s), columns {list(df.columns)}
{preview}

# AUTOMATED CONCERNS
{chr(10).join('- ' + f for f in findings)}

Reply with ONLY JSON:
{{"verdict": "ok" | "retry",
  "reason": "one sentence explaining the judgement",
  "hint": "if retry, a specific instruction for fixing the code"}}

Use "retry" only when a different query would plausibly do better."""

    result = invoke_json(llm, prompt, default={"verdict": "ok", "reason": "review unavailable"})
    verdict = str(result.get("verdict", "ok")).lower()
    reason = str(result.get("reason", "; ".join(findings)))

    if verdict != "retry":
        # Judged acceptable, but the structural concern is still worth showing.
        state["critique"] = {"ok": True, "reason": reason}
        state["warning"] = "; ".join(findings)
        print(f"[CRITIC] accepted: {reason}")
        return state

    hint = str(result.get("hint", "")).strip()
    state["critique"] = {"ok": False, "reason": reason, "hint": hint}

    if state.get("attempts", 1) < config.MAX_ANALYSIS_ATTEMPTS:
        # Reuse the reflexion edge: the Analysis Agent already rewrites on error.
        state["error"] = f"The result did not answer the question: {reason}"
        if hint:
            state["error"] += f" Suggested fix: {hint}"
        print(f"[CRITIC] rejected, retrying: {reason}")
    else:
        # Out of retries. Keep the result, but do not present it as clean.
        state["warning"] = reason
        print(f"[CRITIC] concern stands after {state.get('attempts')} attempts: {reason}")

    return state

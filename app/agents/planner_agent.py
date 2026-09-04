"""Planner Agent - splits a compound question into ordered analysis steps.

Most questions are one step, and forcing a plan onto them adds latency for
nothing. So the planner's default answer is a single step, and it only
decomposes when the question genuinely needs intermediate results.

Each step is executed by the Analysis Agent in turn; a step can reference the
previous step's output as `prev`, which is what makes decomposition worth doing
rather than just asking three separate questions.
"""

from app import config
from app.agents.llm import get_llm, invoke_json


def _single_step(question):
    return [{"step": question, "purpose": "Answer the question directly."}]


def run(state):
    question = state.get("corrected_question") or state["question"]

    # Planning happens once; retries and later steps must not re-plan.
    if state.get("plan"):
        return state

    state["step_index"] = 0
    state["step_results"] = []

    llm = get_llm()
    if not config.PLANNER_ENABLED or not llm:
        state["plan"] = _single_step(question)
        return state

    prompt = f"""You break data questions into the minimum number of analysis steps.

Most questions need exactly ONE step. Only split when a later part genuinely
needs an earlier part's result - for example "find the top 3 regions, then show
their monthly trend" needs the top 3 before the trend can be filtered.

Do NOT split when:
- The question is one aggregation, however wordy.
- The parts are independent (one step can compute both columns).
- Splitting would just repeat the same groupby with different columns.

Maximum {config.MAX_PLAN_STEPS} steps. Prefer one.

# DATA
{state.get('schema_digest', '')}

# QUESTION
{question}

Reply with ONLY JSON:
{{"steps": [{{"step": "what to compute, phrased as an instruction",
             "purpose": "why this step is needed"}}]}}

Later steps may refer to the previous step's result as `prev`."""

    result = invoke_json(llm, prompt, default=None)

    steps = []
    if result:
        for raw in result.get("steps", [])[:config.MAX_PLAN_STEPS]:
            text = str(raw.get("step", "")).strip() if isinstance(raw, dict) else str(raw).strip()
            if text:
                steps.append({
                    "step": text,
                    "purpose": (raw.get("purpose", "") if isinstance(raw, dict) else ""),
                })

    state["plan"] = steps or _single_step(question)
    if len(state["plan"]) > 1:
        print(f"[PLANNER] {len(state['plan'])} steps:")
        for i, step in enumerate(state["plan"], 1):
            print(f"          {i}. {step['step']}")
    return state

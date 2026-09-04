"""Clarification Agent - asks instead of guessing when a question is ambiguous.

Deliberately conservative. Interrupting someone who asked a clear question is
worse than picking a sensible default, so this only fires when the question
cannot be resolved against the actual columns - and it always offers concrete
options drawn from the schema rather than an open-ended "what do you mean?".
"""

from app import config
from app.agents.llm import get_llm, invoke_json


def run(state):
    state["needs_clarification"] = False

    if not config.CLARIFY_ENABLED:
        return state

    # A follow-up refers to the previous turn, which is context, not ambiguity.
    if state.get("history"):
        return state

    llm = get_llm()
    if not llm:
        return state

    question = state.get("corrected_question") or state["question"]

    prompt = f"""You decide whether a data question can be answered as asked.

Be strict about staying out of the way: ONLY ask for clarification when the
question cannot be answered without guessing between materially different
analyses. If a reasonable default exists, take it and do not ask.

Ask when:
- A superlative has no metric ("top performers" - by revenue? by units?) AND
  several numeric columns could plausibly be meant.
- A named entity matches no column or value in the data.

Do NOT ask when:
- The question names a column, even loosely.
- Only one numeric column could be meant.
- A sensible default exists (e.g. "sales" when there is one revenue column).
- The question is broad but answerable ("summarise this data").

# COLUMNS AVAILABLE
{state.get('schema_digest', '')}

# QUESTION
{question}

Reply with ONLY JSON:
{{"needs_clarification": false}}
or
{{"needs_clarification": true,
  "question": "one short question",
  "options": ["concrete option 1", "concrete option 2", "concrete option 3"]}}

Each option must be a complete rephrased question the user could run as-is."""

    result = invoke_json(llm, prompt, default={"needs_clarification": False})

    if not result.get("needs_clarification"):
        return state

    options = [str(o) for o in result.get("options", []) if str(o).strip()][:4]
    if not options:
        # An ambiguity we cannot offer choices for is not actionable.
        return state

    state["needs_clarification"] = True
    state["clarification_question"] = result.get("question", "Could you be more specific?")
    state["clarification_options"] = options
    print(f"[CLARIFY] Asking: {state['clarification_question']}")
    return state

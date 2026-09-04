"""Analysis Agent - turns a question (or one plan step) into pandas code.

Three things shape the prompt beyond the schema:

  * retrieved context from the RAG Agent - certified definitions the code must
    follow, and past analyses that worked on this workbook
  * the current plan step, when the Planner split the question up
  * the previous failure, on a retry - which is what makes the reflexion edge
    corrective rather than a re-roll
"""

import re

from app.agents.llm import get_llm

_SYSTEM_RULES = """You write pandas code to answer questions about a spreadsheet.

# AVAILABLE NAMES
- `pd`   : the pandas module
- `np`   : the numpy module
- one DataFrame per sheet, bound to the sheet name shown in the schema
- `df`   : alias for the primary sheet
- `prev` : the previous plan step's result, when there was one

# HARD RULES
1. Assign your final answer to a variable named `result`.
2. `result` must be a DataFrame, a Series, or a single scalar value. Prefer a
   DataFrame with named columns - it is rendered as a table and a chart.
3. Use ONLY the column names listed in the schema. Never invent one.
4. No imports. No file, network, or system access. `pd` and `np` are already available.
5. No print(), no plotting, no display(). Just compute `result` - charts are
   rendered separately from whatever data you return, so "as a pie chart" or
   "show a bar chart of X" is an instruction about the data shape (e.g. group by
   the right category), not an instruction to plot anything yourself.
6. Date columns may be stored as text. Convert explicitly when you need date parts:
   `pd.to_datetime(df['order_date'], errors='coerce')`
7. Reset the index after a groupby so the grouping keys stay as columns:
   `df.groupby('region', as_index=False)['revenue'].sum()`
8. Give computed columns readable snake_case names.

# OUTPUT
Return ONLY the Python code. No markdown fences, no commentary, no explanation."""


def _strip_fences(code: str) -> str:
    if "```" in code:
        blocks = re.findall(r"```(?:python|py)?\s*(.*?)\s*```", code, re.DOTALL)
        if blocks:
            return blocks[0].strip()
        code = code.replace("```python", "").replace("```py", "").replace("```", "")
    return code.strip()


def _current_step(state):
    """The instruction for this pass: a plan step, or the whole question."""
    plan = state.get("plan") or []
    index = state.get("step_index", 0)
    if plan and index < len(plan):
        return plan[index]["step"]
    return state.get("corrected_question") or state["question"]


def run(state):
    llm = get_llm()
    state["attempts"] = state.get("attempts", 0) + 1
    step = _current_step(state)
    state["current_step"] = step

    if not llm:
        state["code"] = "result = df.head(50)"
        state["code_explanation"] = (
            "No GROQ_API_KEY is configured, so the agent returned a preview of the "
            "primary sheet instead of a generated analysis."
        )
        return state

    primary = state.get("profile", {}).get("primary_sheet", "df")

    rag_block = ""
    if state.get("rag_context"):
        rag_block = f"\n{state['rag_context']}\n"

    plan_block = ""
    plan = state.get("plan") or []
    if len(plan) > 1:
        completed = "\n".join(
            f"  step {i + 1}: {r['step']}  ->  {r['columns']}"
            for i, r in enumerate(state.get("step_results", []))
        )
        plan_block = f"""
# THIS IS STEP {state.get('step_index', 0) + 1} OF {len(plan)}
Full plan:
{chr(10).join(f"  {i}. {s['step']}" for i, s in enumerate(plan, 1))}
Completed so far:
{completed or '  (none)'}
The previous step's result is available as `prev`."""

    retry_block = ""
    if state.get("error"):
        retry_block = f"""
# PREVIOUS ATTEMPT FAILED - FIX IT
Code:
{state.get('code', '')}

Problem:
{state['error']}

Rewrite the code so it does not hit this problem. Check the schema again: the
column names or dtypes may not be what the failed attempt assumed."""

    prompt = f"""{_SYSTEM_RULES}

# SCHEMA - workbook "{state.get('filename', 'upload')}"
# (`df` is the primary sheet, `{primary}`)
{state.get('schema_digest', '')}
{rag_block}
# CONVERSATION SO FAR
{state.get('history', [])}
{plan_block}{retry_block}

# WHAT TO COMPUTE
{step}

# CODE
"""

    state["code"] = _strip_fences(llm.invoke(prompt).content)
    state["error"] = None
    return state

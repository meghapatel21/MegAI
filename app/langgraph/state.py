from typing import Any, List, Optional, TypedDict


class BIState(TypedDict, total=False):
    # --- request ---
    tenant_id: str
    user_id: str
    dataset_id: str
    question: str
    history: List[Any]

    # --- profile agent ---
    filename: str
    profile: dict           # {sheets: [{name, row_count, columns:[...]}], primary_sheet}
    schema_digest: str      # profile rendered for the prompt
    corrected_question: str

    # --- clarify agent ---
    needs_clarification: bool
    clarification_question: str
    clarification_options: List[str]

    # --- rag agent ---
    rag_context: str        # definitions + examples, formatted for the prompt
    rag_definitions: List[dict]   # [{term, definition, score}]
    rag_examples: List[dict]      # [{question, code, score, same_file}]

    # --- planner agent ---
    plan: List[dict]        # [{step, purpose}]
    step_index: int
    current_step: str
    step_results: List[dict]      # [{step, code, columns, rows}]

    # --- analysis / impact / execute ---
    code: str               # generated pandas
    code_explanation: str
    impact: dict            # {approved, estimated_cost, ...}
    attempts: int           # retries within the current step
    error: Optional[str]    # set by impact, execute, or critic; consumed on retry
    result: Any             # list of row dicts
    row_count: int
    truncated: bool

    # --- critic agent ---
    critique: dict          # {ok, reason, hint}
    warning: Optional[str]  # concern that survived the retry budget

    # --- insight agent ---
    insight: str

    # --- bi agent ---
    response: dict

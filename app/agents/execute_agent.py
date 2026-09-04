"""Execute Agent - runs approved pandas code against the uploaded sheets.

The code has already passed the Impact Agent's AST audit; the restricted
namespace here is the second layer, not the only one. Nothing touches a
database: every sheet is an in-memory DataFrame loaded from the dataset store.
"""

import numpy as np
import pandas as pd

from app import config
from app.storage.dataset_store import store

# A deliberately small builtin surface. Anything not listed is simply absent,
# so generated code cannot reach open() or the interpreter.
#
# __import__ IS included, which looks like a contradiction - it isn't. The
# Impact Agent's AST audit already rejects any generated code that references
# the name `__import__` or uses an `import` statement, before this module ever
# runs. So the LLM's own code can never reach it here. It still has to be
# present as a real, callable builtin because numpy/pandas C internals probe
# `frame.f_builtins['__import__']` as part of unrelated operations - e.g.
# plain `str(some_dtype)` does this - and raise a bare KeyError if it is
# missing, which has nothing to do with the generated code doing anything
# unsafe.
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "int": int, "isinstance": isinstance, "len": len,
    "list": list, "map": map, "max": max, "min": min, "pow": pow,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "True": True, "False": False, "None": None,
    "__import__": __import__,
}


def _normalise(result, question: str) -> pd.DataFrame:
    """Coerce whatever the code produced into a DataFrame for rendering."""
    if isinstance(result, pd.DataFrame):
        return result.reset_index(drop=True) if result.index.name is None else result.reset_index()

    if isinstance(result, pd.Series):
        name = result.name or "value"
        frame = result.rename(name).reset_index()
        # A Series from groupby carries the grouping key in the index.
        if frame.columns[0] in (0, "index"):
            frame.columns = ["category", name]
        return frame

    if isinstance(result, (np.generic,)):
        result = result.item()

    if isinstance(result, (list, tuple, set)):
        return pd.DataFrame({"value": list(result)})

    if isinstance(result, dict):
        try:
            return pd.DataFrame(result)
        except ValueError:
            return pd.DataFrame([result])

    # Scalar answer, e.g. "what is total revenue".
    label = "answer"
    for word in ("revenue", "total", "average", "count", "sum", "profit", "cost"):
        if word in question.lower():
            label = word
            break
    return pd.DataFrame([{label: result}])


def run(state):
    frames = store.load(state["dataset_id"])

    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
    }
    namespace.update(frames)
    primary = state.get("profile", {}).get("primary_sheet")
    namespace["df"] = frames[primary] if primary in frames else next(iter(frames.values()))

    # Multi-step plans let a step build on the one before it.
    previous = state.get("step_results") or []
    namespace["prev"] = pd.DataFrame(previous[-1]["rows"]) if previous else None

    try:
        exec(state["code"], namespace)
    except Exception as exc:
        # Handed back to the Analysis Agent through the graph's reflexion edge.
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["result"] = []
        print(f"[EXECUTE] failed on attempt {state.get('attempts')}: {state['error']}")
        return state

    if "result" not in namespace:
        state["error"] = "the code ran but never assigned `result`"
        state["result"] = []
        return state

    frame = _normalise(namespace["result"], state.get("corrected_question", state["question"]))

    state["row_count"] = int(len(frame))
    state["truncated"] = len(frame) > config.MAX_RESULT_ROWS
    if state["truncated"]:
        frame = frame.head(config.MAX_RESULT_ROWS)

    state["result"] = frame.to_dict("records")
    state["error"] = None
    return state

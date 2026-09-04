"""BI Agent - turns the computed rows into KPIs, a chart hint, and an explanation.

Every number here is derived from the result set. Nothing is a placeholder: if a
trend cannot be computed from the data, the response says so rather than
inventing a percentage.
"""

import math

import numpy as np
import pandas as pd

from app.insights import chart_intent
from app.memory.mem0_client import memory
from app.rag import query_memory

_ID_HINTS = ("_id", "id_", "index", "code", "number", "zip", "postal")
_CURRENCY_HINTS = ("revenue", "sales", "spend", "price", "budget", "cost", "amount",
                   "profit", "margin", "value", "salary", "expense")


def _looks_like_id(column: str) -> bool:
    low = column.lower()
    return low == "id" or any(hint in low for hint in _ID_HINTS)


def _pick_metric(df: pd.DataFrame):
    """Choose the column that best represents 'the answer'."""
    numeric = [c for c in df.select_dtypes("number").columns if not _looks_like_id(c)]
    if not numeric:
        return None
    for hint in _CURRENCY_HINTS:
        for col in numeric:
            if hint in col.lower():
                return col
    return numeric[0]


def _find_period_column(df: pd.DataFrame):
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in ("date", "month", "year", "quarter", "week", "period", "day")):
            return col
    return None


def _trend(df: pd.DataFrame, metric: str):
    """First-to-last change over an ordered period column, or None."""
    period = _find_period_column(df)
    if not period or metric is None or len(df) < 2:
        return None
    try:
        ordered = df.sort_values(period)
        first = float(ordered[metric].iloc[0])
        last = float(ordered[metric].iloc[-1])
        # A missing value at either endpoint (a real possibility - the first or
        # last period's total can be NaN, e.g. no rows that period) makes the
        # comparison meaningless, not just a zero: nan == 0 is False, so this
        # must be checked before the zero-division guard, not folded into it.
        if math.isnan(first) or math.isnan(last) or first == 0:
            return None
        return {
            "pct": (last - first) / abs(first) * 100,
            "from_period": str(ordered[period].iloc[0]),
            "to_period": str(ordered[period].iloc[-1]),
        }
    except Exception:
        return None


# What each explicit chart type needs from the result to be renderable at all.
_CHART_REQUIREMENTS = {
    "line": lambda s: s["has_metric"] and (s["has_categorical"] or s["has_period"]),
    "area": lambda s: s["has_metric"] and (s["has_categorical"] or s["has_period"]),
    "bar": lambda s: s["has_metric"] and s["has_categorical"],
    "pie": lambda s: s["has_metric"] and s["has_categorical"],
    "donut": lambda s: s["has_metric"] and s["has_categorical"],
    "scatter": lambda s: s["has_multi_numeric"],
    "histogram": lambda s: s["has_metric"],
    "frequency": lambda s: s["has_categorical"],
    "table": lambda s: True,
}

# Plain-English version of the same requirement, for the fallback explanation.
_CHART_NEEDS = {
    "pie": "a numeric total and a category to group it by",
    "donut": "a numeric total and a category to group it by",
    "bar": "a numeric total and a category to group it by",
    "line": "a numeric total and a date or category to plot along the x-axis",
    "area": "a numeric total and a date or category to plot along the x-axis",
    "scatter": "two numeric columns to plot against each other",
    "histogram": "a numeric column to show the distribution of",
    "frequency": "a category column to count",
}


def _json_safe(value):
    """Recursively replace NaN/Infinity with None.

    A missing value in a spreadsheet is completely normal input - it becomes
    NaN the moment pandas reads the sheet, and any groupby/merge over it
    propagates that NaN into the result. Strict JSON has no representation for
    NaN or Infinity, so returning them as-is 500s the whole response the first
    time a user's real (imperfect) data reaches this function, regardless of
    whether the field that crashes is `data`, a KPI, or a trend percentage.
    """
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _chart_shape(df: pd.DataFrame, metric):
    categorical = [c for c in df.columns if c not in df.select_dtypes("number").columns]
    return {
        "has_metric": metric is not None,
        "has_categorical": bool(categorical),
        "has_period": _find_period_column(df) is not None,
        "has_multi_numeric": len(df.select_dtypes("number").columns) >= 2,
    }


def _auto_chart(df: pd.DataFrame, metric, shape) -> str:
    if len(df) == 1:
        return "single_value" if metric else "table"
    if metric and shape["has_period"]:
        return "line"
    if metric and shape["has_categorical"]:
        return "pie" if 2 <= len(df) <= 8 else "bar"
    if shape["has_categorical"]:
        return "frequency"
    return "table"


def _chart_hint(df: pd.DataFrame, metric, requested=None):
    """Pick a chart type.

    Honors an explicit request ("show me a bar chart") when the result actually
    has the shape that chart type needs. When it does not - a pie chart request
    on a result with no category to group by, say - falls back to the automatic
    pick and returns a note explaining why, so the UI can tell the user rather
    than silently ignoring what they asked for.

    Returns (chart_type, note_or_None).
    """
    if df.empty:
        return "none", None

    shape = _chart_shape(df, metric)

    if requested:
        # A single-row result is a number, not a chart, regardless of what was
        # asked - a pie of one slice or a bar of one bar communicates nothing.
        if len(df) == 1 and requested not in ("single_value", "table"):
            auto = _auto_chart(df, metric, shape)
            return auto, (
                f"you asked for a {requested} chart, but the result is a single "
                f"value, so it's shown as a number instead"
            )

        check = _CHART_REQUIREMENTS.get(requested)
        if check and check(shape):
            return requested, None

        auto = _auto_chart(df, metric, shape)
        need = _CHART_NEEDS.get(requested, "a different shape of result")
        return auto, (
            f"you asked for a {requested} chart, but this result doesn't have "
            f"{need}; showing a {auto} chart instead"
        )

    return _auto_chart(df, metric, shape), None


def run(state):
    # The Clarify Agent stopped before any analysis ran: the response is a
    # question back to the user, not a result.
    if state.get("needs_clarification"):
        state["response"] = {
            "status": "needs_clarification",
            "question": state.get("clarification_question", "Could you be more specific?"),
            "options": state.get("clarification_options", []),
            "data": [],
            "kpis": {},
            "code": "",
            "reasoning": (
                f"**{state.get('clarification_question', 'Could you clarify?')}**\n\n"
                "That question could be read a few ways, and the answers would differ. "
                "Pick one below, or rephrase."
            ),
        }
        return state

    # A failure that survived every reflexion attempt still needs a response.
    if state.get("error"):
        state["response"] = {
            "status": "failed",
            "error": state["error"],
            "code": state.get("code", ""),
            "data": [],
            "kpis": {},
            "reasoning": (
                f"The agent could not produce a working analysis after "
                f"{state.get('attempts', 1)} attempt(s).\n\n"
                f"**Last error:** {state['error']}\n\n"
                "Try rephrasing the question, or check that the columns you are "
                "asking about exist in the uploaded file."
            ),
        }
        return state

    df = pd.DataFrame(state.get("result", []))
    metric = _pick_metric(df) if not df.empty else None

    question = state.get("corrected_question") or state["question"]
    requested_chart = chart_intent.detect_requested_chart(question)
    chart_type, chart_note = _chart_hint(df, metric, requested_chart)

    kpis = {
        "row_count": state.get("row_count", len(df)),
        "column_count": len(df.columns),
    }

    if metric is not None:
        total = float(df[metric].sum())
        kpis.update({
            "primary_label": metric.replace("_", " ").title(),
            "primary_val": total,
            "is_currency": any(h in metric.lower() for h in _CURRENCY_HINTS),
            "mean_val": float(df[metric].mean()),
        })
    else:
        kpis.update({
            "primary_label": "Rows Returned",
            "primary_val": float(len(df)),
            "is_currency": False,
        })

    trend = _trend(df, metric)
    if trend:
        kpis["trend_pct"] = trend["pct"]
        kpis["trend_label"] = f"{trend['from_period']} to {trend['to_period']}"

    reasoning_lines = [
        "### Analysis Summary",
        f"- **Interpreted question:** \"{state.get('corrected_question', state['question'])}\"",
        f"- **Source:** `{state.get('filename', 'uploaded file')}` — "
        f"sheet `{state.get('profile', {}).get('primary_sheet', 'n/a')}`",
        f"- **Rows returned:** {state.get('row_count', len(df)):,}",
    ]
    if metric is not None:
        reasoning_lines.append(
            f"- **Primary metric:** `{metric}` (total {kpis['primary_val']:,.2f})"
        )
    if trend:
        reasoning_lines.append(
            f"- **Trend:** {trend['pct']:+.1f}% from {trend['from_period']} to {trend['to_period']}"
        )
    if metric is None and not df.empty:
        reasoning_lines.append(
            "- **Note:** no numeric column in the result, so the row count is reported instead of a metric."
        )
    if state.get("truncated"):
        reasoning_lines.append(
            f"- **Note:** output truncated to the first {len(df):,} rows."
        )
    if state.get("attempts", 1) > 1:
        reasoning_lines.append(
            f"- **Self-correction:** the first {state['attempts'] - 1} attempt(s) failed "
            "and were rewritten automatically."
        )
    impact = state.get("impact") or {}
    if impact.get("approved"):
        reasoning_lines.append(
            f"- **Governance:** approved by the Impact Agent "
            f"({impact.get('estimated_cost', 'low')} cost, "
            f"{impact.get('table_operations', 0)} table operation(s))."
        )

    # --- what RAG contributed --------------------------------------------
    definitions = state.get("rag_definitions") or []
    examples = state.get("rag_examples") or []
    if definitions:
        reasoning_lines.append(
            "- **Definitions applied:** " + ", ".join(f"`{d['term']}`" for d in definitions)
        )
    if examples:
        same_file = sum(1 for e in examples if e["same_file"])
        reasoning_lines.append(
            f"- **Retrieved context:** {len(examples)} similar past analysis/analyses "
            f"({same_file} from this workbook) used as worked examples."
        )

    plan = state.get("plan") or []
    if len(plan) > 1:
        reasoning_lines.append("- **Plan:** " + " → ".join(s["step"] for s in plan))

    if state.get("warning"):
        reasoning_lines.append(f"- **⚠️ Check this result:** {state['warning']}")

    if chart_note:
        reasoning_lines.append(f"- **Chart:** {chart_note}.")
    elif requested_chart:
        reasoning_lines.append(f"- **Chart:** shown as a {chart_type} chart, as requested.")

    state["response"] = _json_safe({
        "status": "ok",
        "kpis": kpis,
        "data": df.to_dict("records"),
        "code": state.get("code", ""),
        "chart_hint": chart_type,
        "requested_chart": requested_chart,
        "chart_note": chart_note,
        "metric_column": metric,
        "reasoning": "\n".join(reasoning_lines),
        "insight": state.get("insight", ""),
        "warning": state.get("warning"),
        "plan": [s["step"] for s in plan] if len(plan) > 1 else [],
        "step_results": [
            {"step": r["step"], "code": r["code"], "columns": r["columns"]}
            for r in (state.get("step_results") or [])
        ] if len(plan) > 1 else [],
        "definitions_used": definitions,
        "examples_used": [
            {"question": e["question"], "score": e["score"], "same_file": e["same_file"]}
            for e in examples
        ],
    })

    # Feed the RAG corpus: only clean runs become future few-shot examples, so
    # the agent never learns from code the Critic rejected.
    if not state.get("warning"):
        query_memory.remember(
            question=state.get("corrected_question") or state["question"],
            code=state.get("code", ""),
            profile=state.get("profile", {}),
            row_count=len(df),
        )

    try:
        summary = f"Q: {state['question']} | A: {kpis['primary_label']} = {kpis['primary_val']:,.2f}"
        memory.add(summary, user_id=state.get("user_id", "anonymous"))
    except Exception as exc:
        print(f"[MEMORY] add skipped: {exc}")

    return state

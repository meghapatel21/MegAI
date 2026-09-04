"""Overview Agent - the automatic briefing produced the moment a file lands.

Runs once per upload, outside the question-answering graph: there is no question
yet, so there is nothing for the Planner or Analysis agents to do. The findings
come from `profiler`, which is pure pandas; this agent only writes the summary
over them, so the report still appears with no API key - just without the prose.
"""

from app.agents.llm import get_llm
from app.insights import profiler
from app.storage.dataset_store import store


def _facts_for_prompt(sheet):
    """Render the computed findings compactly, so the model summarises rather than guesses."""
    lines = [f"Sheet `{sheet['name']}`: {sheet['rows']:,} rows x {sheet['columns']} columns"]

    roles = sheet["roles"]
    for role in ("numeric", "categorical", "temporal"):
        if roles[role]:
            lines.append(f"{role.title()} columns: {', '.join(roles[role][:12])}")

    if sheet["time_coverage"]:
        coverage = sheet["time_coverage"]
        lines.append(
            f"Time span: {coverage['from']} to {coverage['to']} "
            f"({coverage['distinct_months']} distinct months)"
        )

    for metric in sheet["headline_metrics"]:
        lines.append(f"{metric['column']}: total {metric['total']:,.2f}, mean {metric['mean']:,.2f}")

    for breakdown in sheet["breakdowns"]:
        top = ", ".join(f"{r['name']} ({r['share']}%)" for r in breakdown["rows"])
        lines.append(
            f"{breakdown['metric']} by {breakdown['category']} "
            f"({breakdown['distinct']} distinct): {top}"
        )

    for pair in sheet["correlations"]:
        lines.append(f"{pair['a']} and {pair['b']} move {pair['direction']} (r={pair['r']})")

    for finding in sheet["quality"]:
        lines.append(f"DATA QUALITY [{finding['severity']}]: {finding['issue']}")

    return "\n".join(lines)


def run(dataset_id):
    """Build the upload briefing. Returns the overview dict with a `summary`."""
    meta = store.get_meta(dataset_id)
    frames = store.load(dataset_id)

    overview = profiler.build_overview(frames, meta["profile"])
    overview["filename"] = meta.get("filename", "your file")
    overview["summary"] = ""

    llm = get_llm()
    if not llm or not overview["sheets"]:
        return overview

    primary = next(
        (s for s in overview["sheets"] if s["name"] == overview["primary_sheet"]),
        overview["sheets"][0],
    )

    other_sheets = ""
    if overview["sheet_count"] > 1:
        others = [s for s in overview["sheets"] if s["name"] != primary["name"]]
        other_sheets = "\n\nOther sheets in the same workbook:\n" + "\n".join(
            f"- `{s['name']}`: {s['rows']:,} rows, columns {', '.join(c['name'] for c in s['column_summaries'][:8])}"
            for s in others[:4]
        )

    prompt = f"""You are a data analyst who has just been handed a spreadsheet.
Write the short briefing you would give a colleague before they start asking
questions about it.

Every fact below was computed from the actual file. Summarise and interpret
them - do not invent anything that is not here.

# FILE: {overview['filename']}
{_facts_for_prompt(primary)}{other_sheets}

Write 3-5 sentences covering, in this order:
1. What this dataset appears to be about, and what one row represents.
2. The scale and time span.
3. The one or two most notable things in the numbers, with specific figures.
4. Any data-quality issue that would affect analysis - say so plainly.

Rules:
- State WHAT the data shows, never WHY. You cannot see causes.
- If two columns correlate, say they move together - never that one causes the other.
- No preamble, no headings, no bullet points. Plain prose.
- Do not list the column names back; the reader can see them.
"""

    try:
        overview["summary"] = llm.invoke(prompt).content.strip()
    except Exception as exc:
        print(f"[OVERVIEW] narrative skipped: {exc}")

    return overview

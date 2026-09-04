"""Detects when a user explicitly asked for a specific chart type.

Deliberately separate from analysis: this is about *rendering* intent, not
about which columns to compute, so it runs as plain keyword matching on the
raw question - no LLM call, no latency, no cost. When nothing is asked for,
the BI Agent's shape-based auto-pick is unchanged.
"""

import re

# Canonical types the app can render. Checked in this order so a more specific
# phrase ("line chart") is not shadowed by a shorter one appearing later.
_PATTERNS = [
    ("scatter", r"\bscatter\s*(plot|chart|graph)?\b"),
    ("histogram", r"\bhistogram\b|\bdistribution\s+chart\b"),
    # Donut is checked before pie: a donut is a pie with a hole, and asking
    # for one should not silently render the other.
    ("donut", r"\bdonut\s*(chart|graph)?\b|\bdoughnut\s*(chart|graph)?\b"),
    ("pie", r"\bpie\s*(chart|graph)?\b"),
    ("area", r"\barea\s*(chart|graph)\b"),
    ("line", r"\bline\s*(chart|graph|plot)\b"),
    ("bar", r"\bbar\s*(chart|graph)\b|\bcolumn\s*chart\b"),
    ("frequency", r"\bfrequency\s*(chart|graph|distribution)\b|\bcount\s*chart\b"),
    ("table", r"\btable\b|\braw\s+data\b|\bno\s+chart\b|\bwithout\s+(a\s+)?chart\b|"
              r"\bwithout\s+(a\s+)?visuali[sz]ation\b"),
]

_COMPILED = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _PATTERNS]


def detect_requested_chart(question):
    """Return the canonical chart type explicitly named in the question, or None.

    None means "no explicit request" - the caller should fall back to picking
    a chart from the shape of the result, exactly as before this existed.
    """
    if not question:
        return None
    for name, pattern in _COMPILED:
        if pattern.search(question):
            return name
    return None

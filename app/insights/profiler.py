"""Deterministic dataset profiling - the findings that need no LLM.

Everything here is computed from the data itself, so an upload produces a useful
report even with no API key, no network, and no cost. The Overview Agent layers
a narrative on top; this module is what that narrative is grounded in.

Cost note: every finding here is a full-column pass and there are a dozen of
them, so a large sheet used to spend most of its upload in this module. Two
things bound that now - date columns are parsed once and reused rather than
re-parsed by each section, and above `config.PROFILE_SAMPLE_ROWS` the purely
descriptive passes run on a sample. Figures a reader would quote back (row
counts, totals, breakdowns, the date range) stay exact either way.
"""

import pandas as pd

from app import config

_ID_HINTS = ("_id", "id_", "uuid", "guid", "index", "code", "number", "zip", "postal")
_CURRENCY_HINTS = ("revenue", "sales", "spend", "price", "budget", "cost", "amount",
                   "profit", "margin", "salary", "expense", "value", "total")
_DATE_HINTS = ("date", "time", "month", "year", "quarter", "week", "day", "period",
               "created", "updated", "signup", "expiry")

# Values probed before committing to parsing a whole column as dates.
_DATE_PROBE_ROWS = 2000


def _looks_like_id(name, series):
    low = name.lower()
    if low == "id" or any(hint in low for hint in _ID_HINTS):
        return True

    # Cardinality alone cannot identify a key: a float metric such as `revenue`
    # is normally all-distinct too, and classifying it as an identifier would
    # drop it from every metric, breakdown and correlation. Only an all-distinct
    # *integer* column is key-shaped.
    if not pd.api.types.is_integer_dtype(series):
        return False

    non_null = series.dropna()
    if len(non_null) < 10:
        return False
    return non_null.nunique() == len(non_null)


def _parse_dates(values):
    """Parse date-ish values, cheapest strategy first.

    `format="mixed"` infers a format per element, which measured ~50% slower
    than letting pandas infer one format for the whole column. Uploaded date
    columns are overwhelmingly uniform, so the uniform path is tried first and
    the per-element path is kept as the fallback for columns that genuinely do
    mix formats.
    """
    try:
        parsed = pd.to_datetime(values, errors="coerce")
        if parsed.notna().mean() > 0.8:
            return parsed
    except Exception:
        parsed = None

    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except Exception:
        return parsed


def _as_datetime_uncached(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    non_null = series.dropna()
    if non_null.empty:
        return None

    # Reject on a strided sample first. A column is nominated as date-ish by
    # its *name*, so a text column called `updated_by` would otherwise pay for
    # a full parse just to be thrown away.
    if len(non_null) > _DATE_PROBE_ROWS:
        stride = max(1, len(non_null) // _DATE_PROBE_ROWS)
        probe = _parse_dates(non_null.iloc[::stride].head(_DATE_PROBE_ROWS))
        if probe is None or probe.notna().mean() <= 0.8:
            return None

    parsed = _parse_dates(non_null)
    if parsed is None:
        return None
    # Require most values to parse; otherwise it is text that happens to look datey.
    return parsed if parsed.notna().mean() > 0.8 else None


def _as_datetime(series, cache=None, key=None):
    """Parse a column as dates, returning None when it clearly is not one.

    `cache` memoises per column: classification, the column summary and the
    time-coverage section all want the same parse, and parsing a large column
    three times was plainly visible in upload time.
    """
    if cache is not None and key in cache:
        return cache[key]

    parsed = _as_datetime_uncached(series)

    if cache is not None:
        cache[key] = parsed
    return parsed


def _metric_priority(name):
    """Rank numeric columns so the business metric leads, not whatever is leftmost.

    A sheet with `marketing_spend` before `revenue` should still headline on
    revenue, since that is what a reader means by "the" number.
    """
    low = name.lower()
    for rank, hint in enumerate(("revenue", "sales", "profit", "amount", "total",
                                 "value", "price", "spend", "cost", "budget")):
        if hint in low:
            return rank
    return len(_CURRENCY_HINTS) + 1


def classify_columns(df, dt_cache=None):
    """Split columns into the roles the rest of the report reasons about."""
    numeric, categorical, temporal, identifier = [], [], [], []

    for name in df.columns:
        series = df[name]
        low = name.lower()

        if (any(hint in low for hint in _DATE_HINTS)
                and _as_datetime(series, dt_cache, name) is not None):
            temporal.append(name)
        elif _looks_like_id(name, series):
            identifier.append(name)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(name)
        elif series.nunique(dropna=True) <= max(50, 0.2 * len(series)):
            categorical.append(name)
        else:
            identifier.append(name)  # free text / high-cardinality

    # Business metrics first, so every downstream section leads with the column
    # a reader cares about rather than the one that happens to be leftmost.
    numeric.sort(key=_metric_priority)

    return {"numeric": numeric, "categorical": categorical,
            "temporal": temporal, "identifier": identifier}


def quality_findings(df, roles, total_rows=None):
    """Data-quality problems worth telling the user about, most severe first.

    `total_rows` is the size of the sheet `df` was sampled from, when it was
    sampled. Counts are then reported as shares, so a sample never states a
    precise count it did not actually see.
    """
    findings = []
    rows = len(df)
    if rows == 0:
        return [{"severity": "high", "issue": "The sheet has no rows."}]

    sampled = total_rows is not None and total_rows != rows

    duplicates = int(df.duplicated().sum())
    if duplicates:
        share = duplicates / rows
        findings.append({
            "severity": "high" if share > 0.05 else "medium",
            "issue": (f"about {share:.1%} of rows are duplicated rows, estimated "
                      f"from a {rows:,}-row sample."
                      if sampled else
                      f"{duplicates:,} fully duplicated row(s) ({share:.1%} of the sheet)."),
            "columns": [],
        })

    for name in df.columns:
        series = df[name]
        missing = series.isna()
        null_share = float(missing.mean())
        if null_share == 1.0:
            findings.append({"severity": "high",
                             "issue": f"`{name}` is entirely empty.", "columns": [name]})
        elif null_share > 0.4:
            findings.append({"severity": "medium",
                             "issue": f"`{name}` is {null_share:.0%} empty.", "columns": [name]})

        non_null = series[~missing]
        if len(non_null) > 1 and non_null.nunique() == 1:
            findings.append({
                "severity": "low",
                "issue": f"`{name}` holds a single value ({non_null.iloc[0]!r}) in every row.",
                "columns": [name],
            })

    # Outliers on real metrics only - flagging an ID's "outliers" is noise.
    for name in roles["numeric"]:
        series = df[name].dropna()
        if len(series) < 20:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        extreme = series[(series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)]
        if len(extreme):
            amount = (f"about {len(extreme) / rows:.1%} extreme value(s)"
                      if sampled else f"{len(extreme):,} extreme value(s)")
            findings.append({
                "severity": "low",
                "issue": (f"`{name}` has {amount} "
                          f"(e.g. {extreme.max():,.2f} against a median of {series.median():,.2f})."),
                "columns": [name],
            })

        negatives = int((series < 0).sum())
        if negatives and any(h in name.lower() for h in _CURRENCY_HINTS):
            amount = (f"about {negatives / rows:.1%} negative value(s)"
                      if sampled else f"{negatives:,} negative value(s)")
            findings.append({
                "severity": "medium",
                "issue": f"`{name}` contains {amount}, which is unusual for an amount.",
                "columns": [name],
            })

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(findings, key=lambda f: order[f["severity"]])[:12]


def column_summaries(df, roles, full=None, dt_cache=None):
    """Per-column detail. `full` is the unsampled sheet, when `df` is a sample.

    Aggregates over numeric columns are vectorised and cheap, so they come from
    the full sheet and stay exact. Cardinality and top-value counts are the
    expensive hashing passes, so those come from `df`.
    """
    exact = full if full is not None else df
    summaries = []

    for name in df.columns:
        series = df[name]
        exact_series = exact[name]
        entry = {
            "name": name,
            "role": next(r for r, cols in roles.items() if name in cols),
            "null_pct": round(float(exact_series.isna().mean() * 100), 1),
            "unique": int(series.nunique(dropna=True)),
        }

        if name in roles["numeric"]:
            non_null = exact_series.dropna()
            if not non_null.empty:
                entry.update({
                    "min": float(non_null.min()), "max": float(non_null.max()),
                    "mean": float(non_null.mean()), "median": float(non_null.median()),
                    "sum": float(non_null.sum()),
                })
        elif name in roles["temporal"]:
            parsed = _as_datetime(exact_series, dt_cache, name)
            if parsed is not None and parsed.notna().any():
                entry["from"] = str(parsed.min().date())
                entry["to"] = str(parsed.max().date())
        elif name in roles["categorical"]:
            non_null = series.dropna()
            if not non_null.empty:
                counts = non_null.value_counts().head(5)
                entry["top_values"] = [
                    {"value": str(v), "count": int(c),
                     "pct": round(float(c / len(non_null) * 100), 1)}
                    for v, c in counts.items()
                ]
        summaries.append(entry)
    return summaries


def headline_metrics(df, roles):
    """The numbers a person would want on screen immediately."""
    metrics = []
    for name in roles["numeric"][:4]:
        series = df[name].dropna()
        if series.empty:
            continue
        metrics.append({
            "column": name,
            "label": name.replace("_", " ").title(),
            "total": float(series.sum()),
            "mean": float(series.mean()),
            "is_currency": any(h in name.lower() for h in _CURRENCY_HINTS),
        })
    return metrics


def top_breakdowns(df, roles, limit=3):
    """Leading categories by the primary metric - the first thing anyone asks."""
    if not roles["numeric"] or not roles["categorical"]:
        return []

    metric = next(
        (c for h in _CURRENCY_HINTS for c in roles["numeric"] if h in c.lower()),
        roles["numeric"][0],
    )

    breakdowns = []
    for category in roles["categorical"][:3]:
        try:
            grouped = (df.groupby(category, dropna=True)[metric]
                         .sum().sort_values(ascending=False))
        except Exception:
            continue
        if grouped.empty or len(grouped) < 2:
            continue
        total = float(grouped.sum())
        breakdowns.append({
            "category": category,
            "metric": metric,
            "rows": [
                {"name": str(k), "value": float(v),
                 "share": round(float(v / total * 100), 1) if total else 0.0}
                for k, v in grouped.head(limit).items()
            ],
            "distinct": int(len(grouped)),
        })
    return breakdowns


def correlations(df, roles, threshold=0.5):
    """Strong numeric relationships. Association only - never presented as cause.

    Uses Spearman (rank) rather than Pearson deliberately. Uploaded spreadsheets
    contain typos and stray values, and Pearson is not robust to them: a single
    mistyped row in 400 collapsed a real 0.998 relationship to 0.149 in testing,
    while the rank correlation held at 0.997. Hiding a genuine relationship
    because of one bad cell is the worse failure for an overview report.
    """
    metrics = [c for c in roles["numeric"] if df[c].notna().sum() > 10]
    if len(metrics) < 2:
        return []

    try:
        matrix = df[metrics].corr(method="spearman", numeric_only=True)
    except Exception:
        return []

    pairs = []
    for i, a in enumerate(metrics):
        for b in metrics[i + 1:]:
            value = matrix.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({"a": a, "b": b, "r": round(float(value), 2),
                              "direction": "together" if value > 0 else "inversely"})
    return sorted(pairs, key=lambda p: -abs(p["r"]))[:5]


def time_coverage(df, roles, dt_cache=None):
    if not roles["temporal"]:
        return None
    column = roles["temporal"][0]
    parsed = _as_datetime(df[column], dt_cache, column)
    if parsed is None or parsed.dropna().empty:
        return None
    parsed = parsed.dropna()
    return {
        "column": column,
        "from": str(parsed.min().date()),
        "to": str(parsed.max().date()),
        "days": int((parsed.max() - parsed.min()).days),
        "distinct_months": int(parsed.dt.to_period("M").nunique()),
    }


def suggested_questions(roles, coverage):
    """Questions guaranteed to be answerable, because they name real columns."""
    numeric, categorical = roles["numeric"], roles["categorical"]
    questions = []

    if numeric and categorical:
        questions.append(f"What is total {numeric[0]} by {categorical[0]}?")
        questions.append(f"Which {categorical[0]} has the highest {numeric[0]}?")
    if numeric and coverage:
        questions.append(f"Show the monthly trend of {numeric[0]}")
    if len(categorical) >= 2 and numeric:
        questions.append(f"Break down {numeric[0]} by {categorical[0]} and {categorical[1]}")
    if len(numeric) >= 2:
        questions.append(f"How does {numeric[0]} compare with {numeric[1]}?")
    if categorical:
        questions.append(f"How many records are there per {categorical[0]}?")
    return questions[:6]


def _descriptive_sample(df):
    """A sample for the descriptive passes, or the frame itself when it is small."""
    cap = config.PROFILE_SAMPLE_ROWS
    if not cap or len(df) <= cap:
        return df, False
    # Fixed seed: re-uploading the same file should not quietly produce a
    # different briefing each time.
    return df.sample(n=cap, random_state=0), True


def build_overview(frames, profile):
    """Full deterministic report for an uploaded workbook."""
    sheets = []
    for name, df in frames.items():
        sample, sampled = _descriptive_sample(df)

        # One parse per date column of the *full* sheet, shared by the column
        # summaries and time coverage. Keyed per sheet, since two sheets can
        # hold different columns under the same name.
        dt_cache = {}

        # Classification is a heuristic over column shape, so it reads the
        # sample; it gets its own cache because those parses cover fewer rows.
        roles = classify_columns(sample, dt_cache=({} if sampled else dt_cache))
        coverage = time_coverage(df, roles, dt_cache)

        sheets.append({
            "name": name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "roles": roles,
            "quality": quality_findings(sample, roles, total_rows=len(df)),
            "column_summaries": column_summaries(sample, roles, full=df, dt_cache=dt_cache),
            "headline_metrics": headline_metrics(df, roles),
            "breakdowns": top_breakdowns(df, roles),
            "correlations": correlations(sample, roles),
            "time_coverage": coverage,
            "suggested_questions": suggested_questions(roles, coverage),
            "profiled_rows": int(len(sample)),
            "sampled": bool(sampled),
        })

    primary = profile.get("primary_sheet") or (sheets[0]["name"] if sheets else None)
    return {
        "primary_sheet": primary,
        "sheet_count": len(sheets),
        "total_rows": sum(s["rows"] for s in sheets),
        "sheets": sheets,
    }

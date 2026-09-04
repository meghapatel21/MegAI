"""Automatic upload briefing.

The narrative needs an LLM, but every finding underneath it is pure pandas -
so these tests assert the findings, which is the part that must never be wrong.
"""

import numpy as np
import pandas as pd
import pytest

from app.insights import profiler


@pytest.fixture
def messy():
    """A sheet with deliberate, findable problems."""
    rng = np.random.default_rng(11)
    n = 200
    df = pd.DataFrame({
        "order_id": range(1, n + 1),
        "order_date": np.repeat(pd.date_range("2025-01-01", periods=10, freq="MS").astype(str), 20),
        "region": rng.choice(["East", "West", "North"], n),
        "revenue": rng.uniform(100, 900, n).round(2),
        "units": rng.integers(1, 10, n),
        "notes": [f"free text {i}" for i in range(n)],
        "currency": ["USD"] * n,                       # constant
        "discount": [np.nan] * n,                      # entirely empty
    })
    df.loc[:110, "region"] = np.nan                    # heavily null (>40%)
    df.loc[0, "revenue"] = 500_000.0                   # outlier
    df.loc[1, "revenue"] = -250.0                      # negative amount
    df = pd.concat([df, df.iloc[[5, 6]]], ignore_index=True)  # duplicates
    return df


def test_classifies_column_roles(messy):
    roles = profiler.classify_columns(messy)
    assert "revenue" in roles["numeric"]
    assert "units" in roles["numeric"]
    assert "order_date" in roles["temporal"]
    assert "region" in roles["categorical"]
    # A sequential key is not a metric, and free text is not a category.
    assert "order_id" in roles["identifier"]
    assert "notes" in roles["identifier"]


def test_finds_duplicate_rows(messy):
    issues = " ".join(f["issue"] for f in profiler.quality_findings(
        messy, profiler.classify_columns(messy)))
    assert "duplicated row" in issues


def test_finds_empty_and_constant_columns(messy):
    findings = profiler.quality_findings(messy, profiler.classify_columns(messy))
    issues = " ".join(f["issue"] for f in findings)
    assert "`discount` is entirely empty" in issues
    assert "currency" in issues and "single value" in issues


def test_finds_high_null_column(messy):
    issues = " ".join(f["issue"] for f in profiler.quality_findings(
        messy, profiler.classify_columns(messy)))
    assert "`region` is" in issues and "empty" in issues


def test_finds_outlier_and_negative_amount(messy):
    issues = " ".join(f["issue"] for f in profiler.quality_findings(
        messy, profiler.classify_columns(messy)))
    assert "extreme value" in issues
    assert "negative value" in issues


def test_findings_are_ordered_by_severity(messy):
    findings = profiler.quality_findings(messy, profiler.classify_columns(messy))
    rank = {"high": 0, "medium": 1, "low": 2}
    severities = [rank[f["severity"]] for f in findings]
    assert severities == sorted(severities)


def test_reports_time_coverage(messy):
    coverage = profiler.time_coverage(messy, profiler.classify_columns(messy))
    assert coverage["column"] == "order_date"
    assert coverage["from"] == "2025-01-01"
    assert coverage["distinct_months"] == 10


def test_breakdown_uses_a_currency_column_over_a_count(messy):
    breakdowns = profiler.top_breakdowns(messy, profiler.classify_columns(messy))
    assert breakdowns
    assert breakdowns[0]["metric"] == "revenue", "should prefer revenue over units"
    assert breakdowns[0]["category"] == "region"
    assert sum(r["share"] for r in breakdowns[0]["rows"]) <= 100.5


def test_detects_correlation():
    n = 200
    base = np.linspace(1, 100, n)
    df = pd.DataFrame({
        "spend": base,
        "clicks": base * 3 + np.random.default_rng(3).normal(0, 2, n),
        "unrelated": np.random.default_rng(4).normal(0, 1, n),
    })
    pairs = profiler.correlations(df, profiler.classify_columns(df))
    assert pairs and pairs[0]["r"] > 0.9
    assert {pairs[0]["a"], pairs[0]["b"]} == {"spend", "clicks"}
    assert "unrelated" not in {p["a"] for p in pairs} | {p["b"] for p in pairs}


def test_correlation_is_reported_as_association_not_cause():
    n = 100
    base = np.linspace(1, 50, n)
    df = pd.DataFrame({"a": base, "b": base * 2})
    pair = profiler.correlations(df, profiler.classify_columns(df))[0]
    assert pair["direction"] in ("together", "inversely")
    assert "cause" not in str(pair).lower()


def test_suggested_questions_name_real_columns(messy):
    roles = profiler.classify_columns(messy)
    coverage = profiler.time_coverage(messy, roles)
    questions = profiler.suggested_questions(roles, coverage)
    assert questions
    known = set(messy.columns)
    for question in questions:
        assert any(col in question for col in known), question


def test_build_overview_covers_every_sheet(messy):
    frames = {"orders": messy, "lookup": pd.DataFrame({"k": [1, 2], "v": ["a", "b"]})}
    overview = profiler.build_overview(frames, {"primary_sheet": "orders"})
    assert overview["sheet_count"] == 2
    assert overview["primary_sheet"] == "orders"
    assert overview["total_rows"] == len(messy) + 2
    assert {s["name"] for s in overview["sheets"]} == {"orders", "lookup"}


def test_handles_an_empty_sheet():
    overview = profiler.build_overview({"blank": pd.DataFrame()}, {"primary_sheet": "blank"})
    sheet = overview["sheets"][0]
    assert sheet["rows"] == 0
    assert sheet["quality"][0]["severity"] == "high"


def test_handles_a_sheet_with_no_numeric_columns():
    df = pd.DataFrame({"name": ["a", "b", "c"], "city": ["x", "y", "z"]})
    overview = profiler.build_overview({"s": df}, {"primary_sheet": "s"})
    sheet = overview["sheets"][0]
    assert sheet["headline_metrics"] == []
    assert sheet["breakdowns"] == []
    assert sheet["correlations"] == []


def test_text_that_looks_datey_is_not_treated_as_a_date():
    df = pd.DataFrame({"date_note": ["not a date"] * 10, "v": range(10)})
    roles = profiler.classify_columns(df)
    assert "date_note" not in roles["temporal"]


def test_correlation_survives_a_single_bad_row():
    """Rank correlation, not Pearson: one typo must not hide a real relationship."""
    n = 400
    rng = np.random.default_rng(5)
    spend = rng.uniform(100, 5000, n).round(2)
    df = pd.DataFrame({"spend": spend, "revenue": (spend * 3 + rng.normal(0, 300, n)).round(2)})

    clean = profiler.correlations(df, profiler.classify_columns(df))
    assert clean and clean[0]["r"] > 0.9

    df.loc[0, "revenue"] = 900_000.0  # one mistyped cell
    dirty = profiler.correlations(df, profiler.classify_columns(df))
    assert dirty and dirty[0]["r"] > 0.9, "a single outlier must not erase the relationship"


def test_revenue_outranks_spend_as_the_headline_metric():
    df = pd.DataFrame({
        "marketing_spend": [10.0, 20.0, 30.0],
        "revenue": [100.0, 200.0, 300.0],
        "region": ["a", "b", "c"],
    })
    roles = profiler.classify_columns(df)
    assert roles["numeric"][0] == "revenue", "leftmost column should not win by default"
    assert profiler.headline_metrics(df, roles)[0]["column"] == "revenue"
    assert "revenue" in profiler.suggested_questions(roles, None)[0]

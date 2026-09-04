import pytest

from app.insights.chart_intent import detect_requested_chart

EXPLICIT = [
    ("show me a bar chart of revenue by region", "bar"),
    ("give me a bar graph", "bar"),
    ("plot this as a column chart", "bar"),
    ("show a pie chart of sales by category", "pie"),
    # A donut is its own type: asking for one must not render a plain pie,
    # and asking for a pie must not render a donut.
    ("give me a donut chart", "donut"),
    ("show it as a doughnut", "donut"),
    ("as a pie graph please", "pie"),
    ("show the trend as a line chart", "line"),
    ("line graph of revenue over time", "line"),
    ("plot a line plot of daily sessions", "line"),
    ("show it as an area chart", "area"),
    ("scatter plot of spend vs revenue", "scatter"),
    ("give me a scatterplot", "scatter"),
    ("show a histogram of order values", "histogram"),
    ("distribution chart of ages", "histogram"),
    ("just give me a table", "table"),
    ("show the raw data, no chart", "table"),
    ("show me the numbers without a chart", "table"),
    ("without visualization please", "table"),
    ("frequency chart of ticket types", "frequency"),
    ("count chart by department", "frequency"),
]

IMPLICIT = [
    "total revenue by region",
    "what is the average order value",
    "show me revenue for the East region",
    "how many customers signed up last month",
    "compare marketing spend across channels",
    "what's the trend in revenue",  # "trend" alone must not force a chart type
    "list the top 10 products by revenue",
]


@pytest.mark.parametrize("question,expected", EXPLICIT)
def test_detects_explicit_chart_request(question, expected):
    assert detect_requested_chart(question) == expected


@pytest.mark.parametrize("question", IMPLICIT)
def test_returns_none_when_nothing_explicit(question):
    assert detect_requested_chart(question) is None


def test_returns_none_for_empty_or_missing_question():
    assert detect_requested_chart("") is None
    assert detect_requested_chart(None) is None


def test_does_not_false_positive_on_words_containing_line():
    for question in ["compare our airline bookings", "show the pipeline status",
                      "what is on our timeline", "outline the top products"]:
        assert detect_requested_chart(question) is None, question


def test_is_case_insensitive():
    assert detect_requested_chart("SHOW ME A BAR CHART") == "bar"
    assert detect_requested_chart("Pie Chart of revenue") == "pie"


def test_first_matching_pattern_wins_when_multiple_named():
    # Not a real-world phrasing, but the behaviour should be deterministic.
    result = detect_requested_chart("scatter plot or pie chart, whichever - scatter first")
    assert result == "scatter"

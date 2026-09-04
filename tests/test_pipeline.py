"""Impact -> Execute -> BI, plus the reflexion routing.

The Analysis Agent is stubbed with fixed code so these tests never call an LLM.
"""

import ast
import types

import pytest

from app import config
from app.agents import bi_agent, critic_agent, execute_agent, impact_agent

GROUP_BY_REGION = "result = df.groupby('region', as_index=False)['revenue'].sum()"


def run_chain(state, code):
    """One pass of the real pipeline: impact -> execute -> critic -> bi."""
    state["code"] = code
    state = impact_agent.run(state)
    if not state.get("error"):
        state = execute_agent.run(state)
    if not state.get("error"):
        state = critic_agent.run(state)
    return bi_agent.run(state)


# --- execution ------------------------------------------------------------

def test_groupby_produces_rows_and_kpis(base_state):
    response = run_chain(base_state, GROUP_BY_REGION)["response"]
    assert response["status"] == "ok"
    assert len(response["data"]) == 4
    assert response["metric_column"] == "revenue"
    assert response["kpis"]["is_currency"] is True
    assert response["kpis"]["primary_val"] > 0


def test_scalar_result_is_wrapped(base_state):
    response = run_chain(base_state, "result = df['revenue'].sum()")["response"]
    assert response["status"] == "ok"
    assert len(response["data"]) == 1
    assert response["chart_hint"] == "single_value"


def test_series_result_is_wrapped(base_state):
    response = run_chain(base_state, "result = df.groupby('region')['revenue'].mean()")["response"]
    assert response["status"] == "ok"
    assert len(response["data"]) == 4


def test_cross_sheet_merge(base_state):
    code = ("result = q1_orders.merge(customers, on='customer_id', how='left')"
            ".groupby('country', as_index=False)['revenue'].sum()")
    response = run_chain(base_state, code)["response"]
    assert response["status"] == "ok"
    assert {r["country"] for r in response["data"]} <= {"US", "UK", "India"}


def test_execute_cannot_reach_builtins(base_state):
    """Even if the audit were bypassed, the namespace has no dangerous builtins."""
    base_state["code"] = "result = open('x')"
    state = execute_agent.run(base_state)  # deliberately skips impact_agent
    assert state["error"] and "NameError" in state["error"]


# --- honest reporting -----------------------------------------------------

def test_failure_is_reported_not_fabricated(base_state):
    response = run_chain(base_state, "result = df.groupby('nope')['revenue'].sum()")["response"]
    assert response["status"] == "failed"
    assert "KeyError" in response["error"]
    assert response["kpis"] == {}
    assert response["data"] == []


def test_trend_only_when_time_column_present(base_state):
    with_time = run_chain(dict(base_state), (
        "m = pd.to_datetime(df['order_date']).dt.to_period('M').astype(str)\n"
        "result = df.assign(month=m).groupby('month', as_index=False)['revenue'].sum()"
    ))["response"]
    assert with_time["chart_hint"] == "line"
    assert "trend_pct" in with_time["kpis"]

    without_time = run_chain(dict(base_state), GROUP_BY_REGION)["response"]
    assert "trend_pct" not in without_time["kpis"], "must not invent a trend"


def test_reasoning_cites_the_source_file(base_state):
    response = run_chain(base_state, GROUP_BY_REGION)["response"]
    assert "sales_report.xlsx" in response["reasoning"]
    assert "Governance" in response["reasoning"]


# --- reflexion routing ----------------------------------------------------

@pytest.fixture(scope="module")
def routing():
    """Load graph.py's predicates without importing langgraph."""
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "app" / "langgraph" / "graph.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    mod = types.ModuleType("routing")
    mod.__dict__["config"] = config
    exec(compile(ast.Module(body=fns, type_ignores=[]), "graph.py", "exec"), mod.__dict__)
    return mod


@pytest.fixture(autouse=True)
def isolate_rag(monkeypatch):
    """Keep the few-shot corpus out of these tests' way and off real disk."""
    from app.rag import query_memory
    monkeypatch.setattr(query_memory, "remember", lambda **kwargs: None)


@pytest.mark.parametrize("error,attempts,expected", [
    (None, 1, "execute"),
    ("rejected", 1, "analysis"),
    ("rejected", config.MAX_ANALYSIS_ATTEMPTS, "bi"),
])
def test_impact_routing(routing, error, attempts, expected):
    assert routing._after_impact({"error": error, "attempts": attempts}) == expected


@pytest.mark.parametrize("error,attempts,expected", [
    (None, 1, "critic"),
    ("boom", 1, "analysis"),
    ("boom", config.MAX_ANALYSIS_ATTEMPTS, "bi"),
])
def test_execute_routing(routing, error, attempts, expected):
    assert routing._after_execute({"error": error, "attempts": attempts}) == expected


@pytest.mark.parametrize("error,expected", [
    (None, "advance"),
    ("result did not answer the question", "analysis"),
])
def test_critic_routing(routing, error, expected):
    # The Critic only sets `error` when a retry is still affordable, so the
    # routing itself does not need to re-check the attempt budget.
    assert routing._after_critic({"error": error, "attempts": 1}) == expected


@pytest.mark.parametrize("step_index,steps,expected", [
    (1, 1, "insight"),   # only step finished
    (1, 3, "analysis"),  # two steps left
    (3, 3, "insight"),   # last step finished
])
def test_advance_routing(routing, step_index, steps, expected):
    state = {"step_index": step_index, "plan": [{"step": f"s{i}"} for i in range(steps)]}
    assert routing._after_advance(state) == expected


def test_clarify_routing(routing):
    assert routing._after_clarify({"needs_clarification": True}) == "bi"
    assert routing._after_clarify({"needs_clarification": False}) == "rag"


def test_advance_records_step_and_resets_budget(routing):
    state = {
        "plan": [{"step": "one"}, {"step": "two"}],
        "step_index": 0,
        "current_step": "one",
        "code": "result = df",
        "result": [{"region": "East", "revenue": 10}],
        "attempts": 2,
    }
    out = routing.advance(state)
    assert out["step_index"] == 1
    assert out["attempts"] == 0, "each step gets a fresh retry budget"
    assert out["step_results"][0]["columns"] == ["region", "revenue"]


def test_advance_keeps_budget_on_final_step(routing):
    state = {
        "plan": [{"step": "only"}],
        "step_index": 0,
        "current_step": "only",
        "code": "result = df",
        "result": [{"a": 1}],
        "attempts": 2,
    }
    out = routing.advance(state)
    assert out["step_index"] == 1
    assert out["attempts"] == 2, "no reset once the plan is done"


def drive(routing, base_state, codes, plan=None):
    """Walk the graph edges by hand against the real agents.

    Only the Analysis Agent is stubbed (by `codes`); impact, execute, critic,
    advance and bi are the real implementations.
    """
    state = dict(base_state, attempts=0, step_index=0, step_results=[],
                 plan=plan or [{"step": base_state["question"]}])
    node, path, guard = "analysis", [], 0

    while True:
        guard += 1
        assert guard < 60, f"graph did not terminate: {path}"

        if node == "analysis":
            state["attempts"] += 1
            # Mirrors the real agent: a multi-step plan selects by step, while a
            # single-step run walks the list as successive retries.
            index = (state["step_index"] if len(state["plan"]) > 1
                     else state["attempts"] - 1)
            state["code"] = codes[min(index, len(codes) - 1)]
            state["current_step"] = state["plan"][state["step_index"]]["step"]
            state["error"] = None
            path.append(f"analysis#{state['attempts']}")
            node = "impact"
        elif node == "impact":
            state = impact_agent.run(state)
            path.append("impact")
            node = routing._after_impact(state)
        elif node == "execute":
            state = execute_agent.run(state)
            path.append("execute")
            node = routing._after_execute(state)
        elif node == "critic":
            state = critic_agent.run(state)
            path.append("critic")
            node = routing._after_critic(state)
        elif node == "advance":
            state = routing.advance(state)
            path.append("advance")
            node = routing._after_advance(state)
        elif node == "insight":
            path.append("insight")
            node = "bi"
        else:
            state = bi_agent.run(state)
            path.append("bi")
            return state, path


def test_recovers_after_failures(routing, base_state):
    state, path = drive(routing, base_state, [
        "result = df.groupby('missing')['revenue'].sum()",  # runtime error
        "import os\nresult = os.listdir('.')",              # governance block
        GROUP_BY_REGION,                                    # finally valid
    ])
    assert state["attempts"] == 3
    assert state["response"]["status"] == "ok"
    assert "Self-correction" in state["response"]["reasoning"]
    assert path.count("analysis#3") == 1


def test_gives_up_after_max_attempts(routing, base_state):
    state, _ = drive(routing, base_state, ["result = df['nope'].sum()"])
    assert state["attempts"] == config.MAX_ANALYSIS_ATTEMPTS
    assert state["response"]["status"] == "failed"
    assert state["response"]["kpis"] == {}


def test_rejected_code_never_executes(routing, base_state):
    _, path = drive(routing, base_state, ["result = df.to_csv('/tmp/leak.csv')"])
    assert "execute" not in path
    assert path.count("impact") == config.MAX_ANALYSIS_ATTEMPTS


# --- multi-step plans -----------------------------------------------------

def test_multi_step_plan_passes_prev_between_steps(routing, base_state):
    """Step 2 must be able to build on step 1's output via `prev`."""
    plan = [
        {"step": "revenue by region"},
        {"step": "keep only the top 2 regions"},
    ]
    state, path = drive(routing, base_state, [
        "result = df.groupby('region', as_index=False)['revenue'].sum()",
        "result = prev.nlargest(2, 'revenue')",
    ], plan=plan)

    assert state["response"]["status"] == "ok"
    assert len(state["response"]["data"]) == 2, "second step should narrow to 2 rows"
    assert path.count("advance") == 2
    assert state["response"]["plan"] == [p["step"] for p in plan]
    assert len(state["response"]["step_results"]) == 2


def test_each_plan_step_gets_its_own_retry_budget(routing, base_state):
    plan = [{"step": "one"}, {"step": "two"}]
    # Step 1 succeeds on the first try; the codes list is indexed by attempt,
    # so a fresh budget means step 2 starts back at index 0 and also succeeds.
    state, path = drive(routing, base_state, [
        "result = df.groupby('region', as_index=False)['revenue'].sum()",
    ], plan=plan)
    assert state["response"]["status"] == "ok"
    assert path.count("advance") == 2


def test_prev_is_none_on_the_first_step(base_state):
    base_state["code"] = "result = pd.DataFrame([{'is_none': prev is None}])"
    state = execute_agent.run(base_state)
    assert state["result"][0]["is_none"] is True


# --- critic ---------------------------------------------------------------

def test_critic_flags_empty_result(base_state):
    state = run_chain(base_state, "result = df[df['revenue'] > 1e12]")
    # No LLM in tests, so the Critic reports the concern rather than retrying.
    assert state.get("warning")
    assert "no rows" in state["warning"]


def test_critic_flags_all_nan_column(base_state):
    base_state["code"] = "result = df.assign(broken=np.nan)[['region', 'broken']].head(5)"
    state = impact_agent.run(base_state)
    state = execute_agent.run(state)
    state = critic_agent.run(state)
    assert state["warning"] and "NaN" in state["warning"]


def test_critic_flags_single_row_when_breakdown_requested(base_state):
    base_state["question"] = "revenue by region"
    base_state["corrected_question"] = "revenue by region"
    base_state["current_step"] = "revenue by region"
    base_state["code"] = "result = df['revenue'].sum()"
    state = impact_agent.run(base_state)
    state = execute_agent.run(state)
    state = critic_agent.run(state)
    assert state["warning"] and "breakdown" in state["warning"]


def test_critic_passes_a_clean_result(base_state):
    state = run_chain(base_state, GROUP_BY_REGION)
    assert state["critique"]["ok"] is True
    assert not state.get("warning")


def test_critic_does_not_run_after_a_hard_error(base_state):
    base_state["error"] = "KeyError: 'x'"
    state = critic_agent.run(base_state)
    assert state["error"] == "KeyError: 'x'", "the reflexion edge owns hard errors"


def test_warning_surfaces_in_the_response(base_state):
    state = run_chain(base_state, "result = df[df['revenue'] > 1e12]")
    assert "Check this result" in state["response"]["reasoning"]


# --- explicit chart requests -----------------------------------------------

def test_honors_explicit_bar_chart_request(base_state):
    base_state["question"] = "show me a bar chart of revenue by region"
    base_state["corrected_question"] = "show me a bar chart of revenue by region"
    response = run_chain(base_state, GROUP_BY_REGION)["response"]
    assert response["chart_hint"] == "bar"
    assert response["requested_chart"] == "bar"
    assert response["chart_note"] is None
    assert "as requested" in response["reasoning"]


def test_honors_explicit_pie_chart_request(base_state):
    base_state["question"] = "pie chart of revenue by region"
    base_state["corrected_question"] = "pie chart of revenue by region"
    response = run_chain(base_state, GROUP_BY_REGION)["response"]
    assert response["chart_hint"] == "pie"
    assert response["requested_chart"] == "pie"


def test_honors_explicit_line_chart_request_over_time(base_state):
    base_state["question"] = "show a line chart of revenue by month"
    base_state["corrected_question"] = "show a line chart of revenue by month"
    code = ("m = pd.to_datetime(df['order_date']).dt.to_period('M').astype(str)\n"
            "result = df.assign(month=m).groupby('month', as_index=False)['revenue'].sum()")
    response = run_chain(base_state, code)["response"]
    assert response["chart_hint"] == "line"
    assert response["requested_chart"] == "line"


def test_honors_explicit_scatter_request(base_state):
    base_state["question"] = "scatter plot of revenue vs quantity"
    base_state["corrected_question"] = "scatter plot of revenue vs quantity"
    response = run_chain(base_state, "result = df[['revenue', 'customer_id']]")["response"]
    assert response["chart_hint"] == "scatter"
    assert response["requested_chart"] == "scatter"


def test_honors_explicit_table_request(base_state):
    base_state["question"] = "just give me a table of revenue by region"
    base_state["corrected_question"] = "just give me a table of revenue by region"
    response = run_chain(base_state, GROUP_BY_REGION)["response"]
    assert response["chart_hint"] == "table"
    assert response["requested_chart"] == "table"


def test_falls_back_with_note_when_request_is_incompatible(base_state):
    # Scalar result has nothing to group by, so a pie chart is impossible.
    base_state["question"] = "show me a pie chart of total revenue"
    base_state["corrected_question"] = "show me a pie chart of total revenue"
    response = run_chain(base_state, "result = df['revenue'].sum()")["response"]
    assert response["requested_chart"] == "pie"
    assert response["chart_hint"] != "pie"
    assert response["chart_note"] is not None
    assert "pie" in response["chart_note"]
    assert "you asked for a pie chart" in response["reasoning"].lower() or \
           "pie chart" in response["reasoning"].lower()


def test_falls_back_with_note_for_scatter_with_one_numeric_column(base_state):
    base_state["question"] = "scatter plot of revenue by region"
    base_state["corrected_question"] = "scatter plot of revenue by region"
    response = run_chain(base_state, GROUP_BY_REGION)["response"]
    assert response["requested_chart"] == "scatter"
    assert response["chart_hint"] != "scatter"
    assert response["chart_note"] is not None


def test_no_explicit_request_behaves_exactly_as_before(base_state):
    response = run_chain(base_state, GROUP_BY_REGION)["response"]
    assert response["requested_chart"] is None
    assert response["chart_note"] is None
    assert response["chart_hint"] == "pie"  # unchanged auto behaviour: 4 rows, metric+category


def test_single_row_ignores_incompatible_chart_request(base_state):
    base_state["question"] = "bar chart of total revenue"
    base_state["corrected_question"] = "bar chart of total revenue"
    response = run_chain(base_state, "result = df['revenue'].sum()")["response"]
    assert response["chart_hint"] == "single_value"
    assert response["requested_chart"] == "bar"
    assert "single value" in response["chart_note"]


def test_single_row_honors_table_request():
    from app.agents import bi_agent
    df_hint, note = bi_agent._chart_hint(
        __import__("pandas").DataFrame([{"revenue": 100}]), "revenue", requested="table"
    )
    assert df_hint == "table" and note is None


# --- NaN / Infinity can never reach the JSON response ----------------------

def test_response_is_json_safe_with_missing_values(base_state):
    """An entirely-empty column (all NaN) must not 500 the response.

    Regression: df.to_dict("records") on a NaN-containing column produces
    Python float('nan'), which Starlette's JSONResponse refuses to encode
    (ValueError: Out of range float values are not JSON compliant).
    """
    import json
    base_state["code"] = "result = df.assign(empty_col=np.nan)[['region', 'revenue', 'empty_col']]"
    response = run_chain(base_state, base_state["code"])["response"]
    assert response["status"] == "ok"
    assert any(row["empty_col"] is None for row in response["data"])
    json.dumps(response)  # must not raise


def test_response_is_json_safe_with_infinite_values(base_state):
    import json
    base_state["code"] = "result = df.assign(bad=1 / (df['revenue'] - df['revenue']))[['region', 'bad']]"
    response = run_chain(base_state, base_state["code"])["response"]
    assert response["status"] == "ok"
    assert all(row["bad"] is None for row in response["data"])
    json.dumps(response)


def test_trend_is_omitted_not_nan_when_an_endpoint_is_missing(base_state):
    """A NaN trend must not silently become a `trend_pct: None` key either -
    the render layer treats `"trend_pct" in kpis` as "safe to format as a
    percentage", so the key must be absent, not present-with-None."""
    code = (
        "m = pd.to_datetime(df['order_date']).dt.to_period('M').astype(str)\n"
        "g = df.assign(month=m).groupby('month', as_index=False)['revenue'].sum()\n"
        "g.loc[0, 'revenue'] = np.nan\n"  # blank out the first period's total
        "result = g"
    )
    response = run_chain(base_state, code)["response"]
    assert response["status"] == "ok"
    assert "trend_pct" not in response["kpis"]


def test_json_safe_handles_nested_structures():
    from app.agents.bi_agent import _json_safe
    assert _json_safe(float("nan")) is None
    assert _json_safe(float("inf")) is None
    assert _json_safe(float("-inf")) is None
    assert _json_safe(3.5) == 3.5
    assert _json_safe({"a": float("nan"), "b": [1, float("inf"), {"c": float("nan")}]}) == {
        "a": None, "b": [1, None, {"c": None}]
    }
    assert _json_safe("text") == "text"
    assert _json_safe(None) is None

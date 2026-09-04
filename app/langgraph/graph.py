"""Agent graph.

  profile ─▶ clarify ─┬─(ambiguous)──────────────────────────────────▶ bi ─▶ END
                      │
                      └─▶ rag ─▶ planner ─▶ analysis ─▶ impact ─▶ execute ─▶ critic
                                              ▲            │         │         │
                                              │            │         │         ├─(ok)─▶ advance
                                              └────────────┴─────────┴─────────┘         │
                                                      reflexion on error          ┌──────┴──────┐
                                                                          (next step)     (done)
                                                                                │            │
                                                                                ▼            ▼
                                                                            analysis      insight ─▶ bi ─▶ END

Three things can send work back to the Analysis Agent, all through the same
edge and the same `state["error"]` field:

  * Impact  - the generated code failed the safety audit
  * Execute - the code raised
  * Critic  - the code ran but did not answer the question

Every path is bounded. `attempts` caps retries within a step, `step_index` only
moves forward, and both Impact and Execute fall through to the BI Agent once the
budget is spent - so the graph always terminates.
"""

from langgraph.graph import END, StateGraph

from app import config
from app.agents import (
    analysis_agent,
    bi_agent,
    clarify_agent,
    critic_agent,
    execute_agent,
    impact_agent,
    insight_agent,
    planner_agent,
    profile_agent,
    rag_agent,
)
from app.langgraph.state import BIState


def _retry_or_finish(state):
    """Shared reflexion decision: rewrite the code, or give up and report."""
    return "analysis" if state.get("attempts", 0) < config.MAX_ANALYSIS_ATTEMPTS else "bi"


def _after_clarify(state):
    return "bi" if state.get("needs_clarification") else "rag"


def _after_impact(state):
    return "execute" if not state.get("error") else _retry_or_finish(state)


def _after_execute(state):
    return "critic" if not state.get("error") else _retry_or_finish(state)


def _after_critic(state):
    # The Critic only sets `error` when a retry is still affordable.
    return "analysis" if state.get("error") else "advance"


def _after_advance(state):
    plan = state.get("plan") or []
    return "analysis" if state.get("step_index", 0) < len(plan) else "insight"


def advance(state):
    """Record the finished step and move to the next one, if any."""
    plan = state.get("plan") or []
    results = list(state.get("step_results") or [])
    rows = state.get("result") or []

    results.append({
        "step": state.get("current_step", ""),
        "code": state.get("code", ""),
        "columns": list(rows[0].keys()) if rows else [],
        "rows": rows,
    })
    state["step_results"] = results
    state["step_index"] = state.get("step_index", 0) + 1

    if state["step_index"] < len(plan):
        # Each step gets its own retry budget.
        state["attempts"] = 0
        state["error"] = None
        print(f"[PLAN] step {state['step_index']} of {len(plan)}")

    return state


graph = StateGraph(BIState)
graph.add_node("profile", profile_agent.run)
graph.add_node("clarify", clarify_agent.run)
graph.add_node("rag", rag_agent.run)
graph.add_node("planner", planner_agent.run)
graph.add_node("analysis", analysis_agent.run)
graph.add_node("impact", impact_agent.run)
graph.add_node("execute", execute_agent.run)
graph.add_node("critic", critic_agent.run)
graph.add_node("advance", advance)
graph.add_node("insight", insight_agent.run)
graph.add_node("bi", bi_agent.run)

graph.set_entry_point("profile")
graph.add_edge("profile", "clarify")
graph.add_conditional_edges("clarify", _after_clarify, {"rag": "rag", "bi": "bi"})
graph.add_edge("rag", "planner")
graph.add_edge("planner", "analysis")
graph.add_edge("analysis", "impact")
graph.add_conditional_edges("impact", _after_impact,
                            {"execute": "execute", "analysis": "analysis", "bi": "bi"})
graph.add_conditional_edges("execute", _after_execute,
                            {"critic": "critic", "analysis": "analysis", "bi": "bi"})
graph.add_conditional_edges("critic", _after_critic,
                            {"analysis": "analysis", "advance": "advance"})
graph.add_conditional_edges("advance", _after_advance,
                            {"analysis": "analysis", "insight": "insight"})
graph.add_edge("insight", "bi")
graph.add_edge("bi", END)

# Each step may burn MAX_ANALYSIS_ATTEMPTS, so the ceiling scales with the plan.
bi_graph = graph.compile()
RECURSION_LIMIT = config.MAX_PLAN_STEPS * config.MAX_ANALYSIS_ATTEMPTS * 6 + 20

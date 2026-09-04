"""Impact Agent - governance gate in front of code execution.

Generated code is untrusted: the model that writes it takes the user's question
as input, so a prompt-injected question could otherwise reach `exec`. This agent
parses the code and rejects anything that reads or writes outside the in-memory
DataFrames before the Execute Agent runs it.
"""

import ast

# Callables that give access to the interpreter, the filesystem, or the network.
BANNED_NAMES = {
    "eval", "exec", "compile", "open", "input", "breakpoint", "help",
    "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "exit", "quit", "memoryview", "id", "dir",
}

# Attribute calls that would persist data somewhere or pull it in from outside.
BANNED_ATTRS = {
    "to_csv", "to_excel", "to_pickle", "to_sql", "to_parquet", "to_hdf",
    "to_feather", "to_clipboard", "to_gbq", "to_orc", "to_stata", "to_xml",
    "read_csv", "read_excel", "read_sql", "read_pickle", "read_parquet",
    "read_json", "read_html", "read_clipboard", "read_gbq", "read_hdf",
    "system", "popen", "remove", "unlink", "rmdir", "chmod", "rename",
    "eval", "pipe", "apply_async", "load", "loads", "dump", "dumps",
}


class CodeRejected(Exception):
    """Raised when generated code fails the safety gate."""


class _Auditor(ast.NodeVisitor):
    def __init__(self):
        self.assigns_result = False
        self.table_ops = 0

    def visit_Import(self, node):
        raise CodeRejected("imports are not allowed; pandas is already available as `pd`")

    def visit_ImportFrom(self, node):
        raise CodeRejected("imports are not allowed; pandas is already available as `pd`")

    def visit_Attribute(self, node):
        # Dunder access is the standard route out of a restricted namespace
        # (obj.__class__.__bases__[0].__subclasses__() and friends).
        if node.attr.startswith("__"):
            raise CodeRejected(f"access to dunder attribute '{node.attr}' is not allowed")
        if node.attr in BANNED_ATTRS:
            raise CodeRejected(f"'{node.attr}' is not allowed - the result must stay in memory")
        if node.attr in {"groupby", "merge", "join", "pivot_table", "sort_values", "agg"}:
            self.table_ops += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in BANNED_NAMES:
            raise CodeRejected(f"use of '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "result":
                self.assigns_result = True
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == "result":
            self.assigns_result = True
        self.generic_visit(node)

    # Comprehension/lambda bodies are visited by generic_visit already; the
    # Name and Attribute guards above cover them.


def review(code: str) -> dict:
    """Audit generated code. Raises CodeRejected, or returns an impact summary."""
    if not code or not code.strip():
        raise CodeRejected("no code was generated")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CodeRejected(f"generated code is not valid Python: {exc}") from exc

    auditor = _Auditor()
    auditor.visit(tree)

    if not auditor.assigns_result:
        raise CodeRejected("code must assign its answer to a variable named `result`")

    complexity = "low" if auditor.table_ops <= 1 else "moderate" if auditor.table_ops <= 3 else "high"
    return {
        "approved": True,
        "statements": len(tree.body),
        "table_operations": auditor.table_ops,
        "estimated_cost": complexity,
    }


def run(state):
    try:
        state["impact"] = review(state.get("code", ""))
        state["error"] = None
        print(f"[IMPACT] approved ({state['impact']['estimated_cost']} cost)")
    except CodeRejected as exc:
        # Routed back to the Analysis Agent by the graph, same as a runtime error.
        state["impact"] = {"approved": False, "reason": str(exc)}
        state["error"] = f"Rejected by governance: {exc}"
        print(f"[IMPACT] rejected: {exc}")
    return state

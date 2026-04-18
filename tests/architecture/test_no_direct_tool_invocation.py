"""Heuristic AST guard against bypassing ToolRuntime mediation."""

from __future__ import annotations

from tests.architecture._helpers import (
    format_reference,
    is_approved_direct_invocation_path,
    iter_direct_invocations,
)


def test_no_direct_tool_invoke_or_ainvoke_calls_outside_approved_locations() -> None:
    violations = [
        format_reference(
            reference.path,
            reference.lineno,
            f"direct .{reference.method_name}(...) call is only allowed in "
            "ToolRuntime or tests",
        )
        for reference in iter_direct_invocations()
        if not is_approved_direct_invocation_path(reference.path)
    ]

    assert not violations, (
        "Direct tool invocation bypasses runtime mediation:\n" + "\n".join(violations)
    )

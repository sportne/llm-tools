"""Heuristic AST guard against bypassing ToolRuntime mediation."""

from __future__ import annotations

import textwrap
from pathlib import Path

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
            f".{reference.method_name}(...) bypass is only allowed in ToolRuntime or tests",
        )
        for reference in iter_direct_invocations()
        if not is_approved_direct_invocation_path(reference.path)
    ]

    assert not violations, "Tool invocation bypasses runtime mediation:\n" + "\n".join(
        violations
    )


def test_iter_direct_invocations_flags_getattr_and_cached_bound_methods(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        """
        def call(tool, context, payload):
            getattr(tool, "invoke")(context, payload)
            cached = getattr(tool, "ainvoke")
            runner = cached
            return runner(context, payload)
        """,
    )

    references = list(iter_direct_invocations(tmp_path))

    assert [reference.method_name for reference in references] == ["invoke", "ainvoke"]


def test_iter_direct_invocations_flags_class_level_tool_calls(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
        def call(ToolClass, tool, context, payload):
            return ToolClass.invoke(tool, context, payload)
        """,
    )

    references = list(iter_direct_invocations(tmp_path))

    assert [reference.method_name for reference in references] == ["invoke"]


def test_iter_direct_invocations_ignores_dynamic_getattr_and_rebound_names(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        """
        def local_invoke(*args):
            return args

        def call(tool, method_name, context, payload):
            cached = tool.invoke
            cached = local_invoke
            cached(context, payload)
            return getattr(tool, method_name)(context, payload)
        """,
    )

    assert list(iter_direct_invocations(tmp_path)) == []


def _write_module(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return path

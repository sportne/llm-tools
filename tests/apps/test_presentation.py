"""Tests for workbench presentation helpers."""

from __future__ import annotations

from llm_tools.apps.textual_workbench.presentation import (
    append_event,
    build_tool_details_payload,
    extract_execution_record,
    format_event,
    pretty_json,
    sanitize_tool_result,
)
from llm_tools.tool_api import ToolResult


def test_presentation_helpers_cover_basic_and_empty_cases() -> None:
    assert pretty_json(None) == ""
    assert format_event("  hello  ") == "hello"
    assert append_event([], " one ") == "one"
    assert build_tool_details_payload(spec_payload=None, input_schema=None) is None
    assert extract_execution_record(None) is None


def test_extract_execution_record_returns_attached_payload() -> None:
    result = ToolResult(
        ok=True,
        tool_name="demo",
        tool_version="0.1.0",
        output={},
        metadata={"execution_record": {"tool_name": "demo"}},
    )

    assert extract_execution_record(result) == {"tool_name": "demo"}
    assert sanitize_tool_result(result)["metadata"] == {}

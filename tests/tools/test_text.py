"""Unit tests for text-search built-in tools."""

from __future__ import annotations

from llm_tools.tool_api import (
    ToolContext,
    ToolInvocationRequest,
    ToolRegistry,
    ToolRuntime,
)
from llm_tools.tools.filesystem import SearchTextTool


def _runtime() -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(SearchTextTool())
    return ToolRuntime(registry)


def test_search_text_tool_finds_matches_in_a_single_file(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "notes.txt"
    file_path.write_text("Hello\nworld\nHello again\n", encoding="utf-8")

    result = _runtime().execute(
        ToolInvocationRequest(
            tool_name="search_text",
            arguments={"path": "notes.txt", "query": "Hello"},
        ),
        ToolContext(invocation_id="inv-1", workspace=str(workspace)),
    )

    assert result.ok is True
    matches = result.output["matches"]
    assert [match["line_number"] for match in matches] == [1, 3]
    assert matches[0]["line_text"] == "Hello"


def test_search_text_tool_searches_directory_contents(tmp_path: str) -> None:
    workspace = tmp_path
    (workspace / "keep.py").write_text("value = 42\nanswer = 42\n", encoding="utf-8")
    (workspace / "skip.txt").write_text("42\n", encoding="utf-8")
    nested = workspace / "pkg"
    nested.mkdir()
    (nested / "match.py").write_text("value = 99\n", encoding="utf-8")

    result = _runtime().execute(
        ToolInvocationRequest(
            tool_name="search_text",
            arguments={"path": ".", "query": "value = "},
        ),
        ToolContext(
            invocation_id="inv-2",
            workspace=str(workspace),
            metadata={"source_filters": {"include": ["*.py", "**/*.py"]}},
        ),
    )

    assert result.ok is True
    matches = result.output["matches"]
    assert [item["path"] for item in matches] == ["keep.py", "pkg/match.py"]
    assert matches[0]["line_number"] == 1

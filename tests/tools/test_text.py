"""Unit tests for text-search built-in tools."""

from __future__ import annotations

from llm_tools.tool_api import ToolContext
from llm_tools.tools.text import SearchTextTool


def test_search_text_tool_finds_matches_in_a_single_file(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "notes.txt"
    file_path.write_text("Hello\nworld\nHello again\n", encoding="utf-8")

    result = SearchTextTool().invoke(
        ToolContext(invocation_id="inv-1", workspace=str(workspace)),
        SearchTextTool.input_model(path="notes.txt", query="Hello"),
    )

    assert [match.line_number for match in result.matches] == [1, 3]
    assert result.matches[0].line_text == "Hello"


def test_search_text_tool_searches_directory_contents(tmp_path: str) -> None:
    workspace = tmp_path
    (workspace / "keep.py").write_text("value = 42\nanswer = 42\n", encoding="utf-8")
    (workspace / "skip.txt").write_text("42\n", encoding="utf-8")
    nested = workspace / "pkg"
    nested.mkdir()
    (nested / "match.py").write_text("value = 99\n", encoding="utf-8")

    result = SearchTextTool().invoke(
        ToolContext(
            invocation_id="inv-2",
            workspace=str(workspace),
            metadata={"source_filters": {"include": ["*.py", "**/*.py"]}},
        ),
        SearchTextTool.input_model(path=".", query="value = "),
    )

    assert [item.path for item in result.matches] == ["keep.py", "pkg/match.py"]
    assert result.matches[0].line_number == 1

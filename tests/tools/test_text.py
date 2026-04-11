"""Unit tests for text-search built-in tools."""

from __future__ import annotations

from llm_tools.tool_api import ToolContext
from llm_tools.tools.text import DirectoryTextSearchTool, FileTextSearchTool


def test_file_text_search_tool_finds_case_insensitive_matches(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "notes.txt"
    file_path.write_text("Hello\nworld\nHELLO again\n", encoding="utf-8")

    result = FileTextSearchTool().invoke(
        ToolContext(invocation_id="inv-1", workspace=str(workspace)),
        FileTextSearchTool.input_model(path="notes.txt", query="hello"),
    )

    assert [match.line_number for match in result.matches] == [1, 3]
    assert result.matches[0].matched_texts == ["Hello"]


def test_directory_text_search_tool_filters_by_glob_and_regex(tmp_path: str) -> None:
    workspace = tmp_path
    (workspace / "keep.py").write_text("value = 42\nanswer = 42\n", encoding="utf-8")
    (workspace / "skip.txt").write_text("42\n", encoding="utf-8")
    nested = workspace / "pkg"
    nested.mkdir()
    (nested / "match.py").write_text("value = 99\n", encoding="utf-8")

    result = DirectoryTextSearchTool().invoke(
        ToolContext(invocation_id="inv-2", workspace=str(workspace)),
        DirectoryTextSearchTool.input_model(
            path=".",
            query=r"value\s*=\s*\d+",
            regex=True,
            file_glob="*.py",
        ),
    )

    assert [item.path for item in result.results] == ["keep.py", "pkg/match.py"]
    assert result.results[0].matches[0].line_number == 1

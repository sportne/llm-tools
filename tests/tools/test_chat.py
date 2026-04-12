"""Tests for repository chat tools."""

from __future__ import annotations

from pathlib import Path

from llm_tools.tool_api import (
    ToolContext,
    ToolInvocationRequest,
    ToolRegistry,
    ToolRuntime,
)
from llm_tools.tools.chat import register_chat_tools


def _runtime(tmp_path: Path) -> tuple[ToolRuntime, ToolContext]:
    registry = ToolRegistry()
    register_chat_tools(registry)
    runtime = ToolRuntime(registry)
    context = ToolContext(
        invocation_id="chat-test",
        workspace=str(tmp_path),
        metadata={
            "source_filters": {"include_hidden": False},
            "session_config": {
                "max_context_tokens": 10,
                "max_tool_round_trips": 8,
                "max_tool_calls_per_round": 4,
                "max_total_tool_calls_per_turn": 12,
            },
            "tool_limits": {
                "max_entries_per_call": 50,
                "max_recursive_depth": 5,
                "max_search_matches": 3,
                "max_read_lines": 20,
                "max_file_size_characters": 1000,
                "max_read_file_chars": 10,
                "max_tool_result_chars": 200,
            },
        },
    )
    return runtime, context


def test_chat_tools_list_find_and_search(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\nneedle = 2\n", encoding="utf-8")
    (tmp_path / ".hidden" / "secret.py").write_text("needle\n", encoding="utf-8")
    runtime, context = _runtime(tmp_path)

    listing = runtime.execute(
        ToolInvocationRequest(tool_name="list_directory_recursive", arguments={"path": "."}),
        context.model_copy(),
    )
    assert listing.ok is True
    listing_entries = listing.output["entries"]
    assert any(entry["path"] == "src/app.py" for entry in listing_entries)
    assert all(not entry["path"].startswith(".hidden") for entry in listing_entries)

    find_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="find_files",
            arguments={"path": ".", "pattern": "**/*.py"},
        ),
        context.model_copy(),
    )
    assert find_result.ok is True
    assert find_result.output["matches"] == [
        {
            "path": "src/app.py",
            "name": "app.py",
            "parent_path": "src",
            "is_hidden": False,
        }
    ]

    search_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_text",
            arguments={"path": ".", "query": "needle"},
        ),
        context.model_copy(),
    )
    assert search_result.ok is True
    assert len(search_result.output["matches"]) == 2
    assert search_result.output["matches"][0]["path"] == "src/app.py"


def test_chat_tools_get_file_info_and_read_file_apply_limits(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("abcdefghijklmno", encoding="utf-8")
    runtime, context = _runtime(tmp_path)

    info_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="get_file_info",
            arguments={"path": "notes.txt"},
        ),
        context.model_copy(),
    )
    assert info_result.ok is True
    assert info_result.output["results"][0]["requested_path"] == "notes.txt"
    assert info_result.output["results"][0]["can_read_full"] is False

    read_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="read_file",
            arguments={"path": "notes.txt"},
        ),
        context.model_copy(),
    )
    assert read_result.ok is True
    assert read_result.output["content"] == "abcdefghij"
    assert read_result.output["truncated"] is True

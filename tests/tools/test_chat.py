"""Tests for repository chat tools."""

from __future__ import annotations

from pathlib import Path

from llm_tools.tool_api import (
    ToolContext,
    ToolInvocationRequest,
    ToolRegistry,
    ToolRuntime,
)
from llm_tools.tools import register_filesystem_tools, register_text_tools


def _runtime(tmp_path: Path) -> tuple[ToolRuntime, ToolContext]:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    runtime = ToolRuntime(registry)
    context = ToolContext(
        invocation_id="chat-test",
        workspace=str(tmp_path),
        metadata={
            "source_filters": {"include_hidden": False},
            "tool_limits": {
                "max_entries_per_call": 50,
                "max_recursive_depth": 5,
                "max_search_matches": 3,
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
    (tmp_path / "src" / "app.py").write_text(
        "needle = 1\nneedle = 2\n", encoding="utf-8"
    )
    (tmp_path / ".hidden" / "secret.py").write_text("needle\n", encoding="utf-8")
    runtime, context = _runtime(tmp_path)

    listing = runtime.execute(
        ToolInvocationRequest(
            tool_name="list_directory", arguments={"path": ".", "recursive": True}
        ),
        context.model_copy(),
    )
    assert listing.ok is True
    listing_entries = listing.output["entries"]
    assert any(entry["path"] == "src/app.py" for entry in listing_entries)
    assert all(not entry["path"].startswith(".hidden") for entry in listing_entries)
    hidden_listing = runtime.execute(
        ToolInvocationRequest(
            tool_name="list_directory",
            arguments={"path": ".", "recursive": True, "include_hidden": True},
        ),
        context.model_copy(),
    )
    assert hidden_listing.ok is True
    assert any(
        entry["path"] == ".hidden/secret.py"
        for entry in hidden_listing.output["entries"]
    )

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
    hidden_find_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="find_files",
            arguments={
                "path": ".",
                "pattern": "**/*.py",
                "include_hidden": True,
            },
        ),
        context.model_copy(),
    )
    assert hidden_find_result.ok is True
    assert [match["path"] for match in hidden_find_result.output["matches"]] == [
        ".hidden/secret.py",
        "src/app.py",
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
    hidden_search_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_text",
            arguments={"path": ".", "query": "needle", "include_hidden": True},
        ),
        context.model_copy(),
    )
    assert hidden_search_result.ok is True
    assert [match["path"] for match in hidden_search_result.output["matches"]] == [
        ".hidden/secret.py",
        "src/app.py",
        "src/app.py",
    ]
    hidden_file_search_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_text",
            arguments={"path": ".hidden/secret.py", "query": "needle"},
        ),
        context.model_copy(),
    )
    assert hidden_file_search_result.ok is True
    assert hidden_file_search_result.output["matches"] == []
    hidden_file_search_included_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_text",
            arguments={
                "path": ".hidden/secret.py",
                "query": "needle",
                "include_hidden": True,
            },
        ),
        context.model_copy(),
    )
    assert hidden_file_search_included_result.ok is True
    assert [
        match["path"] for match in hidden_file_search_included_result.output["matches"]
    ] == [".hidden/secret.py"]


def test_chat_tools_treat_gitignored_paths_as_hidden(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(
        "ignored/\n*.log\n!keep.log\n",
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "ignored").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "ignored" / "skip.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "app.log").write_text("needle\n", encoding="utf-8")
    (tmp_path / "keep.log").write_text("needle\n", encoding="utf-8")
    runtime, context = _runtime(tmp_path)

    listing = runtime.execute(
        ToolInvocationRequest(
            tool_name="list_directory", arguments={"path": ".", "recursive": True}
        ),
        context.model_copy(),
    )
    assert listing.ok is True
    assert all(
        entry["path"] not in {"ignored", "ignored/skip.py", "app.log"}
        for entry in listing.output["entries"]
    )

    hidden_listing = runtime.execute(
        ToolInvocationRequest(
            tool_name="list_directory",
            arguments={"path": ".", "recursive": True, "include_hidden": True},
        ),
        context.model_copy(),
    )
    assert hidden_listing.ok is True
    hidden_entries = {
        entry["path"]: entry["is_hidden"] for entry in hidden_listing.output["entries"]
    }
    assert hidden_entries["ignored"] is True
    assert hidden_entries["ignored/skip.py"] is True
    assert hidden_entries["app.log"] is True
    assert hidden_entries["keep.log"] is False

    find_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="find_files",
            arguments={"path": ".", "pattern": "**/*.py"},
        ),
        context.model_copy(),
    )
    assert find_result.ok is True
    assert [match["path"] for match in find_result.output["matches"]] == ["src/app.py"]

    hidden_find_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="find_files",
            arguments={"path": ".", "pattern": "**/*.py", "include_hidden": True},
        ),
        context.model_copy(),
    )
    assert hidden_find_result.ok is True
    hidden_matches = {
        match["path"]: match["is_hidden"]
        for match in hidden_find_result.output["matches"]
    }
    assert hidden_matches["ignored/skip.py"] is True
    assert hidden_matches["src/app.py"] is False

    search_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_text",
            arguments={"path": ".", "query": "needle"},
        ),
        context.model_copy(),
    )
    assert search_result.ok is True
    assert [match["path"] for match in search_result.output["matches"]] == [
        "keep.log",
        "src/app.py",
    ]

    hidden_search_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_text",
            arguments={"path": ".", "query": "needle", "include_hidden": True},
        ),
        context.model_copy(),
    )
    assert hidden_search_result.ok is True
    hidden_text_matches = {
        match["path"]: match["is_hidden"]
        for match in hidden_search_result.output["matches"]
    }
    assert hidden_text_matches["ignored/skip.py"] is True

    direct_search_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_text",
            arguments={"path": "ignored/skip.py", "query": "needle"},
        ),
        context.model_copy(),
    )
    assert direct_search_result.ok is True
    assert direct_search_result.output["matches"] == []

    direct_hidden_search_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_text",
            arguments={
                "path": "ignored/skip.py",
                "query": "needle",
                "include_hidden": True,
            },
        ),
        context.model_copy(),
    )
    assert direct_hidden_search_result.ok is True
    assert [
        match["path"] for match in direct_hidden_search_result.output["matches"]
    ] == ["ignored/skip.py"]
    assert direct_hidden_search_result.output["matches"][0]["is_hidden"] is True


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

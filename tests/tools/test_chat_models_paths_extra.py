"""Focused coverage tests for repository chat helper models and paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import llm_tools.tools.chat._ops as ops_module
from llm_tools.tools.chat._content import LoadedReadableContent, convert_with_markitdown
from llm_tools.tools.chat._ops import (
    find_files_impl,
    list_directory_recursive_impl,
    read_file_impl,
    resolve_search_file_or_directory,
    search_text_impl,
)
from llm_tools.tools.chat._paths import (
    build_entry,
    build_file_match,
    build_text_search_match,
    entry_type,
    is_hidden,
    matches_path_glob,
    matches_patterns,
    normalize_requested_path,
    normalize_required_pattern,
    normalize_required_value,
    resolve_directory_path,
    resolve_file_path,
    should_include_entry,
    should_prune_directory,
)
from llm_tools.tools.chat.models import (
    ChatSessionConfig,
    ChatSourceFilters,
    ChatToolLimits,
    GetFileInfoInputShape,
)


def test_chat_tool_models_cover_validation_errors() -> None:
    with pytest.raises(ValidationError):
        ChatSourceFilters(include=[" "])
    with pytest.raises(ValidationError):
        ChatSessionConfig(max_context_tokens=0)
    with pytest.raises(ValidationError):
        ChatToolLimits(max_entries_per_call=0)
    with pytest.raises(ValidationError):
        ChatToolLimits(max_read_file_chars=0)

    assert GetFileInfoInputShape(path=" src/app.py ").path == "src/app.py"
    assert GetFileInfoInputShape(paths=[" a.py ", "b.py"]).paths == ["a.py", "b.py"]
    with pytest.raises(ValidationError):
        GetFileInfoInputShape()
    with pytest.raises(ValidationError):
        GetFileInfoInputShape(path="a.py", paths=["b.py"])
    with pytest.raises(ValidationError):
        GetFileInfoInputShape(path=" ")
    with pytest.raises(ValidationError):
        GetFileInfoInputShape(paths=[])
    with pytest.raises(ValidationError):
        GetFileInfoInputShape(paths=["ok.py", " "])


def test_chat_path_helpers_cover_matching_and_resolution(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    link_dir = tmp_path / "link-src"
    link_file = tmp_path / "link-app.py"
    link_dir.symlink_to(tmp_path / "src", target_is_directory=True)
    link_file.symlink_to(tmp_path / "src" / "app.py")

    assert matches_patterns(Path("src/app.py"), ["src/*.py"]) is True
    assert normalize_requested_path(" src/app.py ") == "src/app.py"
    assert normalize_requested_path(".") == "."
    with pytest.raises(ValueError):
        normalize_requested_path(" ")
    assert normalize_required_value(" x ", field_name="demo") == "x"
    assert normalize_required_pattern(" **/*.py ") == "**/*.py"
    assert matches_path_glob(Path("src/nested/app.py"), "**/*.py") is True
    assert matches_path_glob(Path("src/app.py"), "src/*.py") is True
    assert matches_path_glob(Path("src/app.py"), "*.md") is False

    include_filters = ChatSourceFilters(include=["src/*.py"])
    exclude_filters = ChatSourceFilters(exclude=["src/app.py"])
    assert (
        should_include_entry(Path("src/app.py"), source_filters=include_filters) is True
    )
    assert (
        should_include_entry(Path(".hidden/item"), source_filters=ChatSourceFilters())
        is False
    )
    assert (
        should_include_entry(Path("src/app.py"), source_filters=exclude_filters)
        is False
    )
    assert (
        should_prune_directory(Path(".hidden"), source_filters=ChatSourceFilters())
        is True
    )
    assert should_prune_directory(Path("src"), source_filters=exclude_filters) is False
    assert is_hidden(Path(".hidden/item")) is True

    assert resolve_directory_path(tmp_path, "src").resolved_path == "src"
    assert resolve_file_path(tmp_path, "src/app.py").resolved_path == "src/app.py"
    with pytest.raises(ValueError):
        resolve_directory_path(tmp_path, str((tmp_path / "src").resolve()))
    with pytest.raises(ValueError):
        resolve_directory_path(tmp_path, "missing")
    with pytest.raises(ValueError):
        resolve_file_path(tmp_path, "src")
    with pytest.raises(ValueError):
        resolve_directory_path(tmp_path, "link-src")
    with pytest.raises(ValueError):
        resolve_file_path(tmp_path, "link-app.py")

    built_entry = build_entry(tmp_path, tmp_path / "src", depth=1)
    built_match = build_file_match(tmp_path, tmp_path / "src" / "app.py")
    built_text_match = build_text_search_match(
        tmp_path,
        tmp_path / "src" / "app.py",
        line_number=1,
        line_text="print('hi')",
    )
    assert built_entry.entry_type == "directory"
    assert built_match.parent_path == "src"
    assert built_text_match.path == "src/app.py"
    assert entry_type(link_file) == "symlink"


def test_chat_ops_cover_recursive_listing_and_file_edge_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "ignored").mkdir()
    (tmp_path / "src" / "keep.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "ignored" / "skip.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "single.txt").write_text("needle\nneedle\n", encoding="utf-8")
    (tmp_path / "dir-link").symlink_to(tmp_path / "src", target_is_directory=True)

    with pytest.raises(ValueError):
        list_directory_recursive_impl(
            tmp_path,
            ".",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
            max_depth=0,
        )

    listing = list_directory_recursive_impl(
        tmp_path,
        ".",
        source_filters=ChatSourceFilters(exclude=["ignored/**"]),
        tool_limits=ChatToolLimits(max_entries_per_call=20, max_recursive_depth=5),
        max_depth=3,
    )
    assert all(entry.path != "ignored/skip.py" for entry in listing.entries)
    assert any(entry.path == "dir-link" for entry in listing.entries)
    assert all(not entry.path.startswith("dir-link/") for entry in listing.entries)

    with pytest.raises(ValueError):
        resolve_search_file_or_directory(tmp_path, "dir-link")

    monkeypatch.setattr(
        ops_module,
        "load_readable_content",
        lambda path: LoadedReadableContent(
            read_kind="unsupported",
            status="unsupported",
            content=None,
            error_message="unsupported",
        ),
    )
    unsupported_search = search_text_impl(
        tmp_path,
        "needle",
        "single.txt",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_file_size_characters=100),
    )
    assert unsupported_search.matches == []

    monkeypatch.setattr(
        ops_module,
        "load_readable_content",
        lambda path: LoadedReadableContent(
            read_kind="text",
            status="ok",
            content="needle\nneedle\n",
        ),
    )
    too_large_search = search_text_impl(
        tmp_path,
        "needle",
        "single.txt",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_file_size_characters=1),
    )
    assert too_large_search.matches == []

    monkeypatch.setattr(
        ops_module,
        "load_readable_content",
        lambda path: LoadedReadableContent(
            read_kind="unsupported",
            status="unsupported",
            content=None,
            error_message="unsupported",
        ),
    )
    unsupported_read = read_file_impl(
        tmp_path,
        "single.txt",
        session_config=ChatSessionConfig(),
        tool_limits=ChatToolLimits(max_file_size_characters=100),
        start_char=None,
        end_char=None,
    )
    assert unsupported_read.status == "unsupported"
    assert unsupported_read.error_message == "unsupported"


def test_convert_with_markitdown_covers_attribute_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "doc.pdf"
    path.write_text("placeholder", encoding="utf-8")
    results = [
        "direct text",
        SimpleNamespace(text_content="text-content"),
        SimpleNamespace(markdown="markdown-content"),
        SimpleNamespace(text="plain-text"),
        SimpleNamespace(),
    ]

    class _FakeMarkItDown:
        def convert(self, raw_path: str) -> object:
            assert raw_path == str(path)
            return results.pop(0)

    fake_module = SimpleNamespace(MarkItDown=lambda: _FakeMarkItDown())
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)

    assert convert_with_markitdown(path) == "direct text"
    assert convert_with_markitdown(path) == "text-content"
    assert convert_with_markitdown(path) == "markdown-content"
    assert convert_with_markitdown(path) == "plain-text"
    with pytest.raises(RuntimeError):
        convert_with_markitdown(path)


def test_chat_ops_cover_remaining_control_flow_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "src").mkdir()
    file_path = tmp_path / "src" / "one.py"
    file_path.write_text("skip\nneedle\nneedle\n", encoding="utf-8")
    (tmp_path / "unsupported.py").write_text("placeholder\n", encoding="utf-8")
    (tmp_path / "too-large.py").write_text("placeholder\n", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "file-link.py").symlink_to(file_path)
    os.mkfifo(tmp_path / "events.pipe")
    outside_file = tmp_path.parent / "outside-chat-search.txt"
    outside_file.write_text("outside\n", encoding="utf-8")

    recursive = list_directory_recursive_impl(
        tmp_path,
        ".",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_entries_per_call=1, max_recursive_depth=5),
        max_depth=3,
    )
    assert recursive.truncated is True
    depth_limited = list_directory_recursive_impl(
        tmp_path,
        "src",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_entries_per_call=20, max_recursive_depth=5),
        max_depth=1,
    )
    assert depth_limited.max_depth_applied == 1

    with pytest.raises(ValueError):
        resolve_search_file_or_directory(tmp_path, "../outside-chat-search.txt")

    file_matches = find_files_impl(
        tmp_path,
        "**/*.py",
        ".",
        source_filters=ChatSourceFilters(include_hidden=False),
        tool_limits=ChatToolLimits(max_entries_per_call=20),
    )
    assert [match.path for match in file_matches.matches] == [
        "src/one.py",
        "too-large.py",
        "unsupported.py",
    ]

    def _load_content(path: Path) -> LoadedReadableContent:
        if path.name == "unsupported.py":
            return LoadedReadableContent(
                read_kind="text",
                status="error",
                content=None,
                error_message="unsupported",
            )
        if path.name == "too-large.py":
            return LoadedReadableContent(
                read_kind="text",
                status="ok",
                content="needle\nneedle\n",
            )
        return LoadedReadableContent(
            read_kind="text",
            status="ok",
            content="needle\n",
        )

    monkeypatch.setattr(ops_module, "load_readable_content", _load_content)
    directory_search = search_text_impl(
        tmp_path,
        "needle",
        ".",
        source_filters=ChatSourceFilters(include_hidden=False),
        tool_limits=ChatToolLimits(max_search_matches=5, max_file_size_characters=7),
    )
    assert [match.path for match in directory_search.matches] == ["src/one.py"]

    monkeypatch.setattr(
        ops_module,
        "load_readable_content",
        lambda path: LoadedReadableContent(
            read_kind="text",
            status="ok",
            content="skip\nneedle\nneedle\n",
        ),
    )
    single_file_search = search_text_impl(
        tmp_path,
        "needle",
        "src/one.py",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_search_matches=1, max_file_size_characters=100),
    )
    assert single_file_search.truncated is True
    assert len(single_file_search.matches) == 1

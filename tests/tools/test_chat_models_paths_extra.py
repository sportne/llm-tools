"""Focused coverage tests for shared filesystem and text helper modules."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import llm_tools.tools.filesystem._ops as filesystem_ops_module
import llm_tools.tools.text._ops as text_ops_module
from llm_tools.tools.filesystem._content import (
    LoadedReadableContent,
    convert_with_markitdown,
)
from llm_tools.tools.filesystem._ops import (
    find_files_impl,
    list_directory_impl,
    read_file_impl,
    resolve_search_file_or_directory,
)
from llm_tools.tools.filesystem._paths import (
    build_entry,
    build_file_match,
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
from llm_tools.tools.filesystem.models import (
    GetFileInfoInputShape,
    SourceFilters,
    ToolLimits,
)
from llm_tools.tools.text._ops import build_text_search_match, search_text_impl


def test_helper_models_cover_validation_errors() -> None:
    with pytest.raises(ValidationError):
        SourceFilters(include=[" "])
    with pytest.raises(ValidationError):
        ToolLimits(max_entries_per_call=0)
    with pytest.raises(ValidationError):
        ToolLimits(max_read_file_chars=0)

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


def test_path_helpers_cover_matching_and_resolution(tmp_path: Path) -> None:
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

    include_filters = SourceFilters(include=["src/*.py"])
    exclude_filters = SourceFilters(exclude=["src/app.py"])
    assert (
        should_include_entry(Path("src/app.py"), source_filters=include_filters) is True
    )
    assert (
        should_include_entry(Path(".hidden/item"), source_filters=SourceFilters())
        is False
    )
    assert (
        should_include_entry(Path("src/app.py"), source_filters=exclude_filters)
        is False
    )
    assert (
        should_prune_directory(Path(".hidden"), source_filters=SourceFilters()) is True
    )
    assert should_prune_directory(Path("src"), source_filters=exclude_filters) is False
    assert (
        should_prune_directory(Path(".llm_tools"), source_filters=SourceFilters())
        is True
    )
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
    with pytest.raises(ValueError):
        resolve_file_path(tmp_path, "link-src/app.py")
    with pytest.raises(ValueError, match="internal tool-managed"):
        resolve_directory_path(tmp_path, ".llm_tools")
    with pytest.raises(ValueError, match="internal tool-managed"):
        resolve_file_path(tmp_path, ".llm_tools/cache/read_file/content.md")

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


def test_ops_cover_recursive_listing_and_file_edge_cases(
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
        list_directory_impl(
            tmp_path,
            ".",
            source_filters=SourceFilters(),
            tool_limits=ToolLimits(),
            recursive=True,
            max_depth=0,
        )

    listing = list_directory_impl(
        tmp_path,
        ".",
        source_filters=SourceFilters(exclude=["ignored/**"]),
        tool_limits=ToolLimits(max_entries_per_call=20, max_recursive_depth=5),
        recursive=True,
        max_depth=3,
    )
    assert all(entry.path != "ignored/skip.py" for entry in listing.entries)
    assert any(entry.path == "dir-link" for entry in listing.entries)
    assert all(not entry.path.startswith("dir-link/") for entry in listing.entries)

    with pytest.raises(ValueError):
        resolve_search_file_or_directory(tmp_path, "dir-link")

    monkeypatch.setattr(
        text_ops_module,
        "load_readable_content",
        lambda path, **kwargs: LoadedReadableContent(
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
        source_filters=SourceFilters(),
        tool_limits=ToolLimits(max_file_size_characters=100),
    )
    assert unsupported_search.matches == []

    monkeypatch.setattr(
        text_ops_module,
        "load_readable_content",
        lambda path, **kwargs: LoadedReadableContent(
            read_kind="text",
            status="ok",
            content="needle\nneedle\n",
        ),
    )
    too_large_search = search_text_impl(
        tmp_path,
        "needle",
        "single.txt",
        source_filters=SourceFilters(),
        tool_limits=ToolLimits(max_file_size_characters=1),
    )
    assert too_large_search.matches == []

    monkeypatch.setattr(
        filesystem_ops_module,
        "load_readable_content",
        lambda path, **kwargs: LoadedReadableContent(
            read_kind="unsupported",
            status="unsupported",
            content=None,
            error_message="unsupported",
        ),
    )
    unsupported_read = read_file_impl(
        tmp_path,
        "single.txt",
        tool_limits=ToolLimits(max_file_size_characters=100),
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


def test_ops_cover_remaining_control_flow_edges(
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

    recursive = list_directory_impl(
        tmp_path,
        ".",
        source_filters=SourceFilters(),
        tool_limits=ToolLimits(max_entries_per_call=1, max_recursive_depth=5),
        recursive=True,
        max_depth=3,
    )
    assert recursive.truncated is True
    depth_limited = list_directory_impl(
        tmp_path,
        "src",
        source_filters=SourceFilters(),
        tool_limits=ToolLimits(max_entries_per_call=20, max_recursive_depth=5),
        recursive=True,
        max_depth=1,
    )
    assert depth_limited.max_depth_applied == 1

    with pytest.raises(ValueError):
        resolve_search_file_or_directory(tmp_path, "../outside-chat-search.txt")

    file_matches = find_files_impl(
        tmp_path,
        "**/*.py",
        ".",
        source_filters=SourceFilters(include_hidden=False),
        tool_limits=ToolLimits(max_entries_per_call=20, max_files_scanned=20),
    )
    assert [match.path for match in file_matches.matches] == [
        "src/one.py",
        "too-large.py",
        "unsupported.py",
    ]

    def _load_content(path: Path, **kwargs: object) -> LoadedReadableContent:
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

    monkeypatch.setattr(text_ops_module, "load_readable_content", _load_content)
    directory_search = search_text_impl(
        tmp_path,
        "needle",
        ".",
        source_filters=SourceFilters(include_hidden=False),
        tool_limits=ToolLimits(
            max_search_matches=5,
            max_file_size_characters=7,
            max_files_scanned=20,
        ),
    )
    assert [match.path for match in directory_search.matches] == ["src/one.py"]

    monkeypatch.setattr(
        text_ops_module,
        "load_readable_content",
        lambda path, **kwargs: LoadedReadableContent(
            read_kind="text",
            status="ok",
            content="skip\nneedle\nneedle\n",
        ),
    )
    single_file_search = search_text_impl(
        tmp_path,
        "needle",
        "src/one.py",
        source_filters=SourceFilters(),
        tool_limits=ToolLimits(max_search_matches=1, max_file_size_characters=100),
    )
    assert single_file_search.truncated is True
    assert len(single_file_search.matches) == 1


def test_find_and_search_honor_recursive_depth_and_scan_budgets(tmp_path: Path) -> None:
    (tmp_path / "nested" / "deeper").mkdir(parents=True)
    (tmp_path / "shallow.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "nested" / "deeper" / "deep.py").write_text(
        "needle\n", encoding="utf-8"
    )
    (tmp_path / "other.py").write_text("miss\n", encoding="utf-8")

    file_matches = find_files_impl(
        tmp_path,
        "**/*.py",
        ".",
        source_filters=SourceFilters(),
        tool_limits=ToolLimits(
            max_entries_per_call=20,
            max_recursive_depth=1,
            max_files_scanned=20,
        ),
    )
    assert file_matches.truncated is True
    assert [match.path for match in file_matches.matches] == ["other.py", "shallow.py"]

    directory_search = search_text_impl(
        tmp_path,
        "needle",
        ".",
        source_filters=SourceFilters(),
        tool_limits=ToolLimits(
            max_recursive_depth=1,
            max_search_matches=20,
            max_file_size_characters=100,
            max_files_scanned=20,
        ),
    )
    assert directory_search.truncated is True
    assert [match.path for match in directory_search.matches] == ["shallow.py"]

    scan_limited_search = search_text_impl(
        tmp_path,
        "absent",
        ".",
        source_filters=SourceFilters(),
        tool_limits=ToolLimits(
            max_recursive_depth=5,
            max_search_matches=20,
            max_file_size_characters=100,
            max_files_scanned=1,
        ),
    )
    assert scan_limited_search.matches == []
    assert scan_limited_search.truncated is True

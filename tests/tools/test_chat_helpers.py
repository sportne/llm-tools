"""Additional tests for repository chat helper modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_tools.tools.chat._content import (
    LoadedReadableContent,
    build_file_info_result,
    count_lines,
    dump_json,
    effective_full_read_char_limit,
    estimate_token_count,
    is_within_character_limit,
    load_readable_content,
    markitdown_cache_path,
    normalize_range,
    read_searchable_text,
)
from llm_tools.tools.chat._ops import (
    find_files_impl,
    get_file_info_impl,
    list_directory_impl,
    read_file_impl,
    resolve_search_file_or_directory,
    search_text_impl,
)
from llm_tools.tools.chat.models import (
    ChatSessionConfig,
    ChatSourceFilters,
    ChatToolLimits,
)


def test_content_helpers_cover_text_binary_markitdown_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text_file = tmp_path / "demo.txt"
    text_file.write_text("a\nb\n", encoding="utf-8")
    assert read_searchable_text(text_file) == "a\nb\n"

    binary_file = tmp_path / "demo.bin"
    binary_file.write_bytes(b"\x00abc")
    assert read_searchable_text(binary_file) is None

    unsupported = load_readable_content(binary_file)
    assert unsupported.status == "unsupported"

    doc_file = tmp_path / "doc.pdf"
    doc_file.write_bytes(b"\xff\xfe\xfd")

    fake_cache = tmp_path / "cache.md"
    fake_cache.write_text("cached markdown", encoding="utf-8")
    monkeypatch.setattr(
        "llm_tools.tools.chat._content.markitdown_cache_path",
        lambda path: fake_cache,
    )
    cached = load_readable_content(doc_file)
    assert cached.content == "cached markdown"

    fake_cache.unlink()
    monkeypatch.setattr(
        "llm_tools.tools.chat._content.convert_with_markitdown",
        lambda path: "converted markdown",
    )
    converted = load_readable_content(doc_file)
    assert converted.content == "converted markdown"
    assert fake_cache.read_text(encoding="utf-8") == "converted markdown"

    fake_cache.unlink()
    monkeypatch.setattr(
        "llm_tools.tools.chat._content.convert_with_markitdown",
        lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    failed = load_readable_content(doc_file)
    assert failed.status == "error"
    assert failed.error_message == "boom"

    assert markitdown_cache_path(doc_file).suffix == ".md"
    assert count_lines("", max_read_lines=5) == 0
    assert count_lines("a\nb\nc", max_read_lines=2) is None
    assert estimate_token_count("Hello, world_2") == 2
    assert is_within_character_limit(
        "abc", tool_limits=ChatToolLimits(max_file_size_characters=3)
    )
    assert not is_within_character_limit(
        "abcd", tool_limits=ChatToolLimits(max_file_size_characters=3)
    )
    assert (
        effective_full_read_char_limit(
            ChatSessionConfig(max_context_tokens=3),
            ChatToolLimits(max_read_file_chars=None),
        )
        == 12
    )
    assert (
        effective_full_read_char_limit(
            ChatSessionConfig(),
            ChatToolLimits(max_read_file_chars=7),
        )
        == 7
    )
    assert normalize_range(start_char=None, end_char=None, character_count=5) == (0, 5)
    assert normalize_range(start_char=1, end_char=10, character_count=5) == (1, 5)
    with pytest.raises(ValueError):
        normalize_range(start_char=-1, end_char=None, character_count=5)
    with pytest.raises(ValueError):
        normalize_range(start_char=0, end_char=-1, character_count=5)
    with pytest.raises(ValueError):
        normalize_range(start_char=2, end_char=2, character_count=5)
    with pytest.raises(ValueError):
        normalize_range(start_char=6, end_char=None, character_count=5)
    assert dump_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_build_file_info_result_and_ops_edge_cases(tmp_path: Path) -> None:
    root = tmp_path
    source_filters = ChatSourceFilters(include_hidden=False)
    tool_limits = ChatToolLimits(
        max_entries_per_call=1,
        max_recursive_depth=2,
        max_search_matches=1,
        max_file_size_characters=3,
        max_read_file_chars=2,
    )
    session_config = ChatSessionConfig(max_context_tokens=2)

    (root / "src").mkdir()
    (root / "docs").mkdir()
    (root / ".hidden").mkdir()
    (root / "README.md").write_text("root\n", encoding="utf-8")
    (root / "src" / "one.py").write_text("hit\nhit\n", encoding="utf-8")
    (root / "src" / "two.py").write_text("miss\n", encoding="utf-8")
    (root / ".hidden" / "three.py").write_text("hit\n", encoding="utf-8")

    listing = list_directory_impl(
        root,
        ".",
        source_filters=source_filters,
        tool_limits=tool_limits,
    )
    assert listing.truncated is True
    assert len(listing.entries) == 1

    file_search = find_files_impl(
        root,
        "**/*.py",
        ".",
        source_filters=source_filters,
        tool_limits=tool_limits,
    )
    assert file_search.truncated is True
    assert file_search.matches[0].path == "src/one.py"

    text_search = search_text_impl(
        root,
        "hit",
        ".",
        source_filters=source_filters,
        tool_limits=ChatToolLimits(
            max_search_matches=1,
            max_file_size_characters=100,
        ),
    )
    assert text_search.truncated is True
    assert text_search.matches[0].path == "src/one.py"

    single_file_search = search_text_impl(
        root,
        "hit",
        "src/one.py",
        source_filters=source_filters,
        tool_limits=ChatToolLimits(max_search_matches=5, max_file_size_characters=100),
    )
    assert len(single_file_search.matches) == 2

    with pytest.raises(ValueError):
        resolve_search_file_or_directory(root, str((root / "src" / "one.py").resolve()))
    with pytest.raises(ValueError):
        resolve_search_file_or_directory(root, "missing.txt")

    info_result = get_file_info_impl(
        root,
        ["src/one.py", "src/two.py"],
        session_config=session_config,
        tool_limits=ChatToolLimits(max_file_size_characters=100, max_read_file_chars=4),
    )
    assert len(info_result.results) == 2

    too_large_read = read_file_impl(
        root,
        "src/one.py",
        session_config=session_config,
        tool_limits=tool_limits,
        start_char=None,
        end_char=None,
    )
    assert too_large_read.status == "too_large"

    ok_read = read_file_impl(
        root,
        "src/two.py",
        session_config=session_config,
        tool_limits=ChatToolLimits(max_file_size_characters=100, max_read_file_chars=2),
        start_char=0,
        end_char=10,
    )
    assert ok_read.status == "ok"
    assert ok_read.truncated is True
    assert ok_read.content == "mi"

    file_info = build_file_info_result(
        requested_path="src/one.py",
        resolved_path="src/one.py",
        candidate_file=root / "src" / "one.py",
        resolved_file=(root / "src" / "one.py").resolve(),
        relative_candidate_path=Path("src/one.py"),
        session_config=session_config,
        tool_limits=ChatToolLimits(
            max_file_size_characters=100, max_read_file_chars=100
        ),
        loaded_content=LoadedReadableContent(
            read_kind="text", status="ok", content="hit\n"
        ),
    )
    assert file_info.can_read_full is True
    assert file_info.line_count == 1

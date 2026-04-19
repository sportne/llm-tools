"""Unit tests for filesystem built-in tools."""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from llm_tools.tool_api import ToolContext
from llm_tools.tools.filesystem import ListDirectoryTool, ReadFileTool, WriteFileTool
from llm_tools.tools.filesystem import tools as filesystem_tools


def test_read_file_tool_reads_text_files(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "hello.txt"
    file_path.write_text("hello world", encoding="utf-8")

    tool = ReadFileTool()
    context = ToolContext(invocation_id="inv-1", workspace=str(workspace))
    result = tool.invoke(
        context,
        ReadFileTool.input_model(path="hello.txt"),
    )

    assert result.content == "hello world"
    assert result.read_kind == "text"
    assert result.start_char == 0
    assert result.end_char == 11
    assert result.content_char_count == 11
    assert result.character_count == 11
    assert result.truncated is False
    assert result.resolved_path == "hello.txt"
    assert context.artifacts == ["hello.txt"]
    assert context.source_provenance[0].source_id == "workspace:hello.txt"
    assert context.source_provenance[0].metadata["path"] == "hello.txt"


def test_read_file_tool_automatically_converts_non_text_files(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path
    file_path = workspace / "report.docx"
    file_path.write_bytes(b"\xff\xfe\x00\x10")
    cache_root = workspace / ".cache-root"

    fake_module = ModuleType("markitdown")
    convert_calls: list[str] = []

    class FakeMarkItDown:
        def convert(self, path: str) -> SimpleNamespace:
            convert_calls.append(path)
            return SimpleNamespace(text_content=f"converted:{path}")

    fake_module.MarkItDown = FakeMarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)
    monkeypatch.setattr(
        filesystem_tools, "_get_read_file_cache_root", lambda root: cache_root
    )

    tool = ReadFileTool()
    context = ToolContext(invocation_id="inv-2", workspace=str(workspace))
    first = tool.invoke(
        context,
        ReadFileTool.input_model(path="report.docx"),
    )
    second = tool.invoke(
        ToolContext(invocation_id="inv-3", workspace=str(workspace)),
        ReadFileTool.input_model(path="report.docx"),
    )

    assert first.read_kind == "markitdown"
    assert first.content == f"converted:{file_path.resolve()}"
    assert second.content == first.content
    assert convert_calls == [str(file_path.resolve())]
    assert context.artifacts == ["report.docx"]
    assert any(cache_root.iterdir())


def test_read_file_tool_rebuilds_cached_markdown_when_source_changes(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path
    file_path = workspace / "report.docx"
    file_path.write_bytes(b"\xff\xfe\x00\x10")
    cache_root = workspace / ".cache-root"

    fake_module = ModuleType("markitdown")
    convert_calls: list[str] = []

    class FakeMarkItDown:
        def convert(self, path: str) -> SimpleNamespace:
            convert_calls.append(path)
            return SimpleNamespace(
                text_content=f"converted-{len(convert_calls)}:{path}"
            )

    fake_module.MarkItDown = FakeMarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)
    monkeypatch.setattr(
        filesystem_tools, "_get_read_file_cache_root", lambda root: cache_root
    )

    tool = ReadFileTool()
    first = tool.invoke(
        ToolContext(invocation_id="inv-4", workspace=str(workspace)),
        ReadFileTool.input_model(path="report.docx"),
    )
    file_path.write_bytes(b"\xff\xfe\x00\x11")
    stat = file_path.stat()
    os.utime(file_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = tool.invoke(
        ToolContext(invocation_id="inv-5", workspace=str(workspace)),
        ReadFileTool.input_model(path="report.docx"),
    )

    assert first.content != second.content
    assert convert_calls == [str(file_path.resolve()), str(file_path.resolve())]


def test_read_file_tool_rejects_large_input_before_conversion(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path
    file_path = workspace / "report.docx"
    file_path.write_bytes(b"x" * 32)

    fake_module = ModuleType("markitdown")
    convert_calls: list[str] = []

    class FakeMarkItDown:
        def convert(self, path: str) -> SimpleNamespace:
            convert_calls.append(path)
            return SimpleNamespace(text_content=f"converted:{path}")

    fake_module.MarkItDown = FakeMarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)

    result = ReadFileTool().invoke(
        ToolContext(
            invocation_id="inv-oversize",
            workspace=str(workspace),
            metadata={"tool_limits": {"max_read_input_bytes": 8}},
        ),
        ReadFileTool.input_model(path="report.docx"),
    )

    assert result.status == "too_large"
    assert result.error_message == "File exceeds the configured readable byte limit"
    assert convert_calls == []


def test_read_file_tool_returns_bounded_character_ranges(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "hello.txt"
    file_path.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")

    result = ReadFileTool().invoke(
        ToolContext(invocation_id="inv-6", workspace=str(workspace)),
        ReadFileTool.input_model(path="hello.txt", start_char=4, end_char=13),
    )

    assert result.content == "two\nthree"
    assert result.start_char == 4
    assert result.end_char == 13
    assert result.character_count == len(file_path.read_text(encoding="utf-8"))
    assert result.truncated is True


def test_read_file_tool_truncates_large_default_ranges(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "many-chars.txt"
    file_path.write_text("x" * 5000, encoding="utf-8")

    result = ReadFileTool().invoke(
        ToolContext(invocation_id="inv-7", workspace=str(workspace)),
        ReadFileTool.input_model(path="many-chars.txt"),
    )

    assert result.content == "x" * 4000
    assert result.start_char == 0
    assert result.end_char == 4000
    assert result.character_count == 5000
    assert result.truncated is True


def test_read_file_tool_rejects_invalid_character_ranges(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "hello.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="end_char"):
        ReadFileTool().invoke(
            ToolContext(invocation_id="inv-8", workspace=str(workspace)),
            ReadFileTool.input_model(path="hello.txt", start_char=3, end_char=2),
        )


def test_write_file_tool_writes_text_and_creates_parents(tmp_path: str) -> None:
    workspace = tmp_path
    tool = WriteFileTool()
    context = ToolContext(invocation_id="inv-3", workspace=str(workspace))

    result = tool.invoke(
        context,
        WriteFileTool.input_model(
            path="nested/output.txt",
            content="hello",
            create_parents=True,
        ),
    )

    target = workspace / "nested" / "output.txt"
    assert result.created is True
    assert result.resolved_path == "nested/output.txt"
    assert result.bytes_written == len(b"hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert context.artifacts == ["nested/output.txt"]


def test_write_file_tool_rejects_existing_files_without_overwrite(
    tmp_path: str,
) -> None:
    workspace = tmp_path
    target = workspace / "exists.txt"
    target.write_text("old", encoding="utf-8")

    tool = WriteFileTool()
    with pytest.raises(FileExistsError, match="already exists"):
        tool.invoke(
            ToolContext(invocation_id="inv-4", workspace=str(workspace)),
            WriteFileTool.input_model(path="exists.txt", content="new"),
        )


def test_write_file_tool_rejects_absolute_and_symlinked_paths(tmp_path: str) -> None:
    workspace = tmp_path
    (workspace / "real").mkdir()
    (workspace / "real" / "target.txt").write_text("old", encoding="utf-8")
    (workspace / "link-dir").symlink_to(workspace / "real", target_is_directory=True)
    (workspace / "link-file.txt").symlink_to(workspace / "real" / "target.txt")
    (workspace / "dangling.txt").symlink_to(workspace / "missing.txt")

    tool = WriteFileTool()
    with pytest.raises(ValueError, match="relative"):
        tool.invoke(
            ToolContext(invocation_id="inv-abs", workspace=str(workspace)),
            WriteFileTool.input_model(
                path=str((workspace / "absolute.txt").resolve()), content="new"
            ),
        )
    with pytest.raises(ValueError, match="symlinked directory"):
        tool.invoke(
            ToolContext(invocation_id="inv-link-dir", workspace=str(workspace)),
            WriteFileTool.input_model(path="link-dir/out.txt", content="new"),
        )
    with pytest.raises(ValueError, match="symlinked file"):
        tool.invoke(
            ToolContext(invocation_id="inv-link-file", workspace=str(workspace)),
            WriteFileTool.input_model(
                path="link-file.txt", content="new", overwrite=True
            ),
        )
    with pytest.raises(ValueError, match="symlinked file"):
        tool.invoke(
            ToolContext(invocation_id="inv-dangling", workspace=str(workspace)),
            WriteFileTool.input_model(path="dangling.txt", content="new"),
        )


def test_list_directory_tool_lists_entries_in_stable_order(tmp_path: str) -> None:
    workspace = tmp_path
    (workspace / "b.txt").write_text("b", encoding="utf-8")
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    hidden = workspace / ".hidden.txt"
    hidden.write_text("hidden", encoding="utf-8")
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "note.txt").write_text("nested", encoding="utf-8")

    tool = ListDirectoryTool()
    result = tool.invoke(
        ToolContext(invocation_id="inv-5", workspace=str(workspace)),
        ListDirectoryTool.input_model(path=".", recursive=True),
    )

    assert [entry.path for entry in result.entries] == [
        "a.txt",
        "b.txt",
        "nested",
        "nested/note.txt",
    ]
    assert all(not entry.path.startswith(".") for entry in result.entries)
    assert result.recursive is True
    assert result.max_depth_applied == 12

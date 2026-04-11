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
    assert result.content_format == "text"
    assert result.line_start == 1
    assert result.line_end == 1
    assert result.total_lines == 1
    assert result.truncated is False
    assert result.cached_markdown_path is None
    assert result.used_cached_conversion is False
    assert result.resolved_path == str(file_path.resolve())
    assert context.artifacts == [str(file_path.resolve())]


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
        filesystem_tools, "_get_read_file_cache_root", lambda: cache_root
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

    assert first.content_format == "markdown"
    assert first.content == f"converted:{file_path.resolve()}"
    assert first.cached_markdown_path is not None
    assert first.used_cached_conversion is False
    assert second.content == first.content
    assert second.cached_markdown_path == first.cached_markdown_path
    assert second.used_cached_conversion is True
    assert convert_calls == [str(file_path.resolve())]
    assert context.artifacts == [str(file_path.resolve()), first.cached_markdown_path]


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
        filesystem_tools, "_get_read_file_cache_root", lambda: cache_root
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
    assert second.used_cached_conversion is False
    assert convert_calls == [str(file_path.resolve()), str(file_path.resolve())]


def test_read_file_tool_returns_bounded_line_ranges(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "hello.txt"
    file_path.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")

    result = ReadFileTool().invoke(
        ToolContext(invocation_id="inv-6", workspace=str(workspace)),
        ReadFileTool.input_model(path="hello.txt", line_start=2, line_end=4),
    )

    assert result.content == "two\nthree\nfour"
    assert result.line_start == 2
    assert result.line_end == 4
    assert result.total_lines == 5
    assert result.truncated is False


def test_read_file_tool_truncates_large_default_ranges(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "many-lines.txt"
    file_path.write_text(
        "\n".join(f"line {index}" for index in range(1, 251)),
        encoding="utf-8",
    )

    result = ReadFileTool().invoke(
        ToolContext(invocation_id="inv-7", workspace=str(workspace)),
        ReadFileTool.input_model(path="many-lines.txt"),
    )

    returned_lines = result.content.splitlines()
    assert len(returned_lines) == 200
    assert returned_lines[0] == "line 1"
    assert returned_lines[-1] == "line 200"
    assert result.line_end == 200
    assert result.total_lines == 250
    assert result.truncated is True


def test_read_file_tool_rejects_invalid_line_ranges(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "hello.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="line_end"):
        ReadFileTool().invoke(
            ToolContext(invocation_id="inv-8", workspace=str(workspace)),
            ReadFileTool.input_model(path="hello.txt", line_start=3, line_end=2),
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
    assert result.bytes_written == len(b"hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert context.artifacts == [str(target.resolve())]


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

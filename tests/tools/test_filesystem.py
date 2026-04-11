"""Unit tests for filesystem built-in tools."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from llm_tools.tool_api import ToolContext
from llm_tools.tools.filesystem import ListDirectoryTool, ReadFileTool, WriteFileTool


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
    assert result.mode == "text"
    assert result.resolved_path == str(file_path.resolve())
    assert context.artifacts == [str(file_path.resolve())]


def test_read_file_tool_uses_markitdown_for_conversion(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path
    file_path = workspace / "report.docx"
    file_path.write_text("unused", encoding="utf-8")

    fake_module = ModuleType("markitdown")

    class FakeMarkItDown:
        def convert(self, path: str) -> SimpleNamespace:
            return SimpleNamespace(text_content=f"converted:{path}")

    fake_module.MarkItDown = FakeMarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)

    tool = ReadFileTool()
    result = tool.invoke(
        ToolContext(invocation_id="inv-2", workspace=str(workspace)),
        ReadFileTool.input_model(path="report.docx", mode="convert"),
    )

    assert result.mode == "convert"
    assert result.content == f"converted:{file_path.resolve()}"


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

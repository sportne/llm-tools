"""Unit tests for filesystem built-in tools."""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from llm_tools.tool_api import (
    ErrorCode,
    SideEffectClass,
    ToolContext,
    ToolInvocationRequest,
    ToolPolicy,
    ToolRegistry,
    ToolRuntime,
)
from llm_tools.tools.filesystem import (
    FindFilesTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from llm_tools.tools.filesystem import tools as filesystem_tools


def _runtime(*, allow_write: bool = False) -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirectoryTool())
    allowed_side_effects = {SideEffectClass.NONE, SideEffectClass.LOCAL_READ}
    if allow_write:
        allowed_side_effects.add(SideEffectClass.LOCAL_WRITE)
    return ToolRuntime(
        registry,
        policy=ToolPolicy(allowed_side_effects=allowed_side_effects),
    )


def _invoke_read(context: ToolContext, **arguments: object):
    result = _runtime().execute(
        ToolInvocationRequest(tool_name="read_file", arguments=arguments),
        context,
    )
    assert result.ok is True, result.error
    return ReadFileTool.output_model.model_validate(result.output), result


def _invoke_write(context: ToolContext, **arguments: object):
    result = _runtime(allow_write=True).execute(
        ToolInvocationRequest(tool_name="write_file", arguments=arguments),
        context,
    )
    assert result.ok is True, result.error
    return WriteFileTool.output_model.model_validate(result.output), result


def _invoke_list(context: ToolContext, **arguments: object):
    result = _runtime().execute(
        ToolInvocationRequest(tool_name="list_directory", arguments=arguments),
        context,
    )
    assert result.ok is True, result.error
    return ListDirectoryTool.output_model.model_validate(result.output), result


def test_find_files_model_facing_guidance_explains_recursive_globs() -> None:
    schema = FindFilesTool.input_model.model_json_schema()

    assert "**/*.py" in FindFilesTool.spec.description
    assert "*.py" in FindFilesTool.spec.description
    assert "**/*.py" in schema["properties"]["pattern"]["description"]
    assert "search root" in schema["properties"]["pattern"]["description"]
    assert "Hidden paths are excluded" in FindFilesTool.spec.description


def test_read_file_tool_reads_text_files(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "hello.txt"
    file_path.write_text("hello world", encoding="utf-8")

    result, tool_result = _invoke_read(
        ToolContext(invocation_id="inv-1", workspace=str(workspace)),
        path="hello.txt",
    )

    assert result.content == "hello world"
    assert result.read_kind == "text"
    assert result.start_char == 0
    assert result.end_char == 11
    assert result.content_char_count == 11
    assert result.character_count == 11
    assert result.truncated is False
    assert result.resolved_path == "hello.txt"
    assert tool_result.artifacts == ["[REDACTED]"]
    assert tool_result.source_provenance[0].source_id == "workspace:hello.txt"
    assert tool_result.source_provenance[0].metadata["path"] == "hello.txt"


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

    first, first_result = _invoke_read(
        ToolContext(invocation_id="inv-2", workspace=str(workspace)),
        path="report.docx",
    )
    second, _ = _invoke_read(
        ToolContext(invocation_id="inv-3", workspace=str(workspace)),
        path="report.docx",
    )

    assert first.read_kind == "markitdown"
    assert first.content == f"converted:{file_path.resolve()}"
    assert second.content == first.content
    assert convert_calls == [str(file_path.resolve())]
    assert first_result.artifacts == ["[REDACTED]"]
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

    first, _ = _invoke_read(
        ToolContext(invocation_id="inv-4", workspace=str(workspace)),
        path="report.docx",
    )
    file_path.write_bytes(b"\xff\xfe\x00\x11")
    stat = file_path.stat()
    os.utime(file_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second, _ = _invoke_read(
        ToolContext(invocation_id="inv-5", workspace=str(workspace)),
        path="report.docx",
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

    result = _runtime().execute(
        ToolInvocationRequest(tool_name="read_file", arguments={"path": "report.docx"}),
        ToolContext(
            invocation_id="inv-oversize",
            workspace=str(workspace),
            metadata={"tool_limits": {"max_read_input_bytes": 8}},
        ),
    )

    assert result.ok is True, result.error
    output = ReadFileTool.output_model.model_validate(result.output)
    assert output.status == "too_large"
    assert output.error_message == "File exceeds the configured readable byte limit"
    assert convert_calls == []


def test_read_file_tool_returns_bounded_character_ranges(tmp_path: str) -> None:
    workspace = tmp_path
    file_path = workspace / "hello.txt"
    file_path.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")

    result, _ = _invoke_read(
        ToolContext(invocation_id="inv-6", workspace=str(workspace)),
        path="hello.txt",
        start_char=4,
        end_char=13,
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

    result, _ = _invoke_read(
        ToolContext(invocation_id="inv-7", workspace=str(workspace)),
        path="many-chars.txt",
    )

    assert result.content == "x" * 4000
    assert result.start_char == 0
    assert result.end_char == 4000
    assert result.character_count == 5000
    assert result.truncated is True


def test_read_file_tool_rejects_invalid_character_ranges(tmp_path: str) -> None:
    workspace = tmp_path
    (workspace / "hello.txt").write_text("hello", encoding="utf-8")

    result = _runtime().execute(
        ToolInvocationRequest(
            tool_name="read_file",
            arguments={"path": "hello.txt", "start_char": 3, "end_char": 2},
        ),
        ToolContext(invocation_id="inv-8", workspace=str(workspace)),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    assert result.error.details["failure_reason"] == (
        "filesystem_target_invalid_or_unavailable"
    )


def test_write_file_tool_writes_text_and_creates_parents(tmp_path: str) -> None:
    workspace = tmp_path

    result, tool_result = _invoke_write(
        ToolContext(invocation_id="inv-3", workspace=str(workspace)),
        path="nested/output.txt",
        content="hello",
        create_parents=True,
    )

    target = workspace / "nested" / "output.txt"
    assert result.created is True
    assert result.resolved_path == "nested/output.txt"
    assert result.bytes_written == len(b"hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert tool_result.artifacts == ["[REDACTED]"]


def test_write_file_tool_rejects_existing_files_without_overwrite(
    tmp_path: str,
) -> None:
    workspace = tmp_path
    target = workspace / "exists.txt"
    target.write_text("old", encoding="utf-8")

    result = _runtime(allow_write=True).execute(
        ToolInvocationRequest(
            tool_name="write_file",
            arguments={"path": "exists.txt", "content": "new"},
        ),
        ToolContext(invocation_id="inv-4", workspace=str(workspace)),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    assert result.error.details["failure_reason"] == (
        "filesystem_target_invalid_or_unavailable"
    )


def test_write_file_tool_rejects_absolute_and_symlinked_paths(tmp_path: str) -> None:
    workspace = tmp_path
    (workspace / "real").mkdir()
    (workspace / "real" / "target.txt").write_text("old", encoding="utf-8")
    (workspace / "link-dir").symlink_to(workspace / "real", target_is_directory=True)
    (workspace / "link-file.txt").symlink_to(workspace / "real" / "target.txt")
    (workspace / "dangling.txt").symlink_to(workspace / "missing.txt")

    cases = [
        (
            {"path": str((workspace / "absolute.txt").resolve()), "content": "new"},
            "relative",
        ),
        ({"path": "link-dir/out.txt", "content": "new"}, "symlinked directory"),
        (
            {"path": "link-file.txt", "content": "new", "overwrite": True},
            "symlinked file",
        ),
        ({"path": "dangling.txt", "content": "new"}, "symlinked file"),
    ]

    for index, (arguments, _expected) in enumerate(cases, start=1):
        result = _runtime(allow_write=True).execute(
            ToolInvocationRequest(tool_name="write_file", arguments=arguments),
            ToolContext(invocation_id=f"inv-link-{index}", workspace=str(workspace)),
        )
        assert result.ok is False
        assert result.error is not None
        assert result.error.code is ErrorCode.EXECUTION_FAILED
        assert result.error.details["failure_reason"] == (
            "filesystem_target_invalid_or_unavailable"
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

    result, _ = _invoke_list(
        ToolContext(invocation_id="inv-5", workspace=str(workspace)),
        path=".",
        recursive=True,
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

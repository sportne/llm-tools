"""Architecture checks for runtime mediation of built-in tools."""

from __future__ import annotations

from pathlib import Path

from tests.architecture._helpers import builtin_runtime, filesystem_context

from llm_tools.tool_api import (
    ErrorCode,
    SideEffectClass,
    ToolInvocationRequest,
    ToolResult,
)


def test_runtime_returns_structured_result_envelopes_for_filesystem_builtins(
    tmp_path: Path,
) -> None:
    runtime = builtin_runtime()
    write_context = filesystem_context(tmp_path, invocation_id="arch-write")

    write_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="write_file",
            arguments={
                "path": "docs/note.txt",
                "content": "hello runtime",
                "create_parents": True,
            },
        ),
        write_context,
    )

    assert isinstance(write_result, ToolResult)
    assert write_result.ok is True
    assert write_result.error is None
    assert write_result.output is not None
    assert write_result.output["path"] == "docs/note.txt"
    assert write_result.output["bytes_written"] == len(b"hello runtime")
    assert write_result.logs == ["[REDACTED]"]

    execution_record = write_result.metadata.get("execution_record")
    assert isinstance(execution_record, dict)
    assert execution_record["tool_name"] == "write_file"
    assert execution_record["request"]["arguments"]["path"] == "docs/note.txt"
    assert execution_record["ok"] is True
    assert execution_record["error_code"] is None

    read_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="read_file",
            arguments={"path": "docs/note.txt"},
        ),
        filesystem_context(tmp_path, invocation_id="arch-read"),
    )

    assert isinstance(read_result, ToolResult)
    assert read_result.ok is True
    assert read_result.error is None
    assert read_result.output is not None
    assert read_result.output["content"] == "hello runtime"
    assert read_result.output["status"] == "ok"
    assert read_result.metadata["execution_record"]["tool_name"] == "read_file"
    assert read_result.metadata["execution_record"]["ok"] is True


def test_runtime_normalizes_policy_denial_for_blocked_write_operations(
    tmp_path: Path,
) -> None:
    runtime = builtin_runtime(
        allowed_side_effects={
            SideEffectClass.NONE,
            SideEffectClass.LOCAL_READ,
            SideEffectClass.EXTERNAL_READ,
        }
    )

    result = runtime.execute(
        ToolInvocationRequest(
            tool_name="write_file",
            arguments={
                "path": "blocked.txt",
                "content": "nope",
            },
        ),
        filesystem_context(tmp_path, invocation_id="arch-denied"),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.POLICY_DENIED
    assert result.error.details["tool_name"] == "write_file"
    assert result.metadata["execution_record"]["policy_decision"]["reason"] == (
        "side effect not allowed"
    )


def test_runtime_normalizes_builtin_input_validation_failures(tmp_path: Path) -> None:
    runtime = builtin_runtime()

    result = runtime.execute(
        ToolInvocationRequest(
            tool_name="write_file",
            arguments={"path": "missing-content.txt"},
        ),
        filesystem_context(tmp_path, invocation_id="arch-invalid"),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.INPUT_VALIDATION_ERROR
    assert result.error.details["tool_name"] == "write_file"
    assert result.metadata["execution_record"]["error_code"] == (
        ErrorCode.INPUT_VALIDATION_ERROR.value
    )


def test_runtime_normalizes_missing_tool_lookup() -> None:
    runtime = builtin_runtime()

    result = runtime.execute(
        ToolInvocationRequest(tool_name="missing_tool", arguments={}),
        filesystem_context(Path.cwd(), invocation_id="arch-missing"),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.TOOL_NOT_FOUND
    assert result.tool_name == "missing_tool"
    assert result.tool_version == "unknown"
    assert result.metadata["execution_record"]["tool_name"] == "missing_tool"
    assert result.metadata["execution_record"]["error_code"] == (
        ErrorCode.TOOL_NOT_FOUND.value
    )

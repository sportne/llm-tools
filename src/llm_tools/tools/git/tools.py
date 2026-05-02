"""Git built-in tool registration and compatibility exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from llm_tools.tool_api import ToolExecutionContext, ToolRegistry
from llm_tools.tools.git import _shared
from llm_tools.tools.git._shared import (
    GIT_COMMAND_OUTPUT_CHAR_LIMIT,
    GIT_COMMAND_TIMEOUT_SECONDS,
    GIT_DISCOVERY_OUTPUT_CHAR_LIMIT,
    GIT_ERROR_MESSAGE_CHAR_LIMIT,
    MAX_GIT_LOG_LIMIT,
    GitCommandInput,
    GitCommandResult,
    _build_git_env,
    _collect_process_output,
    _decode_git_output,
    _format_git_failure_message,
    _GitProcessResult,
    _stream_reader,
    _terminate_process,
    _truncate_git_text,
    _validate_git_diff_ref,
)
from llm_tools.tools.git._shared import (
    _run_git_subprocess as _shared_run_git_subprocess,
)
from llm_tools.tools.git.run_git_diff import (
    RunGitDiffInput,
    RunGitDiffOutput,
    RunGitDiffTool,
)
from llm_tools.tools.git.run_git_log import (
    RunGitLogInput,
    RunGitLogOutput,
    RunGitLogTool,
)
from llm_tools.tools.git.run_git_status import RunGitStatusOutput, RunGitStatusTool

_shared_module = cast(Any, _shared)
subprocess = _shared_module.subprocess
threading = _shared_module.threading
queue = _shared_module.queue
time = _shared_module.time
_ORIGINAL_RUN_GIT_SUBPROCESS = _shared_run_git_subprocess


def register_git_tools(registry: ToolRegistry) -> None:
    """Register the built-in git tool set."""
    registry.register(RunGitStatusTool())
    registry.register(RunGitDiffTool())
    registry.register(RunGitLogTool())


def _run_git_subprocess(
    root: Path,
    args: list[str],
    *,
    output_limit_bytes: int,
) -> _GitProcessResult:
    _shared_module._collect_process_output = _collect_process_output
    return _ORIGINAL_RUN_GIT_SUBPROCESS(
        root,
        args,
        output_limit_bytes=output_limit_bytes,
    )


def _sync_shared_git_helpers() -> None:
    _shared_module._run_git_subprocess = _run_git_subprocess


def _resolve_git_repository_root(context: ToolExecutionContext, path: str) -> Path:
    _sync_shared_git_helpers()
    return _shared._resolve_git_repository_root(context, path)


def _run_git_command(root: Path, args: list[str]) -> GitCommandResult:
    _sync_shared_git_helpers()
    return _shared._run_git_command(root, args)


__all__ = [
    "GIT_COMMAND_OUTPUT_CHAR_LIMIT",
    "GIT_COMMAND_TIMEOUT_SECONDS",
    "GIT_DISCOVERY_OUTPUT_CHAR_LIMIT",
    "GIT_ERROR_MESSAGE_CHAR_LIMIT",
    "MAX_GIT_LOG_LIMIT",
    "GitCommandInput",
    "GitCommandResult",
    "RunGitDiffInput",
    "RunGitDiffOutput",
    "RunGitDiffTool",
    "RunGitLogInput",
    "RunGitLogOutput",
    "RunGitLogTool",
    "RunGitStatusOutput",
    "RunGitStatusTool",
    "register_git_tools",
    "queue",
    "subprocess",
    "threading",
    "time",
    "_GitProcessResult",
    "_build_git_env",
    "_collect_process_output",
    "_decode_git_output",
    "_format_git_failure_message",
    "_resolve_git_repository_root",
    "_run_git_command",
    "_run_git_subprocess",
    "_stream_reader",
    "_terminate_process",
    "_truncate_git_text",
    "_validate_git_diff_ref",
]

"""Shared helpers for git tool implementations."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel

from llm_tools.tool_api import ToolExecutionContext
from llm_tools.tool_api.execution import get_workspace_root, resolve_workspace_path

GIT_COMMAND_TIMEOUT_SECONDS = 10
GIT_COMMAND_OUTPUT_CHAR_LIMIT = 16_000
GIT_DISCOVERY_OUTPUT_CHAR_LIMIT = 4_096
GIT_ERROR_MESSAGE_CHAR_LIMIT = 512
MAX_GIT_LOG_LIMIT = 100
_TRUNCATED_SUFFIX = "\n...(truncated)"


@dataclass(slots=True)
class _GitProcessResult:
    stdout: str
    stderr: str
    returncode: int
    truncated: bool


@dataclass(slots=True)
class GitCommandResult:
    text: str
    truncated: bool


class GitCommandInput(BaseModel):
    path: str = "."


def _build_git_env() -> dict[str, str]:
    env = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()


def _decode_git_output(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _truncate_git_text(text: str, *, truncated: bool) -> str:
    if not truncated:
        return text

    limit = max(GIT_COMMAND_OUTPUT_CHAR_LIMIT - len(_TRUNCATED_SUFFIX), 0)
    return f"{text[:limit]}{_TRUNCATED_SUFFIX}"


def _format_git_failure_message(command_name: str, result: _GitProcessResult) -> str:
    detail = result.stderr.strip() or result.stdout.strip() or f"{command_name} failed"
    if len(detail) > GIT_ERROR_MESSAGE_CHAR_LIMIT:
        limit = max(GIT_ERROR_MESSAGE_CHAR_LIMIT - len(_TRUNCATED_SUFFIX), 0)
        detail = f"{detail[:limit]}{_TRUNCATED_SUFFIX}"
    if result.truncated:
        return f"{command_name} failed after exceeding the output limit. {detail}"
    return detail


def _stream_reader(
    stream: BinaryIO,
    *,
    stream_name: str,
    event_queue: queue.Queue[tuple[str, bytes | None]],
) -> None:
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            event_queue.put((stream_name, chunk))
    finally:
        event_queue.put((stream_name, None))


def _collect_process_output(
    process: subprocess.Popen[bytes],
    *,
    output_limit_bytes: int,
    timeout_seconds: int,
) -> tuple[bytes, bytes, bool]:
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("git subprocess did not expose stdout/stderr pipes.")

    event_queue: queue.Queue[tuple[str, bytes | None]] = queue.Queue()
    readers = [
        threading.Thread(
            target=_stream_reader,
            kwargs={
                "stream": process.stdout,
                "stream_name": "stdout",
                "event_queue": event_queue,
            },
            daemon=True,
        ),
        threading.Thread(
            target=_stream_reader,
            kwargs={
                "stream": process.stderr,
                "stream_name": "stderr",
                "event_queue": event_queue,
            },
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    buffers = {
        "stdout": bytearray(),
        "stderr": bytearray(),
    }
    closed_streams: set[str] = set()
    truncated = False
    deadline = time.monotonic() + timeout_seconds

    try:
        while len(closed_streams) < len(readers):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process(process)
                raise RuntimeError(
                    f"git command timed out after {timeout_seconds} seconds."
                )

            try:
                stream_name, chunk = event_queue.get(timeout=remaining)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue

            if chunk is None:
                closed_streams.add(stream_name)
                continue

            total_captured = len(buffers["stdout"]) + len(buffers["stderr"])
            allowed = max(output_limit_bytes - total_captured, 0)
            if allowed > 0:
                buffers[stream_name].extend(chunk[:allowed])
            if len(chunk) > allowed:
                truncated = True

        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            _terminate_process(process)
            process.wait(timeout=1)
    finally:
        for reader in readers:
            reader.join(timeout=1)
        process.stdout.close()
        process.stderr.close()

    return bytes(buffers["stdout"]), bytes(buffers["stderr"]), truncated


def _run_git_subprocess(
    root: Path,
    args: list[str],
    *,
    output_limit_bytes: int,
) -> _GitProcessResult:
    try:
        process = subprocess.Popen(
            ["git", *args],
            cwd=root,
            env=_build_git_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git executable not found.") from exc

    stdout, stderr, truncated = _collect_process_output(
        process,
        output_limit_bytes=output_limit_bytes,
        timeout_seconds=GIT_COMMAND_TIMEOUT_SECONDS,
    )
    return _GitProcessResult(
        stdout=_decode_git_output(stdout),
        stderr=_decode_git_output(stderr),
        returncode=process.returncode if process.returncode is not None else 1,
        truncated=truncated,
    )


def _resolve_git_repository_root(context: ToolExecutionContext, path: str) -> Path:
    workspace_root = get_workspace_root(context)
    target_root = resolve_workspace_path(
        context,
        path,
        expect_directory=True,
        must_exist=True,
    )
    completed = _run_git_subprocess(
        target_root,
        ["rev-parse", "--show-toplevel"],
        output_limit_bytes=GIT_DISCOVERY_OUTPUT_CHAR_LIMIT,
    )
    if completed.returncode != 0:
        raise RuntimeError(_format_git_failure_message("git rev-parse", completed))
    if completed.truncated:
        raise RuntimeError("git rev-parse exceeded the discovery output limit.")

    repo_root_text = completed.stdout.strip()
    if repo_root_text == "":
        raise RuntimeError("git rev-parse returned an empty repository root.")

    repo_root = Path(repo_root_text).resolve()
    if not repo_root.is_dir():
        raise RuntimeError(f"Resolved git repository root '{repo_root}' is invalid.")
    if not repo_root.is_relative_to(workspace_root):
        raise ValueError(
            "Git repository root "
            f"'{repo_root}' resolves outside the workspace root '{workspace_root}'."
        )
    return repo_root


def _validate_git_diff_ref(ref: str) -> str:
    normalized = ref.strip()
    if normalized == "":
        raise ValueError("Git diff ref must not be empty.")
    if normalized.startswith("-"):
        raise ValueError("Git diff ref must not start with '-'.")
    if any(char in normalized for char in ("\x00", "\r", "\n")):
        raise ValueError("Git diff ref contains invalid control characters.")
    return normalized


def _run_git_command(root: Path, args: list[str]) -> GitCommandResult:
    completed = _run_git_subprocess(
        root,
        args,
        output_limit_bytes=GIT_COMMAND_OUTPUT_CHAR_LIMIT,
    )
    if completed.returncode != 0:
        command_name = f"git {' '.join(args[:2])}".strip()
        raise RuntimeError(_format_git_failure_message(command_name, completed))
    return GitCommandResult(
        text=_truncate_git_text(completed.stdout, truncated=completed.truncated),
        truncated=completed.truncated,
    )

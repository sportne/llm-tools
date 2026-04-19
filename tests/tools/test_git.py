"""Unit tests for git built-in tools."""

from __future__ import annotations

import os
import queue
import subprocess
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from llm_tools.tool_api import ToolContext, ToolRegistry
from llm_tools.tools.git import RunGitDiffTool, RunGitLogTool, RunGitStatusTool
from llm_tools.tools.git import tools as git_tools


class _DummyProcess:
    def __init__(
        self,
        *,
        stdout: BytesIO | None,
        stderr: BytesIO | None,
        returncode: int | None = 0,
        poll_result: int | None = 0,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self._poll_result = poll_result
        self.killed = False
        self.wait_calls: list[int] = []

    def poll(self) -> int | None:
        return self._poll_result

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: int) -> int:
        self.wait_calls.append(timeout)
        return 0


class _WaitTimeoutThenExitProcess(_DummyProcess):
    def wait(self, timeout: int) -> int:
        self.wait_calls.append(timeout)
        if len(self.wait_calls) == 1:
            raise subprocess.TimeoutExpired(cmd="git", timeout=timeout)
        return 0


class _NoOpThread:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        del timeout
        return None


class _EmptyQueue:
    def put(self, item: tuple[str, bytes | None]) -> None:
        del item

    def get(self, timeout: float | None = None) -> tuple[str, bytes | None]:
        del timeout
        raise queue.Empty


def test_build_git_env_strips_git_prefixed_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_DIR", str(Path("git-dir")))
    monkeypatch.setenv("GIT_WORK_TREE", str(Path("git-tree")))
    monkeypatch.setenv("HOME", str(Path("home-dir")))
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    env = git_tools._build_git_env()

    assert "GIT_DIR" not in env
    assert "GIT_WORK_TREE" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["HOME"] == str(Path("home-dir"))


@pytest.mark.parametrize(
    ("poll_result", "expected_killed"),
    [
        (None, True),
        (0, False),
    ],
)
def test_terminate_process_only_kills_running_process(
    poll_result: int | None, expected_killed: bool
) -> None:
    process = _DummyProcess(stdout=BytesIO(), stderr=BytesIO(), poll_result=poll_result)

    git_tools._terminate_process(cast(subprocess.Popen[bytes], process))

    assert process.killed is expected_killed


def test_decode_git_output_replaces_invalid_utf8() -> None:
    assert git_tools._decode_git_output(b"bad:\xff") == "bad:\ufffd"


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            git_tools._GitProcessResult(
                stdout="stdout fallback",
                stderr="fatal: nope",
                returncode=1,
                truncated=False,
            ),
            "fatal: nope",
        ),
        (
            git_tools._GitProcessResult(
                stdout="stdout fallback",
                stderr="",
                returncode=1,
                truncated=False,
            ),
            "stdout fallback",
        ),
        (
            git_tools._GitProcessResult(
                stdout="",
                stderr="",
                returncode=1,
                truncated=False,
            ),
            "git status failed",
        ),
    ],
)
def test_format_git_failure_message_prefers_stderr_then_stdout_then_default(
    result: git_tools._GitProcessResult, expected: str
) -> None:
    assert git_tools._format_git_failure_message("git status", result) == expected


def test_format_git_failure_message_truncates_and_marks_limited_output() -> None:
    oversized = "x" * (git_tools.GIT_ERROR_MESSAGE_CHAR_LIMIT + 50)
    result = git_tools._GitProcessResult(
        stdout="",
        stderr=oversized,
        returncode=1,
        truncated=True,
    )

    message = git_tools._format_git_failure_message("git status", result)

    assert message.startswith("git status failed after exceeding the output limit. ")
    assert message.endswith("...(truncated)")


def test_stream_reader_emits_payload_and_closure_marker() -> None:
    events: queue.Queue[tuple[str, bytes | None]] = queue.Queue()

    git_tools._stream_reader(
        BytesIO(b"payload"),
        stream_name="stdout",
        event_queue=events,
    )

    assert events.get_nowait() == ("stdout", b"payload")
    assert events.get_nowait() == ("stdout", None)


def test_collect_process_output_rejects_missing_pipes() -> None:
    process = _DummyProcess(stdout=None, stderr=None)

    with pytest.raises(RuntimeError, match="did not expose stdout/stderr pipes"):
        git_tools._collect_process_output(
            cast(subprocess.Popen[bytes], process),
            output_limit_bytes=16,
            timeout_seconds=1,
        )


def test_collect_process_output_truncates_without_failing() -> None:
    process = _DummyProcess(
        stdout=BytesIO(b"abcdef"),
        stderr=BytesIO(b"ghij"),
        returncode=0,
        poll_result=None,
    )

    stdout, stderr, truncated = git_tools._collect_process_output(
        cast(subprocess.Popen[bytes], process),
        output_limit_bytes=5,
        timeout_seconds=1,
    )

    assert stdout == b"abcde"
    assert stderr == b""
    assert truncated is True
    assert process.killed is False
    assert process.wait_calls == [1]


def test_collect_process_output_times_out_when_no_events_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _DummyProcess(
        stdout=BytesIO(),
        stderr=BytesIO(),
        returncode=None,
        poll_result=None,
    )
    monotonic_calls = {"count": 0}

    def fake_monotonic() -> float:
        monotonic_calls["count"] += 1
        return 10.0 if monotonic_calls["count"] <= 2 else 12.0

    monkeypatch.setattr(git_tools.threading, "Thread", _NoOpThread)
    monkeypatch.setattr(git_tools.queue, "Queue", _EmptyQueue)
    monkeypatch.setattr(git_tools.time, "monotonic", fake_monotonic)

    with pytest.raises(RuntimeError, match="timed out after 1 seconds"):
        git_tools._collect_process_output(
            cast(subprocess.Popen[bytes], process),
            output_limit_bytes=16,
            timeout_seconds=1,
        )

    assert process.killed is True


def test_collect_process_output_kills_process_if_wait_times_out() -> None:
    process = _WaitTimeoutThenExitProcess(
        stdout=BytesIO(b"ok"),
        stderr=BytesIO(b""),
        returncode=0,
        poll_result=None,
    )

    stdout, stderr, truncated = git_tools._collect_process_output(
        cast(subprocess.Popen[bytes], process),
        output_limit_bytes=16,
        timeout_seconds=1,
    )

    assert stdout == b"ok"
    assert stderr == b""
    assert truncated is False
    assert process.killed is True
    assert process.wait_calls == [1, 1]


def test_run_git_subprocess_invokes_popen_and_decodes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path.resolve()
    recorded: dict[str, object] = {}
    process = _DummyProcess(
        stdout=BytesIO(b"ignored"),
        stderr=BytesIO(b"ignored"),
        returncode=7,
        poll_result=7,
    )

    def fake_popen(
        args: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        stdout: int,
        stderr: int,
    ) -> _DummyProcess:
        recorded["args"] = args
        recorded["cwd"] = cwd
        recorded["env"] = env
        recorded["stdout"] = stdout
        recorded["stderr"] = stderr
        return process

    monkeypatch.setattr(git_tools.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        git_tools,
        "_collect_process_output",
        lambda proc, output_limit_bytes, timeout_seconds: (b"out\xff", b"err", True),
    )

    result = git_tools._run_git_subprocess(workspace, ["status"], output_limit_bytes=64)

    assert recorded["args"] == ["git", "status"]
    assert recorded["cwd"] == workspace
    assert recorded["stdout"] == subprocess.PIPE
    assert recorded["stderr"] == subprocess.PIPE
    assert cast(dict[str, str], recorded["env"])["GIT_TERMINAL_PROMPT"] == "0"
    assert result == git_tools._GitProcessResult(
        stdout="out\ufffd",
        stderr="err",
        returncode=7,
        truncated=True,
    )


def test_run_git_subprocess_reports_missing_git_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_popen(*args: object, **kwargs: object) -> _DummyProcess:
        del args, kwargs
        raise FileNotFoundError

    monkeypatch.setattr(git_tools.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="git executable not found"):
        git_tools._run_git_subprocess(
            tmp_path.resolve(), ["status"], output_limit_bytes=64
        )


@pytest.mark.parametrize(
    "completed",
    [
        git_tools._GitProcessResult(
            stdout="", stderr="fatal", returncode=1, truncated=False
        ),
        git_tools._GitProcessResult(
            stdout="repo\n", stderr="", returncode=0, truncated=True
        ),
        git_tools._GitProcessResult(
            stdout="\n", stderr="", returncode=0, truncated=False
        ),
    ],
)
def test_resolve_git_repository_root_rejects_failed_or_invalid_discovery_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed: git_tools._GitProcessResult,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr(
        git_tools, "_run_git_subprocess", lambda *args, **kwargs: completed
    )

    with pytest.raises(RuntimeError):
        git_tools._resolve_git_repository_root(
            ToolContext(invocation_id="inv-resolve-1", workspace=str(workspace)),
            ".",
        )


def test_resolve_git_repository_root_rejects_non_directory_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo_file = tmp_path / "repo.txt"
    repo_file.write_text("nope", encoding="utf-8")

    monkeypatch.setattr(
        git_tools,
        "_run_git_subprocess",
        lambda *args, **kwargs: git_tools._GitProcessResult(
            stdout=f"{repo_file}\n",
            stderr="",
            returncode=0,
            truncated=False,
        ),
    )

    with pytest.raises(RuntimeError, match="is invalid"):
        git_tools._resolve_git_repository_root(
            ToolContext(invocation_id="inv-resolve-2", workspace=str(workspace)),
            ".",
        )


def test_resolve_git_repository_root_rejects_parent_repo_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_repo = tmp_path / "outside-repo"
    outside_repo.mkdir()

    monkeypatch.setattr(
        git_tools,
        "_run_git_subprocess",
        lambda root, args, output_limit_bytes: git_tools._GitProcessResult(
            stdout=f"{outside_repo}\n",
            stderr="",
            returncode=0,
            truncated=False,
        ),
    )

    with pytest.raises(ValueError, match="outside the workspace root"):
        git_tools._resolve_git_repository_root(
            ToolContext(invocation_id="inv-4", workspace=str(workspace)),
            ".",
        )


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        (" HEAD~1 ", "HEAD~1"),
        ("", ValueError),
        ("--output=/tmp/leak", ValueError),
        ("HEAD\nmain", ValueError),
    ],
)
def test_validate_git_diff_ref(ref: str, expected: str | type[Exception]) -> None:
    if isinstance(expected, str):
        assert git_tools._validate_git_diff_ref(ref) == expected
    else:
        with pytest.raises(expected):
            git_tools._validate_git_diff_ref(ref)


def test_run_git_status_tool_invokes_git_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path.resolve()
    recorded: dict[str, object] = {}

    def fake_resolve(context: ToolContext, path: str) -> Path:
        del context
        recorded["path"] = path
        return workspace

    def fake_run(root: Path, args: list[str]) -> git_tools.GitCommandResult:
        recorded["root"] = root
        recorded["args"] = args
        return git_tools.GitCommandResult(text="clean\n", truncated=False)

    monkeypatch.setattr(git_tools, "_resolve_git_repository_root", fake_resolve)
    monkeypatch.setattr(git_tools, "_run_git_command", fake_run)

    result = RunGitStatusTool().invoke(
        ToolContext(invocation_id="inv-1", workspace=str(workspace)),
        RunGitStatusTool.input_model(path="."),
    )

    assert result.status_text == "clean\n"
    assert result.truncated is False
    assert recorded == {
        "path": ".",
        "root": workspace,
        "args": ["status", "--short", "--branch"],
    }


def test_run_git_diff_tool_supports_staged_and_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path.resolve()
    calls: list[list[str]] = []

    monkeypatch.setattr(
        git_tools,
        "_resolve_git_repository_root",
        lambda context, path: workspace,
    )

    def fake_run(root: Path, args: list[str]) -> git_tools.GitCommandResult:
        assert root == workspace
        calls.append(args)
        return git_tools.GitCommandResult(text="diff\n", truncated=False)

    monkeypatch.setattr(git_tools, "_run_git_command", fake_run)
    tool = RunGitDiffTool()
    context = ToolContext(invocation_id="inv-2", workspace=str(workspace))

    staged = tool.invoke(context, RunGitDiffTool.input_model(staged=True))
    compared = tool.invoke(context, RunGitDiffTool.input_model(ref="HEAD~1"))

    assert staged.diff_text == "diff\n"
    assert compared.diff_text == "diff\n"
    assert calls == [
        ["diff", "--no-ext-diff", "--no-textconv", "--staged"],
        ["diff", "--no-ext-diff", "--no-textconv", "HEAD~1"],
    ]


def test_run_git_diff_tool_rejects_option_like_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path.resolve()
    monkeypatch.setattr(
        git_tools,
        "_resolve_git_repository_root",
        lambda context, path: workspace,
    )

    with pytest.raises(ValueError, match="must not start with '-'"):
        RunGitDiffTool().invoke(
            ToolContext(invocation_id="inv-3", workspace=str(workspace)),
            RunGitDiffTool.input_model(ref="--output=/tmp/leak"),
        )


def test_run_git_log_tool_rejects_large_limits() -> None:
    with pytest.raises(ValidationError):
        RunGitLogTool.input_model(limit=git_tools.MAX_GIT_LOG_LIMIT + 1)


def test_run_git_command_returns_truncated_output_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path.resolve()
    oversized = "x" * (git_tools.GIT_COMMAND_OUTPUT_CHAR_LIMIT + 32)

    monkeypatch.setattr(
        git_tools,
        "_run_git_subprocess",
        lambda root, args, output_limit_bytes: git_tools._GitProcessResult(
            stdout=oversized,
            stderr="",
            returncode=0,
            truncated=True,
        ),
    )

    result = git_tools._run_git_command(workspace, ["status", "--short", "--branch"])

    assert result.truncated is True
    assert result.text.endswith("...(truncated)")
    assert len(result.text) == git_tools.GIT_COMMAND_OUTPUT_CHAR_LIMIT


def test_run_git_command_raises_with_formatted_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path.resolve()
    monkeypatch.setattr(
        git_tools,
        "_run_git_subprocess",
        lambda root, args, output_limit_bytes: git_tools._GitProcessResult(
            stdout="",
            stderr="fatal: bad revision",
            returncode=1,
            truncated=False,
        ),
    )

    with pytest.raises(RuntimeError, match="fatal: bad revision"):
        git_tools._run_git_command(workspace, ["diff", "HEAD~1"])


def test_run_git_log_tool_invokes_git_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path.resolve()
    calls: list[list[str]] = []

    monkeypatch.setattr(
        git_tools,
        "_resolve_git_repository_root",
        lambda context, path: workspace,
    )

    def fake_run(root: Path, args: list[str]) -> git_tools.GitCommandResult:
        assert root == workspace
        calls.append(args)
        return git_tools.GitCommandResult(text="abc123 commit\n", truncated=False)

    monkeypatch.setattr(git_tools, "_run_git_command", fake_run)

    result = RunGitLogTool().invoke(
        ToolContext(invocation_id="inv-log", workspace=str(workspace)),
        RunGitLogTool.input_model(limit=5),
    )

    assert result.log_text == "abc123 commit\n"
    assert result.truncated is False
    assert calls == [["log", "--max-count", "5", "--oneline", "--decorate"]]


def test_run_git_log_tool_raises_for_non_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path.resolve()
    monkeypatch.setattr(
        git_tools,
        "_resolve_git_repository_root",
        lambda context, path: workspace,
    )

    def fake_run(root: Path, args: list[str]) -> git_tools.GitCommandResult:
        del root, args
        raise RuntimeError("fatal: not a git repository")

    monkeypatch.setattr(git_tools, "_run_git_command", fake_run)

    with pytest.raises(RuntimeError, match="not a git repository"):
        RunGitLogTool().invoke(
            ToolContext(invocation_id="inv-5", workspace=str(workspace)),
            RunGitLogTool.input_model(limit=5),
        )


def test_register_git_tools_registers_all_builtins() -> None:
    registry = ToolRegistry()

    git_tools.register_git_tools(registry)

    assert {spec.name for spec in registry.list_tools()} >= {
        "run_git_status",
        "run_git_diff",
        "run_git_log",
    }

"""Unit tests for git built-in tools."""

from __future__ import annotations

import subprocess

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
from llm_tools.tools.git import RunGitDiffTool, RunGitLogTool, RunGitStatusTool


def _runtime() -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(RunGitStatusTool())
    registry.register(RunGitDiffTool())
    registry.register(RunGitLogTool())
    return ToolRuntime(
        registry,
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
            allow_subprocess=True,
        ),
    )


def test_run_git_status_tool_invokes_git_status(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path
    recorded: dict[str, object] = {}

    def fake_run(
        args: list[str],
        *,
        cwd: object,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: object = None,
    ) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        recorded["cwd"] = cwd
        recorded["capture_output"] = capture_output
        recorded["text"] = text
        recorded["check"] = check
        recorded["timeout"] = timeout
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="clean\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _runtime().execute(
        ToolInvocationRequest(tool_name="run_git_status", arguments={"path": "."}),
        ToolContext(invocation_id="inv-1", workspace=str(workspace)),
    )

    assert result.ok is True
    assert result.output == {
        "resolved_root": str(workspace.resolve()),
        "status_text": "clean\n",
    }
    assert recorded["args"] == ["git", "status", "--short", "--branch"]
    assert recorded["cwd"] == workspace.resolve()


def test_run_git_diff_tool_supports_staged_and_ref(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path
    calls: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        cwd: object,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: object = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check, timeout
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="diff\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runtime = _runtime()
    context = ToolContext(invocation_id="inv-2", workspace=str(workspace))

    staged = runtime.execute(
        ToolInvocationRequest(tool_name="run_git_diff", arguments={"staged": True}),
        context,
    )
    compared = runtime.execute(
        ToolInvocationRequest(tool_name="run_git_diff", arguments={"ref": "HEAD~1"}),
        context,
    )

    assert staged.ok is True
    assert staged.output["diff_text"] == "diff\n"
    assert compared.ok is True
    assert compared.output["diff_text"] == "diff\n"
    assert calls == [
        ["git", "diff", "--staged"],
        ["git", "diff", "HEAD~1"],
    ]


def test_run_git_log_tool_raises_for_non_zero_exit(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path

    def fake_run(
        args: list[str],
        *,
        cwd: object,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: object = None,
    ) -> subprocess.CompletedProcess[str]:
        del args, cwd, capture_output, text, check, timeout
        return subprocess.CompletedProcess(
            args=["git", "log"],
            returncode=1,
            stdout="",
            stderr="fatal: not a git repository",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _runtime().execute(
        ToolInvocationRequest(tool_name="run_git_log", arguments={"limit": 5}),
        ToolContext(invocation_id="inv-3", workspace=str(workspace)),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    assert "not a git repository" in result.error.details["exception_message"]

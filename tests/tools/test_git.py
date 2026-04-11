"""Unit tests for git built-in tools."""

from __future__ import annotations

import subprocess

import pytest

from llm_tools.tool_api import ToolContext
from llm_tools.tools.git import RunGitDiffTool, RunGitLogTool, RunGitStatusTool
from llm_tools.tools.git import tools as git_tools


def test_run_git_status_tool_invokes_git_status(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path
    recorded: dict[str, object] = {}

    def fake_run(
        args: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        recorded["cwd"] = cwd
        recorded["capture_output"] = capture_output
        recorded["text"] = text
        recorded["check"] = check
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="clean\n")

    monkeypatch.setattr(git_tools.subprocess, "run", fake_run)

    result = RunGitStatusTool().invoke(
        ToolContext(invocation_id="inv-1", workspace=str(workspace)),
        RunGitStatusTool.input_model(path="."),
    )

    assert result.status_text == "clean\n"
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
        cwd: str,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="diff\n")

    monkeypatch.setattr(git_tools.subprocess, "run", fake_run)
    tool = RunGitDiffTool()
    context = ToolContext(invocation_id="inv-2", workspace=str(workspace))

    staged = tool.invoke(context, RunGitDiffTool.input_model(staged=True))
    compared = tool.invoke(context, RunGitDiffTool.input_model(ref="HEAD~1"))

    assert staged.diff_text == "diff\n"
    assert compared.diff_text == "diff\n"
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
        cwd: str,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del args, cwd, capture_output, text, check
        return subprocess.CompletedProcess(
            args=["git", "log"],
            returncode=1,
            stdout="",
            stderr="fatal: not a git repository",
        )

    monkeypatch.setattr(git_tools.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="not a git repository"):
        RunGitLogTool().invoke(
            ToolContext(invocation_id="inv-3", workspace=str(workspace)),
            RunGitLogTool.input_model(limit=5),
        )

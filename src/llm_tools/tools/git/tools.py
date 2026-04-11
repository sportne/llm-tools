"""Git built-in tool implementations."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from llm_tools.tool_api import (
    SideEffectClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolSpec,
)
from llm_tools.tools._path_utils import resolve_workspace_path


def _run_git_command(root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "git failed"
        raise RuntimeError(message)
    return completed.stdout


class GitCommandInput(BaseModel):
    path: str = "."


class GitCommandOutput(BaseModel):
    resolved_root: str
    text: str


class RunGitStatusOutput(BaseModel):
    resolved_root: str
    status_text: str


class RunGitStatusTool(Tool[GitCommandInput, RunGitStatusOutput]):
    spec = ToolSpec(
        name="run_git_status",
        description="Run a non-interactive git status command in the workspace.",
        tags=["git", "status", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
        requires_subprocess=True,
    )
    input_model = GitCommandInput
    output_model = RunGitStatusOutput

    def invoke(self, context: ToolContext, args: GitCommandInput) -> RunGitStatusOutput:
        root = resolve_workspace_path(
            context,
            args.path,
            expect_directory=True,
            must_exist=True,
        )
        status_text = _run_git_command(root, ["status", "--short", "--branch"])
        context.logs.append(f"Ran git status in '{root}'.")
        return RunGitStatusOutput(resolved_root=str(root), status_text=status_text)


class RunGitDiffInput(BaseModel):
    path: str = "."
    ref: str | None = None
    staged: bool = False


class RunGitDiffOutput(BaseModel):
    resolved_root: str
    diff_text: str


class RunGitDiffTool(Tool[RunGitDiffInput, RunGitDiffOutput]):
    spec = ToolSpec(
        name="run_git_diff",
        description="Run a non-interactive git diff command in the workspace.",
        tags=["git", "diff", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
        requires_subprocess=True,
    )
    input_model = RunGitDiffInput
    output_model = RunGitDiffOutput

    def invoke(self, context: ToolContext, args: RunGitDiffInput) -> RunGitDiffOutput:
        root = resolve_workspace_path(
            context,
            args.path,
            expect_directory=True,
            must_exist=True,
        )
        command = ["diff"]
        if args.staged:
            command.append("--staged")
        if args.ref is not None:
            command.append(args.ref)
        diff_text = _run_git_command(root, command)
        context.logs.append(f"Ran git diff in '{root}'.")
        return RunGitDiffOutput(resolved_root=str(root), diff_text=diff_text)


class RunGitLogInput(BaseModel):
    path: str = "."
    limit: int = Field(default=10, ge=1)


class RunGitLogOutput(BaseModel):
    resolved_root: str
    log_text: str


class RunGitLogTool(Tool[RunGitLogInput, RunGitLogOutput]):
    spec = ToolSpec(
        name="run_git_log",
        description="Run a non-interactive git log command in the workspace.",
        tags=["git", "log", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
        requires_subprocess=True,
    )
    input_model = RunGitLogInput
    output_model = RunGitLogOutput

    def invoke(self, context: ToolContext, args: RunGitLogInput) -> RunGitLogOutput:
        root = resolve_workspace_path(
            context,
            args.path,
            expect_directory=True,
            must_exist=True,
        )
        log_text = _run_git_command(
            root,
            ["log", "--max-count", str(args.limit), "--oneline", "--decorate"],
        )
        context.logs.append(f"Ran git log in '{root}'.")
        return RunGitLogOutput(resolved_root=str(root), log_text=log_text)


def register_git_tools(registry: ToolRegistry) -> None:
    """Register the built-in git tool set."""
    registry.register(RunGitStatusTool())
    registry.register(RunGitDiffTool())
    registry.register(RunGitLogTool())

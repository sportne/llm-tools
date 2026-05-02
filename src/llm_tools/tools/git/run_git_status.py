"""Git status tool."""

from __future__ import annotations

from pydantic import BaseModel

from llm_tools.tool_api import (
    RiskLevel,
    SideEffectClass,
    Tool,
    ToolExecutionContext,
    ToolSpec,
)
from llm_tools.tools.git._shared import (
    GIT_COMMAND_TIMEOUT_SECONDS,
    GitCommandInput,
)


class RunGitStatusOutput(BaseModel):
    resolved_root: str
    status_text: str
    truncated: bool = False


class RunGitStatusTool(Tool[GitCommandInput, RunGitStatusOutput]):
    spec = ToolSpec(
        name="run_git_status",
        description="Run a non-interactive git status command in the workspace.",
        tags=["git", "status", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        timeout_seconds=GIT_COMMAND_TIMEOUT_SECONDS,
        risk_level=RiskLevel.MEDIUM,
        requires_filesystem=True,
        requires_subprocess=True,
        retain_output_in_execution_record=False,
    )
    input_model = GitCommandInput
    output_model = RunGitStatusOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: GitCommandInput,
    ) -> RunGitStatusOutput:
        from llm_tools.tools.git import tools as git_tools

        root = git_tools._resolve_git_repository_root(context, args.path)
        status = git_tools._run_git_command(root, ["status", "--short", "--branch"])
        context.log(f"Ran git status in '{root}'.")
        return RunGitStatusOutput(
            resolved_root=str(root),
            status_text=status.text,
            truncated=status.truncated,
        )

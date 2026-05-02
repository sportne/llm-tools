"""Git log tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tool_api import (
    RiskLevel,
    SideEffectClass,
    Tool,
    ToolExecutionContext,
    ToolSpec,
)
from llm_tools.tools.git._shared import (
    GIT_COMMAND_TIMEOUT_SECONDS,
    MAX_GIT_LOG_LIMIT,
)


class RunGitLogInput(BaseModel):
    path: str = "."
    limit: int = Field(default=10, ge=1, le=MAX_GIT_LOG_LIMIT)


class RunGitLogOutput(BaseModel):
    resolved_root: str
    log_text: str
    truncated: bool = False


class RunGitLogTool(Tool[RunGitLogInput, RunGitLogOutput]):
    spec = ToolSpec(
        name="run_git_log",
        description="Run a non-interactive git log command in the workspace.",
        tags=["git", "log", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        timeout_seconds=GIT_COMMAND_TIMEOUT_SECONDS,
        risk_level=RiskLevel.MEDIUM,
        requires_filesystem=True,
        requires_subprocess=True,
        retain_output_in_execution_record=False,
    )
    input_model = RunGitLogInput
    output_model = RunGitLogOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: RunGitLogInput,
    ) -> RunGitLogOutput:
        from llm_tools.tools.git import tools as git_tools

        root = git_tools._resolve_git_repository_root(context, args.path)
        log = git_tools._run_git_command(
            root,
            ["log", "--max-count", str(args.limit), "--oneline", "--decorate"],
        )
        context.log(f"Ran git log in '{root}'.")
        return RunGitLogOutput(
            resolved_root=str(root),
            log_text=log.text,
            truncated=log.truncated,
        )

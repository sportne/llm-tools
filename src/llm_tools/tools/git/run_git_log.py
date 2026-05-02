"""Git log tool."""

from __future__ import annotations

from llm_tools.tool_api import (
    RiskLevel,
    SideEffectClass,
    Tool,
    ToolExecutionContext,
    ToolSpec,
)
from llm_tools.tools.git._shared import (
    GIT_COMMAND_TIMEOUT_SECONDS,
)
from llm_tools.tools.git.run_git_log_models import (
    RunGitLogInput as RunGitLogInput,
)
from llm_tools.tools.git.run_git_log_models import (
    RunGitLogOutput as RunGitLogOutput,
)


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

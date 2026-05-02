"""Git diff tool."""

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
from llm_tools.tools.git.run_git_diff_models import (
    RunGitDiffInput as RunGitDiffInput,
)
from llm_tools.tools.git.run_git_diff_models import (
    RunGitDiffOutput as RunGitDiffOutput,
)


class RunGitDiffTool(Tool[RunGitDiffInput, RunGitDiffOutput]):
    spec = ToolSpec(
        name="run_git_diff",
        description="Run a non-interactive git diff command in the workspace.",
        tags=["git", "diff", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        timeout_seconds=GIT_COMMAND_TIMEOUT_SECONDS,
        risk_level=RiskLevel.MEDIUM,
        requires_filesystem=True,
        requires_subprocess=True,
        retain_output_in_execution_record=False,
    )
    input_model = RunGitDiffInput
    output_model = RunGitDiffOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: RunGitDiffInput,
    ) -> RunGitDiffOutput:
        from llm_tools.tools.git import tools as git_tools

        root = git_tools._resolve_git_repository_root(context, args.path)
        command = ["diff", "--no-ext-diff", "--no-textconv"]
        if args.staged:
            command.append("--staged")
        if args.ref is not None:
            command.append(git_tools._validate_git_diff_ref(args.ref))
        diff = git_tools._run_git_command(root, command)
        context.log(f"Ran git diff in '{root}'.")
        return RunGitDiffOutput(
            resolved_root=str(root),
            diff_text=diff.text,
            truncated=diff.truncated,
        )

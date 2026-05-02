"""Find-files filesystem tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.filesystem._ops import find_files_impl
from llm_tools.tools.filesystem._shared import (
    require_repository_metadata,
    source_filters_for_call,
)
from llm_tools.tools.filesystem.models import FileSearchResult


class FindFilesInput(BaseModel):
    path: str = Field(
        default=".",
        description=(
            "Workspace-relative directory to search under. Use '.' for the "
            "workspace root."
        ),
    )
    pattern: str = Field(
        description=(
            "Workspace-relative glob pattern matched against full relative file "
            "paths under path. Use recursive globs such as '**/*.py' to find "
            "Python files in nested directories. A pattern like '*.py' only "
            "matches files directly at the search root."
        )
    )
    include_hidden: bool = Field(
        default=False,
        description=(
            "Include hidden files and directories such as '.github' or '.env' "
            "when true. Hidden paths are excluded by default."
        ),
    )


class FindFilesOutput(FileSearchResult):
    pass


class FindFilesTool(Tool[FindFilesInput, FindFilesOutput]):
    spec = ToolSpec(
        name="find_files",
        description=(
            "Find files under a workspace-relative directory using path globs. "
            "Use recursive patterns like '**/*.py' for files in nested "
            "directories; '*.py' only matches files directly at the search root. "
            "Hidden paths are excluded unless include_hidden is true."
        ),
        tags=["filesystem", "read", "search"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = FindFilesInput
    output_model = FindFilesOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: FindFilesInput,
    ) -> FindFilesOutput:
        root_path, source_filters, tool_limits = require_repository_metadata(context)
        result = find_files_impl(
            root_path,
            args.pattern,
            args.path,
            source_filters=source_filters_for_call(
                source_filters,
                include_hidden=args.include_hidden,
            ),
            tool_limits=tool_limits,
        )
        context.log(f"Searched for file pattern '{args.pattern}'.")
        return FindFilesOutput.model_validate(result.model_dump(mode="json"))

"""Find-files filesystem tool."""

from __future__ import annotations

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.filesystem._ops import find_files_impl
from llm_tools.tools.filesystem._shared import (
    require_repository_metadata,
    source_filters_for_call,
)
from llm_tools.tools.filesystem.find_files_models import (
    FindFilesInput as FindFilesInput,
)
from llm_tools.tools.filesystem.find_files_models import (
    FindFilesOutput as FindFilesOutput,
)


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

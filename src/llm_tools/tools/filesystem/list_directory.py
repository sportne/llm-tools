"""List-directory filesystem tool."""

from __future__ import annotations

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.filesystem._ops import list_directory_impl
from llm_tools.tools.filesystem._shared import (
    require_repository_metadata,
    source_filters_for_call,
)
from llm_tools.tools.filesystem.list_directory_models import (
    ListDirectoryInput as ListDirectoryInput,
)
from llm_tools.tools.filesystem.list_directory_models import (
    ListDirectoryOutput as ListDirectoryOutput,
)


class ListDirectoryTool(Tool[ListDirectoryInput, ListDirectoryOutput]):
    spec = ToolSpec(
        name="list_directory",
        description="List immediate or recursive children of one directory.",
        tags=["filesystem", "read", "list"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = ListDirectoryInput
    output_model = ListDirectoryOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: ListDirectoryInput,
    ) -> ListDirectoryOutput:
        root_path, source_filters, tool_limits = require_repository_metadata(context)
        result = list_directory_impl(
            root_path,
            args.path,
            source_filters=source_filters_for_call(
                source_filters,
                include_hidden=args.include_hidden,
            ),
            tool_limits=tool_limits,
            recursive=args.recursive,
            max_depth=args.max_depth,
        )
        context.log(f"Listed directory '{args.path}'.")
        return ListDirectoryOutput.model_validate(result.model_dump(mode="json"))

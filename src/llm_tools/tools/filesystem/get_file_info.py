"""Get-file-info filesystem tool."""

from __future__ import annotations

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.filesystem._ops import get_file_info_impl
from llm_tools.tools.filesystem._shared import require_repository_metadata
from llm_tools.tools.filesystem.get_file_info_models import (
    GetFileInfoInput as GetFileInfoInput,
)
from llm_tools.tools.filesystem.get_file_info_models import (
    GetFileInfoOutput as GetFileInfoOutput,
)
from llm_tools.tools.filesystem.models import FileInfoResult


class GetFileInfoTool(Tool[GetFileInfoInput, GetFileInfoOutput]):
    spec = ToolSpec(
        name="get_file_info",
        description=(
            "Inspect one file or a small batch of files before deciding whether or "
            "how to read them."
        ),
        tags=["filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
        writes_internal_workspace_cache=True,
    )
    input_model = GetFileInfoInput
    output_model = GetFileInfoOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: GetFileInfoInput,
    ) -> GetFileInfoOutput:
        from llm_tools.tools.filesystem import tools as filesystem_tools

        root_path, _, tool_limits = require_repository_metadata(context)
        path_argument: str | list[str] = (
            args.path if args.path is not None else args.paths or []
        )
        result = get_file_info_impl(
            root_path,
            path_argument,
            tool_limits=tool_limits,
            cache_root=filesystem_tools._get_read_file_cache_root(root_path),
        )
        context.log("Collected file metadata.")
        if isinstance(result, FileInfoResult):
            return GetFileInfoOutput(results=[result])
        return GetFileInfoOutput(results=result.results)

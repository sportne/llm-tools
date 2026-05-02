"""Search-text filesystem tool."""

from __future__ import annotations

from pydantic import BaseModel

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tool_api.execution import get_workspace_root
from llm_tools.tools.filesystem._shared import (
    require_repository_metadata,
    source_filters_for_call,
)
from llm_tools.tools.filesystem.search_text_models import TextSearchResult
from llm_tools.tools.filesystem.search_text_ops import search_text_impl


class SearchTextInput(BaseModel):
    path: str = "."
    query: str
    include_hidden: bool = False


class SearchTextOutput(TextSearchResult):
    pass


class SearchTextTool(Tool[SearchTextInput, SearchTextOutput]):
    spec = ToolSpec(
        name="search_text",
        description="Search readable file contents for a literal substring.",
        tags=["filesystem", "search", "read", "text"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
        writes_internal_workspace_cache=True,
    )
    input_model = SearchTextInput
    output_model = SearchTextOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: SearchTextInput,
    ) -> SearchTextOutput:
        from llm_tools.tools.filesystem import tools as filesystem_tools

        root_path = get_workspace_root(context)
        _, source_filters, tool_limits = require_repository_metadata(context)
        result = search_text_impl(
            root_path,
            args.query,
            args.path,
            source_filters=source_filters_for_call(
                source_filters,
                include_hidden=args.include_hidden,
            ),
            tool_limits=tool_limits,
            cache_root=filesystem_tools._get_read_file_cache_root(root_path),
        )
        context.log(f"Searched text for '{args.query}'.")
        return SearchTextOutput.model_validate(result.model_dump(mode="json"))

"""Text-search built-in tool implementations."""

from __future__ import annotations

from pydantic import BaseModel

from llm_tools.tool_api import (
    SideEffectClass,
    Tool,
    ToolExecutionContext,
    ToolRegistry,
    ToolSpec,
)
from llm_tools.tool_api.execution import get_workspace_root
from llm_tools.tools.filesystem._content import _get_read_file_cache_root
from llm_tools.tools.filesystem.models import SourceFilters, ToolLimits
from llm_tools.tools.text._ops import search_text_impl
from llm_tools.tools.text.models import TextSearchResult


def _require_repository_metadata(
    context: ToolExecutionContext,
) -> tuple[SourceFilters, ToolLimits]:
    metadata = context.metadata
    source_filters = SourceFilters.model_validate(metadata.get("source_filters", {}))
    tool_limits = ToolLimits.model_validate(metadata.get("tool_limits", {}))
    return source_filters, tool_limits


def _source_filters_for_call(
    source_filters: SourceFilters, *, include_hidden: bool
) -> SourceFilters:
    if not include_hidden:
        return source_filters
    return source_filters.model_copy(update={"include_hidden": True})


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
        tags=["text", "search", "read"],
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
        root_path = get_workspace_root(context)
        source_filters, tool_limits = _require_repository_metadata(context)
        result = search_text_impl(
            root_path,
            args.query,
            args.path,
            source_filters=_source_filters_for_call(
                source_filters,
                include_hidden=args.include_hidden,
            ),
            tool_limits=tool_limits,
            cache_root=_get_read_file_cache_root(root_path),
        )
        context.log(f"Searched text for '{args.query}'.")
        return SearchTextOutput.model_validate(result.model_dump(mode="json"))


def register_text_tools(registry: ToolRegistry) -> None:
    """Register the built-in text search tool set."""
    registry.register(SearchTextTool())


__all__ = [
    "SearchTextTool",
    "register_text_tools",
]

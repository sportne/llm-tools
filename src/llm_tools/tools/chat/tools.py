"""Tool classes exposing the repository chat tool set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_tools.tool_api import (
    SideEffectClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolSpec,
)
from llm_tools.tools.chat._content import dump_json
from llm_tools.tools.chat._ops import (
    find_files_impl,
    get_file_info_impl,
    list_directory_impl,
    list_directory_recursive_impl,
    read_file_impl,
    search_text_impl,
)
from llm_tools.tools.chat.models import (
    ChatSessionConfig,
    ChatSourceFilters,
    ChatToolLimits,
    DirectoryListingResult,
    FileInfoResult,
    FileReadResult,
    FileSearchResult,
    GetFileInfoInputShape,
    TextSearchResult,
)


def _require_chat_metadata(
    context: ToolContext,
) -> tuple[Path, ChatSourceFilters, ChatSessionConfig, ChatToolLimits]:
    workspace = Path(context.workspace or Path.cwd()).resolve()
    metadata = context.metadata
    source_filters = ChatSourceFilters.model_validate(metadata.get("source_filters", {}))
    session_config = ChatSessionConfig.model_validate(metadata.get("session_config", {}))
    tool_limits = ChatToolLimits.model_validate(metadata.get("tool_limits", {}))
    return workspace, source_filters, session_config, tool_limits


class ListDirectoryInput(BaseModel):
    path: str = "."


class ListDirectoryOutput(DirectoryListingResult):
    pass


class ListDirectoryTool(Tool[ListDirectoryInput, ListDirectoryOutput]):
    spec = ToolSpec(
        name="list_directory",
        description="List immediate children of one directory.",
        tags=["chat", "filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = ListDirectoryInput
    output_model = ListDirectoryOutput

    def invoke(self, context: ToolContext, args: ListDirectoryInput) -> ListDirectoryOutput:
        root_path, source_filters, _, tool_limits = _require_chat_metadata(context)
        result = list_directory_impl(
            root_path,
            args.path,
            source_filters=source_filters,
            tool_limits=tool_limits,
        )
        context.logs.append(f"Listed directory '{args.path}'.")
        return ListDirectoryOutput.model_validate(result.model_dump(mode="json"))


class ListDirectoryRecursiveInput(BaseModel):
    path: str = "."
    max_depth: int | None = Field(default=None, gt=0)


class ListDirectoryRecursiveOutput(DirectoryListingResult):
    pass


class ListDirectoryRecursiveTool(
    Tool[ListDirectoryRecursiveInput, ListDirectoryRecursiveOutput]
):
    spec = ToolSpec(
        name="list_directory_recursive",
        description="List one directory subtree as a flat depth-first result.",
        tags=["chat", "filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = ListDirectoryRecursiveInput
    output_model = ListDirectoryRecursiveOutput

    def invoke(
        self,
        context: ToolContext,
        args: ListDirectoryRecursiveInput,
    ) -> ListDirectoryRecursiveOutput:
        root_path, source_filters, _, tool_limits = _require_chat_metadata(context)
        result = list_directory_recursive_impl(
            root_path,
            args.path,
            source_filters=source_filters,
            tool_limits=tool_limits,
            max_depth=args.max_depth,
        )
        context.logs.append(f"Recursively listed directory '{args.path}'.")
        return ListDirectoryRecursiveOutput.model_validate(
            result.model_dump(mode="json")
        )


class FindFilesInput(BaseModel):
    path: str = "."
    pattern: str


class FindFilesOutput(FileSearchResult):
    pass


class FindFilesTool(Tool[FindFilesInput, FindFilesOutput]):
    spec = ToolSpec(
        name="find_files",
        description="Find files by root-relative glob pattern.",
        tags=["chat", "filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = FindFilesInput
    output_model = FindFilesOutput

    def invoke(self, context: ToolContext, args: FindFilesInput) -> FindFilesOutput:
        root_path, source_filters, _, tool_limits = _require_chat_metadata(context)
        result = find_files_impl(
            root_path,
            args.pattern,
            args.path,
            source_filters=source_filters,
            tool_limits=tool_limits,
        )
        context.logs.append(f"Searched for file pattern '{args.pattern}'.")
        return FindFilesOutput.model_validate(result.model_dump(mode="json"))


class SearchTextInput(BaseModel):
    path: str = "."
    query: str


class SearchTextOutput(TextSearchResult):
    pass


class SearchTextTool(Tool[SearchTextInput, SearchTextOutput]):
    spec = ToolSpec(
        name="search_text",
        description="Search readable file contents for a literal substring.",
        tags=["chat", "filesystem", "read", "search"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = SearchTextInput
    output_model = SearchTextOutput

    def invoke(self, context: ToolContext, args: SearchTextInput) -> SearchTextOutput:
        root_path, source_filters, _, tool_limits = _require_chat_metadata(context)
        result = search_text_impl(
            root_path,
            args.query,
            args.path,
            source_filters=source_filters,
            tool_limits=tool_limits,
        )
        context.logs.append(f"Searched text for '{args.query}'.")
        return SearchTextOutput.model_validate(result.model_dump(mode="json"))


class GetFileInfoInput(GetFileInfoInputShape):
    pass


class GetFileInfoOutput(BaseModel):
    results: list[FileInfoResult] = Field(default_factory=list)


class GetFileInfoTool(Tool[GetFileInfoInput, GetFileInfoOutput]):
    spec = ToolSpec(
        name="get_file_info",
        description=(
            "Inspect one file or a small batch of files before deciding whether or "
            "how to read them."
        ),
        tags=["chat", "filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = GetFileInfoInput
    output_model = GetFileInfoOutput

    def invoke(self, context: ToolContext, args: GetFileInfoInput) -> GetFileInfoOutput:
        root_path, _, session_config, tool_limits = _require_chat_metadata(context)
        path_argument: str | list[str] = (
            args.path if args.path is not None else args.paths or []
        )
        result = get_file_info_impl(
            root_path,
            path_argument,
            session_config=session_config,
            tool_limits=tool_limits,
        )
        context.logs.append("Collected file metadata.")
        if isinstance(result, FileInfoResult):
            return GetFileInfoOutput(results=[result])
        return GetFileInfoOutput(results=result.results)


class ReadFileInput(BaseModel):
    path: str
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class ReadFileOutput(FileReadResult):
    pass


class ReadFileTool(Tool[ReadFileInput, ReadFileOutput]):
    spec = ToolSpec(
        name="read_file",
        description="Read text or converted markdown content for one file.",
        tags=["chat", "filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = ReadFileInput
    output_model = ReadFileOutput

    def invoke(self, context: ToolContext, args: ReadFileInput) -> ReadFileOutput:
        root_path, _, session_config, tool_limits = _require_chat_metadata(context)
        result = read_file_impl(
            root_path,
            args.path,
            session_config=session_config,
            tool_limits=tool_limits,
            start_char=args.start_char,
            end_char=args.end_char,
        )
        context.logs.append(f"Read file '{args.path}'.")
        return ReadFileOutput.model_validate(result.model_dump(mode="json"))


def register_chat_tools(registry: ToolRegistry) -> None:
    """Register the repository-chat tool set."""
    registry.register(ListDirectoryTool())
    registry.register(ListDirectoryRecursiveTool())
    registry.register(FindFilesTool())
    registry.register(SearchTextTool())
    registry.register(GetFileInfoTool())
    registry.register(ReadFileTool())


def format_tool_result_for_model(result: dict[str, Any], *, max_chars: int) -> str:
    """Return a bounded plain-text tool result message for chat transcripts."""
    rendered = dump_json(result)
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[:max_chars]}...(truncated)"

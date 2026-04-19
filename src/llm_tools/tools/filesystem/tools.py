"""Filesystem built-in tool implementations."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from llm_tools.tool_api import (
    SideEffectClass,
    SourceProvenanceRef,
    Tool,
    ToolExecutionContext,
    ToolRegistry,
    ToolSpec,
)
from llm_tools.tool_api.execution import get_workspace_root
from llm_tools.tools._path_utils import relative_display_path
from llm_tools.tools.filesystem._content import (
    _get_cached_conversion_paths,
    _get_read_file_cache_root,
    _read_cached_conversion,
    _write_cached_conversion,
)
from llm_tools.tools.filesystem._ops import (
    find_files_impl,
    get_file_info_impl,
    list_directory_impl,
    read_file_impl,
)
from llm_tools.tools.filesystem._paths import resolve_writable_file_path
from llm_tools.tools.filesystem.models import (
    DirectoryListingResult,
    FileInfoResult,
    FileReadResult,
    FileSearchResult,
    GetFileInfoInputShape,
    SourceFilters,
    ToolLimits,
)


def _append_local_source_provenance(
    context: ToolExecutionContext, *, relative_path: str
) -> None:
    source_id = f"workspace:{relative_path}"
    context.add_source_provenance(
        SourceProvenanceRef(
            source_kind="local_file",
            source_id=source_id,
            content_hash=hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
            whole_source_reproduction_allowed=True,
            metadata={"path": relative_path},
        )
    )


def _require_repository_metadata(
    context: ToolExecutionContext,
) -> tuple[Path, SourceFilters, ToolLimits]:
    workspace = get_workspace_root(context)
    metadata = context.metadata
    source_filters = SourceFilters.model_validate(metadata.get("source_filters", {}))
    tool_limits = ToolLimits.model_validate(metadata.get("tool_limits", {}))
    return workspace, source_filters, tool_limits


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
        tags=["filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
        writes_internal_workspace_cache=True,
    )
    input_model = ReadFileInput
    output_model = ReadFileOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: ReadFileInput,
    ) -> ReadFileOutput:
        root_path, _, tool_limits = _require_repository_metadata(context)
        result = read_file_impl(
            root_path,
            args.path,
            tool_limits=tool_limits,
            start_char=args.start_char,
            end_char=args.end_char,
            cache_root=_get_read_file_cache_root(root_path),
        )
        context.log(f"Read file '{result.resolved_path}'.")
        context.add_artifact(result.resolved_path)
        _append_local_source_provenance(context, relative_path=result.resolved_path)
        return ReadFileOutput.model_validate(result.model_dump(mode="json"))


class WriteFileInput(BaseModel):
    path: str
    content: str
    encoding: str = "utf-8"
    overwrite: bool = False
    create_parents: bool = False


class WriteFileOutput(BaseModel):
    path: str
    resolved_path: str
    bytes_written: int
    created: bool


class WriteFileTool(Tool[WriteFileInput, WriteFileOutput]):
    spec = ToolSpec(
        name="write_file",
        description="Write text content to a file inside the workspace.",
        tags=["filesystem", "write"],
        side_effects=SideEffectClass.LOCAL_WRITE,
        idempotent=False,
        requires_filesystem=True,
    )
    input_model = WriteFileInput
    output_model = WriteFileOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: WriteFileInput,
    ) -> WriteFileOutput:
        root_path = get_workspace_root(context)
        resolved_request = resolve_writable_file_path(root_path, args.path)
        target = resolved_request.resolved
        created = not target.exists()
        if target.exists() and not args.overwrite:
            raise FileExistsError(
                f"Path '{resolved_request.resolved_path}' already exists."
            )

        if not target.parent.exists():
            if not args.create_parents:
                parent_path = relative_display_path(root_path, target.parent)
                raise FileNotFoundError(
                    f"Parent directory '{parent_path}' does not exist."
                )
            target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(args.content, encoding=args.encoding)
        bytes_written = len(args.content.encode(args.encoding))
        context.log(f"Wrote file '{resolved_request.resolved_path}'.")
        context.add_artifact(resolved_request.resolved_path)
        return WriteFileOutput(
            path=args.path,
            resolved_path=resolved_request.resolved_path,
            bytes_written=bytes_written,
            created=created,
        )


class ListDirectoryInput(BaseModel):
    path: str = "."
    recursive: bool = False
    max_depth: int | None = Field(default=None, gt=0)


class ListDirectoryOutput(DirectoryListingResult):
    pass


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
        root_path, source_filters, tool_limits = _require_repository_metadata(context)
        result = list_directory_impl(
            root_path,
            args.path,
            source_filters=source_filters,
            tool_limits=tool_limits,
            recursive=args.recursive,
            max_depth=args.max_depth,
        )
        context.log(f"Listed directory '{args.path}'.")
        return ListDirectoryOutput.model_validate(result.model_dump(mode="json"))


class FindFilesInput(BaseModel):
    path: str = "."
    pattern: str


class FindFilesOutput(FileSearchResult):
    pass


class FindFilesTool(Tool[FindFilesInput, FindFilesOutput]):
    spec = ToolSpec(
        name="find_files",
        description="Find files by root-relative glob pattern.",
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
        root_path, source_filters, tool_limits = _require_repository_metadata(context)
        result = find_files_impl(
            root_path,
            args.pattern,
            args.path,
            source_filters=source_filters,
            tool_limits=tool_limits,
        )
        context.log(f"Searched for file pattern '{args.pattern}'.")
        return FindFilesOutput.model_validate(result.model_dump(mode="json"))


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
        root_path, _, tool_limits = _require_repository_metadata(context)
        path_argument: str | list[str] = (
            args.path if args.path is not None else args.paths or []
        )
        result = get_file_info_impl(
            root_path,
            path_argument,
            tool_limits=tool_limits,
            cache_root=_get_read_file_cache_root(root_path),
        )
        context.log("Collected file metadata.")
        if isinstance(result, FileInfoResult):
            return GetFileInfoOutput(results=[result])
        return GetFileInfoOutput(results=result.results)


def register_filesystem_tools(registry: ToolRegistry) -> None:
    """Register the built-in filesystem tool set."""
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirectoryTool())
    registry.register(FindFilesTool())
    registry.register(GetFileInfoTool())


__all__ = [
    "FindFilesTool",
    "GetFileInfoTool",
    "ListDirectoryTool",
    "ReadFileTool",
    "WriteFileTool",
    "register_filesystem_tools",
    "_get_read_file_cache_root",
    "_get_cached_conversion_paths",
    "_read_cached_conversion",
    "_write_cached_conversion",
]

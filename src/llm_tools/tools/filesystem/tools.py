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
from llm_tools.tools.filesystem._content import (
    _get_cached_conversion_paths,
    _get_read_file_cache_root,
    _read_cached_conversion,
    _write_cached_conversion,
)
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
    context: ToolExecutionContext, *, resolved_path: Path
) -> None:
    source_id = str(resolved_path.resolve())
    context.add_source_provenance(
        SourceProvenanceRef(
            source_kind="local_file",
            source_id=source_id,
            content_hash=hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
            whole_source_reproduction_allowed=True,
            metadata={"path": source_id},
        )
    )


def _require_repository_metadata(
    context: ToolExecutionContext,
) -> tuple[SourceFilters, ToolLimits]:
    metadata = context.metadata
    source_filters = SourceFilters.model_validate(metadata.get("source_filters", {}))
    tool_limits = ToolLimits.model_validate(metadata.get("tool_limits", {}))
    return source_filters, tool_limits


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
    )
    input_model = ReadFileInput
    output_model = ReadFileOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: ReadFileInput,
    ) -> ReadFileOutput:
        _, tool_limits = _require_repository_metadata(context)
        filesystem = context.services.require_filesystem()
        result = filesystem.read_file(
            args.path,
            tool_limits=tool_limits,
            start_char=args.start_char,
            end_char=args.end_char,
        )
        resolved_path = filesystem.resolve_path(
            args.path,
            expect_directory=False,
            must_exist=True,
        )
        context.log(f"Read file '{args.path}'.")
        context.add_artifact(str(resolved_path))
        _append_local_source_provenance(context, resolved_path=resolved_path)
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
        filesystem = context.services.require_filesystem()
        resolved, created, bytes_written = filesystem.write_text(
            args.path,
            content=args.content,
            encoding=args.encoding,
            overwrite=args.overwrite,
            create_parents=args.create_parents,
        )
        context.log(f"Wrote file '{resolved}'.")
        context.add_artifact(str(resolved))
        return WriteFileOutput(
            path=args.path,
            resolved_path=str(resolved),
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
        source_filters, tool_limits = _require_repository_metadata(context)
        result = context.services.require_filesystem().list_directory(
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
        source_filters, tool_limits = _require_repository_metadata(context)
        result = context.services.require_filesystem().find_files(
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
    )
    input_model = GetFileInfoInput
    output_model = GetFileInfoOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: GetFileInfoInput,
    ) -> GetFileInfoOutput:
        _, tool_limits = _require_repository_metadata(context)
        path_argument: str | list[str] = (
            args.path if args.path is not None else args.paths or []
        )
        result = context.services.require_filesystem().get_file_info(
            path_argument,
            tool_limits=tool_limits,
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

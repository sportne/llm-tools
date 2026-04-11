"""Filesystem built-in tool implementations."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from llm_tools.tool_api import (
    SideEffectClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolSpec,
)
from llm_tools.tools._path_utils import (
    get_workspace_root,
    is_hidden_path,
    relative_display_path,
    resolve_workspace_path,
)

_DEFAULT_MAX_RETURN_LINES = 200


def _get_read_file_cache_root() -> Path:
    """Return the cache root for converted file content."""
    return Path(tempfile.gettempdir()) / "llm_tools" / "read_file_cache"


def _get_cached_conversion_paths(resolved: Path) -> tuple[Path, Path]:
    """Return the markdown and metadata cache paths for a source file."""
    cache_key = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
    cache_dir = _get_read_file_cache_root() / cache_key
    return cache_dir / "content.md", cache_dir / "metadata.json"


def _read_cached_conversion(resolved: Path) -> str | None:
    """Return cached converted markdown when the source has not changed."""
    content_path, metadata_path = _get_cached_conversion_paths(resolved)
    if not content_path.exists() or not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stat = resolved.stat()
    if (
        metadata.get("mtime_ns") != stat.st_mtime_ns
        or metadata.get("size_bytes") != stat.st_size
    ):
        return None

    return content_path.read_text(encoding="utf-8")


def _write_cached_conversion(resolved: Path, markdown: str) -> Path:
    """Persist converted markdown and source metadata for future reads."""
    content_path, metadata_path = _get_cached_conversion_paths(resolved)
    content_path.parent.mkdir(parents=True, exist_ok=True)
    content_path.write_text(markdown, encoding="utf-8")
    stat = resolved.stat()
    metadata_path.write_text(
        json.dumps({"mtime_ns": stat.st_mtime_ns, "size_bytes": stat.st_size}),
        encoding="utf-8",
    )
    return content_path


def _read_text_file(path: Path, encoding: str) -> str | None:
    """Read a file as text when possible, otherwise return None."""
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return None


def _slice_lines(
    content: str,
    *,
    line_start: int,
    line_end: int | None,
) -> tuple[str, int, int, bool]:
    """Return a bounded inclusive line slice from the full content."""
    lines = content.splitlines()
    total_lines = len(lines)
    start_index = min(max(line_start - 1, 0), total_lines)
    max_end_line = line_start + _DEFAULT_MAX_RETURN_LINES - 1
    requested_end = line_end if line_end is not None else max_end_line
    actual_end_line = min(requested_end, max_end_line, total_lines)
    if actual_end_line < line_start:
        return "", total_lines, line_start - 1, False

    sliced = lines[start_index:actual_end_line]
    truncated = actual_end_line < total_lines and (
        line_end is None or line_end > actual_end_line
    )
    return "\n".join(sliced), total_lines, actual_end_line, truncated


class ReadFileInput(BaseModel):
    path: str
    encoding: str = "utf-8"
    line_start: int = Field(default=1, ge=1)
    line_end: int | None = Field(default=None, ge=1)


class ReadFileOutput(BaseModel):
    path: str
    resolved_path: str
    content: str
    content_format: Literal["text", "markdown"]
    line_start: int
    line_end: int
    total_lines: int
    truncated: bool
    cached_markdown_path: str | None = None
    used_cached_conversion: bool = False


class ReadFileTool(Tool[ReadFileInput, ReadFileOutput]):
    spec = ToolSpec(
        name="read_file",
        description=(
            "Read a file, automatically converting non-text files to markdown and "
            "returning a bounded line range."
        ),
        tags=["filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = ReadFileInput
    output_model = ReadFileOutput

    def invoke(self, context: ToolContext, args: ReadFileInput) -> ReadFileOutput:
        if args.line_end is not None and args.line_end < args.line_start:
            raise ValueError(
                "'line_end' must be greater than or equal to 'line_start'."
            )

        resolved = resolve_workspace_path(
            context,
            args.path,
            expect_directory=False,
            must_exist=True,
        )
        content = _read_text_file(resolved, args.encoding)
        content_format: Literal["text", "markdown"] = "text"
        cached_markdown_path: str | None = None
        used_cached_conversion = False

        if content is None:
            content_format = "markdown"
            cached = _read_cached_conversion(resolved)
            if cached is None:
                from markitdown import MarkItDown  # type: ignore[import-not-found]

                content = MarkItDown().convert(str(resolved)).text_content
                cached_path = _write_cached_conversion(resolved, content)
            else:
                content = cached
                cached_path, _ = _get_cached_conversion_paths(resolved)
                used_cached_conversion = True
            cached_markdown_path = str(cached_path)

        content_slice, total_lines, returned_line_end, truncated = _slice_lines(
            content,
            line_start=args.line_start,
            line_end=args.line_end,
        )

        context.logs.append(f"Read file '{resolved}'.")
        context.artifacts.append(str(resolved))
        if cached_markdown_path is not None:
            context.artifacts.append(cached_markdown_path)
        return ReadFileOutput(
            path=args.path,
            resolved_path=str(resolved),
            content=content_slice,
            content_format=content_format,
            line_start=args.line_start,
            line_end=returned_line_end,
            total_lines=total_lines,
            truncated=truncated,
            cached_markdown_path=cached_markdown_path,
            used_cached_conversion=used_cached_conversion,
        )


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

    def invoke(self, context: ToolContext, args: WriteFileInput) -> WriteFileOutput:
        resolved = resolve_workspace_path(
            context,
            args.path,
            expect_directory=False,
            must_exist=False,
        )
        created = not resolved.exists()
        if resolved.exists() and not args.overwrite:
            raise FileExistsError(f"Path '{resolved}' already exists.")

        if not resolved.parent.exists():
            if not args.create_parents:
                raise FileNotFoundError(
                    f"Parent directory '{resolved.parent}' does not exist."
                )
            resolved.parent.mkdir(parents=True, exist_ok=True)

        resolved.write_text(args.content, encoding=args.encoding)
        bytes_written = len(args.content.encode(args.encoding))
        context.logs.append(f"Wrote file '{resolved}'.")
        context.artifacts.append(str(resolved))
        return WriteFileOutput(
            path=args.path,
            resolved_path=str(resolved),
            bytes_written=bytes_written,
            created=created,
        )


class DirectoryEntry(BaseModel):
    name: str
    path: str
    resolved_path: str
    is_dir: bool
    size_bytes: int | None = None


class ListDirectoryInput(BaseModel):
    path: str = "."
    recursive: bool = False
    include_hidden: bool = False


class ListDirectoryOutput(BaseModel):
    root: str
    resolved_root: str
    entries: list[DirectoryEntry] = Field(default_factory=list)


class ListDirectoryTool(Tool[ListDirectoryInput, ListDirectoryOutput]):
    spec = ToolSpec(
        name="list_directory",
        description="List files and directories inside the workspace.",
        tags=["filesystem", "list"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = ListDirectoryInput
    output_model = ListDirectoryOutput

    def invoke(
        self, context: ToolContext, args: ListDirectoryInput
    ) -> ListDirectoryOutput:
        root = resolve_workspace_path(
            context,
            args.path,
            expect_directory=True,
            must_exist=True,
        )
        workspace_root = get_workspace_root(context)
        iterator = root.rglob("*") if args.recursive else root.iterdir()
        entries: list[DirectoryEntry] = []

        for item in iterator:
            if not args.include_hidden and is_hidden_path(workspace_root, item):
                continue
            stat = item.stat()
            entries.append(
                DirectoryEntry(
                    name=item.name,
                    path=relative_display_path(workspace_root, item),
                    resolved_path=str(item),
                    is_dir=item.is_dir(),
                    size_bytes=None if item.is_dir() else stat.st_size,
                )
            )

        entries.sort(key=lambda entry: entry.path)
        context.logs.append(f"Listed directory '{root}'.")
        return ListDirectoryOutput(
            root=args.path,
            resolved_root=str(root),
            entries=entries,
        )


def register_filesystem_tools(registry: ToolRegistry) -> None:
    """Register the built-in filesystem tool set."""
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirectoryTool())

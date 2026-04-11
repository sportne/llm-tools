"""Filesystem built-in tool implementations."""

from __future__ import annotations

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


class ReadFileInput(BaseModel):
    path: str
    mode: Literal["text", "convert"] = "text"
    encoding: str = "utf-8"


class ReadFileOutput(BaseModel):
    path: str
    resolved_path: str
    content: str
    mode: Literal["text", "convert"]


class ReadFileTool(Tool[ReadFileInput, ReadFileOutput]):
    spec = ToolSpec(
        name="read_file",
        description="Read a text file or convert a supported document to text.",
        tags=["filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = ReadFileInput
    output_model = ReadFileOutput

    def invoke(self, context: ToolContext, args: ReadFileInput) -> ReadFileOutput:
        resolved = resolve_workspace_path(
            context,
            args.path,
            expect_directory=False,
            must_exist=True,
        )
        if args.mode == "convert":
            from markitdown import MarkItDown  # type: ignore[import-not-found]

            content = MarkItDown().convert(str(resolved)).text_content
        else:
            content = resolved.read_text(encoding=args.encoding)

        context.logs.append(f"Read file '{resolved}'.")
        context.artifacts.append(str(resolved))
        return ReadFileOutput(
            path=args.path,
            resolved_path=str(resolved),
            content=content,
            mode=args.mode,
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

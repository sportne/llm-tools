"""Text search built-in tool implementations."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

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


def _compile_pattern(
    query: str, *, case_sensitive: bool, regex: bool
) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(query if regex else re.escape(query), flags)


def _read_searchable_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class TextMatch(BaseModel):
    line_number: int
    line_text: str
    matched_texts: list[str] = Field(default_factory=list)


class FileTextSearchInput(BaseModel):
    path: str
    query: str
    case_sensitive: bool = False
    regex: bool = False


class FileTextSearchOutput(BaseModel):
    path: str
    matches: list[TextMatch] = Field(default_factory=list)


class FileTextSearchTool(Tool[FileTextSearchInput, FileTextSearchOutput]):
    spec = ToolSpec(
        name="file_text_search",
        description="Search one file for matching text.",
        tags=["text", "search", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = FileTextSearchInput
    output_model = FileTextSearchOutput

    def invoke(
        self, context: ToolContext, args: FileTextSearchInput
    ) -> FileTextSearchOutput:
        resolved = resolve_workspace_path(
            context,
            args.path,
            expect_directory=False,
            must_exist=True,
        )
        pattern = _compile_pattern(
            args.query,
            case_sensitive=args.case_sensitive,
            regex=args.regex,
        )
        matches = _search_file(resolved, pattern)
        context.logs.append(f"Searched file '{resolved}' for text.")
        return FileTextSearchOutput(path=args.path, matches=matches)


class DirectoryTextSearchInput(BaseModel):
    path: str = "."
    query: str
    recursive: bool = True
    include_hidden: bool = False
    case_sensitive: bool = False
    regex: bool = False
    file_glob: str | None = None
    max_matches_per_file: int | None = Field(default=None, ge=1)


class DirectoryTextSearchResult(BaseModel):
    path: str
    resolved_path: str
    matches: list[TextMatch] = Field(default_factory=list)


class DirectoryTextSearchOutput(BaseModel):
    root: str
    results: list[DirectoryTextSearchResult] = Field(default_factory=list)


class DirectoryTextSearchTool(
    Tool[DirectoryTextSearchInput, DirectoryTextSearchOutput]
):
    spec = ToolSpec(
        name="directory_text_search",
        description="Search files under a directory for matching text.",
        tags=["text", "search", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = DirectoryTextSearchInput
    output_model = DirectoryTextSearchOutput

    def invoke(
        self, context: ToolContext, args: DirectoryTextSearchInput
    ) -> DirectoryTextSearchOutput:
        root = resolve_workspace_path(
            context,
            args.path,
            expect_directory=True,
            must_exist=True,
        )
        workspace_root = get_workspace_root(context)
        pattern = _compile_pattern(
            args.query,
            case_sensitive=args.case_sensitive,
            regex=args.regex,
        )
        iterator = root.rglob("*") if args.recursive else root.iterdir()
        results: list[DirectoryTextSearchResult] = []

        for item in iterator:
            if item.is_dir():
                continue
            if not args.include_hidden and is_hidden_path(workspace_root, item):
                continue
            relative_path = relative_display_path(workspace_root, item)
            if args.file_glob is not None and not fnmatch.fnmatch(
                relative_path, args.file_glob
            ):
                continue
            matches = _search_file(item, pattern)
            if args.max_matches_per_file is not None:
                matches = matches[: args.max_matches_per_file]
            if not matches:
                continue
            results.append(
                DirectoryTextSearchResult(
                    path=relative_path,
                    resolved_path=str(item),
                    matches=matches,
                )
            )

        results.sort(key=lambda result: result.path)
        context.logs.append(f"Searched directory '{root}' for text.")
        return DirectoryTextSearchOutput(root=args.path, results=results)


def _search_file(path: Path, pattern: re.Pattern[str]) -> list[TextMatch]:
    matches: list[TextMatch] = []
    for line_number, line_text in enumerate(
        _read_searchable_text(path).splitlines(), start=1
    ):
        found = [match.group(0) for match in pattern.finditer(line_text)]
        if found:
            matches.append(
                TextMatch(
                    line_number=line_number,
                    line_text=line_text,
                    matched_texts=found,
                )
            )
    return matches


def register_text_tools(registry: ToolRegistry) -> None:
    """Register the built-in text search tool set."""
    registry.register(FileTextSearchTool())
    registry.register(DirectoryTextSearchTool())

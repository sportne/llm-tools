"""Filesystem built-in tool registration and compatibility exports."""

from __future__ import annotations

from llm_tools.tool_api import ToolRegistry
from llm_tools.tools.filesystem._content import (
    _get_cached_conversion_paths,
    _get_read_file_cache_root,
    _read_cached_conversion,
    _write_cached_conversion,
)
from llm_tools.tools.filesystem.find_files import FindFilesTool
from llm_tools.tools.filesystem.get_file_info import GetFileInfoTool
from llm_tools.tools.filesystem.list_directory import ListDirectoryTool
from llm_tools.tools.filesystem.read_file import ReadFileTool
from llm_tools.tools.filesystem.search_text import SearchTextTool
from llm_tools.tools.filesystem.write_file import WriteFileTool


def register_filesystem_tools(registry: ToolRegistry) -> None:
    """Register the built-in filesystem tool set."""
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirectoryTool())
    registry.register(FindFilesTool())
    registry.register(GetFileInfoTool())
    registry.register(SearchTextTool())


__all__ = [
    "FindFilesTool",
    "GetFileInfoTool",
    "ListDirectoryTool",
    "ReadFileTool",
    "SearchTextTool",
    "WriteFileTool",
    "register_filesystem_tools",
    "_get_read_file_cache_root",
    "_get_cached_conversion_paths",
    "_read_cached_conversion",
    "_write_cached_conversion",
]

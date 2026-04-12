"""Operational implementations for repository-style text search tools."""

from __future__ import annotations

from pathlib import Path

from llm_tools.tools.filesystem._content import (
    LoadedReadableContent,
    is_within_character_limit,
    load_readable_content,
)
from llm_tools.tools.filesystem._ops import resolve_search_file_or_directory
from llm_tools.tools.filesystem._paths import (
    ResolvedRootPath,
    resolve_directory_path,
    should_include_entry,
    should_prune_directory,
)
from llm_tools.tools.filesystem.models import SourceFilters, ToolLimits
from llm_tools.tools.text.models import TextSearchMatch, TextSearchResult


def build_text_search_match(
    root_path: Path,
    path: Path,
    *,
    line_number: int,
    line_text: str,
) -> TextSearchMatch:
    """Return one text-search match payload."""
    relative_path = path.relative_to(root_path)
    return TextSearchMatch(
        path=relative_path.as_posix(),
        line_number=line_number,
        line_text=line_text,
        is_hidden=any(part.startswith(".") for part in relative_path.parts),
    )


def search_text_impl(
    root_path: Path,
    query: str,
    path: str,
    *,
    source_filters: SourceFilters,
    tool_limits: ToolLimits,
) -> TextSearchResult:
    """Return matching text lines within one root-confined directory or file."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("search_text query must not be empty.")
    normalized_path, resolved_root, candidate_target, resolved_relative = (
        resolve_search_file_or_directory(root_path, path)
    )
    if candidate_target.resolve().is_file():
        return _search_single_file(
            normalized_path=normalized_path,
            resolved_root=resolved_root,
            candidate_target=candidate_target,
            resolved_relative=resolved_relative,
            query=normalized_query,
            tool_limits=tool_limits,
        )
    return _search_directory(
        root_path=root_path,
        path=path,
        query=normalized_query,
        source_filters=source_filters,
        tool_limits=tool_limits,
    )


def _search_single_file(
    *,
    normalized_path: str,
    resolved_root: Path,
    candidate_target: Path,
    resolved_relative: Path,
    query: str,
    tool_limits: ToolLimits,
) -> TextSearchResult:
    resolved_path = (
        resolved_relative.as_posix() if resolved_relative.as_posix() else "."
    )
    resolved_request = ResolvedRootPath(
        root=resolved_root,
        candidate=candidate_target,
        resolved=candidate_target.resolve(),
        requested_path=normalized_path,
        resolved_path=resolved_path,
    )
    loaded_content = load_readable_content(resolved_request.resolved)
    return _search_loaded_content(
        resolved_request=resolved_request,
        query=query,
        tool_limits=tool_limits,
        loaded_content=loaded_content,
    )


def _search_directory(
    *,
    root_path: Path,
    path: str,
    query: str,
    source_filters: SourceFilters,
    tool_limits: ToolLimits,
) -> TextSearchResult:
    resolved_request = resolve_directory_path(root_path, path)
    matches: list[TextSearchMatch] = []
    truncated = False

    def walk_directory(directory: Path) -> bool:
        nonlocal truncated
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative_child = child.relative_to(resolved_request.root)
            if child.is_symlink():
                continue
            if child.is_dir():
                if should_prune_directory(
                    relative_child, source_filters=source_filters
                ):
                    continue
                if not walk_directory(child):
                    return False
                continue
            if not child.is_file():
                continue
            if not should_include_entry(relative_child, source_filters=source_filters):
                continue
            loaded_content = load_readable_content(child)
            if loaded_content.status != "ok" or loaded_content.content is None:
                continue
            if not is_within_character_limit(
                loaded_content.content, tool_limits=tool_limits
            ):
                continue
            for line_number, line_text in enumerate(
                loaded_content.content.splitlines(),
                start=1,
            ):
                if query not in line_text:
                    continue
                if len(matches) >= tool_limits.max_search_matches:
                    truncated = True
                    return False
                matches.append(
                    build_text_search_match(
                        resolved_request.root,
                        child,
                        line_number=line_number,
                        line_text=line_text,
                    )
                )
        return True

    walk_directory(resolved_request.resolved)
    return TextSearchResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        query=query,
        matches=matches,
        truncated=truncated,
    )


def _search_loaded_content(
    *,
    resolved_request: ResolvedRootPath,
    query: str,
    tool_limits: ToolLimits,
    loaded_content: LoadedReadableContent,
) -> TextSearchResult:
    matches: list[TextSearchMatch] = []
    truncated = False
    if loaded_content.status != "ok" or loaded_content.content is None:
        return TextSearchResult(
            requested_path=resolved_request.requested_path,
            resolved_path=resolved_request.resolved_path,
            query=query,
            matches=[],
            truncated=False,
        )
    if not is_within_character_limit(loaded_content.content, tool_limits=tool_limits):
        return TextSearchResult(
            requested_path=resolved_request.requested_path,
            resolved_path=resolved_request.resolved_path,
            query=query,
            matches=[],
            truncated=False,
        )
    for line_number, line_text in enumerate(
        loaded_content.content.splitlines(), start=1
    ):
        if query not in line_text:
            continue
        if len(matches) >= tool_limits.max_search_matches:
            truncated = True
            break
        matches.append(
            build_text_search_match(
                resolved_request.root,
                resolved_request.resolved,
                line_number=line_number,
                line_text=line_text,
            )
        )
    return TextSearchResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        query=query,
        matches=matches,
        truncated=truncated,
    )

"""Operational implementations for repository-style filesystem tools."""

from __future__ import annotations

from pathlib import Path

from llm_tools.tools.filesystem._content import (
    build_file_info_result,
    effective_full_read_char_limit,
    estimate_token_count,
    is_within_character_limit,
    line_range_for_character_range,
    load_readable_content,
    normalize_range,
)
from llm_tools.tools.filesystem._paths import (
    GitignoreMatcher,
    build_entry,
    build_file_match,
    matches_path_glob,
    normalize_requested_path,
    normalize_required_pattern,
    resolve_directory_path,
    resolve_file_path,
    should_include_entry,
    should_prune_directory,
)
from llm_tools.tools.filesystem.models import (
    DirectoryEntry,
    DirectoryListingResult,
    FileInfoBatchResult,
    FileInfoResult,
    FileMatch,
    FileReadResult,
    FileSearchResult,
    SourceFilters,
    ToolLimits,
)


def list_directory_impl(
    root_path: Path,
    path: str,
    *,
    source_filters: SourceFilters,
    tool_limits: ToolLimits,
    recursive: bool,
    max_depth: int | None,
) -> DirectoryListingResult:
    """Return deterministic matching children of one directory under the root."""
    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be greater than 0 when provided")
    resolved_request = resolve_directory_path(root_path, path)
    gitignore_matcher = GitignoreMatcher.from_root(resolved_request.root)
    applied_max_depth = (
        1
        if not recursive
        else min(
            max_depth if max_depth is not None else tool_limits.max_recursive_depth,
            tool_limits.max_recursive_depth,
        )
    )
    entries: list[DirectoryEntry] = []
    truncated = False

    def walk_directory(directory: Path, *, depth: int) -> bool:
        nonlocal truncated
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative_child = child.relative_to(resolved_request.root)
            child_is_dir = child.is_dir() and not child.is_symlink()
            if should_include_entry(
                relative_child,
                source_filters=source_filters,
                gitignore_matcher=gitignore_matcher,
                is_dir=child_is_dir,
            ):
                if len(entries) >= tool_limits.max_entries_per_call:
                    truncated = True
                    return False
                entries.append(
                    build_entry(
                        resolved_request.root,
                        child,
                        depth=depth,
                        gitignore_matcher=gitignore_matcher,
                    )
                )
            if depth >= applied_max_depth:
                continue
            if child.is_symlink() or not child.is_dir():
                continue
            if should_prune_directory(
                relative_child,
                source_filters=source_filters,
                gitignore_matcher=gitignore_matcher,
            ):
                continue
            if not walk_directory(child, depth=depth + 1):
                return False
        return True

    walk_directory(resolved_request.resolved, depth=1)
    return DirectoryListingResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        recursive=recursive,
        max_depth_applied=applied_max_depth,
        entries=entries,
        truncated=truncated,
    )


def find_files_impl(
    root_path: Path,
    pattern: str,
    path: str,
    *,
    source_filters: SourceFilters,
    tool_limits: ToolLimits,
) -> FileSearchResult:
    """Return matching files beneath one root-confined directory subtree."""
    normalized_pattern = normalize_required_pattern(pattern)
    resolved_request = resolve_directory_path(root_path, path)
    gitignore_matcher = GitignoreMatcher.from_root(resolved_request.root)
    matches: list[FileMatch] = []
    truncated = False
    files_scanned = 0

    def walk_directory(directory: Path, *, depth: int) -> bool:
        nonlocal truncated, files_scanned
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative_child = child.relative_to(resolved_request.root)
            if child.is_symlink():
                continue
            if child.is_dir():
                if depth >= tool_limits.max_recursive_depth:
                    truncated = True
                    continue
                if should_prune_directory(
                    relative_child,
                    source_filters=source_filters,
                    gitignore_matcher=gitignore_matcher,
                ):
                    continue
                if not walk_directory(child, depth=depth + 1):
                    return False
                continue
            if not child.is_file():
                continue
            files_scanned += 1
            if files_scanned > tool_limits.max_files_scanned:
                truncated = True
                return False
            if not should_include_entry(
                relative_child,
                source_filters=source_filters,
                gitignore_matcher=gitignore_matcher,
            ):
                continue
            if not matches_path_glob(relative_child, normalized_pattern):
                continue
            if len(matches) >= tool_limits.max_entries_per_call:
                truncated = True
                return False
            matches.append(
                build_file_match(
                    resolved_request.root,
                    child,
                    gitignore_matcher=gitignore_matcher,
                )
            )
        return True

    walk_directory(resolved_request.resolved, depth=1)
    return FileSearchResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        pattern=normalized_pattern,
        matches=matches,
        truncated=truncated,
    )


def resolve_search_file_or_directory(
    root_path: Path,
    path: str,
) -> tuple[str, Path, Path, Path]:
    """Resolve a search target that may be a file or a directory."""
    normalized_path = normalize_requested_path(path)
    requested_path = Path(normalized_path)
    if requested_path.is_absolute():
        raise ValueError("Tool paths must be relative to the configured root.")
    resolved_root = root_path.resolve()
    candidate_target = (resolved_root / requested_path).resolve(strict=False)
    if not candidate_target.is_relative_to(resolved_root):
        raise ValueError("Requested tool path escapes the configured root.")
    if not candidate_target.exists():
        raise ValueError(
            f"Requested file or directory does not exist: {normalized_path}"
        )
    if candidate_target.is_dir():
        resolved_request = resolve_directory_path(root_path, path)
    else:
        resolved_request = resolve_file_path(root_path, path)
    resolved_relative = resolved_request.resolved.relative_to(resolved_request.root)
    return (
        normalized_path,
        resolved_request.root,
        resolved_request.resolved,
        resolved_relative,
    )


def get_file_info_impl(
    root_path: Path,
    path: str | list[str],
    *,
    tool_limits: ToolLimits,
    cache_root: Path | None = None,
) -> FileInfoResult | FileInfoBatchResult:
    """Return deterministic metadata for one or more root-confined files."""
    if isinstance(path, str):
        return _get_single_file_info(
            root_path,
            path,
            tool_limits=tool_limits,
            cache_root=cache_root,
        )
    return FileInfoBatchResult(
        results=[
            _get_single_file_info(
                root_path,
                item,
                tool_limits=tool_limits,
                cache_root=cache_root,
            )
            for item in path
        ]
    )


def _get_single_file_info(
    root_path: Path,
    path: str,
    *,
    tool_limits: ToolLimits,
    cache_root: Path | None,
) -> FileInfoResult:
    resolved_request = resolve_file_path(root_path, path)
    loaded_content = load_readable_content(
        resolved_request.resolved,
        tool_limits=tool_limits,
        cache_root=cache_root,
    )
    return build_file_info_result(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        candidate_file=resolved_request.candidate,
        resolved_file=resolved_request.resolved,
        relative_candidate_path=resolved_request.candidate.relative_to(
            resolved_request.root
        ),
        tool_limits=tool_limits,
        loaded_content=loaded_content,
    )


def read_file_impl(
    root_path: Path,
    path: str,
    *,
    tool_limits: ToolLimits,
    start_char: int | None,
    end_char: int | None,
    cache_root: Path | None = None,
) -> FileReadResult:
    """Return bounded text or markdown content for one root-confined file."""
    resolved_request = resolve_file_path(root_path, path)
    loaded_content = load_readable_content(
        resolved_request.resolved,
        tool_limits=tool_limits,
        cache_root=cache_root,
    )
    full_read_char_limit = effective_full_read_char_limit(tool_limits)
    if loaded_content.status != "ok" or loaded_content.content is None:
        return FileReadResult(
            requested_path=resolved_request.requested_path,
            resolved_path=resolved_request.resolved_path,
            read_kind=loaded_content.read_kind,
            status=loaded_content.status,
            content=None,
            file_size_bytes=resolved_request.resolved.stat().st_size,
            max_read_input_bytes=tool_limits.max_read_input_bytes,
            max_file_size_characters=tool_limits.max_file_size_characters,
            full_read_char_limit=full_read_char_limit,
            error_message=loaded_content.error_message,
        )
    content = loaded_content.content
    character_count = len(content)
    if not is_within_character_limit(content, tool_limits=tool_limits):
        return FileReadResult(
            requested_path=resolved_request.requested_path,
            resolved_path=resolved_request.resolved_path,
            read_kind=loaded_content.read_kind,
            status="too_large",
            content=None,
            character_count=character_count,
            file_size_bytes=resolved_request.resolved.stat().st_size,
            max_read_input_bytes=tool_limits.max_read_input_bytes,
            max_file_size_characters=tool_limits.max_file_size_characters,
            full_read_char_limit=full_read_char_limit,
            estimated_token_count=estimate_token_count(content),
            error_message="File exceeds the configured readable character limit",
        )
    normalized_start, normalized_end = normalize_range(
        start_char=start_char,
        end_char=end_char,
        character_count=character_count,
    )
    truncated_end = min(normalized_end, normalized_start + full_read_char_limit)
    truncated = truncated_end < character_count or truncated_end < normalized_end
    content_slice = content[normalized_start:truncated_end]
    line_start, line_end = line_range_for_character_range(
        content,
        start_char=normalized_start,
        end_char=truncated_end,
    )
    return FileReadResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        read_kind=loaded_content.read_kind,
        status="ok",
        content=content_slice,
        truncated=truncated,
        content_char_count=len(content_slice),
        character_count=character_count,
        start_char=normalized_start,
        end_char=truncated_end,
        line_start=line_start,
        line_end=line_end,
        file_size_bytes=resolved_request.resolved.stat().st_size,
        max_read_input_bytes=tool_limits.max_read_input_bytes,
        max_file_size_characters=tool_limits.max_file_size_characters,
        full_read_char_limit=full_read_char_limit,
        estimated_token_count=estimate_token_count(content_slice),
    )

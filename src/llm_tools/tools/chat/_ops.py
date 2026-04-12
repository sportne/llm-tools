"""Operational implementations for deterministic repository chat tools."""

from __future__ import annotations

from pathlib import Path

from llm_tools.tools.chat._content import (
    LoadedReadableContent,
    build_file_info_result,
    effective_full_read_char_limit,
    estimate_token_count,
    is_within_character_limit,
    load_readable_content,
    normalize_range,
)
from llm_tools.tools.chat._paths import (
    ResolvedRootPath,
    build_entry,
    build_file_match,
    build_text_search_match,
    matches_path_glob,
    normalize_requested_path,
    normalize_required_pattern,
    normalize_required_value,
    resolve_directory_path,
    resolve_file_path,
    should_include_entry,
    should_prune_directory,
)
from llm_tools.tools.chat.models import (
    ChatSessionConfig,
    ChatSourceFilters,
    ChatToolLimits,
    DirectoryEntry,
    DirectoryListingResult,
    FileInfoBatchResult,
    FileInfoResult,
    FileMatch,
    FileReadResult,
    FileSearchResult,
    TextSearchMatch,
    TextSearchResult,
)


def list_directory_impl(
    root_path: Path,
    path: str,
    *,
    source_filters: ChatSourceFilters,
    tool_limits: ChatToolLimits,
) -> DirectoryListingResult:
    """Return the immediate matching children of one directory under the root."""
    resolved_request = resolve_directory_path(root_path, path)
    entries: list[DirectoryEntry] = []
    truncated = False
    for child in sorted(
        resolved_request.resolved.iterdir(), key=lambda item: item.name
    ):
        relative_child = child.relative_to(resolved_request.root)
        if not should_include_entry(relative_child, source_filters=source_filters):
            continue
        if len(entries) >= tool_limits.max_entries_per_call:
            truncated = True
            break
        entries.append(build_entry(resolved_request.root, child, depth=1))
    return DirectoryListingResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        recursive=False,
        max_depth_applied=1,
        entries=entries,
        truncated=truncated,
    )


def list_directory_recursive_impl(
    root_path: Path,
    path: str,
    *,
    source_filters: ChatSourceFilters,
    tool_limits: ChatToolLimits,
    max_depth: int | None,
) -> DirectoryListingResult:
    """Return a flat deterministic recursive listing under one directory."""
    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be greater than 0 when provided")
    resolved_request = resolve_directory_path(root_path, path)
    applied_max_depth = min(
        max_depth if max_depth is not None else tool_limits.max_recursive_depth,
        tool_limits.max_recursive_depth,
    )
    entries: list[DirectoryEntry] = []
    truncated = False

    def walk_directory(directory: Path, *, depth: int) -> bool:
        nonlocal truncated
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative_child = child.relative_to(resolved_request.root)
            if should_include_entry(relative_child, source_filters=source_filters):
                if len(entries) >= tool_limits.max_entries_per_call:
                    truncated = True
                    return False
                entries.append(build_entry(resolved_request.root, child, depth=depth))
            if depth >= applied_max_depth:
                continue
            if child.is_symlink() or not child.is_dir():
                continue
            if should_prune_directory(relative_child, source_filters=source_filters):
                continue
            if not walk_directory(child, depth=depth + 1):
                return False
        return True

    walk_directory(resolved_request.resolved, depth=1)
    return DirectoryListingResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        recursive=True,
        max_depth_applied=applied_max_depth,
        entries=entries,
        truncated=truncated,
    )


def find_files_impl(
    root_path: Path,
    pattern: str,
    path: str,
    *,
    source_filters: ChatSourceFilters,
    tool_limits: ChatToolLimits,
) -> FileSearchResult:
    """Return matching files beneath one root-confined directory subtree."""
    normalized_pattern = normalize_required_pattern(pattern)
    resolved_request = resolve_directory_path(root_path, path)
    matches: list[FileMatch] = []
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
            if not matches_path_glob(relative_child, normalized_pattern):
                continue
            if len(matches) >= tool_limits.max_entries_per_call:
                truncated = True
                return False
            matches.append(build_file_match(resolved_request.root, child))
        return True

    walk_directory(resolved_request.resolved)
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
        raise ValueError("Chat tool paths must be relative to the configured root.")
    resolved_root = root_path.resolve()
    candidate_target = resolved_root / requested_path
    resolved_target = candidate_target.resolve()
    if not resolved_target.is_relative_to(resolved_root):
        raise ValueError("Requested chat tool path escapes the configured root.")
    resolved_relative = resolved_target.relative_to(resolved_root)
    if candidate_target.is_symlink():
        symlink_kind = (
            "directory"
            if resolved_target.exists() and resolved_target.is_dir()
            else "file"
        )
        raise ValueError(
            f"Requested path must not be a symlinked {symlink_kind}: {normalized_path}"
        )
    if not candidate_target.exists():
        raise ValueError(
            f"Requested file or directory does not exist: {normalized_path}"
        )
    return normalized_path, resolved_root, candidate_target, resolved_relative


def search_text_impl(
    root_path: Path,
    query: str,
    path: str,
    *,
    source_filters: ChatSourceFilters,
    tool_limits: ChatToolLimits,
) -> TextSearchResult:
    """Return matching text lines within one root-confined directory or file."""
    normalized_query = normalize_required_value(query, field_name="search_text query")
    normalized_path, resolved_root, candidate_target, resolved_relative = (
        resolve_search_file_or_directory(root_path, path)
    )
    if candidate_target.resolve().is_file():
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
            query=normalized_query,
            tool_limits=tool_limits,
            loaded_content=loaded_content,
        )

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
                if normalized_query not in line_text:
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
        query=normalized_query,
        matches=matches,
        truncated=truncated,
    )


def _search_loaded_content(
    *,
    resolved_request: ResolvedRootPath,
    query: str,
    tool_limits: ChatToolLimits,
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


def get_file_info_impl(
    root_path: Path,
    path: str | list[str],
    *,
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
) -> FileInfoResult | FileInfoBatchResult:
    """Return deterministic metadata for one or more root-confined files."""
    if isinstance(path, str):
        return _get_single_file_info(
            root_path,
            path,
            session_config=session_config,
            tool_limits=tool_limits,
        )
    return FileInfoBatchResult(
        results=[
            _get_single_file_info(
                root_path,
                item,
                session_config=session_config,
                tool_limits=tool_limits,
            )
            for item in path
        ]
    )


def _get_single_file_info(
    root_path: Path,
    path: str,
    *,
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
) -> FileInfoResult:
    resolved_request = resolve_file_path(root_path, path)
    loaded_content = load_readable_content(resolved_request.resolved)
    return build_file_info_result(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        candidate_file=resolved_request.candidate,
        resolved_file=resolved_request.resolved,
        relative_candidate_path=resolved_request.candidate.relative_to(
            resolved_request.root
        ),
        session_config=session_config,
        tool_limits=tool_limits,
        loaded_content=loaded_content,
    )


def read_file_impl(
    root_path: Path,
    path: str,
    *,
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
    start_char: int | None,
    end_char: int | None,
) -> FileReadResult:
    """Return bounded text or markdown content for one root-confined file."""
    resolved_request = resolve_file_path(root_path, path)
    loaded_content = load_readable_content(resolved_request.resolved)
    full_read_char_limit = effective_full_read_char_limit(session_config, tool_limits)
    if loaded_content.status != "ok" or loaded_content.content is None:
        return FileReadResult(
            requested_path=resolved_request.requested_path,
            resolved_path=resolved_request.resolved_path,
            read_kind=loaded_content.read_kind,
            status=loaded_content.status,
            content=None,
            file_size_bytes=resolved_request.resolved.stat().st_size,
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
        file_size_bytes=resolved_request.resolved.stat().st_size,
        max_file_size_characters=tool_limits.max_file_size_characters,
        full_read_char_limit=full_read_char_limit,
        estimated_token_count=estimate_token_count(content_slice),
    )

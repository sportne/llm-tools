"""Shared path and filter helpers for repository-style filesystem tools."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from llm_tools.tools.filesystem.models import (
    DirectoryEntry,
    DirectoryEntryType,
    FileMatch,
    SourceFilters,
)


@dataclass(frozen=True, slots=True)
class ResolvedRootPath:
    """Normalized and root-confined request path metadata."""

    root: Path
    candidate: Path
    resolved: Path
    requested_path: str
    resolved_path: str


def matches_patterns(path: Path, patterns: list[str]) -> bool:
    """Return whether the relative path matches any configured glob pattern."""
    path_match = PurePosixPath(path.as_posix())
    return any(path_match.match(pattern) for pattern in patterns)


def normalize_requested_path(path: str) -> str:
    """Return a stable non-empty POSIX-style requested path."""
    cleaned = path.strip()
    if not cleaned:
        raise ValueError("Requested tool path must not be empty.")
    if cleaned == ".":
        return "."
    return PurePosixPath(cleaned).as_posix()


def normalize_required_value(value: str, *, field_name: str) -> str:
    """Return a stable non-empty value for required string inputs."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty.")
    return cleaned


def normalize_required_pattern(pattern: str) -> str:
    """Return a stable non-empty file glob pattern."""
    return PurePosixPath(
        normalize_required_value(pattern, field_name="find_files pattern")
    ).as_posix()


def matches_path_glob(relative_path: Path, pattern: str) -> bool:
    """Return whether a relative path matches a recursive glob pattern."""
    path_parts = PurePosixPath(relative_path.as_posix()).parts
    pattern_parts = PurePosixPath(pattern).parts

    def match_parts(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)

        pattern_part = pattern_parts[pattern_index]
        if pattern_part == "**":
            if pattern_index == len(pattern_parts) - 1:
                return True
            return any(
                match_parts(next_index, pattern_index + 1)
                for next_index in range(path_index, len(path_parts) + 1)
            )

        if path_index >= len(path_parts):
            return False
        if not fnmatch.fnmatchcase(path_parts[path_index], pattern_part):
            return False
        return match_parts(path_index + 1, pattern_index + 1)

    return match_parts(0, 0)


def resolve_root_confined_path(
    root_path: Path,
    path: str,
    *,
    expected_kind: str,
    reject_symlink: bool,
) -> ResolvedRootPath:
    """Resolve a relative path inside the root and enforce kind constraints."""
    normalized_request = normalize_requested_path(path)
    requested_path = Path(normalized_request)
    if requested_path.is_absolute():
        raise ValueError("Tool paths must be relative to the configured root.")

    resolved_root = root_path.resolve()
    candidate_target = resolved_root / requested_path
    if reject_symlink and candidate_target.is_symlink():
        raise ValueError(
            "Requested path must not be a symlinked "
            f"{expected_kind}: {normalized_request}"
        )

    resolved_target = candidate_target.resolve()
    if not resolved_target.is_relative_to(resolved_root):
        raise ValueError("Requested tool path escapes the configured root.")
    if not resolved_target.exists():
        raise ValueError(
            f"Requested {expected_kind} does not exist: {normalized_request}"
        )

    kind_check = (
        resolved_target.is_dir
        if expected_kind == "directory"
        else resolved_target.is_file
    )
    if not kind_check():
        raise ValueError(
            f"Requested path is not a {expected_kind}: {normalized_request}"
        )

    resolved_relative = resolved_target.relative_to(resolved_root)
    resolved_path = (
        resolved_relative.as_posix() if resolved_relative.as_posix() else "."
    )
    return ResolvedRootPath(
        root=resolved_root,
        candidate=candidate_target,
        resolved=resolved_target,
        requested_path=normalized_request,
        resolved_path=resolved_path,
    )


def resolve_directory_path(root_path: Path, path: str) -> ResolvedRootPath:
    """Resolve one directory path inside the configured root."""
    return resolve_root_confined_path(
        root_path,
        path,
        expected_kind="directory",
        reject_symlink=True,
    )


def resolve_file_path(root_path: Path, path: str) -> ResolvedRootPath:
    """Resolve one file path inside the configured root."""
    return resolve_root_confined_path(
        root_path,
        path,
        expected_kind="file",
        reject_symlink=True,
    )


def is_hidden(path: Path) -> bool:
    """Return whether any path component is hidden."""
    return any(part.startswith(".") for part in path.parts if part not in {".", ".."})


def entry_type(path: Path) -> DirectoryEntryType:
    """Return the stable entry-type label for one filesystem path."""
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def should_include_entry(
    relative_path: Path,
    *,
    source_filters: SourceFilters,
) -> bool:
    """Return whether an entry should be exposed to the model."""
    if is_hidden(relative_path) and not source_filters.include_hidden:
        return False
    if matches_patterns(relative_path, source_filters.exclude):
        return False
    return not (
        source_filters.include
        and not matches_patterns(relative_path, source_filters.include)
    )


def should_prune_directory(
    relative_path: Path,
    *,
    source_filters: SourceFilters,
) -> bool:
    """Return whether a directory subtree should be skipped entirely."""
    if is_hidden(relative_path) and not source_filters.include_hidden:
        return True
    return matches_patterns(relative_path, source_filters.exclude)


def build_entry(root_path: Path, path: Path, *, depth: int) -> DirectoryEntry:
    """Return one directory-entry payload."""
    relative_path = path.relative_to(root_path)
    return DirectoryEntry(
        path=relative_path.as_posix(),
        name=path.name,
        entry_type=entry_type(path),
        depth=depth,
        is_hidden=is_hidden(relative_path),
        is_symlink=path.is_symlink(),
    )


def build_file_match(root_path: Path, path: Path) -> FileMatch:
    """Return one file-search match payload."""
    relative_path = path.relative_to(root_path)
    parent_path = relative_path.parent.as_posix()
    return FileMatch(
        path=relative_path.as_posix(),
        name=path.name,
        parent_path="." if parent_path == "." else parent_path,
        is_hidden=is_hidden(relative_path),
    )

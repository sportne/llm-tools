"""Shared path and filter helpers for repository-style filesystem tools."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pathspec import GitIgnoreSpec

from llm_tools.tools.filesystem.models import (
    DirectoryEntry,
    DirectoryEntryType,
    FileMatch,
    SourceFilters,
)

_INTERNAL_TOOL_PATH_PART = ".llm_tools"


@dataclass(frozen=True, slots=True)
class GitignoreSpecEntry:
    """One parsed .gitignore file scoped to a root-relative base path."""

    base_path: Path
    spec: GitIgnoreSpec


@dataclass(frozen=True, slots=True)
class GitignoreMatcher:
    """Per-call matcher for root and nested .gitignore rules."""

    entries: tuple[GitignoreSpecEntry, ...] = ()

    @classmethod
    def from_root(cls, root_path: Path) -> GitignoreMatcher:
        """Build a matcher from .gitignore files under one workspace root."""
        resolved_root = root_path.resolve()
        entries: list[GitignoreSpecEntry] = []
        for directory, dirnames, filenames in os.walk(resolved_root, followlinks=False):
            directory_path = Path(directory)
            relative_directory = directory_path.relative_to(resolved_root)
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not (directory_path / dirname).is_symlink()
                and not is_internal_tool_path(relative_directory / dirname)
            ]
            if ".gitignore" not in filenames:
                continue
            gitignore_path = directory_path / ".gitignore"
            try:
                lines = gitignore_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            entries.append(
                GitignoreSpecEntry(
                    base_path=relative_directory,
                    spec=GitIgnoreSpec.from_lines(lines),
                )
            )
        entries.sort(key=lambda entry: len(entry.base_path.parts))
        return cls(tuple(entries))

    def is_ignored(self, relative_path: Path, *, is_dir: bool = False) -> bool:
        """Return whether a root-relative path is ignored by any .gitignore."""
        ignored = False
        for entry in self.entries:
            relative_to_base = _relative_to_gitignore_base(
                relative_path,
                entry.base_path,
            )
            if relative_to_base is None:
                continue
            path_text = relative_to_base.as_posix()
            if not path_text or path_text == ".":
                continue
            if is_dir and not path_text.endswith("/"):
                path_text += "/"
            result = entry.spec.check_file(path_text)
            if result.include is True:
                ignored = True
            elif result.include is False:
                ignored = False
        return ignored


@dataclass(frozen=True, slots=True)
class ResolvedRootPath:
    """Normalized and root-confined request path metadata."""

    root: Path
    candidate: Path
    resolved: Path
    requested_path: str
    resolved_path: str


def _relative_to_gitignore_base(relative_path: Path, base_path: Path) -> Path | None:
    """Return relative_path scoped to one .gitignore base, if applicable."""
    if not base_path.parts:
        return relative_path
    try:
        return relative_path.relative_to(base_path)
    except ValueError:
        return None


def matches_patterns(path: Path, patterns: list[str]) -> bool:
    """Return whether the relative path matches any configured glob pattern."""
    path_match = PurePosixPath(path.as_posix())
    return any(path_match.match(pattern) for pattern in patterns)


def normalize_requested_path(path: str) -> str:
    """Return a stable non-empty POSIX-style requested path."""
    cleaned = path.strip()
    if not cleaned:
        raise ValueError("Requested tool path must not be empty.")
    if _looks_like_windows_rooted_path(cleaned):
        raise ValueError("Tool paths must be relative to the configured root.")
    cleaned = cleaned.replace("\\", "/")
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
        normalize_required_value(pattern, field_name="find_files pattern").replace(
            "\\", "/"
        )
    ).as_posix()


def _looks_like_windows_rooted_path(path: str) -> bool:
    """Return whether a request uses a Windows drive or UNC root."""
    if len(path) >= 2 and path[0].isalpha() and path[1] == ":":
        return True
    return path.startswith(("\\\\", "//"))


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


def _normalized_request_path(path: str) -> tuple[str, Path, Path]:
    """Normalize one user path and reject absolute or internal tool paths."""
    normalized_request = normalize_requested_path(path)
    requested_path = Path(normalized_request)
    if requested_path.is_absolute():
        raise ValueError("Tool paths must be relative to the configured root.")
    if requested_path.parts and requested_path.parts[0] == _INTERNAL_TOOL_PATH_PART:
        raise ValueError(
            "Tool paths must not target internal tool-managed directories."
        )
    return (
        normalized_request,
        requested_path,
        Path() if normalized_request == "." else requested_path,
    )


def _resolve_relative_display_path(root_path: Path, resolved_target: Path) -> str:
    """Return a stable root-relative display path."""
    resolved_relative = resolved_target.relative_to(root_path)
    return resolved_relative.as_posix() if resolved_relative.as_posix() else "."


def _validate_root_confined_candidate(
    resolved_root: Path,
    requested_path: Path,
    *,
    normalized_request: str,
    final_kind: str,
) -> Path:
    """Validate a relative candidate path stays root-confined and symlink-free."""
    current = resolved_root
    for index, part in enumerate(requested_path.parts):
        if part == ".":
            continue
        current = current / part
        is_final = index == len(requested_path.parts) - 1
        if current.is_symlink():
            if is_final:
                raise ValueError(
                    "Requested path must not be a symlinked "
                    f"{final_kind}: {normalized_request}"
                )
            raise ValueError(
                "Requested path must not traverse a symlinked directory: "
                f"{normalized_request}"
            )
        if not current.resolve(strict=False).is_relative_to(resolved_root):
            raise ValueError("Requested tool path escapes the configured root.")
        if not is_final and current.exists() and not current.is_dir():
            raise ValueError(
                "Requested path has a non-directory parent component: "
                f"{normalized_request}"
            )
    return resolved_root / requested_path


def resolve_root_confined_path(
    root_path: Path,
    path: str,
    *,
    expected_kind: str,
    reject_symlink: bool,
) -> ResolvedRootPath:
    """Resolve a relative path inside the root and enforce kind constraints."""
    normalized_request, _requested_path, normalized_parts = _normalized_request_path(
        path
    )
    del reject_symlink

    resolved_root = root_path.resolve()
    candidate_target = _validate_root_confined_candidate(
        resolved_root,
        normalized_parts,
        normalized_request=normalized_request,
        final_kind=expected_kind,
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

    return ResolvedRootPath(
        root=resolved_root,
        candidate=candidate_target,
        resolved=resolved_target,
        requested_path=normalized_request,
        resolved_path=_resolve_relative_display_path(resolved_root, resolved_target),
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


def resolve_writable_file_path(root_path: Path, path: str) -> ResolvedRootPath:
    """Resolve one writable file path inside the configured root."""
    normalized_request, _, normalized_parts = _normalized_request_path(path)
    resolved_root = root_path.resolve()
    candidate_target = _validate_root_confined_candidate(
        resolved_root,
        normalized_parts,
        normalized_request=normalized_request,
        final_kind="file",
    )
    resolved_target = candidate_target.resolve(strict=False)
    if not resolved_target.is_relative_to(resolved_root):
        raise ValueError("Requested tool path escapes the configured root.")
    if resolved_target.exists() and resolved_target.is_dir():
        raise IsADirectoryError(
            f"Requested path is a directory, not a file: {normalized_request}"
        )
    return ResolvedRootPath(
        root=resolved_root,
        candidate=candidate_target,
        resolved=resolved_target,
        requested_path=normalized_request,
        resolved_path=_resolve_relative_display_path(resolved_root, resolved_target),
    )


def is_internal_tool_path(path: Path) -> bool:
    """Return whether a path targets the tool-managed cache subtree."""
    return bool(path.parts) and path.parts[0] == _INTERNAL_TOOL_PATH_PART


def is_hidden(
    path: Path,
    *,
    gitignore_matcher: GitignoreMatcher | None = None,
    is_dir: bool = False,
) -> bool:
    """Return whether any path component is hidden."""
    return any(
        part.startswith(".") for part in path.parts if part not in {".", ".."}
    ) or (
        gitignore_matcher is not None
        and gitignore_matcher.is_ignored(path, is_dir=is_dir)
    )


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
    gitignore_matcher: GitignoreMatcher | None = None,
    is_dir: bool = False,
) -> bool:
    """Return whether an entry should be exposed to the model."""
    if is_internal_tool_path(relative_path):
        return False
    if (
        is_hidden(
            relative_path,
            gitignore_matcher=gitignore_matcher,
            is_dir=is_dir,
        )
        and not source_filters.include_hidden
    ):
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
    gitignore_matcher: GitignoreMatcher | None = None,
) -> bool:
    """Return whether a directory subtree should be skipped entirely."""
    if is_internal_tool_path(relative_path):
        return True
    if (
        is_hidden(
            relative_path,
            gitignore_matcher=gitignore_matcher,
            is_dir=True,
        )
        and not source_filters.include_hidden
    ):
        return True
    return matches_patterns(relative_path, source_filters.exclude)


def build_entry(
    root_path: Path,
    path: Path,
    *,
    depth: int,
    gitignore_matcher: GitignoreMatcher | None = None,
) -> DirectoryEntry:
    """Return one directory-entry payload."""
    relative_path = path.relative_to(root_path)
    path_type = entry_type(path)
    return DirectoryEntry(
        path=relative_path.as_posix(),
        name=path.name,
        entry_type=path_type,
        depth=depth,
        is_hidden=is_hidden(
            relative_path,
            gitignore_matcher=gitignore_matcher,
            is_dir=path_type == "directory",
        ),
        is_symlink=path.is_symlink(),
    )


def build_file_match(
    root_path: Path,
    path: Path,
    *,
    gitignore_matcher: GitignoreMatcher | None = None,
) -> FileMatch:
    """Return one file-search match payload."""
    relative_path = path.relative_to(root_path)
    parent_path = relative_path.parent.as_posix()
    return FileMatch(
        path=relative_path.as_posix(),
        name=path.name,
        parent_path="." if parent_path == "." else parent_path,
        is_hidden=is_hidden(relative_path, gitignore_matcher=gitignore_matcher),
    )

"""Local filesystem discovery for portable skill packages."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from llm_tools.skills_api.models import (
    SkillDiscoveryOptions,
    SkillDiscoveryResult,
    SkillError,
    SkillMetadata,
    SkillRoot,
    SkillScope,
)

SKILL_FILENAME = "SKILL.md"
MAX_NAME_LEN = 64
MAX_DESCRIPTION_LEN = 1024
_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SCOPE_RANK: dict[SkillScope, int] = {
    SkillScope.ENTERPRISE: 0,
    SkillScope.USER: 1,
    SkillScope.PROJECT: 2,
    SkillScope.BUNDLED: 3,
}


def discover_skills(
    roots: Iterable[SkillRoot],
    options: SkillDiscoveryOptions | None = None,
) -> SkillDiscoveryResult:
    """Scan local roots for valid skill metadata and path-specific errors."""
    resolved_options = options or SkillDiscoveryOptions()
    skills: list[SkillMetadata] = []
    errors: list[SkillError] = []
    seen_paths: set[Path] = set()

    for root in roots:
        root_path = _canonical_path(root.path)
        if not root_path.exists() or not root_path.is_dir():
            continue
        for candidate in _iter_skill_files(root_path, resolved_options):
            canonical_candidate = _canonical_path(candidate)
            if canonical_candidate in seen_paths:
                continue
            seen_paths.add(canonical_candidate)
            try:
                skills.append(_parse_skill_metadata(canonical_candidate, root.scope))
            except ValueError as exc:
                errors.append(SkillError(path=canonical_candidate, message=str(exc)))
            except OSError as exc:
                errors.append(
                    SkillError(
                        path=canonical_candidate,
                        message=f"failed to read skill file: {exc}",
                    )
                )

    skills.sort(key=lambda skill: (_SCOPE_RANK[skill.scope], skill.name, skill.path))
    return SkillDiscoveryResult(skills=tuple(skills), errors=tuple(errors))


def bundled_skill_root() -> SkillRoot:
    """Return the package's bundled example skill root."""
    return SkillRoot(path=Path(__file__).parent / "bundled", scope=SkillScope.BUNDLED)


def _iter_skill_files(root: Path, options: SkillDiscoveryOptions) -> Iterable[Path]:
    visited_dirs: set[Path] = set()
    stack: list[tuple[Path, int]] = [(root, 0)]

    while stack:
        directory, depth = stack.pop()
        canonical_dir = _canonical_path(directory)
        if canonical_dir in visited_dirs:
            continue
        if len(visited_dirs) >= options.max_directories_per_root:
            return
        visited_dirs.add(canonical_dir)

        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            continue

        for child in children:
            if child.is_symlink() and not options.follow_symlinks:
                continue
            if child.name == SKILL_FILENAME and child.is_file():
                yield child
                continue
            if depth >= options.max_depth:
                continue
            if options.ignore_hidden_directories and child.name.startswith("."):
                continue
            if child.is_dir():
                stack.append((child, depth + 1))


def _parse_skill_metadata(path: Path, scope: SkillScope) -> SkillMetadata:
    text = path.read_text(encoding="utf-8")
    frontmatter = _extract_frontmatter(text)
    if frontmatter is None:
        msg = "missing YAML frontmatter delimited by ---"
        raise ValueError(msg)
    try:
        # BaseLoader avoids YAML 1.1 scalar coercion, so names like "off" remain
        # strings while still refusing arbitrary object construction.
        raw = yaml.load(frontmatter, Loader=yaml.BaseLoader)  # noqa: S506
    except yaml.YAMLError as exc:
        msg = f"invalid YAML frontmatter: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = "frontmatter must be a mapping"
        raise ValueError(msg)

    name = _required_metadata_string(raw, "name", MAX_NAME_LEN)
    if not _NAME_RE.fullmatch(name):
        msg = "invalid name: use only letters, digits, '_', '-', '.', and ':'"
        raise ValueError(msg)
    description = _required_metadata_string(raw, "description", MAX_DESCRIPTION_LEN)
    return SkillMetadata(
        name=name,
        description=description,
        path=path,
        scope=scope,
    )


def _extract_frontmatter(text: str) -> str | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[1:index])
    return None


def _required_metadata_string(
    raw: dict[Any, Any],
    field: str,
    max_len: int,
) -> str:
    value = raw.get(field)
    if not isinstance(value, str):
        msg = f"missing or invalid `{field}`"
        raise ValueError(msg)
    cleaned = value.strip()
    if not cleaned:
        msg = f"missing or invalid `{field}`"
        raise ValueError(msg)
    if "\n" in cleaned or "\r" in cleaned:
        msg = f"invalid `{field}`: must be single-line"
        raise ValueError(msg)
    if len(cleaned) > max_len:
        msg = f"invalid `{field}`: must be at most {max_len} characters"
        raise ValueError(msg)
    return cleaned


def _canonical_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)

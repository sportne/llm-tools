"""Path helpers for Assistant app local state and user-provided paths."""

from __future__ import annotations

from pathlib import Path


def expand_app_path(path: Path | str) -> Path:
    """Return a user-expanded path without resolving or requiring existence."""
    return Path(path).expanduser()


def expanded_path_text(path: Path | str) -> str:
    """Return a native string for a user-expanded app path."""
    return str(expand_app_path(path))

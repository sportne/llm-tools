"""Public readable-content helpers shared by filesystem-adjacent features."""

from __future__ import annotations

from pathlib import Path

from llm_tools.tools.filesystem._content import (
    MARKITDOWN_EXTENSIONS,
    PROJECT_EXTENSIONS,
    ConversionBackend,
    LoadedReadableContent,
    _get_conversion_backend,
    _get_read_file_cache_root,
    _read_cached_conversion,
    _write_cached_conversion,
    load_readable_content,
)

CONVERTIBLE_DOCUMENT_EXTENSIONS = MARKITDOWN_EXTENSIONS | PROJECT_EXTENSIONS


def get_read_file_cache_root(workspace_root: Path) -> Path:
    """Return the shared readable-content cache root."""
    return _get_read_file_cache_root(workspace_root)


def get_conversion_backend(path: Path) -> ConversionBackend | None:
    """Return the conversion backend for a supported non-text file."""
    return _get_conversion_backend(path)


def read_cached_conversion(path: Path, *, cache_root: Path) -> str | None:
    """Return cached converted markdown when the source has not changed."""
    return _read_cached_conversion(path, cache_root=cache_root)


def write_cached_conversion(path: Path, markdown: str, *, cache_root: Path) -> Path:
    """Persist converted markdown and source metadata for future reads."""
    return _write_cached_conversion(path, markdown, cache_root=cache_root)


__all__ = [
    "CONVERTIBLE_DOCUMENT_EXTENSIONS",
    "ConversionBackend",
    "LoadedReadableContent",
    "get_conversion_backend",
    "get_read_file_cache_root",
    "load_readable_content",
    "read_cached_conversion",
    "write_cached_conversion",
]

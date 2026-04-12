"""Readable-content and file-metadata helpers for repository filesystem tools."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from llm_tools.tools.filesystem._paths import is_hidden
from llm_tools.tools.filesystem.models import (
    FileInfoResult,
    FileInfoStatus,
    FileReadKind,
    ToolLimits,
)

MARKITDOWN_EXTENSIONS = {
    ".doc",
    ".docx",
    ".epub",
    ".html",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rtf",
    ".xls",
    ".xlsx",
}
DEFAULT_MAX_READ_FILE_CHARS = 4000
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class LoadedReadableContent:
    """Deterministic text representation for one readable file."""

    read_kind: FileReadKind
    status: FileInfoStatus
    content: str | None
    error_message: str | None = None


def _get_read_file_cache_root() -> Path:
    """Return the cache root for converted file content."""
    return Path(tempfile.gettempdir()) / "llm_tools" / "read_file_cache"


def _get_cached_conversion_paths(resolved: Path) -> tuple[Path, Path]:
    """Return the markdown and metadata cache paths for a source file."""
    cache_key = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
    cache_dir = _get_read_file_cache_root() / cache_key
    return cache_dir / "content.md", cache_dir / "metadata.json"


def _read_cached_conversion(resolved: Path) -> str | None:
    """Return cached converted markdown when the source has not changed."""
    content_path, metadata_path = _get_cached_conversion_paths(resolved)
    if not content_path.exists() or not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stat = resolved.stat()
    if (
        metadata.get("mtime_ns") != stat.st_mtime_ns
        or metadata.get("size_bytes") != stat.st_size
    ):
        return None
    return content_path.read_text(encoding="utf-8")


def _write_cached_conversion(resolved: Path, markdown: str) -> Path:
    """Persist converted markdown and source metadata for future reads."""
    content_path, metadata_path = _get_cached_conversion_paths(resolved)
    content_path.parent.mkdir(parents=True, exist_ok=True)
    content_path.write_text(markdown, encoding="utf-8")
    stat = resolved.stat()
    metadata_path.write_text(
        json.dumps({"mtime_ns": stat.st_mtime_ns, "size_bytes": stat.st_size}),
        encoding="utf-8",
    )
    return content_path


def read_searchable_text(path: Path) -> str | None:
    """Read text content when it is UTF-8-like and non-binary."""
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if "\x00" in content:
        return None
    return content


def load_readable_content(path: Path) -> LoadedReadableContent:
    """Return deterministic readable content for one file."""
    text_content = read_searchable_text(path)
    if text_content is not None:
        return LoadedReadableContent(
            read_kind="text", status="ok", content=text_content
        )

    if path.suffix.lower() not in MARKITDOWN_EXTENSIONS:
        return LoadedReadableContent(
            read_kind="unsupported",
            status="unsupported",
            content=None,
            error_message="File type is not supported for repository reads",
        )

    cached = _read_cached_conversion(path)
    if cached is not None:
        return LoadedReadableContent(
            read_kind="markitdown",
            status="ok",
            content=cached,
        )

    try:
        converted = convert_with_markitdown(path)
    except Exception as exc:
        return LoadedReadableContent(
            read_kind="markitdown",
            status="error",
            content=None,
            error_message=str(exc),
        )

    _write_cached_conversion(path, converted)
    return LoadedReadableContent(read_kind="markitdown", status="ok", content=converted)


def convert_with_markitdown(path: Path) -> str:
    """Convert a supported non-text document into markdown text."""
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(path))
    if isinstance(result, str):
        return result
    for attribute in ("text_content", "markdown", "text"):
        value = getattr(result, attribute, None)
        if isinstance(value, str):
            return value
    raise RuntimeError("markitdown conversion did not return readable markdown text")


def count_lines(text: str, *, max_read_lines: int) -> int | None:
    """Count lines, returning None when the count exceeds the configured cap."""
    if not text:
        return 0
    line_count = text.count("\n")
    if not text.endswith("\n"):
        line_count += 1
    if line_count > max_read_lines:
        return None
    return line_count


def estimate_token_count(text: str) -> int:
    """Estimate tokens with a small deterministic lexical tokenizer."""
    return len([match.group(0).lower() for match in TOKEN_RE.finditer(text)])


def is_within_character_limit(
    content: str | None,
    *,
    tool_limits: ToolLimits,
) -> bool:
    """Return whether readable content fits within the configured size cap."""
    return content is not None and len(content) <= tool_limits.max_file_size_characters


def effective_full_read_char_limit(tool_limits: ToolLimits) -> int:
    """Return the full-read cap derived from effective tool limits."""
    configured_limit = tool_limits.max_read_file_chars
    if configured_limit is not None:
        return configured_limit
    return DEFAULT_MAX_READ_FILE_CHARS


def normalize_range(
    *,
    start_char: int | None,
    end_char: int | None,
    character_count: int,
) -> tuple[int, int]:
    """Validate and clamp a character-range request."""
    normalized_start = 0 if start_char is None else start_char
    normalized_end = character_count if end_char is None else end_char
    if normalized_start < 0:
        raise ValueError("start_char must be greater than or equal to 0")
    if normalized_end < 0:
        raise ValueError("end_char must be greater than or equal to 0")
    if end_char is not None and normalized_end <= normalized_start:
        raise ValueError("end_char must be greater than start_char")
    if normalized_start > character_count:
        raise ValueError("start_char must not exceed character_count")
    return normalized_start, min(normalized_end, character_count)


def build_file_info_result(
    *,
    requested_path: str,
    resolved_path: str,
    candidate_file: Path,
    resolved_file: Path,
    relative_candidate_path: Path,
    tool_limits: ToolLimits,
    loaded_content: LoadedReadableContent,
) -> FileInfoResult:
    """Return deterministic metadata for one root-confined file."""
    full_read_char_limit = effective_full_read_char_limit(tool_limits)
    size_bytes = resolved_file.stat().st_size
    content = loaded_content.content
    character_count = len(content) if content is not None else None
    within_size_limit = is_within_character_limit(content, tool_limits=tool_limits)
    return FileInfoResult(
        requested_path=requested_path,
        resolved_path=resolved_path,
        name=candidate_file.name,
        size_bytes=size_bytes,
        is_hidden=is_hidden(relative_candidate_path),
        is_symlink=candidate_file.is_symlink(),
        read_kind=loaded_content.read_kind,
        status=loaded_content.status,
        estimated_token_count=(
            estimate_token_count(content) if content is not None else None
        ),
        character_count=character_count,
        line_count=(
            count_lines(content, max_read_lines=tool_limits.max_read_lines)
            if content is not None
            else None
        ),
        max_file_size_characters=tool_limits.max_file_size_characters,
        within_size_limit=within_size_limit,
        full_read_char_limit=full_read_char_limit,
        can_read_full=within_size_limit
        and character_count is not None
        and character_count <= full_read_char_limit,
        error_message=loaded_content.error_message,
    )


def dump_json(payload: object) -> str:
    """Return a stable compact JSON string."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

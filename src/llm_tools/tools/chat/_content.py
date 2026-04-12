"""Readable-content and file-metadata helpers for repository chat tools."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from llm_tools.tools.chat._paths import is_hidden
from llm_tools.tools.chat.models import (
    ChatSessionConfig,
    ChatToolLimits,
    FileInfoResult,
    FileInfoStatus,
    FileReadKind,
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
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class LoadedReadableContent:
    """Deterministic text representation for one readable file."""

    read_kind: FileReadKind
    status: FileInfoStatus
    content: str | None
    error_message: str | None = None


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
            error_message="File type is not supported for chat reads",
        )

    cache_path = markitdown_cache_path(path)
    if cache_path.exists():
        return LoadedReadableContent(
            read_kind="markitdown",
            status="ok",
            content=cache_path.read_text(encoding="utf-8"),
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(converted, encoding="utf-8")
    return LoadedReadableContent(read_kind="markitdown", status="ok", content=converted)


def markitdown_cache_path(path: Path) -> Path:
    """Return the stable markdown cache path for one source file."""
    cache_root = Path(tempfile.gettempdir()) / "llm_tools-chat-markitdown-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    stat = path.stat()
    cache_key = hashlib.sha256(
        f"{path.resolve().as_posix()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()
    return cache_root / f"{cache_key}.md"


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
    tool_limits: ChatToolLimits,
) -> bool:
    """Return whether readable content fits within the configured size cap."""
    return content is not None and len(content) <= tool_limits.max_file_size_characters


def effective_full_read_char_limit(
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
) -> int:
    """Return the full-read cap derived from config and context window."""
    configured_limit = tool_limits.max_read_file_chars
    if configured_limit is not None:
        return configured_limit
    return max(1, session_config.max_context_tokens * 4)


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
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
    loaded_content: LoadedReadableContent,
) -> FileInfoResult:
    """Return deterministic metadata for one root-confined file."""
    full_read_char_limit = effective_full_read_char_limit(session_config, tool_limits)
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

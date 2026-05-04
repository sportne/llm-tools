"""Shared helpers for Atlassian tool implementations."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

from llm_tools.tool_api import (
    RetryableToolExecutionError,
    SourceProvenanceRef,
    ToolExecutionContext,
)
from llm_tools.tools.filesystem._content import (
    effective_full_read_char_limit,
    estimate_token_count,
    is_within_character_limit,
    line_range_for_character_range,
    load_readable_content,
    normalize_range,
)
from llm_tools.tools.filesystem.models import FileReadKind, FileReadResult, ToolLimits

_JIRA_ENV_KEYS = ("JIRA_BASE_URL", "JIRA_API_TOKEN")
_BITBUCKET_ENV_KEYS = (
    "BITBUCKET_BASE_URL",
    "BITBUCKET_API_TOKEN",
)
_CONFLUENCE_ENV_KEYS = (
    "CONFLUENCE_BASE_URL",
    "CONFLUENCE_API_TOKEN",
)
_INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
_REMOTE_TOOL_TIMEOUT_SECONDS = 30
_REMOTE_COLLECTION_LIMIT = 100
_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


def _search_fetch_limit(limit: int) -> int:
    return limit + 1


def _append_remote_source_provenance(
    context: ToolExecutionContext,
    *,
    source_kind: str,
    source_id: str,
) -> None:
    context.add_source_provenance(
        SourceProvenanceRef(
            source_kind=source_kind,
            source_id=source_id,
            content_hash=hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
            whole_source_reproduction_allowed=True,
            metadata={"source_id": source_id},
        )
    )


def _get_tool_limits(context: ToolExecutionContext) -> ToolLimits:
    return ToolLimits.model_validate(context.metadata.get("tool_limits", {}))


def _get_value(payload: object, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _extract_collection(payload: object) -> list[Any]:
    if isinstance(payload, dict):
        for key in ("values", "results", "lines"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    if isinstance(payload, list):
        return payload
    return []


def _absolute_url(base_url: str, raw_url: str | None) -> str | None:
    if raw_url is None or raw_url == "":
        return None
    return urljoin(base_url.rstrip("/") + "/", raw_url)


def _extract_issue_fields(
    issue: dict[str, Any],
    *,
    requested_fields: list[str] | None = None,
) -> dict[str, Any]:
    fields = cast(dict[str, Any], issue.get("fields") or {})
    issue_type = fields.get("issuetype") or {}
    status = fields.get("status") or {}
    assignee = fields.get("assignee") or {}
    description = fields.get("description")
    requested = {
        name: fields[name] for name in (requested_fields or []) if name in fields
    }
    return {
        "key": issue.get("key", ""),
        "summary": fields.get("summary"),
        "description": description,
        "status": status.get("name"),
        "issue_type": issue_type.get("name"),
        "assignee": assignee.get("displayName"),
        "requested_fields": requested,
    }


def _normalize_remote_exception(exc: Exception) -> Exception:
    status_code = cast(
        int | None,
        _get_value(
            exc,
            "status_code",
            _get_value(_get_value(exc, "response", {}), "status_code"),
        ),
    )
    exception_type = type(exc).__name__.lower()
    message = str(exc).lower()
    if status_code in _RETRYABLE_STATUS_CODES or any(
        token in exception_type or token in message
        for token in ("timeout", "timedout", "connection", "temporarily unavailable")
    ):
        return RetryableToolExecutionError(str(exc))
    return exc


def _normalize_remote_text(text: str) -> tuple[str | None, int, str | None]:
    size_bytes = len(text.encode())
    if "\x00" in text:
        return None, size_bytes, "Remote content is binary."
    return text, size_bytes, None


def _normalize_remote_bytes(raw_bytes: bytes) -> tuple[str | None, int, str | None]:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None, len(raw_bytes), "Remote content is not UTF-8 text."
    if "\x00" in text:
        return None, len(raw_bytes), "Remote content is binary."
    return text, len(raw_bytes), None


def _build_text_read_result(
    *,
    requested_path: str,
    resolved_path: str,
    tool_limits: ToolLimits,
    content: str | None,
    file_size_bytes: int,
    status: str,
    read_kind: FileReadKind,
    error_message: str | None = None,
    start_char: int | None = None,
    end_char: int | None = None,
) -> FileReadResult:
    full_read_char_limit = effective_full_read_char_limit(tool_limits)
    if content is None or status != "ok":
        return FileReadResult(
            requested_path=requested_path,
            resolved_path=resolved_path,
            read_kind=read_kind,
            status=cast(Any, status),
            content=None,
            file_size_bytes=file_size_bytes,
            max_read_input_bytes=tool_limits.max_read_input_bytes,
            max_file_size_characters=tool_limits.max_file_size_characters,
            full_read_char_limit=full_read_char_limit,
            error_message=error_message,
        )

    character_count = len(content)
    if not is_within_character_limit(content, tool_limits=tool_limits):
        return FileReadResult(
            requested_path=requested_path,
            resolved_path=resolved_path,
            read_kind=read_kind,
            status="too_large",
            content=None,
            character_count=character_count,
            file_size_bytes=file_size_bytes,
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
    content_slice = content[normalized_start:truncated_end]
    truncated = truncated_end < character_count or truncated_end < normalized_end
    line_start, line_end = line_range_for_character_range(
        content,
        start_char=normalized_start,
        end_char=truncated_end,
    )
    return FileReadResult(
        requested_path=requested_path,
        resolved_path=resolved_path,
        read_kind=read_kind,
        status="ok",
        content=content_slice,
        truncated=truncated,
        content_char_count=len(content_slice),
        character_count=character_count,
        start_char=normalized_start,
        end_char=truncated_end,
        line_start=line_start,
        line_end=line_end,
        file_size_bytes=file_size_bytes,
        max_read_input_bytes=tool_limits.max_read_input_bytes,
        max_file_size_characters=tool_limits.max_file_size_characters,
        full_read_char_limit=full_read_char_limit,
        estimated_token_count=estimate_token_count(content_slice),
    )


def _build_attachment_read_result(
    *,
    requested_path: str,
    resolved_path: str,
    tool_limits: ToolLimits,
    cached_path: Path,
    start_char: int | None = None,
    end_char: int | None = None,
) -> FileReadResult:
    loaded_content = load_readable_content(
        cached_path,
        tool_limits=tool_limits,
        cache_root=_get_confluence_attachment_cache_root(),
    )
    status = loaded_content.status
    return _build_text_read_result(
        requested_path=requested_path,
        resolved_path=resolved_path,
        tool_limits=tool_limits,
        content=loaded_content.content,
        file_size_bytes=cached_path.stat().st_size,
        status=status,
        read_kind=loaded_content.read_kind,
        error_message=loaded_content.error_message,
        start_char=start_char,
        end_char=end_char,
    )


def _bitbucket_file_to_text(raw_content: object) -> tuple[str | None, int, str | None]:
    if isinstance(raw_content, bytes):
        return _normalize_remote_bytes(raw_content)
    if isinstance(raw_content, str):
        return _normalize_remote_text(raw_content)
    return None, 0, "Bitbucket file payload did not include readable content."


def _confluence_page_body(page: object) -> tuple[str, str | None, str | None]:
    body = _get_value(page, "body", {})
    if isinstance(body, dict):
        for key in ("storage", "view", "editor", "export_view"):
            value = _get_value(body, key, {})
            if isinstance(value, dict):
                content = _get_value(value, "value")
                representation = _get_value(value, "representation", key)
                if isinstance(content, str):
                    return content, cast(str | None, representation), None
    return "", None, "Confluence page did not include readable body content."


def _extract_bitbucket_path(change: object, key: str) -> str | None:
    path_payload = _get_value(change, key, {})
    if isinstance(path_payload, dict):
        path_value = path_payload.get("toString") or path_payload.get("path")
        if isinstance(path_value, str):
            return path_value
    return cast(str | None, _get_value(change, key))


def _extract_first_link_href(payload: object, key: str = "self") -> str | None:
    links = _get_value(payload, "links", {})
    raw_links = _get_value(links, key, [])
    if isinstance(raw_links, list) and raw_links:
        return cast(str | None, _get_value(raw_links[0], "href"))
    if isinstance(raw_links, dict):
        return cast(str | None, _get_value(raw_links, "href"))
    return None


def _sanitize_filename(filename: str) -> str:
    return _INVALID_FILENAME_RE.sub("_", filename).strip() or "attachment"


def _get_confluence_attachment_cache_root() -> Path:
    return Path(tempfile.gettempdir()) / "llm_tools" / "confluence_attachment_cache"


def _get_confluence_attachment_cache_paths(
    *,
    base_url: str,
    page_id: str,
    attachment: object,
) -> tuple[Path, Path]:
    attachment_id = str(_get_value(attachment, "id", "attachment"))
    title = str(_get_value(attachment, "title", attachment_id))
    cache_key = hashlib.sha256(
        f"{base_url}:{page_id}:{attachment_id}".encode()
    ).hexdigest()
    cache_dir = _get_confluence_attachment_cache_root() / cache_key
    return cache_dir / _sanitize_filename(title), cache_dir / "metadata.json"


def _attachment_cache_signature(attachment: object) -> dict[str, Any]:
    version = _get_value(_get_value(attachment, "version", {}), "number")
    size = _get_value(_get_value(attachment, "extensions", {}), "fileSize")
    title = _get_value(attachment, "title")
    return {"version": version, "size": size, "title": title}


def _cached_attachment_is_current(
    cached_path: Path,
    metadata_path: Path,
    *,
    attachment: object,
) -> bool:
    if not cached_path.exists() or not metadata_path.exists():
        return False
    expected = _attachment_cache_signature(attachment)
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(existing == expected)


def _download_confluence_attachment_bytes(
    client: Any,
    *,
    page_id: str,
    attachment: object,
) -> bytes:
    title = str(_get_value(attachment, "title", ""))
    download_payload = client.download_attachments_from_page(
        page_id,
        filename=title or None,
        to_memory=True,
    )
    if isinstance(download_payload, dict):
        if title in download_payload:
            file_obj = download_payload[title]
        elif download_payload:
            file_obj = next(iter(download_payload.values()))
        else:
            raise RuntimeError("Confluence attachment download returned no content.")
        getvalue = getattr(file_obj, "getvalue", None)
        if callable(getvalue):
            return cast(bytes, getvalue())
        read = getattr(file_obj, "read", None)
        if callable(read):
            return cast(bytes, read())
    raise RuntimeError("Confluence attachment download did not return file bytes.")


def _resolve_confluence_attachment(
    client: Any,
    *,
    page_id: str,
    attachment_id: str | None,
    attachment_filename: str | None,
) -> dict[str, Any]:
    if attachment_filename is not None:
        payload = client.get_attachments_from_content(
            page_id=page_id, filename=attachment_filename
        )
        matches = _extract_collection(payload)
        for item in matches:
            if _get_value(item, "title") == attachment_filename:
                return cast(dict[str, Any], item)
        raise FileNotFoundError(
            f"Confluence attachment '{attachment_filename}' was not found on page '{page_id}'."
        )

    payload = client.get_attachments_from_content(page_id=page_id, start=0, limit=200)
    matches = _extract_collection(payload)
    for item in matches:
        if str(_get_value(item, "id")) == str(attachment_id):
            return cast(dict[str, Any], item)
    raise FileNotFoundError(
        f"Confluence attachment '{attachment_id}' was not found on page '{page_id}'."
    )


def _ensure_cached_confluence_attachment(
    client: Any,
    *,
    base_url: str,
    page_id: str,
    attachment: object,
) -> Path:
    cached_path, metadata_path = _get_confluence_attachment_cache_paths(
        base_url=base_url,
        page_id=page_id,
        attachment=attachment,
    )
    if _cached_attachment_is_current(
        cached_path,
        metadata_path,
        attachment=attachment,
    ):
        return cached_path

    attachment_bytes = _download_confluence_attachment_bytes(
        client,
        page_id=page_id,
        attachment=attachment,
    )
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(attachment_bytes)
    metadata_path.write_text(
        json.dumps(_attachment_cache_signature(attachment)),
        encoding="utf-8",
    )
    return cached_path

"""Shared helpers for GitLab tool implementations."""

from __future__ import annotations

import base64
import hashlib
from typing import Any, cast

from llm_tools.tool_api import (
    RetryableToolExecutionError,
    SourceProvenanceRef,
    ToolExecutionContext,
)
from llm_tools.tools.filesystem.models import ToolLimits

GITLAB_ENV_KEYS = ("GITLAB_BASE_URL", "GITLAB_API_TOKEN")
REMOTE_TOOL_TIMEOUT_SECONDS = 30
REMOTE_COLLECTION_LIMIT = 100
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


def search_fetch_limit(limit: int) -> int:
    return limit + 1


def append_remote_source_provenance(
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


def get_tool_limits(context: ToolExecutionContext) -> ToolLimits:
    return ToolLimits.model_validate(context.metadata.get("tool_limits", {}))


def get_value(payload: object, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def normalize_remote_exception(exc: Exception) -> Exception:
    status_code = cast(
        int | None,
        get_value(
            exc,
            "status_code",
            get_value(get_value(exc, "response", {}), "status_code"),
        ),
    )
    exception_type = type(exc).__name__.lower()
    message = str(exc).lower()
    if status_code in RETRYABLE_STATUS_CODES or any(
        token in exception_type or token in message
        for token in ("timeout", "timedout", "connection", "temporarily unavailable")
    ):
        return RetryableToolExecutionError(str(exc))
    return exc


def get_gitlab_project(client: Any, project: str) -> Any:
    projects = getattr(client, "projects", None)
    if projects is None or not hasattr(projects, "get"):
        raise RuntimeError("Configured GitLab client does not support project reads.")
    try:
        return projects.get(project)
    except Exception as exc:
        raise normalize_remote_exception(exc) from exc


def search_project_code(
    project: Any,
    query: str,
    *,
    ref: str | None,
    limit: int,
) -> list[Any]:
    search = getattr(project, "search", None)
    if search is None:
        raise RuntimeError("Configured GitLab project does not support code search.")

    kwargs: dict[str, Any] = {"per_page": limit}
    if ref is not None:
        kwargs["ref"] = ref

    try:
        results = search("blobs", query, **kwargs)
    except TypeError:
        try:
            results = search(scope="blobs", search=query, **kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "Configured GitLab project does not support code search."
            ) from exc
        except Exception as exc:
            raise normalize_remote_exception(exc) from exc
    except Exception as exc:
        raise normalize_remote_exception(exc) from exc
    return list(cast(list[Any], results))


def get_project_file(project: Any, file_path: str, *, ref: str) -> Any:
    files = getattr(project, "files", None)
    if files is None or not hasattr(files, "get"):
        raise RuntimeError("Configured GitLab project does not support file reads.")
    try:
        return files.get(file_path=file_path, ref=ref)
    except Exception as exc:
        raise normalize_remote_exception(exc) from exc


def get_merge_request(project: Any, merge_request_iid: int) -> Any:
    merge_requests = getattr(project, "mergerequests", None)
    if merge_requests is None or not hasattr(merge_requests, "get"):
        raise RuntimeError(
            "Configured GitLab project does not support merge request reads."
        )
    try:
        return merge_requests.get(merge_request_iid)
    except Exception as exc:
        raise normalize_remote_exception(exc) from exc


def get_merge_request_commits(merge_request: Any) -> list[Any]:
    commits = getattr(merge_request, "commits", None)
    if commits is None:
        return []
    try:
        if callable(commits):
            return list(cast(list[Any], commits()))
        if hasattr(commits, "list"):
            return list(cast(list[Any], commits.list()))
    except Exception as exc:
        raise normalize_remote_exception(exc) from exc
    return []


def get_merge_request_changes(merge_request: Any) -> list[Any]:
    changes = getattr(merge_request, "changes", None)
    if changes is None:
        return []
    try:
        if callable(changes):
            payload = changes()
            if isinstance(payload, dict):
                return list(cast(list[Any], payload.get("changes", [])))
            return []
    except Exception as exc:
        raise normalize_remote_exception(exc) from exc
    if isinstance(changes, dict):
        return list(cast(list[Any], changes.get("changes", [])))
    return []


def decode_gitlab_file_content(file_obj: Any) -> tuple[str | None, int, str | None]:
    decoded = getattr(file_obj, "decode", None)
    if callable(decoded):
        payload = decoded()
        if isinstance(payload, str):
            if "\x00" in payload:
                return None, len(payload.encode("utf-8")), "Remote file is binary."
            return payload, len(payload.encode("utf-8")), None
        if isinstance(payload, bytes):
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                return None, len(payload), "Remote file is not UTF-8 text."
            if "\x00" in text:
                return None, len(payload), "Remote file is binary."
            return text, len(payload), None

    raw_content = get_value(file_obj, "content")
    encoding = get_value(file_obj, "encoding")
    if isinstance(raw_content, bytes):
        raw_bytes = raw_content
    elif isinstance(raw_content, str) and str(encoding).lower() == "base64":
        raw_bytes = base64.b64decode(raw_content)
    elif isinstance(raw_content, str):
        if "\x00" in raw_content:
            return None, len(raw_content.encode("utf-8")), "Remote file is binary."
        return raw_content, len(raw_content.encode("utf-8")), None
    else:
        return None, 0, "GitLab file payload did not include readable content."

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None, len(raw_bytes), "Remote file is not UTF-8 text."
    if "\x00" in text:
        return None, len(raw_bytes), "Remote file is binary."
    return text, len(raw_bytes), None


def normalize_project_name(project: Any, requested_project: str) -> str:
    return str(
        get_value(project, "path_with_namespace")
        or get_value(project, "path")
        or requested_project
    )

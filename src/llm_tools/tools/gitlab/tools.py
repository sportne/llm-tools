"""GitLab built-in tool implementations."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from llm_tools.tool_api import (
    RetryableToolExecutionError,
    SideEffectClass,
    SourceProvenanceRef,
    Tool,
    ToolExecutionContext,
    ToolRegistry,
    ToolSpec,
)
from llm_tools.tools.filesystem._content import (
    effective_full_read_char_limit,
    estimate_token_count,
    is_within_character_limit,
    normalize_range,
)
from llm_tools.tools.filesystem.models import FileReadResult, ToolLimits

_GITLAB_ENV_KEYS = ("GITLAB_BASE_URL", "GITLAB_API_TOKEN")
_REMOTE_TOOL_TIMEOUT_SECONDS = 30
_REMOTE_COLLECTION_LIMIT = 100
_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


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


def _get_gitlab_project(client: Any, project: str) -> Any:
    projects = getattr(client, "projects", None)
    if projects is None or not hasattr(projects, "get"):
        raise RuntimeError("Configured GitLab client does not support project reads.")
    try:
        return projects.get(project)
    except Exception as exc:
        raise _normalize_remote_exception(exc) from exc


def _search_project_code(
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
            raise _normalize_remote_exception(exc) from exc
    except Exception as exc:
        raise _normalize_remote_exception(exc) from exc
    return list(cast(list[Any], results))


def _get_project_file(project: Any, file_path: str, *, ref: str) -> Any:
    files = getattr(project, "files", None)
    if files is None or not hasattr(files, "get"):
        raise RuntimeError("Configured GitLab project does not support file reads.")
    try:
        return files.get(file_path=file_path, ref=ref)
    except Exception as exc:
        raise _normalize_remote_exception(exc) from exc


def _get_merge_request(project: Any, merge_request_iid: int) -> Any:
    merge_requests = getattr(project, "mergerequests", None)
    if merge_requests is None or not hasattr(merge_requests, "get"):
        raise RuntimeError(
            "Configured GitLab project does not support merge request reads."
        )
    try:
        return merge_requests.get(merge_request_iid)
    except Exception as exc:
        raise _normalize_remote_exception(exc) from exc


def _get_merge_request_commits(merge_request: Any) -> list[Any]:
    commits = getattr(merge_request, "commits", None)
    if commits is None:
        return []
    try:
        if callable(commits):
            return list(cast(list[Any], commits()))
        if hasattr(commits, "list"):
            return list(cast(list[Any], commits.list()))
    except Exception as exc:
        raise _normalize_remote_exception(exc) from exc
    return []


def _get_merge_request_changes(merge_request: Any) -> list[Any]:
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
        raise _normalize_remote_exception(exc) from exc
    if isinstance(changes, dict):
        return list(cast(list[Any], changes.get("changes", [])))
    return []


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


def _decode_gitlab_file_content(file_obj: Any) -> tuple[str | None, int, str | None]:
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

    raw_content = _get_value(file_obj, "content")
    encoding = _get_value(file_obj, "encoding")
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


def _normalize_project_name(project: Any, requested_project: str) -> str:
    return str(
        _get_value(project, "path_with_namespace")
        or _get_value(project, "path")
        or requested_project
    )


class SearchGitLabCodeInput(BaseModel):
    project: str
    query: str
    ref: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


class GitLabCodeSearchMatch(BaseModel):
    project: str
    path: str
    name: str
    ref: str | None = None
    start_line: int | None = None
    snippet: str | None = None


class SearchGitLabCodeOutput(BaseModel):
    project: str
    query: str
    ref: str | None = None
    matches: list[GitLabCodeSearchMatch] = Field(default_factory=list)
    truncated: bool = False


class SearchGitLabCodeTool(Tool[SearchGitLabCodeInput, SearchGitLabCodeOutput]):
    spec = ToolSpec(
        name="search_gitlab_code",
        description="Search one GitLab project for code matches.",
        tags=["gitlab", "search", "read", "code"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=_REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(_GITLAB_ENV_KEYS),
    )
    input_model = SearchGitLabCodeInput
    output_model = SearchGitLabCodeOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: SearchGitLabCodeInput
    ) -> SearchGitLabCodeOutput:
        client = context.services.require_gitlab().client
        project = _get_gitlab_project(client, args.project)
        project_name = _normalize_project_name(project, args.project)
        raw_matches = _search_project_code(
            project,
            args.query,
            ref=args.ref,
            limit=args.limit,
        )
        matches = [
            GitLabCodeSearchMatch(
                project=project_name,
                path=str(
                    _get_value(raw, "path")
                    or _get_value(raw, "filename")
                    or _get_value(raw, "file_path")
                    or ""
                ),
                name=Path(
                    str(
                        _get_value(raw, "path")
                        or _get_value(raw, "filename")
                        or _get_value(raw, "file_path")
                        or ""
                    )
                ).name,
                ref=cast(str | None, _get_value(raw, "ref", args.ref)),
                start_line=cast(
                    int | None,
                    _get_value(raw, "startline", _get_value(raw, "start_line")),
                ),
                snippet=cast(
                    str | None,
                    _get_value(raw, "data", _get_value(raw, "snippet")),
                ),
            )
            for raw in raw_matches[: args.limit]
        ]
        context.log(
            f"Ran GitLab code search for '{args.query}' in project '{args.project}'."
        )
        return SearchGitLabCodeOutput(
            project=project_name,
            query=args.query,
            ref=args.ref,
            matches=matches,
            truncated=len(raw_matches) > args.limit,
        )


class ReadGitLabFileInput(BaseModel):
    project: str
    file_path: str
    ref: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class ReadGitLabFileOutput(FileReadResult):
    project: str
    ref: str


class ReadGitLabFileTool(Tool[ReadGitLabFileInput, ReadGitLabFileOutput]):
    spec = ToolSpec(
        name="read_gitlab_file",
        description="Read one UTF-8 text file from a GitLab project.",
        tags=["gitlab", "read", "file"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=_REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(_GITLAB_ENV_KEYS),
    )
    input_model = ReadGitLabFileInput
    output_model = ReadGitLabFileOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: ReadGitLabFileInput
    ) -> ReadGitLabFileOutput:
        client = context.services.require_gitlab().client
        project = _get_gitlab_project(client, args.project)
        project_name = _normalize_project_name(project, args.project)
        tool_limits = _get_tool_limits(context)
        effective_ref = (
            args.ref
            or cast(str | None, _get_value(project, "default_branch"))
            or "HEAD"
        )
        full_read_char_limit = effective_full_read_char_limit(tool_limits)
        file_obj = _get_project_file(project, args.file_path, ref=effective_ref)
        content, file_size_bytes, error_message = _decode_gitlab_file_content(file_obj)
        resolved_path = f"{project_name}@{effective_ref}:{args.file_path}"

        if content is None:
            return ReadGitLabFileOutput(
                project=project_name,
                ref=effective_ref,
                requested_path=args.file_path,
                resolved_path=resolved_path,
                read_kind="unsupported",
                status="unsupported",
                content=None,
                file_size_bytes=file_size_bytes,
                max_read_input_bytes=tool_limits.max_read_input_bytes,
                max_file_size_characters=tool_limits.max_file_size_characters,
                full_read_char_limit=full_read_char_limit,
                error_message=error_message,
            )

        character_count = len(content)
        if not is_within_character_limit(content, tool_limits=tool_limits):
            return ReadGitLabFileOutput(
                project=project_name,
                ref=effective_ref,
                requested_path=args.file_path,
                resolved_path=resolved_path,
                read_kind="text",
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
            start_char=args.start_char,
            end_char=args.end_char,
            character_count=character_count,
        )
        truncated_end = min(normalized_end, normalized_start + full_read_char_limit)
        content_slice = content[normalized_start:truncated_end]
        truncated = truncated_end < character_count or truncated_end < normalized_end

        context.log(
            f"Read GitLab file '{args.file_path}' from project '{args.project}'."
        )
        context.add_artifact(resolved_path)
        _append_remote_source_provenance(
            context,
            source_kind="gitlab_file",
            source_id=resolved_path,
        )
        return ReadGitLabFileOutput(
            project=project_name,
            ref=effective_ref,
            requested_path=args.file_path,
            resolved_path=resolved_path,
            read_kind="text",
            status="ok",
            content=content_slice,
            truncated=truncated,
            content_char_count=len(content_slice),
            character_count=character_count,
            start_char=normalized_start,
            end_char=truncated_end,
            file_size_bytes=file_size_bytes,
            max_read_input_bytes=tool_limits.max_read_input_bytes,
            max_file_size_characters=tool_limits.max_file_size_characters,
            full_read_char_limit=full_read_char_limit,
            estimated_token_count=estimate_token_count(content_slice),
        )


class ReadGitLabMergeRequestInput(BaseModel):
    project: str
    merge_request_iid: int = Field(ge=1)
    commit_limit: int = Field(default=20, ge=1, le=_REMOTE_COLLECTION_LIMIT)
    change_limit: int = Field(default=20, ge=1, le=_REMOTE_COLLECTION_LIMIT)


class GitLabMergeRequestCommit(BaseModel):
    id: str | None = None
    short_id: str | None = None
    title: str | None = None
    author_name: str | None = None


class GitLabMergeRequestChange(BaseModel):
    old_path: str | None = None
    new_path: str | None = None
    new_file: bool = False
    renamed_file: bool = False
    deleted_file: bool = False
    diff_excerpt: str | None = None


class ReadGitLabMergeRequestOutput(BaseModel):
    project: str
    merge_request_iid: int
    title: str | None = None
    description: str | None = None
    state: str | None = None
    author: str | None = None
    source_branch: str | None = None
    target_branch: str | None = None
    web_url: str | None = None
    commits: list[GitLabMergeRequestCommit] = Field(default_factory=list)
    commits_truncated: bool = False
    changed_files: list[GitLabMergeRequestChange] = Field(default_factory=list)
    changed_files_truncated: bool = False


class ReadGitLabMergeRequestTool(
    Tool[ReadGitLabMergeRequestInput, ReadGitLabMergeRequestOutput]
):
    spec = ToolSpec(
        name="read_gitlab_merge_request",
        description="Read one GitLab merge request by IID.",
        tags=["gitlab", "read", "merge_request"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=_REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(_GITLAB_ENV_KEYS),
    )
    input_model = ReadGitLabMergeRequestInput
    output_model = ReadGitLabMergeRequestOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: ReadGitLabMergeRequestInput
    ) -> ReadGitLabMergeRequestOutput:
        client = context.services.require_gitlab().client
        project = _get_gitlab_project(client, args.project)
        project_name = _normalize_project_name(project, args.project)
        merge_request = _get_merge_request(project, args.merge_request_iid)
        raw_commits = _get_merge_request_commits(merge_request)
        raw_changes = _get_merge_request_changes(merge_request)

        commits = [
            GitLabMergeRequestCommit(
                id=cast(str | None, _get_value(commit, "id")),
                short_id=cast(str | None, _get_value(commit, "short_id")),
                title=cast(
                    str | None,
                    _get_value(commit, "title", _get_value(commit, "message")),
                ),
                author_name=cast(str | None, _get_value(commit, "author_name")),
            )
            for commit in raw_commits[: args.commit_limit]
        ]
        changed_files = [
            GitLabMergeRequestChange(
                old_path=cast(str | None, _get_value(change, "old_path")),
                new_path=cast(str | None, _get_value(change, "new_path")),
                new_file=bool(_get_value(change, "new_file", False)),
                renamed_file=bool(_get_value(change, "renamed_file", False)),
                deleted_file=bool(_get_value(change, "deleted_file", False)),
                diff_excerpt=(cast(str | None, _get_value(change, "diff")) or "")[:400]
                or None,
            )
            for change in raw_changes[: args.change_limit]
        ]

        context.log(
            "Read GitLab merge request "
            f"'{args.merge_request_iid}' from project '{args.project}'."
        )
        return ReadGitLabMergeRequestOutput(
            project=project_name,
            merge_request_iid=args.merge_request_iid,
            title=cast(str | None, _get_value(merge_request, "title")),
            description=cast(str | None, _get_value(merge_request, "description")),
            state=cast(str | None, _get_value(merge_request, "state")),
            author=cast(
                str | None,
                _get_value(
                    _get_value(merge_request, "author", {}),
                    "name",
                    _get_value(
                        _get_value(merge_request, "author", {}),
                        "username",
                    ),
                ),
            ),
            source_branch=cast(str | None, _get_value(merge_request, "source_branch")),
            target_branch=cast(str | None, _get_value(merge_request, "target_branch")),
            web_url=cast(str | None, _get_value(merge_request, "web_url")),
            commits=commits,
            commits_truncated=len(raw_commits) > args.commit_limit,
            changed_files=changed_files,
            changed_files_truncated=len(raw_changes) > args.change_limit,
        )


def register_gitlab_tools(registry: ToolRegistry) -> None:
    """Register the built-in GitLab tool set."""
    registry.register(SearchGitLabCodeTool())
    registry.register(ReadGitLabFileTool())
    registry.register(ReadGitLabMergeRequestTool())

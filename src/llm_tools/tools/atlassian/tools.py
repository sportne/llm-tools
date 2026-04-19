"""Atlassian built-in tool implementations."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

from pydantic import BaseModel, Field, model_validator

from llm_tools.tool_api import (
    SideEffectClass,
    SourceProvenanceRef,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolSpec,
)
from llm_tools.tools.filesystem._content import (
    effective_full_read_char_limit,
    estimate_token_count,
    is_within_character_limit,
    load_readable_content,
    normalize_range,
)
from llm_tools.tools.filesystem.models import FileReadKind, FileReadResult, ToolLimits

_JIRA_ENV_KEYS = ("JIRA_BASE_URL", "JIRA_USERNAME", "JIRA_API_TOKEN")
_BITBUCKET_ENV_KEYS = (
    "BITBUCKET_BASE_URL",
    "BITBUCKET_USERNAME",
    "BITBUCKET_API_TOKEN",
)
_CONFLUENCE_ENV_KEYS = (
    "CONFLUENCE_BASE_URL",
    "CONFLUENCE_USERNAME",
    "CONFLUENCE_API_TOKEN",
)
_INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


def _append_remote_source_provenance(
    context: ToolContext,
    *,
    source_kind: str,
    source_id: str,
) -> None:
    context.source_provenance.append(
        SourceProvenanceRef(
            source_kind=source_kind,
            source_id=source_id,
            content_hash=hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
            whole_source_reproduction_allowed=True,
            metadata={"source_id": source_id},
        )
    )


def _get_required_env(context: ToolContext, key: str, label: str) -> str:
    value = context.env.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required {label} credential '{key}'.")
    return value


def _get_tool_limits(context: ToolContext) -> ToolLimits:
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


def _build_jira_client(context: ToolContext) -> Any:
    base_url = _get_required_env(context, "JIRA_BASE_URL", "Jira")
    username = _get_required_env(context, "JIRA_USERNAME", "Jira")
    api_token = _get_required_env(context, "JIRA_API_TOKEN", "Jira")

    from atlassian import Jira

    return Jira(
        url=base_url,
        username=username,
        password=api_token,
    )


def _build_bitbucket_client(context: ToolContext) -> Any:
    base_url = _get_required_env(context, "BITBUCKET_BASE_URL", "Bitbucket")
    username = _get_required_env(context, "BITBUCKET_USERNAME", "Bitbucket")
    api_token = _get_required_env(context, "BITBUCKET_API_TOKEN", "Bitbucket")

    from atlassian import Bitbucket

    bitbucket_cls = cast(Any, Bitbucket)
    return bitbucket_cls(
        url=base_url,
        username=username,
        password=api_token,
    )


def _build_confluence_client(context: ToolContext) -> Any:
    base_url = _get_required_env(context, "CONFLUENCE_BASE_URL", "Confluence")
    username = _get_required_env(context, "CONFLUENCE_USERNAME", "Confluence")
    api_token = _get_required_env(context, "CONFLUENCE_API_TOKEN", "Confluence")

    from atlassian import Confluence

    confluence_cls = cast(Any, Confluence)
    return confluence_cls(
        url=base_url,
        username=username,
        password=api_token,
    )


def _absolute_url(base_url: str, raw_url: str | None) -> str | None:
    if raw_url is None or raw_url == "":
        return None
    return urljoin(base_url.rstrip("/") + "/", raw_url)


def _extract_issue_fields(issue: dict[str, Any]) -> dict[str, Any]:
    fields = cast(dict[str, Any], issue.get("fields") or {})
    issue_type = fields.get("issuetype") or {}
    status = fields.get("status") or {}
    assignee = fields.get("assignee") or {}
    description = fields.get("description")
    return {
        "key": issue.get("key", ""),
        "summary": fields.get("summary"),
        "description": description,
        "status": status.get("name"),
        "issue_type": issue_type.get("name"),
        "assignee": assignee.get("displayName"),
        "raw_fields": fields,
    }


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


class JiraIssueSummary(BaseModel):
    key: str
    summary: str | None = None
    status: str | None = None
    issue_type: str | None = None
    assignee: str | None = None


class SearchJiraInput(BaseModel):
    jql: str
    limit: int = Field(default=20, ge=1)


class SearchJiraOutput(BaseModel):
    issues: list[JiraIssueSummary] = Field(default_factory=list)


class SearchJiraTool(Tool[SearchJiraInput, SearchJiraOutput]):
    spec = ToolSpec(
        name="search_jira",
        description="Search Jira issues with JQL.",
        tags=["atlassian", "jira", "search", "read"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        required_secrets=list(_JIRA_ENV_KEYS),
    )
    input_model = SearchJiraInput
    output_model = SearchJiraOutput

    def invoke(self, context: ToolContext, args: SearchJiraInput) -> SearchJiraOutput:
        client = _build_jira_client(context)
        if hasattr(client, "enhanced_jql"):
            payload = client.enhanced_jql(args.jql, limit=args.limit)
        elif hasattr(client, "jql"):
            payload = client.jql(args.jql, limit=args.limit)
        else:
            raise RuntimeError("Configured Jira client does not support JQL search.")

        issues = []
        for issue in cast(list[dict[str, Any]], payload.get("issues", [])):
            normalized = _extract_issue_fields(issue)
            issues.append(
                JiraIssueSummary(
                    key=normalized["key"],
                    summary=normalized["summary"],
                    status=normalized["status"],
                    issue_type=normalized["issue_type"],
                    assignee=normalized["assignee"],
                )
            )

        context.logs.append(f"Ran Jira search for JQL '{args.jql}'.")
        return SearchJiraOutput(issues=issues)


class ReadJiraIssueInput(BaseModel):
    issue_key: str


class ReadJiraIssueOutput(BaseModel):
    key: str
    summary: str | None = None
    description: Any = None
    status: str | None = None
    issue_type: str | None = None
    assignee: str | None = None
    raw_fields: dict[str, Any] = Field(default_factory=dict)


class ReadJiraIssueTool(Tool[ReadJiraIssueInput, ReadJiraIssueOutput]):
    spec = ToolSpec(
        name="read_jira_issue",
        description="Read one Jira issue by key.",
        tags=["atlassian", "jira", "read"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        required_secrets=list(_JIRA_ENV_KEYS),
    )
    input_model = ReadJiraIssueInput
    output_model = ReadJiraIssueOutput

    def invoke(
        self, context: ToolContext, args: ReadJiraIssueInput
    ) -> ReadJiraIssueOutput:
        client = _build_jira_client(context)
        if hasattr(client, "issue"):
            issue = client.issue(args.issue_key)
        elif hasattr(client, "get_issue"):
            issue = client.get_issue(args.issue_key)
        else:
            raise RuntimeError("Configured Jira client does not support issue reads.")

        normalized = _extract_issue_fields(cast(dict[str, Any], issue))
        context.logs.append(f"Read Jira issue '{args.issue_key}'.")
        return ReadJiraIssueOutput(**normalized)


class SearchBitbucketCodeInput(BaseModel):
    project_key: str
    query: str
    limit: int = Field(default=20, ge=1)


class BitbucketCodeMatch(BaseModel):
    repository_slug: str | None = None
    path: str
    line_number: int | None = None
    snippet: str | None = None


class SearchBitbucketCodeOutput(BaseModel):
    project_key: str
    query: str
    matches: list[BitbucketCodeMatch] = Field(default_factory=list)


class SearchBitbucketCodeTool(
    Tool[SearchBitbucketCodeInput, SearchBitbucketCodeOutput]
):
    spec = ToolSpec(
        name="search_bitbucket_code",
        description="Search Bitbucket Server/DC code within one project.",
        tags=["atlassian", "bitbucket", "search", "read"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        required_secrets=list(_BITBUCKET_ENV_KEYS),
    )
    input_model = SearchBitbucketCodeInput
    output_model = SearchBitbucketCodeOutput

    def invoke(
        self, context: ToolContext, args: SearchBitbucketCodeInput
    ) -> SearchBitbucketCodeOutput:
        client = _build_bitbucket_client(context)
        payload = client.search_code(args.project_key, args.query, limit=args.limit)
        matches = []
        for item in _extract_collection(payload):
            matches.append(
                BitbucketCodeMatch(
                    repository_slug=cast(
                        str | None,
                        _get_value(
                            _get_value(item, "repository", {}),
                            "slug",
                            _get_value(item, "repository_slug"),
                        ),
                    ),
                    path=str(
                        _get_value(item, "path")
                        or _get_value(item, "file")
                        or _get_value(item, "filename")
                        or ""
                    ),
                    line_number=cast(
                        int | None,
                        _get_value(item, "line", _get_value(item, "lineNumber")),
                    ),
                    snippet=cast(
                        str | None,
                        _get_value(item, "content", _get_value(item, "snippet")),
                    ),
                )
            )

        context.logs.append(
            f"Ran Bitbucket code search for '{args.query}' in project '{args.project_key}'."
        )
        return SearchBitbucketCodeOutput(
            project_key=args.project_key,
            query=args.query,
            matches=matches,
        )


class ReadBitbucketFileInput(BaseModel):
    project_key: str
    repository_slug: str
    path: str
    ref: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class ReadBitbucketFileOutput(FileReadResult):
    project_key: str
    repository_slug: str
    ref: str


class ReadBitbucketFileTool(Tool[ReadBitbucketFileInput, ReadBitbucketFileOutput]):
    spec = ToolSpec(
        name="read_bitbucket_file",
        description="Read one UTF-8 text file from Bitbucket Server/DC.",
        tags=["atlassian", "bitbucket", "read", "file"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        required_secrets=list(_BITBUCKET_ENV_KEYS),
    )
    input_model = ReadBitbucketFileInput
    output_model = ReadBitbucketFileOutput

    def invoke(
        self, context: ToolContext, args: ReadBitbucketFileInput
    ) -> ReadBitbucketFileOutput:
        client = _build_bitbucket_client(context)
        tool_limits = _get_tool_limits(context)
        effective_ref = args.ref or "HEAD"
        raw_content = client.get_content_of_file(
            args.project_key,
            args.repository_slug,
            args.path,
            at=effective_ref,
        )
        content, file_size_bytes, error_message = _bitbucket_file_to_text(raw_content)
        requested_path = args.path
        resolved_path = (
            f"{args.project_key}/{args.repository_slug}@{effective_ref}:{args.path}"
        )
        result = _build_text_read_result(
            requested_path=requested_path,
            resolved_path=resolved_path,
            tool_limits=tool_limits,
            content=content,
            file_size_bytes=file_size_bytes,
            status="ok" if content is not None else "unsupported",
            read_kind="text" if content is not None else "unsupported",
            error_message=error_message,
            start_char=args.start_char,
            end_char=args.end_char,
        )
        context.logs.append(
            "Read Bitbucket file "
            f"'{args.path}' from '{args.project_key}/{args.repository_slug}'."
        )
        context.artifacts.append(resolved_path)
        _append_remote_source_provenance(
            context,
            source_kind="bitbucket_file",
            source_id=resolved_path,
        )
        return ReadBitbucketFileOutput(
            project_key=args.project_key,
            repository_slug=args.repository_slug,
            ref=effective_ref,
            **result.model_dump(mode="json"),
        )


class BitbucketPullRequestCommit(BaseModel):
    id: str | None = None
    display_id: str | None = None
    message: str | None = None
    author_name: str | None = None


class BitbucketPullRequestChange(BaseModel):
    old_path: str | None = None
    new_path: str | None = None
    change_type: str | None = None
    executable: bool | None = None


class ReadBitbucketPullRequestInput(BaseModel):
    project_key: str
    repository_slug: str
    pull_request_id: int = Field(ge=1)


class ReadBitbucketPullRequestOutput(BaseModel):
    project_key: str
    repository_slug: str
    pull_request_id: int
    title: str | None = None
    description: str | None = None
    state: str | None = None
    author: str | None = None
    source_branch: str | None = None
    target_branch: str | None = None
    web_url: str | None = None
    commits: list[BitbucketPullRequestCommit] = Field(default_factory=list)
    changed_files: list[BitbucketPullRequestChange] = Field(default_factory=list)


class ReadBitbucketPullRequestTool(
    Tool[ReadBitbucketPullRequestInput, ReadBitbucketPullRequestOutput]
):
    spec = ToolSpec(
        name="read_bitbucket_pull_request",
        description="Read one Bitbucket Server/DC pull request.",
        tags=["atlassian", "bitbucket", "read", "pull_request"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        required_secrets=list(_BITBUCKET_ENV_KEYS),
    )
    input_model = ReadBitbucketPullRequestInput
    output_model = ReadBitbucketPullRequestOutput

    def invoke(
        self, context: ToolContext, args: ReadBitbucketPullRequestInput
    ) -> ReadBitbucketPullRequestOutput:
        client = _build_bitbucket_client(context)
        pull_request = client.get_pull_request(
            args.project_key,
            args.repository_slug,
            args.pull_request_id,
        )
        commits_payload = client.get_pull_requests_commits(
            args.project_key,
            args.repository_slug,
            args.pull_request_id,
        )
        changes_payload = client.get_pull_requests_changes(
            args.project_key,
            args.repository_slug,
            args.pull_request_id,
        )
        commits = [
            BitbucketPullRequestCommit(
                id=cast(str | None, _get_value(commit, "id")),
                display_id=cast(str | None, _get_value(commit, "displayId")),
                message=cast(str | None, _get_value(commit, "message")),
                author_name=cast(
                    str | None,
                    _get_value(
                        _get_value(commit, "author", {}),
                        "name",
                        _get_value(commit, "authorName"),
                    ),
                ),
            )
            for commit in _extract_collection(commits_payload)
        ]
        changed_files = [
            BitbucketPullRequestChange(
                old_path=_extract_bitbucket_path(change, "srcPath"),
                new_path=_extract_bitbucket_path(change, "path"),
                change_type=cast(str | None, _get_value(change, "type")),
                executable=cast(bool | None, _get_value(change, "executable")),
            )
            for change in _extract_collection(changes_payload)
        ]
        context.logs.append(
            "Read Bitbucket pull request "
            f"'{args.pull_request_id}' from '{args.project_key}/{args.repository_slug}'."
        )
        return ReadBitbucketPullRequestOutput(
            project_key=args.project_key,
            repository_slug=args.repository_slug,
            pull_request_id=args.pull_request_id,
            title=cast(str | None, _get_value(pull_request, "title")),
            description=cast(str | None, _get_value(pull_request, "description")),
            state=cast(str | None, _get_value(pull_request, "state")),
            author=cast(
                str | None,
                _get_value(
                    _get_value(_get_value(pull_request, "author", {}), "user", {}),
                    "displayName",
                ),
            ),
            source_branch=cast(
                str | None,
                _get_value(_get_value(pull_request, "fromRef", {}), "displayId"),
            ),
            target_branch=cast(
                str | None,
                _get_value(_get_value(pull_request, "toRef", {}), "displayId"),
            ),
            web_url=_extract_first_link_href(pull_request),
            commits=commits,
            changed_files=changed_files,
        )


class SearchConfluenceInput(BaseModel):
    cql: str
    limit: int = Field(default=20, ge=1)


class ConfluenceSearchMatch(BaseModel):
    content_id: str
    title: str | None = None
    content_type: str | None = None
    space_key: str | None = None
    excerpt: str | None = None
    web_url: str | None = None


class SearchConfluenceOutput(BaseModel):
    cql: str
    matches: list[ConfluenceSearchMatch] = Field(default_factory=list)


class SearchConfluenceTool(Tool[SearchConfluenceInput, SearchConfluenceOutput]):
    spec = ToolSpec(
        name="search_confluence",
        description="Search Confluence with CQL.",
        tags=["atlassian", "confluence", "search", "read"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        required_secrets=list(_CONFLUENCE_ENV_KEYS),
    )
    input_model = SearchConfluenceInput
    output_model = SearchConfluenceOutput

    def invoke(
        self, context: ToolContext, args: SearchConfluenceInput
    ) -> SearchConfluenceOutput:
        client = _build_confluence_client(context)
        base_url = _get_required_env(context, "CONFLUENCE_BASE_URL", "Confluence")
        payload = client.cql(args.cql, limit=args.limit, excerpt="highlight")
        matches = []
        for item in _extract_collection(payload):
            content = _get_value(item, "content", item)
            links = _get_value(content, "_links", _get_value(item, "_links", {}))
            matches.append(
                ConfluenceSearchMatch(
                    content_id=str(
                        _get_value(content, "id", _get_value(item, "id", ""))
                    ),
                    title=cast(
                        str | None,
                        _get_value(content, "title", _get_value(item, "title")),
                    ),
                    content_type=cast(
                        str | None,
                        _get_value(content, "type", _get_value(item, "type")),
                    ),
                    space_key=cast(
                        str | None,
                        _get_value(_get_value(content, "space", {}), "key"),
                    ),
                    excerpt=cast(str | None, _get_value(item, "excerpt")),
                    web_url=_absolute_url(
                        base_url,
                        cast(
                            str | None,
                            _get_value(links, "webui", _get_value(links, "tinyui")),
                        ),
                    ),
                )
            )
        context.logs.append(f"Ran Confluence search for CQL '{args.cql}'.")
        return SearchConfluenceOutput(cql=args.cql, matches=matches)


class ReadConfluenceContentInput(BaseModel):
    page_id: str
    attachment_id: str | None = None
    attachment_filename: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_attachment_selector(self) -> ReadConfluenceContentInput:
        if self.attachment_id is not None and self.attachment_filename is not None:
            raise ValueError(
                "provide at most one of attachment_id or attachment_filename"
            )
        return self


class ReadConfluenceContentOutput(FileReadResult):
    page_id: str
    mode: str
    title: str | None = None
    space_key: str | None = None
    web_url: str | None = None
    attachment_id: str | None = None
    attachment_filename: str | None = None
    representation: str | None = None


class ReadConfluenceContentTool(
    Tool[ReadConfluenceContentInput, ReadConfluenceContentOutput]
):
    spec = ToolSpec(
        name="read_confluence_content",
        description="Read a Confluence page or one attachment from a page.",
        tags=["atlassian", "confluence", "read"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        requires_filesystem=True,
        required_secrets=list(_CONFLUENCE_ENV_KEYS),
    )
    input_model = ReadConfluenceContentInput
    output_model = ReadConfluenceContentOutput

    def invoke(
        self, context: ToolContext, args: ReadConfluenceContentInput
    ) -> ReadConfluenceContentOutput:
        client = _build_confluence_client(context)
        base_url = _get_required_env(context, "CONFLUENCE_BASE_URL", "Confluence")
        tool_limits = _get_tool_limits(context)
        page = client.get_page_by_id(args.page_id, expand="body.storage,space,_links")
        page_links = _get_value(page, "_links", {})
        page_web_url = _absolute_url(
            base_url,
            cast(
                str | None,
                _get_value(page_links, "webui", _get_value(page_links, "tinyui")),
            ),
        )
        page_title = cast(str | None, _get_value(page, "title"))
        space_key = cast(str | None, _get_value(_get_value(page, "space", {}), "key"))

        if args.attachment_id is None and args.attachment_filename is None:
            body_content, representation, body_error = _confluence_page_body(page)
            result = _build_text_read_result(
                requested_path=f"page:{args.page_id}",
                resolved_path=page_web_url or f"confluence:page:{args.page_id}",
                tool_limits=tool_limits,
                content=body_content if body_error is None else None,
                file_size_bytes=len(body_content.encode()),
                status="ok" if body_error is None else "error",
                read_kind="text",
                error_message=body_error,
                start_char=args.start_char,
                end_char=args.end_char,
            )
            page_source_id = page_web_url or f"confluence:page:{args.page_id}"
            context.logs.append(f"Read Confluence page '{args.page_id}'.")
            context.artifacts.append(page_source_id)
            _append_remote_source_provenance(
                context,
                source_kind="confluence_page",
                source_id=page_source_id,
            )
            return ReadConfluenceContentOutput(
                page_id=args.page_id,
                mode="page",
                title=page_title,
                space_key=space_key,
                web_url=page_web_url,
                representation=representation,
                **result.model_dump(mode="json"),
            )

        attachment = _resolve_confluence_attachment(
            client,
            page_id=args.page_id,
            attachment_id=args.attachment_id,
            attachment_filename=args.attachment_filename,
        )
        cached_path = _ensure_cached_confluence_attachment(
            client,
            base_url=base_url,
            page_id=args.page_id,
            attachment=attachment,
        )
        attachment_title = cast(str | None, _get_value(attachment, "title"))
        attachment_links = _get_value(attachment, "_links", {})
        attachment_url = _absolute_url(
            base_url,
            cast(str | None, _get_value(attachment_links, "download")),
        )
        result = _build_attachment_read_result(
            requested_path=(
                f"page:{args.page_id}/attachment:{args.attachment_filename or args.attachment_id}"
            ),
            resolved_path=attachment_url or cached_path.name,
            tool_limits=tool_limits,
            cached_path=cached_path,
            start_char=args.start_char,
            end_char=args.end_char,
        )
        attachment_source_id = (
            attachment_url
            or f"confluence:page:{args.page_id}:attachment:{attachment_title}"
        )
        context.logs.append(
            "Read Confluence attachment "
            f"'{attachment_title or args.attachment_id}' from page '{args.page_id}'."
        )
        context.artifacts.append(attachment_source_id)
        _append_remote_source_provenance(
            context,
            source_kind="confluence_attachment",
            source_id=attachment_source_id,
        )
        return ReadConfluenceContentOutput(
            page_id=args.page_id,
            mode="attachment",
            title=page_title,
            space_key=space_key,
            web_url=page_web_url,
            attachment_id=cast(str | None, _get_value(attachment, "id")),
            attachment_filename=attachment_title,
            representation=None,
            **result.model_dump(mode="json"),
        )


def register_atlassian_tools(registry: ToolRegistry) -> None:
    """Register the built-in Atlassian tool set."""
    registry.register(SearchJiraTool())
    registry.register(ReadJiraIssueTool())
    registry.register(SearchBitbucketCodeTool())
    registry.register(ReadBitbucketFileTool())
    registry.register(ReadBitbucketPullRequestTool())
    registry.register(SearchConfluenceTool())
    registry.register(ReadConfluenceContentTool())

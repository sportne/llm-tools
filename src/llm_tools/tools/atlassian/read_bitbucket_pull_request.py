"""Bitbucket pull-request read tool."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.atlassian._shared import (
    _BITBUCKET_ENV_KEYS,
    _REMOTE_COLLECTION_LIMIT,
    _REMOTE_TOOL_TIMEOUT_SECONDS,
    _extract_bitbucket_path,
    _extract_collection,
    _extract_first_link_href,
    _get_value,
    _normalize_remote_exception,
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
    commit_limit: int = Field(default=20, ge=1, le=_REMOTE_COLLECTION_LIMIT)
    change_limit: int = Field(default=20, ge=1, le=_REMOTE_COLLECTION_LIMIT)


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
    commits_truncated: bool = False
    changed_files: list[BitbucketPullRequestChange] = Field(default_factory=list)
    changed_files_truncated: bool = False


class ReadBitbucketPullRequestTool(
    Tool[ReadBitbucketPullRequestInput, ReadBitbucketPullRequestOutput]
):
    spec = ToolSpec(
        name="read_bitbucket_pull_request",
        description="Read one Bitbucket Server/DC pull request.",
        tags=["atlassian", "bitbucket", "read", "pull_request"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=_REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(_BITBUCKET_ENV_KEYS),
    )
    input_model = ReadBitbucketPullRequestInput
    output_model = ReadBitbucketPullRequestOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: ReadBitbucketPullRequestInput
    ) -> ReadBitbucketPullRequestOutput:
        client = context.services.require_bitbucket().client
        try:
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
        except Exception as exc:
            raise _normalize_remote_exception(exc) from exc
        raw_commits = _extract_collection(commits_payload)
        raw_changes = _extract_collection(changes_payload)
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
            for commit in raw_commits[: args.commit_limit]
        ]
        changed_files = [
            BitbucketPullRequestChange(
                old_path=_extract_bitbucket_path(change, "srcPath"),
                new_path=_extract_bitbucket_path(change, "path"),
                change_type=cast(str | None, _get_value(change, "type")),
                executable=cast(bool | None, _get_value(change, "executable")),
            )
            for change in raw_changes[: args.change_limit]
        ]
        context.log(
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
            commits_truncated=len(raw_commits) > args.commit_limit,
            changed_files=changed_files,
            changed_files_truncated=len(raw_changes) > args.change_limit,
        )

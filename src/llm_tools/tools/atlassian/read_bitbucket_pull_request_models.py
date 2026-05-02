"""Bitbucket pull-request read tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.atlassian._shared import (
    _REMOTE_COLLECTION_LIMIT,
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


__all__ = [
    "BitbucketPullRequestChange",
    "BitbucketPullRequestCommit",
    "ReadBitbucketPullRequestInput",
    "ReadBitbucketPullRequestOutput",
]

"""GitLab merge-request read tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.gitlab._shared import (
    REMOTE_COLLECTION_LIMIT,
)


class ReadGitLabMergeRequestInput(BaseModel):
    project: str
    merge_request_iid: int = Field(ge=1)
    commit_limit: int = Field(default=20, ge=1, le=REMOTE_COLLECTION_LIMIT)
    change_limit: int = Field(default=20, ge=1, le=REMOTE_COLLECTION_LIMIT)


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


__all__ = [
    "GitLabMergeRequestChange",
    "GitLabMergeRequestCommit",
    "ReadGitLabMergeRequestInput",
    "ReadGitLabMergeRequestOutput",
]

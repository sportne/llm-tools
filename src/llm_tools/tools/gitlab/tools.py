"""GitLab built-in tool registration and compatibility exports."""

from __future__ import annotations

from llm_tools.tool_api import ToolRegistry
from llm_tools.tools.gitlab._shared import (
    GITLAB_ENV_KEYS,
    REMOTE_COLLECTION_LIMIT,
    REMOTE_TOOL_TIMEOUT_SECONDS,
    append_remote_source_provenance,
    decode_gitlab_file_content,
    get_gitlab_project,
    get_merge_request,
    get_merge_request_changes,
    get_merge_request_commits,
    get_project_file,
    get_tool_limits,
    get_value,
    normalize_project_name,
    normalize_remote_exception,
    search_fetch_limit,
    search_project_code,
)
from llm_tools.tools.gitlab.read_gitlab_file import (
    ReadGitLabFileInput,
    ReadGitLabFileOutput,
    ReadGitLabFileTool,
)
from llm_tools.tools.gitlab.read_gitlab_merge_request import (
    GitLabMergeRequestChange,
    GitLabMergeRequestCommit,
    ReadGitLabMergeRequestInput,
    ReadGitLabMergeRequestOutput,
    ReadGitLabMergeRequestTool,
)
from llm_tools.tools.gitlab.search_gitlab_code import (
    GitLabCodeSearchMatch,
    SearchGitLabCodeInput,
    SearchGitLabCodeOutput,
    SearchGitLabCodeTool,
)

_GITLAB_ENV_KEYS = GITLAB_ENV_KEYS
_REMOTE_TOOL_TIMEOUT_SECONDS = REMOTE_TOOL_TIMEOUT_SECONDS
_REMOTE_COLLECTION_LIMIT = REMOTE_COLLECTION_LIMIT
_search_fetch_limit = search_fetch_limit
_append_remote_source_provenance = append_remote_source_provenance
_get_tool_limits = get_tool_limits
_get_value = get_value
_get_gitlab_project = get_gitlab_project
_search_project_code = search_project_code
_get_project_file = get_project_file
_get_merge_request = get_merge_request
_get_merge_request_commits = get_merge_request_commits
_get_merge_request_changes = get_merge_request_changes
_normalize_remote_exception = normalize_remote_exception
_decode_gitlab_file_content = decode_gitlab_file_content
_normalize_project_name = normalize_project_name


def register_gitlab_tools(registry: ToolRegistry) -> None:
    """Register the built-in GitLab tool set."""
    registry.register(SearchGitLabCodeTool())
    registry.register(ReadGitLabFileTool())
    registry.register(ReadGitLabMergeRequestTool())


__all__ = [
    "GitLabCodeSearchMatch",
    "GitLabMergeRequestChange",
    "GitLabMergeRequestCommit",
    "ReadGitLabFileInput",
    "ReadGitLabFileOutput",
    "ReadGitLabFileTool",
    "ReadGitLabMergeRequestInput",
    "ReadGitLabMergeRequestOutput",
    "ReadGitLabMergeRequestTool",
    "SearchGitLabCodeInput",
    "SearchGitLabCodeOutput",
    "SearchGitLabCodeTool",
    "register_gitlab_tools",
    "_GITLAB_ENV_KEYS",
    "_REMOTE_TOOL_TIMEOUT_SECONDS",
    "_REMOTE_COLLECTION_LIMIT",
    "_search_fetch_limit",
    "_append_remote_source_provenance",
    "_get_tool_limits",
    "_get_value",
    "_get_gitlab_project",
    "_search_project_code",
    "_get_project_file",
    "_get_merge_request",
    "_get_merge_request_commits",
    "_get_merge_request_changes",
    "_normalize_remote_exception",
    "_decode_gitlab_file_content",
    "_normalize_project_name",
]

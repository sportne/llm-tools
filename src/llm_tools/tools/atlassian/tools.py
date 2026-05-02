"""Atlassian built-in tool registration and compatibility exports."""

from __future__ import annotations

from llm_tools.tool_api import ToolRegistry
from llm_tools.tools.atlassian._shared import (
    _absolute_url,
    _append_remote_source_provenance,
    _attachment_cache_signature,
    _bitbucket_file_to_text,
    _build_attachment_read_result,
    _build_text_read_result,
    _cached_attachment_is_current,
    _confluence_page_body,
    _download_confluence_attachment_bytes,
    _ensure_cached_confluence_attachment,
    _extract_bitbucket_path,
    _extract_collection,
    _extract_first_link_href,
    _extract_issue_fields,
    _get_confluence_attachment_cache_paths,
    _get_confluence_attachment_cache_root,
    _get_tool_limits,
    _get_value,
    _normalize_remote_bytes,
    _normalize_remote_exception,
    _normalize_remote_text,
    _resolve_confluence_attachment,
    _sanitize_filename,
    _search_fetch_limit,
)
from llm_tools.tools.atlassian.read_bitbucket_file import (
    ReadBitbucketFileInput,
    ReadBitbucketFileOutput,
    ReadBitbucketFileTool,
)
from llm_tools.tools.atlassian.read_bitbucket_pull_request import (
    BitbucketPullRequestChange,
    BitbucketPullRequestCommit,
    ReadBitbucketPullRequestInput,
    ReadBitbucketPullRequestOutput,
    ReadBitbucketPullRequestTool,
)
from llm_tools.tools.atlassian.read_confluence_attachment import (
    ReadConfluenceAttachmentInput,
    ReadConfluenceAttachmentOutput,
    ReadConfluenceAttachmentTool,
)
from llm_tools.tools.atlassian.read_confluence_page import (
    ReadConfluencePageInput,
    ReadConfluencePageOutput,
    ReadConfluencePageTool,
)
from llm_tools.tools.atlassian.read_jira_issue import (
    ReadJiraIssueInput,
    ReadJiraIssueOutput,
    ReadJiraIssueTool,
)
from llm_tools.tools.atlassian.search_bitbucket_code import (
    BitbucketCodeMatch,
    SearchBitbucketCodeInput,
    SearchBitbucketCodeOutput,
    SearchBitbucketCodeTool,
)
from llm_tools.tools.atlassian.search_confluence import (
    ConfluenceSearchMatch,
    SearchConfluenceInput,
    SearchConfluenceOutput,
    SearchConfluenceTool,
)
from llm_tools.tools.atlassian.search_jira import (
    JiraIssueSummary,
    SearchJiraInput,
    SearchJiraOutput,
    SearchJiraTool,
)
from llm_tools.tools.filesystem.models import ToolLimits


def register_atlassian_tools(registry: ToolRegistry) -> None:
    """Register the built-in Atlassian tool set."""
    registry.register(SearchJiraTool())
    registry.register(ReadJiraIssueTool())
    registry.register(SearchBitbucketCodeTool())
    registry.register(ReadBitbucketFileTool())
    registry.register(ReadBitbucketPullRequestTool())
    registry.register(SearchConfluenceTool())
    registry.register(ReadConfluencePageTool())
    registry.register(ReadConfluenceAttachmentTool())


__all__ = [
    "BitbucketCodeMatch",
    "BitbucketPullRequestChange",
    "BitbucketPullRequestCommit",
    "ConfluenceSearchMatch",
    "JiraIssueSummary",
    "ReadBitbucketFileInput",
    "ReadBitbucketFileOutput",
    "ReadBitbucketFileTool",
    "ReadBitbucketPullRequestInput",
    "ReadBitbucketPullRequestOutput",
    "ReadBitbucketPullRequestTool",
    "ReadConfluenceAttachmentInput",
    "ReadConfluenceAttachmentOutput",
    "ReadConfluenceAttachmentTool",
    "ReadConfluencePageInput",
    "ReadConfluencePageOutput",
    "ReadConfluencePageTool",
    "ReadJiraIssueInput",
    "ReadJiraIssueOutput",
    "ReadJiraIssueTool",
    "SearchBitbucketCodeInput",
    "SearchBitbucketCodeOutput",
    "SearchBitbucketCodeTool",
    "SearchConfluenceInput",
    "SearchConfluenceOutput",
    "SearchConfluenceTool",
    "SearchJiraInput",
    "SearchJiraOutput",
    "SearchJiraTool",
    "ToolLimits",
    "register_atlassian_tools",
    "_absolute_url",
    "_append_remote_source_provenance",
    "_attachment_cache_signature",
    "_bitbucket_file_to_text",
    "_build_attachment_read_result",
    "_build_text_read_result",
    "_cached_attachment_is_current",
    "_confluence_page_body",
    "_download_confluence_attachment_bytes",
    "_ensure_cached_confluence_attachment",
    "_extract_bitbucket_path",
    "_extract_collection",
    "_extract_first_link_href",
    "_extract_issue_fields",
    "_get_confluence_attachment_cache_paths",
    "_get_confluence_attachment_cache_root",
    "_get_tool_limits",
    "_get_value",
    "_normalize_remote_bytes",
    "_normalize_remote_exception",
    "_normalize_remote_text",
    "_resolve_confluence_attachment",
    "_sanitize_filename",
    "_search_fetch_limit",
]

"""Jira search tool."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, Field

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.atlassian._shared import (
    _JIRA_ENV_KEYS,
    _REMOTE_COLLECTION_LIMIT,
    _REMOTE_TOOL_TIMEOUT_SECONDS,
    _extract_issue_fields,
    _normalize_remote_exception,
    _search_fetch_limit,
)


class JiraIssueSummary(BaseModel):
    key: str
    summary: str | None = None
    status: str | None = None
    issue_type: str | None = None
    assignee: str | None = None


class SearchJiraInput(BaseModel):
    jql: str
    limit: int = Field(default=20, ge=1, le=_REMOTE_COLLECTION_LIMIT)


class SearchJiraOutput(BaseModel):
    issues: list[JiraIssueSummary] = Field(default_factory=list)
    truncated: bool = False


class SearchJiraTool(Tool[SearchJiraInput, SearchJiraOutput]):
    spec = ToolSpec(
        name="search_jira",
        description="Search Jira issues with JQL.",
        tags=["atlassian", "jira", "search", "read"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=_REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(_JIRA_ENV_KEYS),
    )
    input_model = SearchJiraInput
    output_model = SearchJiraOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: SearchJiraInput
    ) -> SearchJiraOutput:
        client = context.services.require_jira().client
        fetch_limit = _search_fetch_limit(args.limit)
        try:
            if hasattr(client, "enhanced_jql"):
                payload = client.enhanced_jql(args.jql, limit=fetch_limit)
            elif hasattr(client, "jql"):
                payload = client.jql(args.jql, limit=fetch_limit)
            else:
                raise RuntimeError(
                    "Configured Jira client does not support JQL search."
                )
        except Exception as exc:
            raise _normalize_remote_exception(exc) from exc

        raw_issues = cast(list[dict[str, Any]], payload.get("issues", []))
        issues = []
        for issue in raw_issues[: args.limit]:
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

        context.log(f"Ran Jira search for JQL '{args.jql}'.")
        return SearchJiraOutput(issues=issues, truncated=len(raw_issues) > args.limit)

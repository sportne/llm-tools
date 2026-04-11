"""Atlassian built-in tool implementations."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, Field

from llm_tools.tool_api import (
    SideEffectClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolSpec,
)

_JIRA_ENV_KEYS = ("JIRA_BASE_URL", "JIRA_USERNAME", "JIRA_API_TOKEN")


def _get_required_env(context: ToolContext, key: str) -> str:
    value = context.env.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required Jira credential '{key}'.")
    return value


def _build_jira_client(context: ToolContext) -> Any:
    base_url = _get_required_env(context, "JIRA_BASE_URL")
    username = _get_required_env(context, "JIRA_USERNAME")
    api_token = _get_required_env(context, "JIRA_API_TOKEN")

    from atlassian import Jira  # type: ignore[import-not-found]

    return Jira(
        url=base_url,
        username=username,
        password=api_token,
    )


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


def register_atlassian_tools(registry: ToolRegistry) -> None:
    """Register the built-in Atlassian tool set."""
    registry.register(SearchJiraTool())
    registry.register(ReadJiraIssueTool())

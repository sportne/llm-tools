"""Jira issue read tool."""

from __future__ import annotations

from typing import Any, cast

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.atlassian._shared import (
    _JIRA_ENV_KEYS,
    _REMOTE_TOOL_TIMEOUT_SECONDS,
    _extract_issue_fields,
    _normalize_remote_exception,
)
from llm_tools.tools.atlassian.read_jira_issue_models import (
    ReadJiraIssueInput as ReadJiraIssueInput,
)
from llm_tools.tools.atlassian.read_jira_issue_models import (
    ReadJiraIssueOutput as ReadJiraIssueOutput,
)


class ReadJiraIssueTool(Tool[ReadJiraIssueInput, ReadJiraIssueOutput]):
    spec = ToolSpec(
        name="read_jira_issue",
        description="Read one Jira issue by key.",
        tags=["atlassian", "jira", "read"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=_REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(_JIRA_ENV_KEYS),
    )
    input_model = ReadJiraIssueInput
    output_model = ReadJiraIssueOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: ReadJiraIssueInput
    ) -> ReadJiraIssueOutput:
        client = context.services.require_jira().client
        try:
            if hasattr(client, "issue"):
                issue = client.issue(args.issue_key)
            elif hasattr(client, "get_issue"):
                issue = client.get_issue(args.issue_key)
            else:
                raise RuntimeError(
                    "Configured Jira client does not support issue reads."
                )
        except Exception as exc:
            raise _normalize_remote_exception(exc) from exc

        normalized = _extract_issue_fields(
            cast(dict[str, Any], issue),
            requested_fields=args.requested_fields,
        )
        context.log(f"Read Jira issue '{args.issue_key}'.")
        return ReadJiraIssueOutput(**normalized)

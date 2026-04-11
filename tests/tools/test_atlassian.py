"""Unit tests for Atlassian built-in tools."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from llm_tools.tool_api import ToolContext
from llm_tools.tools.atlassian import ReadJiraIssueTool, SearchJiraTool


def _install_fake_atlassian_module(
    monkeypatch: pytest.MonkeyPatch, jira_cls: Any
) -> None:
    fake_module = ModuleType("atlassian")
    fake_module.Jira = jira_cls
    monkeypatch.setitem(sys.modules, "atlassian", fake_module)


def _context() -> ToolContext:
    return ToolContext(
        invocation_id="inv-1",
        env={
            "JIRA_BASE_URL": "https://example.atlassian.net",
            "JIRA_USERNAME": "user@example.com",
            "JIRA_API_TOKEN": "token",
        },
    )


def test_search_jira_tool_maps_search_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def enhanced_jql(self, jql: str, *, limit: int) -> dict[str, object]:
            assert jql == "project = DEMO"
            assert limit == 5
            return {
                "issues": [
                    {
                        "key": "DEMO-1",
                        "fields": {
                            "summary": "Test issue",
                            "status": {"name": "Open"},
                            "issuetype": {"name": "Task"},
                            "assignee": {"displayName": "Alice"},
                        },
                    }
                ]
            }

    _install_fake_atlassian_module(monkeypatch, FakeJira)

    result = SearchJiraTool().invoke(
        _context(),
        SearchJiraTool.input_model(jql="project = DEMO", limit=5),
    )

    assert len(result.issues) == 1
    assert result.issues[0].key == "DEMO-1"
    assert result.issues[0].summary == "Test issue"


def test_read_jira_issue_tool_maps_issue_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def issue(self, issue_key: str) -> dict[str, object]:
            assert issue_key == "DEMO-2"
            return {
                "key": "DEMO-2",
                "fields": {
                    "summary": "Issue summary",
                    "description": "Issue description",
                    "status": {"name": "Done"},
                    "issuetype": {"name": "Bug"},
                    "assignee": {"displayName": "Bob"},
                    "priority": {"name": "High"},
                },
            }

    _install_fake_atlassian_module(monkeypatch, FakeJira)

    result = ReadJiraIssueTool().invoke(
        _context(),
        ReadJiraIssueTool.input_model(issue_key="DEMO-2"),
    )

    assert result.key == "DEMO-2"
    assert result.summary == "Issue summary"
    assert result.status == "Done"
    assert result.issue_type == "Bug"
    assert result.assignee == "Bob"
    assert result.raw_fields["priority"]["name"] == "High"


def test_jira_tools_require_context_env_credentials() -> None:
    with pytest.raises(ValueError, match="JIRA_BASE_URL"):
        SearchJiraTool().invoke(
            ToolContext(invocation_id="inv-2"),
            SearchJiraTool.input_model(jql="project = DEMO"),
        )


def test_search_jira_tool_falls_back_to_jql_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def jql(self, jql: str, *, limit: int) -> dict[str, object]:
            assert jql == "project = DEMO"
            assert limit == 3
            return {
                "issues": [
                    {
                        "key": "DEMO-3",
                        "fields": {
                            "summary": "Fallback issue",
                            "status": {"name": "Open"},
                            "issuetype": {"name": "Task"},
                            "assignee": {"displayName": "Carol"},
                        },
                    }
                ]
            }

    _install_fake_atlassian_module(monkeypatch, FakeJira)

    result = SearchJiraTool().invoke(
        _context(),
        SearchJiraTool.input_model(jql="project = DEMO", limit=3),
    )

    assert result.issues[0].key == "DEMO-3"
    assert result.issues[0].summary == "Fallback issue"


def test_search_jira_tool_rejects_unsupported_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

    _install_fake_atlassian_module(monkeypatch, FakeJira)

    with pytest.raises(RuntimeError, match="does not support JQL search"):
        SearchJiraTool().invoke(
            _context(),
            SearchJiraTool.input_model(jql="project = DEMO"),
        )


def test_read_jira_issue_tool_falls_back_to_get_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def get_issue(self, issue_key: str) -> dict[str, object]:
            assert issue_key == "DEMO-4"
            return {
                "key": "DEMO-4",
                "fields": {
                    "summary": "Get issue fallback",
                    "description": {"type": "doc"},
                    "status": {"name": "In Progress"},
                    "issuetype": {"name": "Story"},
                    "assignee": {"displayName": "Dana"},
                },
            }

    _install_fake_atlassian_module(monkeypatch, FakeJira)

    result = ReadJiraIssueTool().invoke(
        _context(),
        ReadJiraIssueTool.input_model(issue_key="DEMO-4"),
    )

    assert result.key == "DEMO-4"
    assert result.issue_type == "Story"
    assert result.assignee == "Dana"


def test_read_jira_issue_tool_rejects_unsupported_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

    _install_fake_atlassian_module(monkeypatch, FakeJira)

    with pytest.raises(RuntimeError, match="does not support issue reads"):
        ReadJiraIssueTool().invoke(
            _context(),
            ReadJiraIssueTool.input_model(issue_key="DEMO-5"),
        )

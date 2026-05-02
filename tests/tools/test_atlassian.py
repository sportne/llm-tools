"""Unit tests for Atlassian built-in tools."""

from __future__ import annotations

import io
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from llm_tools.tool_api import (
    ErrorCode,
    SideEffectClass,
    ToolContext,
    ToolInvocationRequest,
    ToolPolicy,
    ToolRegistry,
    ToolRuntime,
)
from llm_tools.tools.atlassian import (
    ReadBitbucketFileTool,
    ReadBitbucketPullRequestTool,
    ReadConfluenceAttachmentTool,
    ReadConfluencePageTool,
    ReadJiraIssueTool,
    SearchBitbucketCodeTool,
    SearchConfluenceTool,
    SearchJiraTool,
)
from llm_tools.tools.atlassian import tools as atlassian_tools


def test_product_facades_reexport_tools() -> None:
    from llm_tools.tools.atlassian import bitbucket, confluence, jira

    assert bitbucket.SearchBitbucketCodeTool is SearchBitbucketCodeTool
    assert bitbucket.ReadBitbucketFileTool is ReadBitbucketFileTool
    assert bitbucket.ReadBitbucketPullRequestTool is ReadBitbucketPullRequestTool
    assert confluence.SearchConfluenceTool is SearchConfluenceTool
    assert confluence.ReadConfluencePageTool is ReadConfluencePageTool
    assert confluence.ReadConfluenceAttachmentTool is ReadConfluenceAttachmentTool
    assert jira.SearchJiraTool is SearchJiraTool
    assert jira.ReadJiraIssueTool is ReadJiraIssueTool


def _atlassian_runtime() -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(SearchJiraTool())
    registry.register(ReadJiraIssueTool())
    registry.register(SearchBitbucketCodeTool())
    registry.register(ReadBitbucketFileTool())
    registry.register(ReadBitbucketPullRequestTool())
    registry.register(SearchConfluenceTool())
    registry.register(ReadConfluencePageTool())
    registry.register(ReadConfluenceAttachmentTool())
    return ToolRuntime(
        registry,
        policy=ToolPolicy(
            allowed_side_effects={
                SideEffectClass.NONE,
                SideEffectClass.LOCAL_READ,
                SideEffectClass.LOCAL_WRITE,
                SideEffectClass.EXTERNAL_READ,
            },
            allow_network=True,
            allow_filesystem=True,
        ),
    )


def _invoke_tool(tool_name: str, context: ToolContext, arguments: dict[str, object]):
    result = _atlassian_runtime().execute(
        ToolInvocationRequest(tool_name=tool_name, arguments=arguments),
        context,
    )
    assert result.ok is True, result.error
    return result


def _execute_tool(tool_name: str, context: ToolContext, arguments: dict[str, object]):
    return _atlassian_runtime().execute(
        ToolInvocationRequest(tool_name=tool_name, arguments=arguments),
        context,
    )


def _install_fake_atlassian_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    jira_cls: Any | None = None,
    bitbucket_cls: Any | None = None,
    confluence_cls: Any | None = None,
) -> None:
    fake_module = ModuleType("atlassian")
    if jira_cls is not None:
        fake_module.Jira = jira_cls
    if bitbucket_cls is not None:
        fake_module.Bitbucket = bitbucket_cls
    if confluence_cls is not None:
        fake_module.Confluence = confluence_cls
    monkeypatch.setitem(sys.modules, "atlassian", fake_module)


def _jira_context() -> ToolContext:
    return ToolContext(
        invocation_id="jira-inv",
        env={
            "JIRA_BASE_URL": "https://example.atlassian.net",
            "JIRA_USERNAME": "user@example.com",
            "JIRA_API_TOKEN": "token",
        },
    )


def _bitbucket_context() -> ToolContext:
    return ToolContext(
        invocation_id="bitbucket-inv",
        env={
            "BITBUCKET_BASE_URL": "https://bitbucket.example.com",
            "BITBUCKET_USERNAME": "user@example.com",
            "BITBUCKET_API_TOKEN": "token",
        },
    )


def _confluence_context() -> ToolContext:
    return ToolContext(
        invocation_id="confluence-inv",
        env={
            "CONFLUENCE_BASE_URL": "https://confluence.example.com",
            "CONFLUENCE_USERNAME": "user@example.com",
            "CONFLUENCE_API_TOKEN": "token",
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
            assert limit == 6
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

    _install_fake_atlassian_module(monkeypatch, jira_cls=FakeJira)

    result = SearchJiraTool.output_model.model_validate(
        _invoke_tool(
            "search_jira", _jira_context(), {"jql": "project = DEMO", "limit": 5}
        ).output
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

    _install_fake_atlassian_module(monkeypatch, jira_cls=FakeJira)

    result = ReadJiraIssueTool.output_model.model_validate(
        _invoke_tool(
            "read_jira_issue",
            _jira_context(),
            {"issue_key": "DEMO-2", "requested_fields": ["priority"]},
        ).output
    )

    assert result.key == "DEMO-2"
    assert result.summary == "Issue summary"
    assert result.status == "Done"
    assert result.issue_type == "Bug"
    assert result.assignee == "Bob"
    assert result.requested_fields["priority"]["name"] == "High"


def test_read_jira_issue_input_rejects_duplicate_or_blank_requested_fields() -> None:
    with pytest.raises(ValueError, match="requested_fields must be unique"):
        ReadJiraIssueTool.input_model(issue_key="DEMO-2", requested_fields=["a", "a"])

    with pytest.raises(ValueError, match="requested_fields must not contain empty"):
        ReadJiraIssueTool.input_model(issue_key="DEMO-2", requested_fields=[" "])


def test_jira_tools_require_context_env_credentials() -> None:
    result = _execute_tool(
        "search_jira",
        ToolContext(invocation_id="jira-missing"),
        {"jql": "project = DEMO"},
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.POLICY_DENIED
    assert set(
        result.error.details["policy_decision"]["metadata"]["missing_secrets"]
    ) == {
        "JIRA_BASE_URL",
        "JIRA_API_TOKEN",
    }


def test_search_jira_tool_falls_back_to_jql_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def jql(self, jql: str, *, limit: int) -> dict[str, object]:
            assert jql == "project = DEMO"
            assert limit == 4
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

    _install_fake_atlassian_module(monkeypatch, jira_cls=FakeJira)

    result = SearchJiraTool.output_model.model_validate(
        _invoke_tool(
            "search_jira", _jira_context(), {"jql": "project = DEMO", "limit": 3}
        ).output
    )

    assert result.issues[0].key == "DEMO-3"
    assert result.issues[0].summary == "Fallback issue"


def test_search_jira_tool_marks_truncated_when_extra_issue_is_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            del kwargs

        def enhanced_jql(self, jql: str, *, limit: int) -> dict[str, object]:
            assert jql == "project = DEMO"
            assert limit == 3
            return {
                "issues": [
                    {"key": "DEMO-1", "fields": {"summary": "One"}},
                    {"key": "DEMO-2", "fields": {"summary": "Two"}},
                    {"key": "DEMO-3", "fields": {"summary": "Three"}},
                ]
            }

    _install_fake_atlassian_module(monkeypatch, jira_cls=FakeJira)

    result = SearchJiraTool.output_model.model_validate(
        _invoke_tool(
            "search_jira", _jira_context(), {"jql": "project = DEMO", "limit": 2}
        ).output
    )

    assert [issue.key for issue in result.issues] == ["DEMO-1", "DEMO-2"]
    assert result.truncated is True


def test_search_jira_tool_rejects_unsupported_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

    _install_fake_atlassian_module(monkeypatch, jira_cls=FakeJira)

    result = _execute_tool(
        "search_jira",
        _jira_context(),
        {"jql": "project = DEMO"},
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    assert "does not support JQL search" in result.error.details["exception_message"]


def test_atlassian_tools_surface_transient_remote_failures_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            del kwargs

        def enhanced_jql(self, jql: str, *, limit: int) -> dict[str, object]:
            del jql, limit
            raise TimeoutError("jira timed out")

    _install_fake_atlassian_module(monkeypatch, jira_cls=FakeJira)

    result = _execute_tool(
        "search_jira",
        _jira_context(),
        {"jql": "project = DEMO"},
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    assert result.error.retryable is True


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

    _install_fake_atlassian_module(monkeypatch, jira_cls=FakeJira)

    result = ReadJiraIssueTool.output_model.model_validate(
        _invoke_tool("read_jira_issue", _jira_context(), {"issue_key": "DEMO-4"}).output
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

    _install_fake_atlassian_module(monkeypatch, jira_cls=FakeJira)

    result = _execute_tool(
        "read_jira_issue",
        _jira_context(),
        {"issue_key": "DEMO-5"},
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    assert "does not support issue reads" in result.error.details["exception_message"]


def test_search_bitbucket_code_tool_maps_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBitbucket:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def search_code(
            self, team: str, search_query: str, *, limit: int
        ) -> dict[str, object]:
            assert team == "PROJ"
            assert search_query == "needle"
            assert limit == 6
            return {
                "values": [
                    {
                        "repository": {"slug": "demo-repo"},
                        "path": "src/main.py",
                        "line": 12,
                        "content": "needle()",
                    }
                ]
            }

    _install_fake_atlassian_module(monkeypatch, bitbucket_cls=FakeBitbucket)

    result = SearchBitbucketCodeTool.output_model.model_validate(
        _invoke_tool(
            "search_bitbucket_code",
            _bitbucket_context(),
            {"project_key": "PROJ", "query": "needle", "limit": 5},
        ).output
    )

    assert result.project_key == "PROJ"
    assert result.matches[0].repository_slug == "demo-repo"
    assert result.matches[0].path == "src/main.py"
    assert result.matches[0].line_number == 12
    assert result.matches[0].snippet == "needle()"


def test_search_bitbucket_code_tool_marks_truncated_when_extra_match_is_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBitbucket:
        def __init__(self, **kwargs: str) -> None:
            del kwargs

        def search_code(
            self, team: str, search_query: str, *, limit: int
        ) -> dict[str, object]:
            assert team == "PROJ"
            assert search_query == "needle"
            assert limit == 3
            return {
                "values": [
                    {"repository": {"slug": "repo"}, "path": "a.py"},
                    {"repository": {"slug": "repo"}, "path": "b.py"},
                    {"repository": {"slug": "repo"}, "path": "c.py"},
                ]
            }

    _install_fake_atlassian_module(monkeypatch, bitbucket_cls=FakeBitbucket)

    result = SearchBitbucketCodeTool.output_model.model_validate(
        _invoke_tool(
            "search_bitbucket_code",
            _bitbucket_context(),
            {"project_key": "PROJ", "query": "needle", "limit": 2},
        ).output
    )

    assert [match.path for match in result.matches] == ["a.py", "b.py"]
    assert result.truncated is True


def test_read_bitbucket_file_tool_reads_text_and_applies_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBitbucket:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def get_content_of_file(
            self,
            project_key: str,
            repository_slug: str,
            filename: str,
            *,
            at: str,
        ) -> bytes:
            assert project_key == "PROJ"
            assert repository_slug == "demo-repo"
            assert filename == "README.md"
            assert at == "main"
            return b"alpha beta gamma"

    _install_fake_atlassian_module(monkeypatch, bitbucket_cls=FakeBitbucket)

    result = ReadBitbucketFileTool.output_model.model_validate(
        _invoke_tool(
            "read_bitbucket_file",
            _bitbucket_context(),
            {
                "project_key": "PROJ",
                "repository_slug": "demo-repo",
                "path": "README.md",
                "ref": "main",
                "start_char": 6,
                "end_char": 10,
            },
        ).output
    )

    assert result.project_key == "PROJ"
    assert result.repository_slug == "demo-repo"
    assert result.ref == "main"
    assert result.content == "beta"
    assert result.truncated is True
    assert result.end_char == 10


def test_read_bitbucket_file_tool_returns_structured_unsupported_for_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBitbucket:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def get_content_of_file(
            self,
            project_key: str,
            repository_slug: str,
            filename: str,
            *,
            at: str,
        ) -> bytes:
            del project_key, repository_slug, filename, at
            return b"\xff\xfe"

    _install_fake_atlassian_module(monkeypatch, bitbucket_cls=FakeBitbucket)

    result = ReadBitbucketFileTool.output_model.model_validate(
        _invoke_tool(
            "read_bitbucket_file",
            _bitbucket_context(),
            {
                "project_key": "PROJ",
                "repository_slug": "demo-repo",
                "path": "binary.bin",
            },
        ).output
    )

    assert result.status == "unsupported"
    assert result.read_kind == "unsupported"
    assert "UTF-8" in (result.error_message or "")


def test_read_bitbucket_pull_request_tool_maps_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBitbucket:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def get_pull_request(
            self, project_key: str, repository_slug: str, pull_request_id: int
        ) -> dict[str, object]:
            assert (project_key, repository_slug, pull_request_id) == (
                "PROJ",
                "demo-repo",
                7,
            )
            return {
                "title": "Add feature",
                "description": "Implements it",
                "state": "OPEN",
                "author": {"user": {"displayName": "Alice"}},
                "fromRef": {"displayId": "feature"},
                "toRef": {"displayId": "main"},
                "links": {"self": [{"href": "https://bitbucket/pull/7"}]},
            }

        def get_pull_requests_commits(
            self, project_key: str, repository_slug: str, pull_request_id: int
        ) -> dict[str, object]:
            del project_key, repository_slug, pull_request_id
            return {
                "values": [
                    {
                        "id": "abc123",
                        "displayId": "abc123",
                        "message": "feat: add feature",
                        "author": {"name": "Alice"},
                    }
                ]
            }

        def get_pull_requests_changes(
            self, project_key: str, repository_slug: str, pull_request_id: int
        ) -> dict[str, object]:
            del project_key, repository_slug, pull_request_id
            return {
                "values": [
                    {
                        "srcPath": {"toString": "old.py"},
                        "path": {"toString": "new.py"},
                        "type": "MOVE",
                        "executable": False,
                    }
                ]
            }

    _install_fake_atlassian_module(monkeypatch, bitbucket_cls=FakeBitbucket)

    result = ReadBitbucketPullRequestTool.output_model.model_validate(
        _invoke_tool(
            "read_bitbucket_pull_request",
            _bitbucket_context(),
            {
                "project_key": "PROJ",
                "repository_slug": "demo-repo",
                "pull_request_id": 7,
            },
        ).output
    )

    assert result.title == "Add feature"
    assert result.author == "Alice"
    assert result.source_branch == "feature"
    assert result.commits[0].message == "feat: add feature"
    assert result.changed_files[0].new_path == "new.py"
    assert result.changed_files[0].change_type == "MOVE"


def test_search_confluence_tool_maps_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfluence:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def cql(self, cql: str, *, limit: int, excerpt: str) -> dict[str, object]:
            assert cql == "type = page"
            assert limit == 5
            assert excerpt == "highlight"
            return {
                "results": [
                    {
                        "content": {
                            "id": "123",
                            "title": "Demo page",
                            "type": "page",
                            "space": {"key": "ENG"},
                            "_links": {"webui": "/spaces/ENG/pages/123"},
                        },
                        "excerpt": "match excerpt",
                    }
                ]
            }

    _install_fake_atlassian_module(monkeypatch, confluence_cls=FakeConfluence)

    result = SearchConfluenceTool.output_model.model_validate(
        _invoke_tool(
            "search_confluence",
            _confluence_context(),
            {"cql": "type = page", "limit": 4},
        ).output
    )

    assert result.cql == "type = page"
    assert result.matches[0].content_id == "123"
    assert result.matches[0].title == "Demo page"
    assert result.matches[0].space_key == "ENG"
    assert (
        result.matches[0].web_url
        == "https://confluence.example.com/spaces/ENG/pages/123"
    )


def test_search_confluence_tool_marks_truncated_when_extra_match_is_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfluence:
        def __init__(self, **kwargs: str) -> None:
            del kwargs

        def cql(self, cql: str, *, limit: int, excerpt: str) -> dict[str, object]:
            assert cql == "type = page"
            assert limit == 3
            assert excerpt == "highlight"
            return {
                "results": [
                    {"content": {"id": "1", "title": "One", "type": "page"}},
                    {"content": {"id": "2", "title": "Two", "type": "page"}},
                    {"content": {"id": "3", "title": "Three", "type": "page"}},
                ]
            }

    _install_fake_atlassian_module(monkeypatch, confluence_cls=FakeConfluence)

    result = SearchConfluenceTool.output_model.model_validate(
        _invoke_tool(
            "search_confluence",
            _confluence_context(),
            {"cql": "type = page", "limit": 2},
        ).output
    )

    assert [match.content_id for match in result.matches] == ["1", "2"]
    assert result.truncated is True


def test_read_confluence_page_tool_reads_page_with_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfluence:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def get_page_by_id(self, page_id: str, *, expand: str) -> dict[str, object]:
            assert page_id == "123"
            assert "body.storage" in expand
            return {
                "title": "Demo page",
                "space": {"key": "ENG"},
                "body": {
                    "storage": {
                        "value": "alpha beta gamma",
                        "representation": "storage",
                    }
                },
                "_links": {"webui": "/spaces/ENG/pages/123"},
            }

    _install_fake_atlassian_module(monkeypatch, confluence_cls=FakeConfluence)

    result = ReadConfluencePageTool.output_model.model_validate(
        _invoke_tool(
            "read_confluence_page",
            _confluence_context(),
            {"page_id": "123", "start_char": 6, "end_char": 10},
        ).output
    )

    assert result.title == "Demo page"
    assert result.content == "beta"
    assert result.truncated is True
    assert result.representation == "storage"


def test_read_confluence_attachment_tool_reads_attachment_and_reuses_cache(
    tmp_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_calls: list[str] = []

    class FakeConfluence:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def get_page_by_id(self, page_id: str, *, expand: str) -> dict[str, object]:
            del expand
            return {
                "title": "Demo page",
                "space": {"key": "ENG"},
                "body": {"storage": {"value": "page body"}},
                "_links": {"webui": "/spaces/ENG/pages/123"},
            }

        def get_attachments_from_content(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["page_id"] == "123"
            if "filename" in kwargs:
                assert kwargs["filename"] == "report.pdf"
            return {
                "results": [
                    {
                        "id": "att-1",
                        "title": "report.pdf",
                        "version": {"number": 2},
                        "extensions": {"fileSize": 4},
                        "_links": {"download": "/download/attachments/123/report.pdf"},
                    }
                ]
            }

        def download_attachments_from_page(
            self,
            page_id: str,
            *,
            filename: str | None,
            to_memory: bool,
        ) -> dict[str, io.BytesIO]:
            assert page_id == "123"
            assert filename == "report.pdf"
            assert to_memory is True
            download_calls.append(filename)
            return {"report.pdf": io.BytesIO(b"\xff\xfe\x00\x10")}

    fake_markitdown = ModuleType("markitdown")
    convert_calls: list[str] = []

    class FakeMarkItDown:
        def convert(self, path: str) -> SimpleNamespace:
            convert_calls.append(path)
            return SimpleNamespace(text_content=f"converted:{path}")

    fake_markitdown.MarkItDown = FakeMarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", fake_markitdown)
    _install_fake_atlassian_module(monkeypatch, confluence_cls=FakeConfluence)
    monkeypatch.setattr(
        atlassian_tools,
        "_get_confluence_attachment_cache_root",
        lambda: tmp_path / "attachment-cache",
    )

    first = ReadConfluenceAttachmentTool.output_model.model_validate(
        _invoke_tool(
            "read_confluence_attachment",
            _confluence_context(),
            {"page_id": "123", "attachment_filename": "report.pdf"},
        ).output
    )
    second = ReadConfluenceAttachmentTool.output_model.model_validate(
        _invoke_tool(
            "read_confluence_attachment",
            _confluence_context(),
            {"page_id": "123", "attachment_filename": "report.pdf"},
        ).output
    )

    assert first.read_kind == "markitdown"
    assert first.content is not None and first.content.startswith("converted:")
    assert second.content == first.content
    assert download_calls == ["report.pdf"]
    assert len(convert_calls) == 1


def test_read_confluence_attachment_tool_selects_attachment_by_id(
    tmp_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfluence:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def get_page_by_id(self, page_id: str, *, expand: str) -> dict[str, object]:
            del expand
            return {
                "title": "Demo page",
                "space": {"key": "ENG"},
                "body": {"storage": {"value": "page body"}},
                "_links": {"webui": "/spaces/ENG/pages/123"},
            }

        def get_attachments_from_content(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["page_id"] == "123"
            return {
                "results": [
                    {
                        "id": "att-9",
                        "title": "note.txt",
                        "version": {"number": 1},
                        "extensions": {"fileSize": 9},
                        "_links": {"download": "/download/attachments/123/note.txt"},
                    }
                ]
            }

        def download_attachments_from_page(
            self,
            page_id: str,
            *,
            filename: str | None,
            to_memory: bool,
        ) -> dict[str, io.BytesIO]:
            assert page_id == "123"
            assert filename == "note.txt"
            assert to_memory is True
            return {"note.txt": io.BytesIO(b"plain text")}

    _install_fake_atlassian_module(monkeypatch, confluence_cls=FakeConfluence)
    monkeypatch.setattr(
        atlassian_tools,
        "_get_confluence_attachment_cache_root",
        lambda: tmp_path / "attachment-cache",
    )

    result = ReadConfluenceAttachmentTool.output_model.model_validate(
        _invoke_tool(
            "read_confluence_attachment",
            _confluence_context(),
            {"page_id": "123", "attachment_id": "att-9"},
        ).output
    )

    assert result.attachment_id == "att-9"
    assert result.attachment_filename == "note.txt"
    assert result.content == "plain text"


def test_read_confluence_attachment_tool_validates_attachment_selector() -> None:
    with pytest.raises(ValueError, match="at most one"):
        ReadConfluenceAttachmentTool.input_model(
            page_id="123",
            attachment_id="att-1",
            attachment_filename="report.pdf",
        )


def test_read_confluence_attachment_tool_requires_selector() -> None:
    with pytest.raises(ValueError, match="provide one"):
        ReadConfluenceAttachmentTool.input_model(page_id="123")


def test_atlassian_helper_value_collection_and_link_fallbacks() -> None:
    payload = SimpleNamespace(answer="value")

    assert atlassian_tools._get_value(payload, "answer") == "value"
    assert atlassian_tools._extract_collection(["a", "b"]) == ["a", "b"]
    assert atlassian_tools._extract_collection(SimpleNamespace()) == []
    assert atlassian_tools._absolute_url("https://example.com", None) is None
    assert (
        atlassian_tools._extract_bitbucket_path(
            SimpleNamespace(path="plain-path"), "path"
        )
        == "plain-path"
    )
    assert (
        atlassian_tools._extract_first_link_href(
            {"links": {"self": {"href": "https://bitbucket/pull/8"}}}
        )
        == "https://bitbucket/pull/8"
    )
    assert atlassian_tools._extract_first_link_href({"links": {"self": None}}) is None
    assert atlassian_tools._get_confluence_attachment_cache_root().name == (
        "confluence_attachment_cache"
    )


def test_atlassian_helper_text_normalization_and_size_guard() -> None:
    assert atlassian_tools._normalize_remote_text("hello") == ("hello", 5, None)
    assert atlassian_tools._normalize_remote_text("hi\x00") == (
        None,
        3,
        "Remote content is binary.",
    )
    assert atlassian_tools._normalize_remote_bytes(b"hi\x00") == (
        None,
        3,
        "Remote content is binary.",
    )

    result = atlassian_tools._build_text_read_result(
        requested_path="remote.txt",
        resolved_path="PROJ/repo@HEAD:remote.txt",
        tool_limits=atlassian_tools.ToolLimits(max_file_size_characters=3),
        content="abcd",
        file_size_bytes=4,
        status="ok",
        read_kind="text",
    )

    assert result.status == "too_large"
    assert result.content is None


def test_atlassian_helper_page_and_bitbucket_file_fallbacks() -> None:
    assert atlassian_tools._bitbucket_file_to_text("plain text") == (
        "plain text",
        10,
        None,
    )
    assert atlassian_tools._bitbucket_file_to_text(object()) == (
        None,
        0,
        "Bitbucket file payload did not include readable content.",
    )
    assert atlassian_tools._confluence_page_body({"body": {"storage": "bad"}}) == (
        "",
        None,
        "Confluence page did not include readable body content.",
    )


def test_cached_attachment_is_current_handles_invalid_metadata(tmp_path) -> None:
    cached_path = tmp_path / "cached.txt"
    metadata_path = tmp_path / "metadata.json"
    cached_path.write_text("cached", encoding="utf-8")
    metadata_path.write_text("{", encoding="utf-8")

    assert not atlassian_tools._cached_attachment_is_current(
        cached_path,
        metadata_path,
        attachment={"title": "cached.txt", "version": {"number": 1}},
    )


def test_download_confluence_attachment_bytes_helper_fallbacks() -> None:
    class ReadOnlyBuffer:
        def read(self) -> bytes:
            return b"via-read"

    class FakeConfluence:
        def __init__(self, payload: object) -> None:
            self.payload = payload

        def download_attachments_from_page(
            self,
            page_id: str,
            *,
            filename: str | None,
            to_memory: bool,
        ) -> object:
            assert page_id == "123"
            assert filename == "report.pdf"
            assert to_memory is True
            return self.payload

    assert (
        atlassian_tools._download_confluence_attachment_bytes(
            FakeConfluence({"other.pdf": io.BytesIO(b"via-getvalue")}),
            page_id="123",
            attachment={"title": "report.pdf"},
        )
        == b"via-getvalue"
    )
    assert (
        atlassian_tools._download_confluence_attachment_bytes(
            FakeConfluence({"other.pdf": ReadOnlyBuffer()}),
            page_id="123",
            attachment={"title": "report.pdf"},
        )
        == b"via-read"
    )

    with pytest.raises(RuntimeError, match="returned no content"):
        atlassian_tools._download_confluence_attachment_bytes(
            FakeConfluence({}),
            page_id="123",
            attachment={"title": "report.pdf"},
        )

    with pytest.raises(RuntimeError, match="did not return file bytes"):
        atlassian_tools._download_confluence_attachment_bytes(
            FakeConfluence({"other.pdf": object()}),
            page_id="123",
            attachment={"title": "report.pdf"},
        )


def test_download_confluence_attachment_bytes_rejects_non_dict_payload() -> None:
    class FakeConfluence:
        def download_attachments_from_page(
            self,
            page_id: str,
            *,
            filename: str | None,
            to_memory: bool,
        ) -> object:
            assert page_id == "123"
            assert filename == "report.pdf"
            assert to_memory is True
            return [b"not-a-dict"]

    with pytest.raises(RuntimeError, match="did not return file bytes"):
        atlassian_tools._download_confluence_attachment_bytes(
            FakeConfluence(),
            page_id="123",
            attachment={"title": "report.pdf"},
        )


def test_resolve_confluence_attachment_raises_for_missing_targets() -> None:
    class FakeConfluence:
        def get_attachments_from_content(self, **kwargs: object) -> dict[str, object]:
            if "filename" in kwargs:
                return {"results": [{"title": "other.pdf"}]}
            return {"results": [{"id": "att-2"}]}

    with pytest.raises(FileNotFoundError, match="missing.pdf"):
        atlassian_tools._resolve_confluence_attachment(
            FakeConfluence(),
            page_id="123",
            attachment_id=None,
            attachment_filename="missing.pdf",
        )

    with pytest.raises(FileNotFoundError, match="att-9"):
        atlassian_tools._resolve_confluence_attachment(
            FakeConfluence(),
            page_id="123",
            attachment_id="att-9",
            attachment_filename=None,
        )

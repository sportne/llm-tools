"""Runtime integration tests for built-in tools."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType

import pytest

import llm_tools.tools.git.tools as git_tools
from llm_tools.tool_api import (
    ErrorCode,
    SideEffectClass,
    ToolContext,
    ToolInvocationRequest,
    ToolPolicy,
    ToolRegistry,
    ToolRuntime,
)
from llm_tools.tools.atlassian import register_atlassian_tools
from llm_tools.tools.filesystem import register_filesystem_tools
from llm_tools.tools.git import register_git_tools
from llm_tools.tools.gitlab import register_gitlab_tools
from llm_tools.tools.text import register_text_tools


def _runtime(registry: ToolRegistry, *, allow_write: bool = False) -> ToolRuntime:
    allowed_side_effects = {
        SideEffectClass.NONE,
        SideEffectClass.LOCAL_READ,
        SideEffectClass.EXTERNAL_READ,
    }
    if allow_write:
        allowed_side_effects.add(SideEffectClass.LOCAL_WRITE)
    return ToolRuntime(
        registry,
        policy=ToolPolicy(allowed_side_effects=allowed_side_effects),
    )


def test_runtime_executes_filesystem_and_text_builtins(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    runtime = _runtime(registry, allow_write=True)
    context = ToolContext(invocation_id="inv-1", workspace=str(tmp_path))

    write_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="write_file",
            arguments={
                "path": "docs/note.txt",
                "content": "hello world",
                "create_parents": True,
            },
        ),
        context,
    )
    read_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="read_file", arguments={"path": "docs/note.txt"}
        ),
        ToolContext(invocation_id="inv-2", workspace=str(tmp_path)),
    )
    list_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="list_directory",
            arguments={"path": ".", "recursive": True},
        ),
        ToolContext(invocation_id="inv-3", workspace=str(tmp_path)),
    )
    search_result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_text",
            arguments={"path": ".", "query": "hello"},
        ),
        ToolContext(invocation_id="inv-4", workspace=str(tmp_path)),
    )

    assert write_result.ok is True
    assert write_result.output == {
        "path": "docs/note.txt",
        "resolved_path": "docs/note.txt",
        "bytes_written": 11,
        "created": True,
    }
    assert read_result.output == {
        "requested_path": "docs/note.txt",
        "resolved_path": "docs/note.txt",
        "content": "hello world",
        "read_kind": "text",
        "status": "ok",
        "truncated": False,
        "content_char_count": 11,
        "character_count": 11,
        "start_char": 0,
        "end_char": 11,
        "file_size_bytes": 11,
        "max_read_input_bytes": 1048576,
        "max_file_size_characters": 262144,
        "full_read_char_limit": 4000,
        "estimated_token_count": 2,
        "error_message": None,
    }
    assert [entry["path"] for entry in list_result.output["entries"]] == [
        "docs",
        "docs/note.txt",
    ]
    assert search_result.output["matches"][0]["path"] == "docs/note.txt"


def test_runtime_normalizes_workspace_root_enforcement_failures(
    tmp_path: str,
) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    registry = ToolRegistry()
    register_filesystem_tools(registry)
    runtime = _runtime(registry)
    result = runtime.execute(
        ToolInvocationRequest(
            tool_name="read_file", arguments={"path": "../outside.txt"}
        ),
        ToolContext(invocation_id="inv-5", workspace=str(tmp_path)),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    assert result.error.details["failure_reason"] == (
        "filesystem_target_invalid_or_unavailable"
    )
    assert "exception_message" not in result.error.details


def test_runtime_executes_git_builtins_with_mocked_subprocess(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ToolRegistry()
    register_git_tools(registry)
    runtime = _runtime(registry)

    def fake_run(
        args: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n")

    monkeypatch.setattr(git_tools.subprocess, "run", fake_run)

    result = runtime.execute(
        ToolInvocationRequest(tool_name="run_git_status", arguments={"path": "."}),
        ToolContext(invocation_id="inv-6", workspace=str(tmp_path)),
    )

    assert result.ok is True
    assert result.output == {
        "resolved_root": str(tmp_path.resolve()),
        "status_text": "ok\n",
    }


def test_runtime_executes_jira_builtins_with_mocked_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def enhanced_jql(self, jql: str, *, limit: int) -> dict[str, object]:
            del jql, limit
            return {
                "issues": [
                    {
                        "key": "DEMO-1",
                        "fields": {
                            "summary": "Issue",
                            "status": {"name": "Open"},
                            "issuetype": {"name": "Task"},
                            "assignee": {"displayName": "Alice"},
                        },
                    }
                ]
            }

    fake_module = ModuleType("atlassian")
    fake_module.Jira = FakeJira
    monkeypatch.setitem(sys.modules, "atlassian", fake_module)

    registry = ToolRegistry()
    register_atlassian_tools(registry)
    runtime = ToolRuntime(
        registry,
        policy=ToolPolicy(
            allowed_side_effects={
                SideEffectClass.NONE,
                SideEffectClass.LOCAL_READ,
                SideEffectClass.EXTERNAL_READ,
            }
        ),
    )
    context = ToolContext(
        invocation_id="inv-7",
        env={
            "JIRA_BASE_URL": "https://example.atlassian.net",
            "JIRA_USERNAME": "user@example.com",
            "JIRA_API_TOKEN": "token",
        },
    )

    result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_jira",
            arguments={"jql": "project = DEMO"},
        ),
        context,
    )

    assert result.ok is True
    assert result.output["issues"][0]["key"] == "DEMO-1"


def test_runtime_executes_bitbucket_builtins_with_mocked_client(
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
            assert limit == 2
            return {
                "values": [
                    {
                        "repository": {"slug": "demo-repo"},
                        "path": "src/main.py",
                        "line": 9,
                        "content": "needle()",
                    }
                ]
            }

    fake_module = ModuleType("atlassian")
    fake_module.Jira = type("FakeJira", (), {})
    fake_module.Bitbucket = FakeBitbucket
    fake_module.Confluence = type("FakeConfluence", (), {})
    monkeypatch.setitem(sys.modules, "atlassian", fake_module)

    registry = ToolRegistry()
    register_atlassian_tools(registry)
    runtime = _runtime(registry)
    context = ToolContext(
        invocation_id="inv-8",
        env={
            "BITBUCKET_BASE_URL": "https://bitbucket.example.com",
            "BITBUCKET_USERNAME": "user@example.com",
            "BITBUCKET_API_TOKEN": "token",
        },
    )

    result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_bitbucket_code",
            arguments={"project_key": "PROJ", "query": "needle", "limit": 2},
        ),
        context,
    )

    assert result.ok is True
    assert result.output == {
        "project_key": "PROJ",
        "query": "needle",
        "matches": [
            {
                "repository_slug": "demo-repo",
                "path": "src/main.py",
                "line_number": 9,
                "snippet": "needle()",
            }
        ],
    }


def test_runtime_executes_confluence_builtins_with_mocked_client(
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
                        "value": "hello confluence",
                        "representation": "storage",
                    }
                },
                "_links": {"webui": "/spaces/ENG/pages/123"},
            }

    fake_module = ModuleType("atlassian")
    fake_module.Jira = type("FakeJira", (), {})
    fake_module.Bitbucket = type("FakeBitbucket", (), {})
    fake_module.Confluence = FakeConfluence
    monkeypatch.setitem(sys.modules, "atlassian", fake_module)

    registry = ToolRegistry()
    register_atlassian_tools(registry)
    runtime = ToolRuntime(
        registry,
        policy=ToolPolicy(
            allowed_side_effects={
                SideEffectClass.NONE,
                SideEffectClass.LOCAL_READ,
                SideEffectClass.EXTERNAL_READ,
            },
            allow_filesystem=True,
        ),
    )
    context = ToolContext(
        invocation_id="inv-9",
        env={
            "CONFLUENCE_BASE_URL": "https://confluence.example.com",
            "CONFLUENCE_USERNAME": "user@example.com",
            "CONFLUENCE_API_TOKEN": "token",
        },
    )

    result = runtime.execute(
        ToolInvocationRequest(
            tool_name="read_confluence_content",
            arguments={"page_id": "123"},
        ),
        context,
    )

    assert result.ok is True
    assert result.output == {
        "page_id": "123",
        "mode": "page",
        "title": "Demo page",
        "space_key": "ENG",
        "web_url": "https://confluence.example.com/spaces/ENG/pages/123",
        "attachment_id": None,
        "attachment_filename": None,
        "representation": "storage",
        "requested_path": "page:123",
        "resolved_path": "https://confluence.example.com/spaces/ENG/pages/123",
        "read_kind": "text",
        "status": "ok",
        "content": "hello confluence",
        "truncated": False,
        "content_char_count": 16,
        "character_count": 16,
        "start_char": 0,
        "end_char": 16,
        "file_size_bytes": 16,
        "max_read_input_bytes": 1048576,
        "max_file_size_characters": 262144,
        "full_read_char_limit": 4000,
        "estimated_token_count": 2,
        "error_message": None,
    }


def test_runtime_denies_confluence_attachment_read_when_filesystem_is_disabled() -> (
    None
):
    registry = ToolRegistry()
    register_atlassian_tools(registry)
    runtime = ToolRuntime(
        registry,
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.EXTERNAL_READ},
            allow_filesystem=False,
        ),
    )

    result = runtime.execute(
        ToolInvocationRequest(
            tool_name="read_confluence_content",
            arguments={"page_id": "123", "attachment_filename": "report.pdf"},
        ),
        ToolContext(
            invocation_id="inv-10",
            env={
                "CONFLUENCE_BASE_URL": "https://confluence.example.com",
                "CONFLUENCE_USERNAME": "user@example.com",
                "CONFLUENCE_API_TOKEN": "token",
            },
        ),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.POLICY_DENIED


def test_runtime_executes_gitlab_builtins_with_mocked_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFile:
        def decode(self) -> str:
            return "hello from gitlab"

    class FakeFiles:
        def get(self, *, file_path: str, ref: str) -> FakeFile:
            assert file_path == "README.md"
            assert ref == "main"
            return FakeFile()

    class FakeProject:
        path_with_namespace = "group/repo"
        default_branch = "main"
        files = FakeFiles()

    class FakeProjects:
        def get(self, project: str) -> FakeProject:
            assert project == "group/repo"
            return FakeProject()

    class FakeGitlab:
        def __init__(self, url: str, *, private_token: str) -> None:
            assert url == "https://gitlab.example.com"
            assert private_token != ""
            self.projects = FakeProjects()

    fake_module = ModuleType("gitlab")
    fake_module.Gitlab = FakeGitlab
    monkeypatch.setitem(sys.modules, "gitlab", fake_module)

    registry = ToolRegistry()
    register_gitlab_tools(registry)
    runtime = _runtime(registry)
    context = ToolContext(
        invocation_id="inv-11",
        env={
            "GITLAB_BASE_URL": "https://gitlab.example.com",
            "GITLAB_API_TOKEN": "token",
        },
    )

    result = runtime.execute(
        ToolInvocationRequest(
            tool_name="read_gitlab_file",
            arguments={"project": "group/repo", "file_path": "README.md"},
        ),
        context,
    )

    assert result.ok is True
    assert result.output == {
        "project": "group/repo",
        "ref": "main",
        "requested_path": "README.md",
        "resolved_path": "group/repo@main:README.md",
        "read_kind": "text",
        "status": "ok",
        "content": "hello from gitlab",
        "truncated": False,
        "content_char_count": 17,
        "character_count": 17,
        "start_char": 0,
        "end_char": 17,
        "file_size_bytes": 17,
        "max_read_input_bytes": 1048576,
        "max_file_size_characters": 262144,
        "full_read_char_limit": 4000,
        "estimated_token_count": 3,
        "error_message": None,
    }


def test_runtime_denies_gitlab_tool_when_network_is_disabled() -> None:
    registry = ToolRegistry()
    register_gitlab_tools(registry)
    runtime = ToolRuntime(
        registry,
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.EXTERNAL_READ},
            allow_network=False,
        ),
    )

    result = runtime.execute(
        ToolInvocationRequest(
            tool_name="search_gitlab_code",
            arguments={"project": "group/repo", "query": "needle"},
        ),
        ToolContext(
            invocation_id="inv-12",
            env={
                "GITLAB_BASE_URL": "https://gitlab.example.com",
                "GITLAB_API_TOKEN": "token",
            },
        ),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.POLICY_DENIED

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
            tool_name="directory_text_search",
            arguments={"path": ".", "query": "hello"},
        ),
        ToolContext(invocation_id="inv-4", workspace=str(tmp_path)),
    )

    assert write_result.ok is True
    assert read_result.output == {
        "path": "docs/note.txt",
        "resolved_path": str((tmp_path / "docs" / "note.txt").resolve()),
        "content": "hello world",
        "mode": "text",
    }
    assert [entry["path"] for entry in list_result.output["entries"]] == [
        "docs",
        "docs/note.txt",
    ]
    assert search_result.output["results"][0]["path"] == "docs/note.txt"


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

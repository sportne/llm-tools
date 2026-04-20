"""Unit tests for GitLab built-in tools."""

from __future__ import annotations

import base64
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
from llm_tools.tools.gitlab import (
    ReadGitLabFileTool,
    ReadGitLabMergeRequestTool,
    SearchGitLabCodeTool,
)
from llm_tools.tools.gitlab import tools as gitlab_tools


def _gitlab_runtime() -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(SearchGitLabCodeTool())
    registry.register(ReadGitLabFileTool())
    registry.register(ReadGitLabMergeRequestTool())
    return ToolRuntime(
        registry,
        policy=ToolPolicy(
            allowed_side_effects={
                SideEffectClass.NONE,
                SideEffectClass.LOCAL_READ,
                SideEffectClass.EXTERNAL_READ,
            },
            allow_network=True,
        ),
    )


def _invoke_tool(tool_name: str, context: ToolContext, arguments: dict[str, object]):
    result = _gitlab_runtime().execute(
        ToolInvocationRequest(tool_name=tool_name, arguments=arguments),
        context,
    )
    assert result.ok is True, result.error
    return result


def _execute_tool(tool_name: str, context: ToolContext, arguments: dict[str, object]):
    return _gitlab_runtime().execute(
        ToolInvocationRequest(tool_name=tool_name, arguments=arguments),
        context,
    )


def _install_fake_gitlab_module(
    monkeypatch: pytest.MonkeyPatch, gitlab_cls: Any
) -> None:
    fake_module = ModuleType("gitlab")
    fake_module.Gitlab = gitlab_cls
    monkeypatch.setitem(sys.modules, "gitlab", fake_module)


def _context() -> ToolContext:
    return ToolContext(
        invocation_id="inv-1",
        env={
            "GITLAB_BASE_URL": "https://gitlab.example.com",
            "GITLAB_API_TOKEN": "token",
        },
    )


def test_search_gitlab_code_tool_maps_project_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProject:
        path_with_namespace = "group/repo"

        def search(
            self, scope: str, query: str, **kwargs: object
        ) -> list[dict[str, object]]:
            assert scope == "blobs"
            assert query == "needle"
            assert kwargs == {"per_page": 3, "ref": "main"}
            return [
                {
                    "path": "src/app.py",
                    "startline": 7,
                    "data": "needle()",
                    "ref": "main",
                }
            ]

    class FakeProjects:
        def get(self, project: str) -> FakeProject:
            assert project == "group/repo"
            return FakeProject()

    class FakeGitlab:
        def __init__(self, url: str, *, private_token: str) -> None:
            assert url == "https://gitlab.example.com"
            assert private_token != ""
            self.projects = FakeProjects()

    _install_fake_gitlab_module(monkeypatch, FakeGitlab)

    tool_result = _invoke_tool(
        "search_gitlab_code",
        _context(),
        {"project": "group/repo", "query": "needle", "ref": "main", "limit": 2},
    )
    result = SearchGitLabCodeTool.output_model.model_validate(tool_result.output)

    assert result.project == "group/repo"
    assert result.matches[0].path == "src/app.py"
    assert result.matches[0].name == "app.py"
    assert result.matches[0].start_line == 7
    assert result.matches[0].snippet == "needle()"


def test_search_gitlab_code_tool_marks_truncated_when_results_exceed_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProject:
        path_with_namespace = "group/repo"

        def search(
            self, scope: str, query: str, **kwargs: object
        ) -> list[dict[str, object]]:
            assert scope == "blobs"
            assert query == "needle"
            assert kwargs == {"per_page": 3}
            return [
                {"path": f"src/file-{index}.py", "snippet": "needle()"}
                for index in range(3)
            ]

    class FakeProjects:
        def get(self, project: str) -> FakeProject:
            assert project == "group/repo"
            return FakeProject()

    class FakeGitlab:
        def __init__(self, url: str, *, private_token: str) -> None:
            del url, private_token
            self.projects = FakeProjects()

    _install_fake_gitlab_module(monkeypatch, FakeGitlab)

    tool_result = _invoke_tool(
        "search_gitlab_code",
        _context(),
        {"project": "group/repo", "query": "needle", "limit": 2},
    )
    result = SearchGitLabCodeTool.output_model.model_validate(tool_result.output)

    assert len(result.matches) == 2
    assert result.truncated is True


def test_read_gitlab_file_tool_reads_text_and_applies_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFile:
        def decode(self) -> str:
            return "alpha beta gamma"

    class FakeFiles:
        def get(self, *, file_path: str, ref: str) -> FakeFile:
            assert file_path == "docs/readme.txt"
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
            self.projects = FakeProjects()

    _install_fake_gitlab_module(monkeypatch, FakeGitlab)

    tool_result = _invoke_tool(
        "read_gitlab_file",
        _context(),
        {
            "project": "group/repo",
            "file_path": "docs/readme.txt",
            "start_char": 6,
            "end_char": 20,
        },
    )
    result = ReadGitLabFileTool.output_model.model_validate(tool_result.output)

    assert result.project == "group/repo"
    assert result.ref == "main"
    assert result.content == "beta gamma"
    assert result.start_char == 6
    assert result.end_char == 16
    assert result.truncated is False
    assert result.resolved_path == "group/repo@main:docs/readme.txt"


def test_read_gitlab_file_tool_returns_structured_unsupported_for_binary_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFile:
        content = base64.b64encode(b"\x00\x01\x02").decode("ascii")
        encoding = "base64"

    class FakeFiles:
        def get(self, *, file_path: str, ref: str) -> FakeFile:
            del file_path, ref
            return FakeFile()

    class FakeProject:
        default_branch = "main"
        path_with_namespace = "group/repo"
        files = FakeFiles()

    class FakeProjects:
        def get(self, project: str) -> FakeProject:
            del project
            return FakeProject()

    class FakeGitlab:
        def __init__(self, url: str, *, private_token: str) -> None:
            del url, private_token
            self.projects = FakeProjects()

    _install_fake_gitlab_module(monkeypatch, FakeGitlab)

    tool_result = _invoke_tool(
        "read_gitlab_file",
        _context(),
        {"project": "group/repo", "file_path": "bin/data.bin"},
    )
    result = ReadGitLabFileTool.output_model.model_validate(tool_result.output)

    assert result.status == "unsupported"
    assert result.read_kind == "unsupported"
    assert result.content is None
    assert "binary" in (result.error_message or "")


def test_read_gitlab_merge_request_tool_maps_metadata_commits_and_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMergeRequest:
        title = "Add feature"
        description = "Implements the feature"
        state = "opened"
        author = {"name": "Alice"}
        source_branch = "feature"
        target_branch = "main"
        web_url = "https://gitlab.example.com/group/repo/-/merge_requests/12"

        def commits(self) -> list[dict[str, object]]:
            return [
                {
                    "id": "abc123",
                    "short_id": "abc123",
                    "title": "feat: add feature",
                    "author_name": "Alice",
                }
            ]

        def changes(self) -> dict[str, object]:
            return {
                "changes": [
                    {
                        "old_path": "old.py",
                        "new_path": "new.py",
                        "renamed_file": True,
                        "diff": "+hello\n-world\n",
                    }
                ]
            }

    class FakeMergeRequests:
        def get(self, merge_request_iid: int) -> FakeMergeRequest:
            assert merge_request_iid == 12
            return FakeMergeRequest()

    class FakeProject:
        path_with_namespace = "group/repo"
        mergerequests = FakeMergeRequests()

    class FakeProjects:
        def get(self, project: str) -> FakeProject:
            assert project == "group/repo"
            return FakeProject()

    class FakeGitlab:
        def __init__(self, url: str, *, private_token: str) -> None:
            del url, private_token
            self.projects = FakeProjects()

    _install_fake_gitlab_module(monkeypatch, FakeGitlab)

    tool_result = _invoke_tool(
        "read_gitlab_merge_request",
        _context(),
        {"project": "group/repo", "merge_request_iid": 12},
    )
    result = ReadGitLabMergeRequestTool.output_model.model_validate(tool_result.output)

    assert result.project == "group/repo"
    assert result.title == "Add feature"
    assert result.author == "Alice"
    assert result.commits[0].title == "feat: add feature"
    assert result.changed_files[0].new_path == "new.py"
    assert result.changed_files[0].renamed_file is True


def test_read_gitlab_merge_request_tool_marks_truncated_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMergeRequest:
        title = "Add feature"
        description = "Implements the feature"
        state = "opened"
        author = {"name": "Alice"}
        source_branch = "feature"
        target_branch = "main"
        web_url = "https://gitlab.example.com/group/repo/-/merge_requests/12"

        def commits(self) -> list[dict[str, object]]:
            return [
                {
                    "id": f"commit-{index}",
                    "short_id": f"c{index}",
                    "title": f"commit {index}",
                    "author_name": "Alice",
                }
                for index in range(3)
            ]

        def changes(self) -> dict[str, object]:
            return {
                "changes": [
                    {
                        "old_path": f"old-{index}.py",
                        "new_path": f"new-{index}.py",
                        "renamed_file": False,
                        "diff": "+hello\n",
                    }
                    for index in range(3)
                ]
            }

    class FakeMergeRequests:
        def get(self, merge_request_iid: int) -> FakeMergeRequest:
            assert merge_request_iid == 12
            return FakeMergeRequest()

    class FakeProject:
        path_with_namespace = "group/repo"
        mergerequests = FakeMergeRequests()

    class FakeProjects:
        def get(self, project: str) -> FakeProject:
            assert project == "group/repo"
            return FakeProject()

    class FakeGitlab:
        def __init__(self, url: str, *, private_token: str) -> None:
            del url, private_token
            self.projects = FakeProjects()

    _install_fake_gitlab_module(monkeypatch, FakeGitlab)

    tool_result = _invoke_tool(
        "read_gitlab_merge_request",
        _context(),
        {
            "project": "group/repo",
            "merge_request_iid": 12,
            "commit_limit": 2,
            "change_limit": 2,
        },
    )
    result = ReadGitLabMergeRequestTool.output_model.model_validate(tool_result.output)

    assert len(result.commits) == 2
    assert len(result.changed_files) == 2
    assert result.commits_truncated is True
    assert result.changed_files_truncated is True


def test_gitlab_tools_surface_transient_remote_failures_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProjects:
        def get(self, project: str):
            del project
            raise TimeoutError("upstream timed out")

    class FakeGitlab:
        def __init__(self, url: str, *, private_token: str) -> None:
            del url, private_token
            self.projects = FakeProjects()

    _install_fake_gitlab_module(monkeypatch, FakeGitlab)

    result = _execute_tool(
        "search_gitlab_code",
        _context(),
        {"project": "group/repo", "query": "needle"},
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    assert result.error.retryable is True


def test_gitlab_tools_require_context_credentials() -> None:
    result = _execute_tool(
        "search_gitlab_code",
        ToolContext(invocation_id="inv-2"),
        {"project": "group/repo", "query": "needle"},
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.POLICY_DENIED
    assert set(
        result.error.details["policy_decision"]["metadata"]["missing_secrets"]
    ) == {
        "GITLAB_BASE_URL",
        "GITLAB_API_TOKEN",
    }


def test_search_gitlab_code_tool_rejects_unsupported_project_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProject:
        path_with_namespace = "group/repo"

    class FakeProjects:
        def get(self, project: str) -> FakeProject:
            del project
            return FakeProject()

    class FakeGitlab:
        def __init__(self, url: str, *, private_token: str) -> None:
            del url, private_token
            self.projects = FakeProjects()

    _install_fake_gitlab_module(monkeypatch, FakeGitlab)

    result = _execute_tool(
        "search_gitlab_code",
        _context(),
        {"project": "group/repo", "query": "needle"},
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    assert "does not support code search" in result.error.details["exception_message"]


def test_get_gitlab_project_rejects_unsupported_client_shape() -> None:
    with pytest.raises(RuntimeError, match="does not support project reads"):
        gitlab_tools._get_gitlab_project(SimpleNamespace(), "group/repo")


def test_search_project_code_falls_back_to_keyword_signature() -> None:
    class FakeProject:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def search(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
            self.calls.append((args, kwargs))
            if args:
                raise TypeError("positional form unsupported")
            return [{"filename": "src/fallback.py", "snippet": "needle"}]

    project = FakeProject()
    result = gitlab_tools._search_project_code(
        project,
        "needle",
        ref=None,
        limit=3,
    )

    assert result == [{"filename": "src/fallback.py", "snippet": "needle"}]
    assert project.calls == [
        (("blobs", "needle"), {"per_page": 3}),
        ((), {"scope": "blobs", "search": "needle", "per_page": 3}),
    ]


def test_search_project_code_rejects_when_both_signatures_fail() -> None:
    class FakeProject:
        def search(self, *args: object, **kwargs: object) -> list[object]:
            del args, kwargs
            raise TypeError("unsupported")

    with pytest.raises(RuntimeError, match="does not support code search"):
        gitlab_tools._search_project_code(
            FakeProject(),
            "needle",
            ref=None,
            limit=2,
        )


def test_get_project_file_rejects_unsupported_client_shape() -> None:
    with pytest.raises(RuntimeError, match="does not support file reads"):
        gitlab_tools._get_project_file(SimpleNamespace(), "README.md", ref="main")


def test_get_merge_request_rejects_unsupported_client_shape() -> None:
    with pytest.raises(RuntimeError, match="does not support merge request reads"):
        gitlab_tools._get_merge_request(SimpleNamespace(), 7)


def test_get_merge_request_commits_covers_none_list_and_fallback_paths() -> None:
    assert gitlab_tools._get_merge_request_commits(SimpleNamespace()) == []

    manager = SimpleNamespace(list=lambda: [{"id": "abc"}])
    assert gitlab_tools._get_merge_request_commits(
        SimpleNamespace(commits=manager)
    ) == [{"id": "abc"}]

    assert (
        gitlab_tools._get_merge_request_commits(
            SimpleNamespace(commits=SimpleNamespace())
        )
        == []
    )


def test_get_merge_request_changes_covers_none_dict_and_fallback_paths() -> None:
    assert gitlab_tools._get_merge_request_changes(SimpleNamespace()) == []

    assert (
        gitlab_tools._get_merge_request_changes(
            SimpleNamespace(changes=lambda: ["not-a-dict"])
        )
        == []
    )

    assert gitlab_tools._get_merge_request_changes(
        SimpleNamespace(changes={"changes": [{"new_path": "x.py"}]})
    ) == [{"new_path": "x.py"}]

    assert (
        gitlab_tools._get_merge_request_changes(
            SimpleNamespace(changes=SimpleNamespace())
        )
        == []
    )


def test_decode_gitlab_file_content_covers_remaining_branches() -> None:
    binary_text = SimpleNamespace(decode=lambda: "hi\x00there")
    assert gitlab_tools._decode_gitlab_file_content(binary_text) == (
        None,
        len(b"hi\x00there"),
        "Remote file is binary.",
    )

    invalid_bytes = SimpleNamespace(decode=lambda: b"\xff\xfe")
    assert gitlab_tools._decode_gitlab_file_content(invalid_bytes) == (
        None,
        2,
        "Remote file is not UTF-8 text.",
    )

    null_bytes = SimpleNamespace(decode=lambda: b"abc\x00def")
    assert gitlab_tools._decode_gitlab_file_content(null_bytes) == (
        None,
        7,
        "Remote file is binary.",
    )

    ok_bytes = SimpleNamespace(decode=lambda: b"plain text")
    assert gitlab_tools._decode_gitlab_file_content(ok_bytes) == (
        "plain text",
        10,
        None,
    )

    raw_bytes = SimpleNamespace(content=b"raw text", encoding=None)
    assert gitlab_tools._decode_gitlab_file_content(raw_bytes) == (
        "raw text",
        8,
        None,
    )

    raw_string = SimpleNamespace(content="raw string", encoding=None)
    assert gitlab_tools._decode_gitlab_file_content(raw_string) == (
        "raw string",
        10,
        None,
    )

    raw_binary_string = SimpleNamespace(content="abc\x00def", encoding=None)
    assert gitlab_tools._decode_gitlab_file_content(raw_binary_string) == (
        None,
        7,
        "Remote file is binary.",
    )

    missing_content = SimpleNamespace(content=None, encoding=None)
    assert gitlab_tools._decode_gitlab_file_content(missing_content) == (
        None,
        0,
        "GitLab file payload did not include readable content.",
    )


def test_decode_gitlab_file_content_raw_bytes_invalid_utf8() -> None:
    invalid_raw_bytes = SimpleNamespace(content=b"\xff", encoding=None)
    assert gitlab_tools._decode_gitlab_file_content(invalid_raw_bytes) == (
        None,
        1,
        "Remote file is not UTF-8 text.",
    )


def test_decode_gitlab_file_content_raw_bytes_with_null_and_without_null() -> None:
    null_raw_bytes = SimpleNamespace(content=b"a\x00b", encoding=None)
    assert gitlab_tools._decode_gitlab_file_content(null_raw_bytes) == (
        None,
        3,
        "Remote file is binary.",
    )

    clean_raw_bytes = SimpleNamespace(content=b"clean", encoding=None)
    assert gitlab_tools._decode_gitlab_file_content(clean_raw_bytes) == (
        "clean",
        5,
        None,
    )


def test_read_gitlab_file_tool_returns_too_large_when_content_exceeds_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFile:
        def decode(self) -> str:
            return "x" * 20

    class FakeFiles:
        def get(self, *, file_path: str, ref: str) -> FakeFile:
            del file_path, ref
            return FakeFile()

    class FakeProject:
        path = "repo-only"
        default_branch = "main"
        files = FakeFiles()

    class FakeProjects:
        def get(self, project: str) -> FakeProject:
            del project
            return FakeProject()

    class FakeGitlab:
        def __init__(self, url: str, *, private_token: str) -> None:
            del url, private_token
            self.projects = FakeProjects()

    _install_fake_gitlab_module(monkeypatch, FakeGitlab)
    context = _context().model_copy(
        update={"metadata": {"tool_limits": {"max_file_size_characters": 5}}}
    )

    tool_result = _invoke_tool(
        "read_gitlab_file",
        context,
        {"project": "group/repo", "file_path": "README.md"},
    )
    result = ReadGitLabFileTool.output_model.model_validate(tool_result.output)

    assert result.project == "repo-only"
    assert result.status == "too_large"
    assert result.content is None
    assert result.character_count == 20
    assert result.estimated_token_count == 1

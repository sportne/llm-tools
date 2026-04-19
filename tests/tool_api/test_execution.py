"""Focused unit tests for runtime execution primitives."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import llm_tools.tool_api.execution as execution_module
from llm_tools.tool_api import ToolContext
from llm_tools.tool_api.execution import (
    ExecutionServices,
    FilesystemBroker,
    SecretView,
    SubprocessBroker,
    ToolExecutionContext,
    _context_accepts_permit,
    _create_execution_context,
    _ExecutionPermit,
    _issue_execution_permit_for_context,
    build_bitbucket_gateway,
    build_confluence_gateway,
    build_gitlab_gateway,
    build_jira_gateway,
    get_workspace_root,
    resolve_workspace_path,
)
from llm_tools.tool_api.models import SourceProvenanceRef


class _FakeExecutionContext:
    invocation_id = "inv-1"
    workspace: str | None = None
    metadata: dict[str, object] = {}
    secrets = SecretView({})
    services = ExecutionServices()

    def log(self, message: str) -> None:
        del message

    def add_artifact(self, artifact: str) -> None:
        del artifact

    def add_source_provenance(self, entry: SourceProvenanceRef) -> None:
        del entry

    def snapshot_logs(self) -> list[str]:
        return []

    def snapshot_artifacts(self) -> list[str]:
        return []

    def snapshot_source_provenance(self) -> list[SourceProvenanceRef]:
        return []


def test_secret_view_supports_mapping_behavior_and_required_lookup() -> None:
    secrets = SecretView({"VISIBLE": "abc", "USER": "alice"})

    assert len(secrets) == 2
    assert list(secrets) == ["VISIBLE", "USER"]
    assert secrets["VISIBLE"] == "abc"
    assert secrets.get_required("USER") == "alice"

    with pytest.raises(KeyError, match="was not granted"):
        _ = secrets["MISSING"]
    with pytest.raises(ValueError, match="Missing required secret"):
        secrets.get_required("MISSING")


def test_private_execution_permit_rejects_non_runtime_issuer() -> None:
    with pytest.raises(RuntimeError, match="may only be issued by ToolRuntime"):
        _ExecutionPermit(object(), _issuer=object())


def test_create_execution_context_returns_runtime_issued_protocol_instance() -> None:
    context = _create_execution_context(
        invocation_id="inv-1",
        workspace=None,
        metadata={"key": "value"},
        secrets=SecretView({"VISIBLE": "abc"}),
        services=ExecutionServices(),
    )

    assert isinstance(context, ToolExecutionContext)
    assert context.invocation_id == "inv-1"
    assert context.metadata == {"key": "value"}

    context.log("line-1")
    context.add_artifact("artifact-1")
    provenance = SourceProvenanceRef(
        source_kind="local_file",
        source_id="file.txt",
        content_hash="sha256:abc",
    )
    context.add_source_provenance(provenance)

    logs = context.snapshot_logs()
    artifacts = context.snapshot_artifacts()
    emitted = context.snapshot_source_provenance()

    assert logs == ["line-1"]
    assert artifacts == ["artifact-1"]
    assert emitted == [provenance]
    logs.append("mutated")
    artifacts.append("mutated")
    emitted.append(
        SourceProvenanceRef(
            source_kind="other",
            source_id="other",
            content_hash="sha256:def",
        )
    )
    assert context.snapshot_logs() == ["line-1"]
    assert context.snapshot_artifacts() == ["artifact-1"]
    assert context.snapshot_source_provenance() == [provenance]


def test_runtime_permit_helpers_require_runtime_issued_context() -> None:
    context = _create_execution_context(
        invocation_id="inv-1",
        workspace=None,
        metadata={},
        secrets=SecretView({}),
        services=ExecutionServices(),
    )

    permit = _issue_execution_permit_for_context(context)

    assert _context_accepts_permit(context, permit) is True
    assert _context_accepts_permit(context, None) is False

    with pytest.raises(RuntimeError, match="was not issued by ToolRuntime"):
        _issue_execution_permit_for_context(_FakeExecutionContext())
    assert _context_accepts_permit(_FakeExecutionContext(), permit) is False


def test_execution_services_require_methods_raise_without_grants() -> None:
    services = ExecutionServices()

    with pytest.raises(RuntimeError, match="Filesystem access"):
        services.require_filesystem()
    with pytest.raises(RuntimeError, match="Subprocess access"):
        services.require_subprocess()
    with pytest.raises(RuntimeError, match="GitLab access"):
        services.require_gitlab()
    with pytest.raises(RuntimeError, match="Jira access"):
        services.require_jira()
    with pytest.raises(RuntimeError, match="Bitbucket access"):
        services.require_bitbucket()
    with pytest.raises(RuntimeError, match="Confluence access"):
        services.require_confluence()


def test_filesystem_broker_delegates_to_ops_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = ToolContext(invocation_id="inv-1", workspace=str(tmp_path))
    broker = FilesystemBroker(context)
    captured: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(name: str):
        def _inner(*args: object, **kwargs: object) -> object:
            captured.append((name, args, kwargs))
            return SimpleNamespace(name=name, args=args, kwargs=kwargs)

        return _inner

    fake_ops = SimpleNamespace(
        read_file_impl=_record("read"),
        list_directory_impl=_record("list"),
        find_files_impl=_record("find"),
        get_file_info_impl=_record("info"),
    )
    monkeypatch.setattr(
        execution_module.importlib, "import_module", lambda name: fake_ops
    )

    read_result = broker.read_file(
        "note.txt", tool_limits={"limit": 1}, start_char=1, end_char=5
    )
    list_result = broker.list_directory(
        ".",
        source_filters={"hidden": False},
        tool_limits={"limit": 2},
        recursive=True,
        max_depth=3,
    )
    find_result = broker.find_files("*.py", ".", source_filters={}, tool_limits={})
    info_result = broker.get_file_info(["note.txt"], tool_limits={"limit": 4})

    assert read_result.name == "read"
    assert list_result.name == "list"
    assert find_result.name == "find"
    assert info_result.name == "info"
    assert [entry[0] for entry in captured] == ["read", "list", "find", "info"]
    assert captured[0][1][0] == tmp_path.resolve()
    assert captured[1][2]["recursive"] is True


def test_filesystem_broker_write_text_handles_create_and_rejects_existing_file(
    tmp_path: Path,
) -> None:
    context = ToolContext(invocation_id="inv-1", workspace=str(tmp_path))
    broker = FilesystemBroker(context)

    resolved, created, bytes_written = broker.write_text(
        "docs/note.txt",
        content="hello",
        encoding="utf-8",
        overwrite=False,
        create_parents=True,
    )

    assert created is True
    assert bytes_written == len(b"hello")
    assert resolved.read_text(encoding="utf-8") == "hello"

    with pytest.raises(FileExistsError, match="already exists"):
        broker.write_text(
            "docs/note.txt",
            content="new",
            encoding="utf-8",
            overwrite=False,
            create_parents=True,
        )


def test_subprocess_broker_validates_workspace_boundary_and_delegates_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolContext(invocation_id="inv-1", workspace=str(tmp_path))
    broker = SubprocessBroker(context)
    seen: dict[str, object] = {}

    def fake_run(
        args: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int | None,
    ) -> subprocess.CompletedProcess[str]:
        seen.update(
            {
                "args": args,
                "cwd": cwd,
                "capture_output": capture_output,
                "text": text,
                "check": check,
                "timeout": timeout,
            }
        )
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="ok", stderr=""
        )

    monkeypatch.setattr(execution_module.subprocess, "run", fake_run)

    completed = broker.run(["git", "status"], cwd=tmp_path, timeout_seconds=5)

    assert completed.stdout == "ok"
    assert seen == {
        "args": ["git", "status"],
        "cwd": tmp_path.resolve(),
        "capture_output": True,
        "text": True,
        "check": False,
        "timeout": 5,
    }

    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="escapes workspace"):
        broker.run(["git", "status"], cwd=outside)


def test_gateway_builders_construct_clients_from_scoped_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gitlab_module = ModuleType("gitlab")
    gitlab_module.Gitlab = lambda url, private_token: {
        "provider": "gitlab",
        "url": url,
        "private_token": private_token,
    }
    atlassian_module = ModuleType("atlassian")
    atlassian_module.Jira = lambda **kwargs: {"provider": "jira", **kwargs}
    atlassian_module.Bitbucket = lambda **kwargs: {"provider": "bitbucket", **kwargs}
    atlassian_module.Confluence = lambda **kwargs: {"provider": "confluence", **kwargs}
    monkeypatch.setitem(sys.modules, "gitlab", gitlab_module)
    monkeypatch.setitem(sys.modules, "atlassian", atlassian_module)

    gitlab_gateway = build_gitlab_gateway(
        SecretView({"GITLAB_BASE_URL": "https://gitlab", "GITLAB_API_TOKEN": "token"})
    )
    jira_gateway = build_jira_gateway(
        SecretView(
            {
                "JIRA_BASE_URL": "https://jira",
                "JIRA_USERNAME": "alice",
                "JIRA_API_TOKEN": "token",
            }
        )
    )
    bitbucket_gateway = build_bitbucket_gateway(
        SecretView(
            {
                "BITBUCKET_BASE_URL": "https://bitbucket",
                "BITBUCKET_USERNAME": "alice",
                "BITBUCKET_API_TOKEN": "token",
            }
        )
    )
    confluence_gateway = build_confluence_gateway(
        SecretView(
            {
                "CONFLUENCE_BASE_URL": "https://confluence",
                "CONFLUENCE_USERNAME": "alice",
                "CONFLUENCE_API_TOKEN": "token",
            }
        )
    )

    assert gitlab_gateway.client == {
        "provider": "gitlab",
        "url": "https://gitlab",
        "private_token": "token",
    }
    assert jira_gateway.client["provider"] == "jira"
    assert bitbucket_gateway.client["provider"] == "bitbucket"
    assert confluence_gateway.client["provider"] == "confluence"


def test_gateway_builders_reject_blank_required_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atlassian_module = ModuleType("atlassian")
    atlassian_module.Jira = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "atlassian", atlassian_module)

    with pytest.raises(ValueError, match="Missing required Jira credential"):
        build_jira_gateway(
            SecretView(
                {
                    "JIRA_BASE_URL": "https://jira",
                    "JIRA_USERNAME": "alice",
                    "JIRA_API_TOKEN": "",
                }
            )
        )


def test_execution_path_helpers_resolve_workspace_success_cases(tmp_path: Path) -> None:
    context = ToolContext(invocation_id="inv-1", workspace=f"  {tmp_path}  ")
    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")

    assert get_workspace_root(context) == tmp_path.resolve()
    assert resolve_workspace_path(context, "note.txt") == note.resolve()
    assert resolve_workspace_path(context, str(note.resolve())) == note.resolve()

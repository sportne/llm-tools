"""Runtime-issued execution contexts, permits, and brokers."""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from llm_tools.tool_api.models import SourceProvenanceRef, ToolContext

_RUNTIME_CONTEXT_ISSUER = object()
_RUNTIME_PERMIT_ISSUER = object()

_JIRA_SECRET_KEYS = ("JIRA_BASE_URL", "JIRA_API_TOKEN")
_BITBUCKET_SECRET_KEYS = (
    "BITBUCKET_BASE_URL",
    "BITBUCKET_API_TOKEN",
)
_CONFLUENCE_SECRET_KEYS = (
    "CONFLUENCE_BASE_URL",
    "CONFLUENCE_API_TOKEN",
)
_GITLAB_SECRET_KEYS = ("GITLAB_BASE_URL", "GITLAB_API_TOKEN")


HostToolContext = ToolContext


class SecretView(Mapping[str, str]):
    """Read-only view over the secrets explicitly granted to one tool."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> str:
        try:
            return self._values[key]
        except KeyError as exc:
            raise KeyError(f"Secret '{key}' was not granted to this tool.") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get_required(self, key: str) -> str:
        """Return one granted secret or raise a stable error."""
        try:
            return self[key]
        except KeyError as exc:
            raise ValueError(f"Missing required secret '{key}'.") from exc


class _ExecutionPermit:
    """Opaque runtime-issued token required for direct tool execution."""

    __slots__ = ("_token",)

    def __init__(self, token: object, *, _issuer: object) -> None:
        if _issuer is not _RUNTIME_PERMIT_ISSUER:
            raise RuntimeError("Execution permits may only be issued by ToolRuntime.")
        self._token = token


class FilesystemBroker:
    """Runtime-controlled filesystem access rooted at one host context."""

    def __init__(self, host_context: HostToolContext) -> None:
        self._host_context = host_context

    def workspace_root(self) -> Path:
        return get_workspace_root(self._host_context)

    def resolve_path(
        self,
        path: str,
        *,
        expect_directory: bool | None = None,
        must_exist: bool = True,
    ) -> Path:
        return resolve_workspace_path(
            self._host_context,
            path,
            expect_directory=expect_directory,
            must_exist=must_exist,
        )

    def read_file(
        self,
        path: str,
        *,
        tool_limits: Any,
        start_char: int | None,
        end_char: int | None,
    ) -> Any:
        ops = importlib.import_module("llm_tools.tools.filesystem._ops")

        return ops.read_file_impl(
            self.workspace_root(),
            path,
            tool_limits=tool_limits,
            start_char=start_char,
            end_char=end_char,
        )

    def list_directory(
        self,
        path: str,
        *,
        source_filters: Any,
        tool_limits: Any,
        recursive: bool,
        max_depth: int | None,
    ) -> Any:
        ops = importlib.import_module("llm_tools.tools.filesystem._ops")

        return ops.list_directory_impl(
            self.workspace_root(),
            path,
            source_filters=source_filters,
            tool_limits=tool_limits,
            recursive=recursive,
            max_depth=max_depth,
        )

    def find_files(
        self,
        pattern: str,
        path: str,
        *,
        source_filters: Any,
        tool_limits: Any,
    ) -> Any:
        ops = importlib.import_module("llm_tools.tools.filesystem._ops")

        return ops.find_files_impl(
            self.workspace_root(),
            pattern,
            path,
            source_filters=source_filters,
            tool_limits=tool_limits,
        )

    def get_file_info(
        self,
        path: str | list[str],
        *,
        tool_limits: Any,
    ) -> Any:
        ops = importlib.import_module("llm_tools.tools.filesystem._ops")

        return ops.get_file_info_impl(
            self.workspace_root(),
            path,
            tool_limits=tool_limits,
        )

    def write_text(
        self,
        path: str,
        *,
        content: str,
        encoding: str,
        overwrite: bool,
        create_parents: bool,
    ) -> tuple[Path, bool, int]:
        resolved = self.resolve_path(
            path,
            expect_directory=False,
            must_exist=False,
        )
        created = not resolved.exists()
        if resolved.exists() and not overwrite:
            raise FileExistsError(f"Path '{resolved}' already exists.")

        if not resolved.parent.exists():
            if not create_parents:
                raise FileNotFoundError(
                    f"Parent directory '{resolved.parent}' does not exist."
                )
            resolved.parent.mkdir(parents=True, exist_ok=True)

        resolved.write_text(content, encoding=encoding)
        return resolved, created, len(content.encode(encoding))


class SubprocessBroker:
    """Runtime-controlled subprocess execution confined to the workspace."""

    def __init__(self, host_context: HostToolContext) -> None:
        self._host_context = host_context

    def run(
        self,
        args: list[str],
        *,
        cwd: Path,
        timeout_seconds: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        workspace_root = get_workspace_root(self._host_context)
        resolved_cwd = cwd.resolve()
        if not resolved_cwd.is_relative_to(workspace_root):
            raise ValueError(
                f"Subprocess cwd '{resolved_cwd}' escapes workspace '{workspace_root}'."
            )
        return subprocess.run(
            args,
            cwd=resolved_cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )


@runtime_checkable
class GitLabGateway(Protocol):
    """Runtime-owned gateway for GitLab-backed tools."""

    @property
    def client(self) -> Any: ...


@runtime_checkable
class JiraGateway(Protocol):
    """Runtime-owned gateway for Jira-backed tools."""

    @property
    def client(self) -> Any: ...


@runtime_checkable
class BitbucketGateway(Protocol):
    """Runtime-owned gateway for Bitbucket-backed tools."""

    @property
    def client(self) -> Any: ...


@runtime_checkable
class ConfluenceGateway(Protocol):
    """Runtime-owned gateway for Confluence-backed tools."""

    @property
    def client(self) -> Any: ...


class _StaticClientGateway:
    """Small concrete gateway wrapper exposing one provider client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        return self._client


@dataclass(slots=True)
class ExecutionServices:
    """Capability brokers and provider gateways granted to one tool."""

    filesystem: FilesystemBroker | None = None
    subprocess: SubprocessBroker | None = None
    gitlab: GitLabGateway | None = None
    jira: JiraGateway | None = None
    bitbucket: BitbucketGateway | None = None
    confluence: ConfluenceGateway | None = None

    def require_filesystem(self) -> FilesystemBroker:
        if self.filesystem is None:
            raise RuntimeError("Filesystem access was not granted to this tool.")
        return self.filesystem

    def require_subprocess(self) -> SubprocessBroker:
        if self.subprocess is None:
            raise RuntimeError("Subprocess access was not granted to this tool.")
        return self.subprocess

    def require_gitlab(self) -> GitLabGateway:
        if self.gitlab is None:
            raise RuntimeError("GitLab access was not granted to this tool.")
        return self.gitlab

    def require_jira(self) -> JiraGateway:
        if self.jira is None:
            raise RuntimeError("Jira access was not granted to this tool.")
        return self.jira

    def require_bitbucket(self) -> BitbucketGateway:
        if self.bitbucket is None:
            raise RuntimeError("Bitbucket access was not granted to this tool.")
        return self.bitbucket

    def require_confluence(self) -> ConfluenceGateway:
        if self.confluence is None:
            raise RuntimeError("Confluence access was not granted to this tool.")
        return self.confluence


@runtime_checkable
class ToolExecutionContext(Protocol):
    """Runtime-issued execution context visible to one tool invocation."""

    invocation_id: str
    workspace: str | None
    metadata: dict[str, Any]
    secrets: SecretView
    services: ExecutionServices

    def log(self, message: str) -> None: ...

    def add_artifact(self, artifact: str) -> None: ...

    def add_source_provenance(self, entry: SourceProvenanceRef) -> None: ...

    def snapshot_logs(self) -> list[str]: ...

    def snapshot_artifacts(self) -> list[str]: ...

    def snapshot_source_provenance(self) -> list[SourceProvenanceRef]: ...


@dataclass(slots=True)
class _RuntimeToolExecutionContext:
    """Private runtime-issued execution context implementation."""

    invocation_id: str
    workspace: str | None
    metadata: dict[str, Any]
    secrets: SecretView
    services: ExecutionServices
    _permit_token: object = field(repr=False)
    _logs: list[str] = field(default_factory=list, repr=False)
    _artifacts: list[str] = field(default_factory=list, repr=False)
    _source_provenance: list[SourceProvenanceRef] = field(
        default_factory=list,
        repr=False,
    )
    _issuer: object = field(repr=False, kw_only=True)

    def __post_init__(self) -> None:
        if self._issuer is not _RUNTIME_CONTEXT_ISSUER:
            raise RuntimeError(
                "Tool execution contexts may only be issued by ToolRuntime."
            )

    def log(self, message: str) -> None:
        """Append one log line to the tool-visible observability sink."""
        self._logs.append(message)

    def add_artifact(self, artifact: str) -> None:
        """Append one artifact reference to the tool-visible observability sink."""
        self._artifacts.append(artifact)

    def add_source_provenance(self, entry: SourceProvenanceRef) -> None:
        """Append one source provenance reference emitted by the tool."""
        self._source_provenance.append(entry)

    def snapshot_logs(self) -> list[str]:
        """Return a stable copy of tool-emitted logs."""
        return list(self._logs)

    def snapshot_artifacts(self) -> list[str]:
        """Return a stable copy of tool-emitted artifacts."""
        return list(self._artifacts)

    def snapshot_source_provenance(self) -> list[SourceProvenanceRef]:
        """Return a stable copy of tool-emitted provenance."""
        return [entry.model_copy(deep=True) for entry in self._source_provenance]


def _secret_values(
    secrets: SecretView,
    keys: Iterable[str],
    *,
    label: str,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for key in keys:
        value = secrets.get_required(key)
        if value == "":
            raise ValueError(f"Missing required {label} credential '{key}'.")
        values[key] = value
    return values


def build_gitlab_gateway(secrets: SecretView) -> GitLabGateway:
    """Construct a GitLab client gateway from scoped secrets."""
    values = _secret_values(secrets, _GITLAB_SECRET_KEYS, label="GitLab")
    import gitlab

    return _StaticClientGateway(
        gitlab.Gitlab(
            values["GITLAB_BASE_URL"], private_token=values["GITLAB_API_TOKEN"]
        )
    )


def build_jira_gateway(secrets: SecretView) -> JiraGateway:
    """Construct a Jira client gateway from scoped secrets."""
    values = _secret_values(secrets, _JIRA_SECRET_KEYS, label="Jira")
    from atlassian import Jira

    username = secrets.get("JIRA_USERNAME")
    kwargs: dict[str, Any] = {
        "url": values["JIRA_BASE_URL"],
    }
    if username:
        kwargs.update(username=username, password=values["JIRA_API_TOKEN"])
    else:
        kwargs["token"] = values["JIRA_API_TOKEN"]
    return _StaticClientGateway(Jira(**kwargs))


def build_bitbucket_gateway(secrets: SecretView) -> BitbucketGateway:
    """Construct a Bitbucket client gateway from scoped secrets."""
    values = _secret_values(secrets, _BITBUCKET_SECRET_KEYS, label="Bitbucket")
    from atlassian import Bitbucket

    username = secrets.get("BITBUCKET_USERNAME")
    bitbucket_cls = cast(Any, Bitbucket)
    kwargs: dict[str, Any] = {
        "url": values["BITBUCKET_BASE_URL"],
    }
    if username:
        kwargs.update(username=username, password=values["BITBUCKET_API_TOKEN"])
    else:
        kwargs["token"] = values["BITBUCKET_API_TOKEN"]
    return _StaticClientGateway(bitbucket_cls(**kwargs))


def build_confluence_gateway(secrets: SecretView) -> ConfluenceGateway:
    """Construct a Confluence client gateway from scoped secrets."""
    values = _secret_values(secrets, _CONFLUENCE_SECRET_KEYS, label="Confluence")
    from atlassian import Confluence

    username = secrets.get("CONFLUENCE_USERNAME")
    confluence_cls = cast(Any, Confluence)
    kwargs: dict[str, Any] = {
        "url": values["CONFLUENCE_BASE_URL"],
    }
    if username:
        kwargs.update(username=username, password=values["CONFLUENCE_API_TOKEN"])
    else:
        kwargs["token"] = values["CONFLUENCE_API_TOKEN"]
    return _StaticClientGateway(confluence_cls(**kwargs))


def _create_execution_context(
    *,
    invocation_id: str,
    workspace: str | None,
    metadata: dict[str, Any],
    secrets: SecretView,
    services: ExecutionServices,
) -> ToolExecutionContext:
    """Return one private runtime-issued execution context."""
    return _RuntimeToolExecutionContext(
        invocation_id=invocation_id,
        workspace=workspace,
        metadata=metadata,
        secrets=secrets,
        services=services,
        _permit_token=object(),
        _issuer=_RUNTIME_CONTEXT_ISSUER,
    )


def _issue_execution_permit_for_context(
    context: ToolExecutionContext,
) -> _ExecutionPermit:
    """Return one permit for one runtime-issued execution context."""
    if not isinstance(context, _RuntimeToolExecutionContext):
        raise RuntimeError("Tool execution context was not issued by ToolRuntime.")
    return _ExecutionPermit(context._permit_token, _issuer=_RUNTIME_PERMIT_ISSUER)


def _context_accepts_permit(
    context: ToolExecutionContext,
    permit: _ExecutionPermit | None,
) -> bool:
    """Return whether the supplied permit matches one runtime-issued context."""
    return (
        permit is not None
        and isinstance(context, _RuntimeToolExecutionContext)
        and permit._token is context._permit_token
    )


def get_workspace_root(context: HostToolContext | ToolExecutionContext) -> Path:
    """Return the resolved workspace root for one execution scope."""
    workspace = (context.workspace or "").strip()
    if workspace == "":
        raise ValueError("No workspace configured for local tool execution.")
    root = Path(workspace).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(
            f"Workspace root '{root}' does not exist or is not a directory."
        )
    return root


def resolve_workspace_path(
    context: HostToolContext | ToolExecutionContext,
    path: str,
    *,
    expect_directory: bool | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a user path relative to the workspace and enforce the root boundary."""
    root = get_workspace_root(context)
    candidate = Path(path).expanduser()
    resolved = (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (root / candidate).resolve(strict=False)
    )

    if not resolved.is_relative_to(root):
        raise ValueError(f"Path '{path}' resolves outside the workspace root '{root}'.")

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path '{resolved}' does not exist.")

    if expect_directory is True and not resolved.is_dir():
        raise NotADirectoryError(f"Path '{resolved}' is not a directory.")

    if expect_directory is False and resolved.exists() and resolved.is_dir():
        raise IsADirectoryError(f"Path '{resolved}' is a directory, not a file.")

    return resolved


@dataclass(slots=True, frozen=True)
class RuntimeInspection:
    """Resolved tool metadata plus policy verdict for one invocation."""

    tool_name: str
    tool_version: str
    policy_decision: Any

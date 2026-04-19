"""Streamlit app shell for the assistant-focused chat experience."""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from llm_tools.apps.assistant_config import (
    StreamlitAssistantConfig,
    load_streamlit_assistant_config,
)
from llm_tools.apps.assistant_prompts import build_assistant_system_prompt
from llm_tools.apps.assistant_runtime import (
    build_assistant_available_tool_specs,
    build_assistant_context,
    build_assistant_executor,
    build_assistant_policy,
    build_live_harness_provider,
    build_tool_capabilities,
    resolve_assistant_default_enabled_tools,
)
from llm_tools.apps.chat_presentation import format_citation, pretty_json
from llm_tools.apps.chat_runtime import create_provider
from llm_tools.apps.streamlit_chat.models import (
    StreamlitInspectorEntry,
    StreamlitPersistedSessionRecord,
    StreamlitPreferences,
    StreamlitRuntimeConfig,
    StreamlitSessionIndex,
    StreamlitSessionSummary,
    StreamlitTranscriptEntry,
)
from llm_tools.harness_api import (
    ApprovalResolution,
    BudgetPolicy,
    FileHarnessStateStore,
    HarnessSessionCreateRequest,
    HarnessSessionInspection,
    HarnessSessionInspectRequest,
    HarnessSessionListRequest,
    HarnessSessionListResult,
    HarnessSessionResumeRequest,
    HarnessSessionRunRequest,
    HarnessSessionService,
    HarnessSessionStopRequest,
)
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import SideEffectClass, ToolSpec
from llm_tools.workflow_api import (
    ChatSessionState,
    ChatTokenUsage,
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowApprovalState,
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
    run_interactive_chat_session_turn,
)
from llm_tools.workflow_api.chat_session import (
    ChatSessionTurnRunner,
    ModelTurnProvider,
)

_APP_STATE_SLOT = "llm_tools_streamlit_assistant_app_state"
_ACTIVE_TURN_STATE_SLOT = "llm_tools_streamlit_assistant_active_turn"
_SECRET_CACHE_STATE_SLOT = "llm_tools_streamlit_assistant_secret_cache"  # noqa: S105
_STREAMLIT_BROWSER_USAGE_STATS_FLAG = "--browser.gatherUsageStats=false"
_STREAMLIT_TOOLBAR_MODE_FLAG = "--client.toolbarMode=minimal"
_POLL_INTERVAL_SECONDS = 1.0
_DEFAULT_THEME_MODE: Literal["dark", "light"] = "light"
_STORAGE_ENV_VAR = "LLM_TOOLS_STREAMLIT_ASSISTANT_STATE_DIR"
_SESSION_STORAGE_DIR_NAME = "sessions"


@dataclass(slots=True)
class AssistantTurnState:
    """Mutable UI state for one assistant session."""

    busy: bool = False
    status_text: str = ""
    pending_approval: ChatWorkflowApprovalState | None = None
    approval_decision_in_flight: bool = False
    active_turn_number: int = 0
    pending_interrupt_draft: str | None = None
    cancelling: bool = False


@dataclass(slots=True)
class AssistantQueuedEvent:
    """One serialized workflow event queued from a worker thread."""

    kind: Literal[
        "status",
        "approval_requested",
        "approval_resolved",
        "inspector",
        "result",
        "error",
        "complete",
    ]
    payload: object
    turn_number: int
    session_id: str


@dataclass(slots=True)
class AssistantActiveTurnHandle:
    """Background runner handle stored in Streamlit session state."""

    session_id: str
    runner: ChatSessionTurnRunner
    event_queue: queue.Queue[AssistantQueuedEvent]
    thread: threading.Thread
    turn_number: int


@dataclass(slots=True)
class StreamlitAssistantTurnOutcome:
    """One processed assistant turn plus any transcript-side system messages."""

    session_state: ChatSessionState
    transcript_entries: list[StreamlitTranscriptEntry]
    token_usage: ChatTokenUsage | None = None


@dataclass(slots=True)
class AssistantWorkspaceState:
    """In-memory assistant app state layered over persisted session files."""

    sessions: dict[str, StreamlitPersistedSessionRecord]
    session_order: list[str]
    active_session_id: str
    preferences: StreamlitPreferences
    turn_states: dict[str, AssistantTurnState] = field(default_factory=dict)
    drafts: dict[str, str] = field(default_factory=dict)
    startup_notices: list[str] = field(default_factory=list)


class AssistantResearchSessionController:
    """Thin app-facing wrapper over the public harness session service."""

    def __init__(
        self,
        *,
        service_factory: Callable[[], HarnessSessionService],
        budget_policy: BudgetPolicy,
        include_replay_by_default: bool,
        list_limit: int,
    ) -> None:
        self._service_factory = service_factory
        self._budget_policy = budget_policy
        self._include_replay_by_default = include_replay_by_default
        self._list_limit = list_limit

    def launch(self, *, prompt: str) -> HarnessSessionInspection:
        service = self._service_factory()
        created = service.create_session(
            HarnessSessionCreateRequest(
                title=_title_from_prompt(prompt),
                intent=prompt,
                budget_policy=self._budget_policy,
            )
        )
        service.run_session(HarnessSessionRunRequest(session_id=created.session_id))
        return service.inspect_session(
            HarnessSessionInspectRequest(
                session_id=created.session_id,
                include_replay=self._include_replay_by_default,
            )
        )

    def list_recent(self) -> HarnessSessionListResult:
        service = self._service_factory()
        return service.list_sessions(
            HarnessSessionListRequest(
                limit=self._list_limit,
                include_replay=self._include_replay_by_default,
            )
        )

    def inspect(self, session_id: str) -> HarnessSessionInspection:
        service = self._service_factory()
        return service.inspect_session(
            HarnessSessionInspectRequest(
                session_id=session_id,
                include_replay=self._include_replay_by_default,
            )
        )

    def resume(
        self,
        session_id: str,
        *,
        approval_resolution: ApprovalResolution | None = None,
    ) -> HarnessSessionInspection:
        service = self._service_factory()
        service.resume_session(
            HarnessSessionResumeRequest(
                session_id=session_id,
                approval_resolution=approval_resolution,
            )
        )
        return self.inspect(session_id)

    def stop(self, session_id: str) -> HarnessSessionInspection:
        service = self._service_factory()
        return service.stop_session(HarnessSessionStopRequest(session_id=session_id))

    @staticmethod
    def summary_text(inspection: HarnessSessionInspection) -> str:
        summary = inspection.summary
        lines = [
            f"Research session: {summary.session_id}",
            f"Stop reason: {summary.stop_reason.value if summary.stop_reason else 'running'}",
            f"Turns: {summary.total_turns}",
            ("Completed tasks: " + (", ".join(summary.completed_task_ids) or "none")),
            f"Active tasks: {', '.join(summary.active_task_ids) or 'none'}",
        ]
        if summary.pending_approval_ids:
            lines.append(
                "Pending approvals: " + ", ".join(summary.pending_approval_ids)
            )
        if summary.latest_decision_summary:
            lines.append(f"Latest decision: {summary.latest_decision_summary}")
        return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by the bootstrap and script entrypoints."""
    parser = argparse.ArgumentParser(
        prog="llm-tools-streamlit-assistant",
        description="Streamlit assistant with optional proprietary-data tools.",
    )
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--directory", dest="directory_override", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--provider")
    parser.add_argument("--model", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--api-base-url", type=str)
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--max-tool-round-trips", type=int)
    parser.add_argument("--max-tool-calls-per-round", type=int)
    parser.add_argument("--max-total-tool-calls-per-turn", type=int)
    parser.add_argument("--max-entries-per-call", type=int)
    parser.add_argument("--max-recursive-depth", type=int)
    parser.add_argument("--max-search-matches", type=int)
    parser.add_argument("--max-read-lines", type=int)
    parser.add_argument("--max-file-size-characters", type=int)
    parser.add_argument("--max-tool-result-chars", type=int)
    return parser


def _resolve_assistant_config(args: argparse.Namespace) -> StreamlitAssistantConfig:
    base_config = (
        load_streamlit_assistant_config(args.config)
        if args.config is not None
        else StreamlitAssistantConfig()
    )
    raw = base_config.model_dump(mode="python")
    raw.setdefault("llm", {})
    raw.setdefault("session", {})
    raw.setdefault("tool_limits", {})
    if args.provider is not None:
        raw["llm"]["provider"] = args.provider
    if args.model is not None:
        raw["llm"]["model_name"] = args.model
    if args.temperature is not None:
        raw["llm"]["temperature"] = args.temperature
    if args.api_base_url is not None:
        raw["llm"]["api_base_url"] = args.api_base_url
    for field_name in (
        "max_context_tokens",
        "max_tool_round_trips",
        "max_tool_calls_per_round",
        "max_total_tool_calls_per_turn",
    ):
        value = getattr(args, field_name)
        if value is not None:
            raw["session"][field_name] = value
    for field_name in (
        "max_entries_per_call",
        "max_recursive_depth",
        "max_search_matches",
        "max_read_lines",
        "max_file_size_characters",
        "max_tool_result_chars",
    ):
        value = getattr(args, field_name)
        if value is not None:
            raw["tool_limits"][field_name] = value
    return StreamlitAssistantConfig.model_validate(raw)


def _resolve_root_argument(
    args: argparse.Namespace,
    config: StreamlitAssistantConfig,
) -> Path | None:
    candidate = args.directory_override or args.directory
    if candidate is None:
        default_root = config.workspace.default_root
        if default_root is None:
            return None
        return Path(default_root).expanduser().resolve()
    resolved_candidate = (
        candidate if isinstance(candidate, Path) else Path(str(candidate))
    )
    return resolved_candidate.expanduser().resolve()


def _streamlit_module() -> Any:  # pragma: no cover
    import streamlit as streamlit

    return streamlit


def _page_config() -> dict[str, object]:  # pragma: no cover
    return {
        "page_title": "llm-tools assistant",
        "page_icon": "AI",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
        "menu_items": {"Get help": None, "Report a bug": None, "About": None},
    }


def _all_tool_specs() -> dict[str, ToolSpec]:
    return build_assistant_available_tool_specs()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _storage_root() -> Path:
    override = os.getenv(_STORAGE_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".llm-tools" / "assistant" / "streamlit").resolve()


def _sessions_dir() -> Path:
    return _storage_root() / _SESSION_STORAGE_DIR_NAME


def _preferences_path() -> Path:
    return _storage_root() / "preferences.json"


def _index_path() -> Path:
    return _storage_root() / "index.json"


def _session_path(session_id: str) -> Path:
    return _sessions_dir() / f"{session_id}.json"


def _research_store_dir(config: StreamlitAssistantConfig) -> Path:
    if config.research.store_dir is not None:
        return Path(config.research.store_dir).expanduser().resolve()
    return (_storage_root() / "research").resolve()


def _read_model_file(path: Path, model_type: Any) -> Any | None:
    if not path.exists():
        return None
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def _sync_summary_fields(record: StreamlitPersistedSessionRecord) -> None:
    record.summary.root_path = record.runtime.root_path
    record.summary.provider = record.runtime.provider
    record.summary.model_name = record.runtime.model_name
    record.summary.message_count = len(
        [entry for entry in record.transcript if entry.role in {"user", "assistant"}]
    )


def _touch_record(record: StreamlitPersistedSessionRecord) -> None:
    _sync_summary_fields(record)
    record.summary.updated_at = _now_iso()


def _remember_runtime_preferences(
    preferences: StreamlitPreferences,
    runtime: StreamlitRuntimeConfig,
) -> None:
    if runtime.root_path is not None:
        preferences.recent_roots = _dedupe_preserve(
            [runtime.root_path, *preferences.recent_roots]
        )[:12]
    provider_key = runtime.provider.value
    preferences.recent_models[provider_key] = _dedupe_preserve(
        [runtime.model_name, *preferences.recent_models.get(provider_key, [])]
    )[:12]
    if runtime.api_base_url:
        preferences.recent_base_urls[provider_key] = _dedupe_preserve(
            [runtime.api_base_url, *preferences.recent_base_urls.get(provider_key, [])]
        )[:12]


def _default_runtime_config(
    config: StreamlitAssistantConfig,
    *,
    root_path: Path | None,
) -> StreamlitRuntimeConfig:
    default_approvals = set(config.policy.require_approval_for).union(
        {SideEffectClass.LOCAL_WRITE, SideEffectClass.EXTERNAL_WRITE}
    )
    return StreamlitRuntimeConfig(
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        api_base_url=config.llm.api_base_url,
        root_path=str(root_path) if root_path is not None else None,
        enabled_tools=sorted(resolve_assistant_default_enabled_tools(config)),
        require_approval_for=default_approvals,
        allow_network=False,
        allow_filesystem=root_path is not None,
        allow_subprocess=False,
        inspector_open=config.ui.inspector_open_by_default,
    )


def _new_session_record(
    session_id: str,
    runtime: StreamlitRuntimeConfig,
) -> StreamlitPersistedSessionRecord:
    now = _now_iso()
    summary = StreamlitSessionSummary(
        session_id=session_id,
        title="New assistant chat",
        created_at=now,
        updated_at=now,
        root_path=runtime.root_path,
        provider=runtime.provider,
        model_name=runtime.model_name,
        message_count=0,
    )
    return StreamlitPersistedSessionRecord(summary=summary, runtime=runtime)


def _load_workspace_state(
    *,
    root_path: Path | None,
    config: StreamlitAssistantConfig,
) -> AssistantWorkspaceState:
    startup_notices: list[str] = []
    try:
        preferences = _read_model_file(_preferences_path(), StreamlitPreferences)
    except Exception as exc:
        preferences = None
        startup_notices.append(f"Unable to load preferences: {exc}")
    if preferences is None:
        preferences = StreamlitPreferences(theme_mode=_DEFAULT_THEME_MODE)
    try:
        index = _read_model_file(_index_path(), StreamlitSessionIndex)
    except Exception as exc:
        index = None
        startup_notices.append(f"Unable to load session index: {exc}")
    if index is None:
        index = StreamlitSessionIndex()

    sessions: dict[str, StreamlitPersistedSessionRecord] = {}
    session_order: list[str] = []
    for session_id in index.session_order:
        try:
            record = _read_model_file(
                _session_path(session_id), StreamlitPersistedSessionRecord
            )
        except Exception as exc:
            startup_notices.append(
                f"Skipped unreadable assistant session {session_id}: {exc}"
            )
            continue
        if record is None:
            startup_notices.append(f"Skipped missing assistant session {session_id}.")
            continue
        sessions[session_id] = record
        session_order.append(session_id)

    if not session_order:
        session_id = f"session-{uuid4().hex[:12]}"
        runtime = _default_runtime_config(config, root_path=root_path)
        record = _new_session_record(session_id, runtime)
        sessions[session_id] = record
        session_order = [session_id]

    active_session_id = index.active_session_id or session_order[0]
    if active_session_id not in sessions:
        active_session_id = session_order[0]

    turn_states = {session_id: AssistantTurnState() for session_id in session_order}
    return AssistantWorkspaceState(
        sessions=sessions,
        session_order=session_order,
        active_session_id=active_session_id,
        preferences=preferences,
        turn_states=turn_states,
        startup_notices=startup_notices,
    )


def _save_workspace_state(app_state: AssistantWorkspaceState) -> None:
    storage_root = _storage_root()
    sessions_dir = _sessions_dir()
    sessions_dir.mkdir(parents=True, exist_ok=True)
    storage_root.mkdir(parents=True, exist_ok=True)

    for session_id in app_state.session_order:
        if session_id not in app_state.sessions:
            continue
        record = app_state.sessions[session_id]
        _sync_summary_fields(record)
        _session_path(session_id).write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )

    index = StreamlitSessionIndex(
        active_session_id=app_state.active_session_id,
        session_order=list(app_state.session_order),
        summaries=[
            app_state.sessions[session_id].summary
            for session_id in app_state.session_order
            if session_id in app_state.sessions
        ],
    )
    _index_path().write_text(index.model_dump_json(indent=2), encoding="utf-8")
    _preferences_path().write_text(
        app_state.preferences.model_dump_json(indent=2),
        encoding="utf-8",
    )

    active_ids = set(app_state.session_order)
    for candidate in sessions_dir.glob("*.json"):
        if candidate.stem not in active_ids:
            candidate.unlink(missing_ok=True)


def _active_session(
    app_state: AssistantWorkspaceState,
) -> StreamlitPersistedSessionRecord:
    return app_state.sessions[app_state.active_session_id]


def _turn_state_for(
    app_state: AssistantWorkspaceState,
    session_id: str,
) -> AssistantTurnState:
    return app_state.turn_states.setdefault(session_id, AssistantTurnState())


def _create_session(
    app_state: AssistantWorkspaceState,
    *,
    template_runtime: StreamlitRuntimeConfig,
) -> str:
    session_id = f"session-{uuid4().hex[:12]}"
    runtime = template_runtime.model_copy(deep=True)
    record = _new_session_record(session_id, runtime)
    app_state.sessions[session_id] = record
    app_state.session_order.insert(0, session_id)
    app_state.turn_states[session_id] = AssistantTurnState()
    app_state.active_session_id = session_id
    _remember_runtime_preferences(app_state.preferences, runtime)
    _save_workspace_state(app_state)
    return session_id


def _delete_session(
    app_state: AssistantWorkspaceState,
    *,
    session_id: str,
    config: StreamlitAssistantConfig,
    root_path: Path | None,
) -> None:
    st: Any | None = None
    try:
        st = _streamlit_module()
    except ModuleNotFoundError:
        handle = None
    else:
        handle = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
    if (
        isinstance(handle, AssistantActiveTurnHandle)
        and handle.session_id == session_id
    ):
        handle.runner.cancel()
        if st is not None:
            st.session_state[_ACTIVE_TURN_STATE_SLOT] = None
    app_state.sessions.pop(session_id, None)
    app_state.turn_states.pop(session_id, None)
    app_state.drafts.pop(session_id, None)
    app_state.session_order = [
        item for item in app_state.session_order if item != session_id
    ]
    _session_path(session_id).unlink(missing_ok=True)
    if not app_state.session_order:
        template_runtime = _default_runtime_config(config, root_path=root_path)
        new_id = f"session-{uuid4().hex[:12]}"
        app_state.sessions[new_id] = _new_session_record(new_id, template_runtime)
        app_state.session_order = [new_id]
        app_state.turn_states[new_id] = AssistantTurnState()
    if app_state.active_session_id not in app_state.sessions:
        app_state.active_session_id = app_state.session_order[0]
    _save_workspace_state(app_state)


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _title_from_prompt(prompt: str) -> str:
    cleaned = " ".join(prompt.strip().split())
    if len(cleaned) <= 60:
        return cleaned
    return f"{cleaned[:57].rstrip()}..."


def _llm_config_for_runtime(
    config: StreamlitAssistantConfig,
    runtime: StreamlitRuntimeConfig,
) -> Any:
    return config.llm.model_copy(
        update={
            "provider": runtime.provider,
            "model_name": runtime.model_name,
            "api_base_url": runtime.api_base_url,
        }
    )


def _create_provider_for_runtime(
    llm_config: Any,
    runtime: StreamlitRuntimeConfig,
    *,
    api_key: str | None,
    model_name: str,
) -> OpenAICompatibleProvider:
    return create_provider(
        llm_config,
        api_key=api_key,
        model_name=model_name,
        mode_strategy=runtime.provider_mode_strategy,
    )


def _current_api_key(llm_config: Any) -> str | None:  # pragma: no cover
    st = _streamlit_module()
    metadata = llm_config.credential_prompt_metadata()
    if not metadata.expects_api_key:
        return None
    env_var = llm_config.api_key_env_var or "OPENAI_API_KEY"
    env_value = os.getenv(env_var)
    if env_value:
        return env_value
    cache = st.session_state.setdefault(_SECRET_CACHE_STATE_SLOT, {})
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[_SECRET_CACHE_STATE_SLOT] = cache
    cached_value = str(cache.get(env_var, "")).strip()
    return cached_value or None


def _missing_api_key_text(llm_config: Any) -> str:
    env_var = llm_config.api_key_env_var or "OPENAI_API_KEY"
    return f"Set {env_var} or enter it in the header controls to use this provider."


def _exposed_tool_names_for_runtime(
    *,
    tool_specs: dict[str, ToolSpec],
    runtime: StreamlitRuntimeConfig,
    root: Path | None,
    env: dict[str, str],
) -> set[str]:
    capability_groups = build_tool_capabilities(
        tool_specs=tool_specs,
        enabled_tools=set(runtime.enabled_tools),
        root_path=runtime.root_path,
        env=env,
        allow_network=runtime.allow_network,
        allow_filesystem=runtime.allow_filesystem and root is not None,
        allow_subprocess=runtime.allow_subprocess and root is not None,
        require_approval_for=set(runtime.require_approval_for),
    )
    return {
        tool.tool_name
        for group in capability_groups.values()
        for tool in group
        if tool.exposed_to_model
    }


def _build_assistant_runner(
    *,
    session_id: str,
    config: StreamlitAssistantConfig,
    runtime: StreamlitRuntimeConfig,
    provider: ModelTurnProvider,
    session_state: ChatSessionState,
    user_message: str,
) -> ChatSessionTurnRunner:
    tool_specs = _all_tool_specs()
    root = Path(runtime.root_path) if runtime.root_path is not None else None
    enabled_tools = set(runtime.enabled_tools)
    policy = build_assistant_policy(
        enabled_tools=enabled_tools,
        tool_specs=tool_specs,
        require_approval_for=set(runtime.require_approval_for),
        allow_network=runtime.allow_network,
        allow_filesystem=runtime.allow_filesystem and root is not None,
        allow_subprocess=runtime.allow_subprocess and root is not None,
        redaction_config=config.policy.redaction,
    )
    registry, executor = build_assistant_executor(policy=policy)
    exposed_tool_names = _exposed_tool_names_for_runtime(
        tool_specs=tool_specs,
        runtime=runtime,
        root=root,
        env=dict(os.environ),
    )
    return run_interactive_chat_session_turn(
        user_message=user_message,
        session_state=session_state,
        executor=executor,
        provider=provider,
        system_prompt=build_assistant_system_prompt(
            tool_registry=registry,
            tool_limits=config.tool_limits,
            enabled_tool_names=exposed_tool_names,
            workspace_enabled=root is not None,
        ),
        base_context=build_assistant_context(
            root_path=root,
            config=config,
            app_name=f"streamlit-assistant-{session_id}",
        ),
        session_config=config.session,
        tool_limits=config.tool_limits,
        redaction_config=config.policy.redaction,
        temperature=config.llm.temperature,
    )


def _serialize_workflow_event(
    event: object,
    *,
    turn_number: int,
    session_id: str,
) -> AssistantQueuedEvent:
    if isinstance(event, ChatWorkflowStatusEvent):
        return AssistantQueuedEvent(
            kind="status",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowApprovalEvent):
        return AssistantQueuedEvent(
            kind="approval_requested",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowApprovalResolvedEvent):
        return AssistantQueuedEvent(
            kind="approval_resolved",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowInspectorEvent):
        return AssistantQueuedEvent(
            kind="inspector",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowResultEvent):
        return AssistantQueuedEvent(
            kind="result",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    raise TypeError(f"Unsupported workflow event type: {type(event)!r}")


def _worker_run_turn(handle: AssistantActiveTurnHandle) -> None:  # pragma: no cover
    try:
        for event in handle.runner:
            handle.event_queue.put(
                _serialize_workflow_event(
                    event,
                    turn_number=handle.turn_number,
                    session_id=handle.session_id,
                )
            )
    except Exception as exc:
        handle.event_queue.put(
            AssistantQueuedEvent(
                kind="error",
                payload=str(exc),
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )
    finally:
        handle.event_queue.put(
            AssistantQueuedEvent(
                kind="complete",
                payload=None,
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )


def _apply_turn_result(
    app_state: AssistantWorkspaceState,
    *,
    session_id: str,
    event: ChatWorkflowResultEvent,
) -> str | None:
    record = app_state.sessions[session_id]
    turn_state = _turn_state_for(app_state, session_id)
    result = ChatWorkflowTurnResult.model_validate(event.result)
    record.workflow_session_state = (
        result.session_state or record.workflow_session_state
    )
    record.token_usage = result.token_usage
    turn_state.pending_approval = None
    turn_state.approval_decision_in_flight = False
    if result.context_warning:
        record.transcript.append(
            StreamlitTranscriptEntry(role="system", text=result.context_warning)
        )
    if result.status == "needs_continuation" and result.continuation_reason:
        record.transcript.append(
            StreamlitTranscriptEntry(role="system", text=result.continuation_reason)
        )
    if result.final_response is not None:
        record.transcript.append(
            StreamlitTranscriptEntry(
                role="assistant",
                text=result.final_response.answer,
                final_response=result.final_response,
            )
        )
    elif result.status == "interrupted":
        interrupted_message = next(
            (
                message
                for message in reversed(result.new_messages)
                if message.role == "assistant"
                and message.completion_state == "interrupted"
            ),
            None,
        )
        if interrupted_message is not None:
            record.transcript.append(
                StreamlitTranscriptEntry(
                    role="assistant",
                    text=interrupted_message.content,
                    assistant_completion_state="interrupted",
                )
            )
        elif result.interruption_reason:
            record.transcript.append(
                StreamlitTranscriptEntry(role="system", text=result.interruption_reason)
            )
    turn_state.status_text = ""
    turn_state.busy = False
    turn_state.cancelling = False
    pending_prompt = turn_state.pending_interrupt_draft
    turn_state.pending_interrupt_draft = None
    _touch_record(record)
    return pending_prompt


def _append_inspector_entry(
    entries: list[StreamlitInspectorEntry], *, label: str, payload: object
) -> None:
    entries.append(StreamlitInspectorEntry(label=label, payload=payload))


def _apply_turn_error(
    app_state: AssistantWorkspaceState,
    *,
    session_id: str,
    error_message: str,
) -> None:
    record = app_state.sessions[session_id]
    turn_state = _turn_state_for(app_state, session_id)
    record.transcript.append(StreamlitTranscriptEntry(role="error", text=error_message))
    turn_state.busy = False
    turn_state.status_text = ""
    turn_state.pending_approval = None
    turn_state.approval_decision_in_flight = False
    turn_state.pending_interrupt_draft = None
    turn_state.cancelling = False
    _touch_record(record)


def _apply_queued_event(
    app_state: AssistantWorkspaceState,
    queued_event: AssistantQueuedEvent,
) -> str | None:
    record = app_state.sessions[queued_event.session_id]
    turn_state = _turn_state_for(app_state, queued_event.session_id)
    inspector_state = record.inspector_state
    if queued_event.kind == "status":
        if turn_state.cancelling:
            return None
        status_event = ChatWorkflowStatusEvent.model_validate(queued_event.payload)
        turn_state.status_text = status_event.status
        return None
    if queued_event.kind == "approval_requested":
        approval_event = ChatWorkflowApprovalEvent.model_validate(queued_event.payload)
        turn_state.pending_approval = approval_event.approval
        turn_state.approval_decision_in_flight = False
        turn_state.status_text = (
            f"approval required for {approval_event.approval.tool_name}"
        )
        record.transcript.append(
            StreamlitTranscriptEntry(
                role="system",
                text=(
                    f"Approval requested for {approval_event.approval.tool_name}: "
                    f"{approval_event.approval.policy_reason}"
                ),
            )
        )
        _touch_record(record)
        return None
    if queued_event.kind == "approval_resolved":
        approval_resolved_event = ChatWorkflowApprovalResolvedEvent.model_validate(
            queued_event.payload
        )
        turn_state.pending_approval = None
        turn_state.approval_decision_in_flight = False
        resolution_text = {
            "approved": "Approved pending approval request.",
            "denied": "Denied pending approval request.",
            "timed_out": "Pending approval request timed out.",
            "cancelled": "Pending approval request was cancelled.",
        }[approval_resolved_event.resolution]
        record.transcript.append(
            StreamlitTranscriptEntry(role="system", text=resolution_text)
        )
        turn_state.status_text = {
            "approved": "resuming turn",
            "denied": "continuing without approval",
            "timed_out": "approval timed out",
            "cancelled": "",
        }[approval_resolved_event.resolution]
        _touch_record(record)
        return None
    if queued_event.kind == "inspector":
        inspector_event = ChatWorkflowInspectorEvent.model_validate(
            queued_event.payload
        )
        label = (
            f"Turn {queued_event.turn_number} Round {inspector_event.round_index} "
            f"{inspector_event.kind.replace('_', ' ')}"
        )
        target = {
            "provider_messages": inspector_state.provider_messages,
            "parsed_response": inspector_state.parsed_responses,
            "tool_execution": inspector_state.tool_executions,
        }[inspector_event.kind]
        _append_inspector_entry(target, label=label, payload=inspector_event.payload)
        _touch_record(record)
        return None
    if queued_event.kind == "result":
        result_event = ChatWorkflowResultEvent.model_validate(queued_event.payload)
        return _apply_turn_result(
            app_state,
            session_id=queued_event.session_id,
            event=result_event,
        )
    if queued_event.kind == "error":
        _apply_turn_error(
            app_state,
            session_id=queued_event.session_id,
            error_message=str(queued_event.payload),
        )
        return None
    if queued_event.kind == "complete":
        if turn_state.busy and turn_state.cancelling:
            pending_prompt = turn_state.pending_interrupt_draft
            turn_state.busy = False
            turn_state.status_text = ""
            turn_state.pending_approval = None
            turn_state.approval_decision_in_flight = False
            turn_state.pending_interrupt_draft = None
            turn_state.cancelling = False
            record.transcript.append(
                StreamlitTranscriptEntry(role="system", text="Stopped active turn.")
            )
            _touch_record(record)
            return pending_prompt
        return None
    raise ValueError(f"Unsupported queued event kind: {queued_event.kind}")


def _drain_active_turn_events(app_state: AssistantWorkspaceState) -> str | None:
    st = _streamlit_module()
    handle = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
    if not isinstance(handle, AssistantActiveTurnHandle):
        return None
    pending_prompt: str | None = None
    while True:
        try:
            queued_event = handle.event_queue.get_nowait()
        except queue.Empty:
            break
        next_prompt = _apply_queued_event(app_state, queued_event)
        if next_prompt is not None:
            pending_prompt = next_prompt
    turn_state = _turn_state_for(app_state, handle.session_id)
    if (
        not handle.thread.is_alive()
        and handle.event_queue.empty()
        and not turn_state.busy
    ):
        st.session_state[_ACTIVE_TURN_STATE_SLOT] = None
    _save_workspace_state(app_state)
    return pending_prompt


def _start_streamlit_turn(
    *,
    app_state: AssistantWorkspaceState,
    session_id: str,
    config: StreamlitAssistantConfig,
    provider: ModelTurnProvider,
    user_message: str,
) -> None:
    st = _streamlit_module()
    record = app_state.sessions[session_id]
    turn_state = _turn_state_for(app_state, session_id)
    turn_number = turn_state.active_turn_number + 1
    runner = _build_assistant_runner(
        session_id=session_id,
        config=config,
        runtime=record.runtime,
        provider=provider,
        session_state=record.workflow_session_state,
        user_message=user_message,
    )
    event_queue: queue.Queue[AssistantQueuedEvent] = queue.Queue()
    handle = AssistantActiveTurnHandle(
        session_id=session_id,
        runner=runner,
        event_queue=event_queue,
        thread=threading.Thread(),
        turn_number=turn_number,
    )
    thread = threading.Thread(target=_worker_run_turn, args=(handle,), daemon=True)
    handle.thread = thread
    turn_state.active_turn_number = turn_number
    turn_state.busy = True
    turn_state.status_text = "thinking"
    turn_state.pending_approval = None
    turn_state.approval_decision_in_flight = False
    turn_state.cancelling = False
    if record.summary.title == "New assistant chat" and not any(
        entry.role == "user" for entry in record.transcript
    ):
        record.summary.title = _title_from_prompt(user_message)
    record.transcript.append(StreamlitTranscriptEntry(role="user", text=user_message))
    _touch_record(record)
    st.session_state[_ACTIVE_TURN_STATE_SLOT] = handle
    thread.start()
    _save_workspace_state(app_state)


def _submit_streamlit_prompt(
    *,
    app_state: AssistantWorkspaceState,
    session_id: str,
    config: StreamlitAssistantConfig,
    prompt: str,
) -> None:
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        return
    record = app_state.sessions[session_id]
    turn_state = _turn_state_for(app_state, session_id)
    if turn_state.busy:
        turn_state.pending_interrupt_draft = cleaned_prompt
        _cancel_active_turn(
            app_state,
            session_id=session_id,
            preserve_pending_prompt=True,
        )
        _save_workspace_state(app_state)
        return
    llm_config = _llm_config_for_runtime(config, record.runtime)
    api_key = _current_api_key(llm_config)
    provider = _create_provider_for_runtime(
        llm_config,
        record.runtime,
        api_key=api_key,
        model_name=record.runtime.model_name,
    )
    _start_streamlit_turn(
        app_state=app_state,
        session_id=session_id,
        config=config,
        provider=provider,
        user_message=cleaned_prompt,
    )


def _resolve_active_approval(
    app_state: AssistantWorkspaceState,
    *,
    session_id: str,
    approved: bool,
) -> None:
    st = _streamlit_module()
    handle = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
    turn_state = _turn_state_for(app_state, session_id)
    if not isinstance(handle, AssistantActiveTurnHandle):
        return
    if handle.session_id != session_id or turn_state.pending_approval is None:
        return
    if not handle.runner.resolve_pending_approval(approved):
        return
    turn_state.approval_decision_in_flight = True
    turn_state.status_text = "approving" if approved else "denying"
    _save_workspace_state(app_state)


def _cancel_active_turn(
    app_state: AssistantWorkspaceState,
    *,
    session_id: str,
    preserve_pending_prompt: bool = False,
) -> None:
    st = _streamlit_module()
    handle = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
    turn_state = _turn_state_for(app_state, session_id)
    if (
        not isinstance(handle, AssistantActiveTurnHandle)
        or handle.session_id != session_id
    ):
        if not preserve_pending_prompt:
            turn_state.pending_interrupt_draft = None
        if turn_state.busy:
            turn_state.pending_approval = None
            turn_state.approval_decision_in_flight = False
            turn_state.cancelling = False
            turn_state.status_text = ""
            turn_state.busy = False
            app_state.sessions[session_id].transcript.append(
                StreamlitTranscriptEntry(role="system", text="Stopped active turn.")
            )
        return
    if not preserve_pending_prompt:
        turn_state.pending_interrupt_draft = None
    turn_state.pending_approval = None
    turn_state.approval_decision_in_flight = False
    turn_state.cancelling = True
    turn_state.status_text = "stopping"
    handle.runner.cancel()


def process_streamlit_assistant_turn(
    *,
    root_path: Path | None,
    config: StreamlitAssistantConfig,
    provider: ModelTurnProvider,
    session_state: ChatSessionState,
    user_message: str,
    approval_resolver: Callable[[ChatWorkflowApprovalState], bool] | None = None,
    runtime_config: StreamlitRuntimeConfig | None = None,
) -> StreamlitAssistantTurnOutcome:
    """Execute one assistant turn using the shared reducers."""
    runtime = runtime_config or _default_runtime_config(config, root_path=root_path)
    transcript_entries: list[StreamlitTranscriptEntry] = []
    token_usage: ChatTokenUsage | None = None
    runner = _build_assistant_runner(
        session_id="test-session",
        config=config,
        runtime=runtime,
        provider=provider,
        session_state=session_state,
        user_message=user_message,
    )
    updated_session_state = session_state
    resolve_approval = approval_resolver or (lambda approval: False)
    for event in runner:
        if isinstance(event, ChatWorkflowApprovalEvent):
            approval_event = ChatWorkflowApprovalEvent.model_validate(
                event.model_dump(mode="json")
            )
            transcript_entries.append(
                StreamlitTranscriptEntry(
                    role="system",
                    text=(
                        f"Approval requested for {approval_event.approval.tool_name}: "
                        f"{approval_event.approval.policy_reason}"
                    ),
                )
            )
            runner.resolve_pending_approval(resolve_approval(approval_event.approval))
            continue
        if isinstance(event, ChatWorkflowApprovalResolvedEvent):
            approval_resolved_event = ChatWorkflowApprovalResolvedEvent.model_validate(
                event.model_dump(mode="json")
            )
            transcript_entries.append(
                StreamlitTranscriptEntry(
                    role="system",
                    text={
                        "approved": "Approved pending approval request.",
                        "denied": "Denied pending approval request.",
                        "timed_out": "Pending approval request timed out.",
                        "cancelled": "Pending approval request was cancelled.",
                    }[approval_resolved_event.resolution],
                )
            )
            continue
        if isinstance(event, ChatWorkflowResultEvent):
            result = ChatWorkflowTurnResult.model_validate(event.result)
            updated_session_state = result.session_state or updated_session_state
            token_usage = result.token_usage
            if result.context_warning:
                transcript_entries.append(
                    StreamlitTranscriptEntry(role="system", text=result.context_warning)
                )
            if result.status == "needs_continuation" and result.continuation_reason:
                transcript_entries.append(
                    StreamlitTranscriptEntry(
                        role="system",
                        text=result.continuation_reason,
                    )
                )
            if result.final_response is not None:
                transcript_entries.append(
                    StreamlitTranscriptEntry(
                        role="assistant",
                        text=result.final_response.answer,
                        final_response=result.final_response,
                    )
                )
            continue
    return StreamlitAssistantTurnOutcome(
        session_state=updated_session_state,
        transcript_entries=transcript_entries,
        token_usage=token_usage,
    )


def _render_theme(preferences: StreamlitPreferences) -> None:  # pragma: no cover
    st = _streamlit_module()
    dark = preferences.theme_mode == "dark"
    accent = "#0f766e" if not dark else "#99f6e4"
    background = "#eef6f5" if not dark else "#071413"
    surface = "#ffffff" if not dark else "#0d1f1e"
    border = "#c6dfdc" if not dark else "#29504d"
    text = "#10201f" if not dark else "#ddf3ef"
    muted = "#506a67" if not dark else "#8db3ae"
    st.markdown(
        f"""
<style>
:root {{
  --assistant-bg: {background};
  --assistant-surface: {surface};
  --assistant-border: {border};
  --assistant-text: {text};
  --assistant-muted: {muted};
  --assistant-accent: {accent};
}}
.stApp, [data-testid="stAppViewContainer"] {{
  background: radial-gradient(circle at top left, color-mix(in srgb, var(--assistant-accent) 12%, transparent), transparent 32%), var(--assistant-bg);
  color: var(--assistant-text);
}}
[data-testid="stSidebar"] > div:first-child {{
  background: color-mix(in srgb, var(--assistant-surface) 96%, transparent);
}}
.stApp [data-testid="stChatMessage"], .assistant-panel {{
  background: var(--assistant-surface);
  border: 1px solid var(--assistant-border);
  border-radius: 1rem;
  color: var(--assistant-text);
}}
.assistant-panel {{ padding: 0.85rem 1rem; margin-bottom: 0.9rem; }}
.assistant-chip {{ display:inline-block; margin:0.12rem 0.35rem 0.12rem 0; padding:0.18rem 0.55rem; border-radius:999px; border:1px solid var(--assistant-border); }}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_fatal_error(exc: Exception) -> None:  # pragma: no cover
    st = _streamlit_module()
    traceback_text = "".join(traceback.format_exception(exc)).rstrip()
    st.error(f"{type(exc).__name__}: {exc}")
    st.text_area(
        "Traceback text",
        value=traceback_text,
        height=320,
        key="fatal-error-traceback",
    )


def _render_transcript_entry(
    entry: StreamlitTranscriptEntry,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    if entry.role == "user":
        with st.chat_message("user"):
            st.markdown(entry.text)
        return
    with st.chat_message("assistant"):
        if entry.role == "system":
            st.caption("System")
            st.markdown(entry.text)
            return
        if entry.role == "error":
            st.caption("Error")
            st.error(entry.text)
            return
        if entry.assistant_completion_state == "interrupted":
            st.caption("Assistant (interrupted)")
            st.markdown(entry.text)
            return
        if entry.final_response is not None:
            st.markdown(entry.final_response.answer)
            if entry.final_response.confidence is not None:
                st.caption(f"Confidence: {entry.final_response.confidence:.2f}")
            if entry.final_response.citations:
                st.markdown("**Sources**")
                for citation in entry.final_response.citations:
                    st.markdown(f"- `{format_citation(citation)}`")
                    if citation.excerpt:
                        st.code(citation.excerpt)
            return
        st.markdown(entry.text)


def _render_summary_chips(
    record: StreamlitPersistedSessionRecord,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    enabled_count = len(record.runtime.enabled_tools)
    root_text = record.runtime.root_path or "no workspace"
    st.markdown(
        "<div class='assistant-panel'>"
        "<div><strong>llm-tools assistant</strong></div>"
        f"<span class='assistant-chip'>model: {record.runtime.model_name}</span>"
        f"<span class='assistant-chip'>provider: {record.runtime.provider.value}</span>"
        f"<span class='assistant-chip'>tools enabled: {enabled_count}</span>"
        f"<span class='assistant-chip'>root: {root_text}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_empty_state(
    record: StreamlitPersistedSessionRecord,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    root_text = record.runtime.root_path or "No workspace selected"
    st.markdown("### Start a new assistant conversation")
    st.caption(f"Current root: {root_text}")
    st.markdown(
        "Ask a normal question for a tool-free answer, or enable local, GitLab, or Atlassian tools in the sidebar when you need proprietary data."
    )


def _visible_transcript_entries(
    entries: list[StreamlitTranscriptEntry],
) -> list[StreamlitTranscriptEntry]:
    return [entry for entry in entries if entry.show_in_transcript]


def _build_research_controller(
    *,
    config: StreamlitAssistantConfig,
    runtime: StreamlitRuntimeConfig,
) -> AssistantResearchSessionController:
    budget_policy = BudgetPolicy(
        max_turns=config.research.default_max_turns,
        max_tool_invocations=config.research.default_max_tool_invocations,
        max_elapsed_seconds=config.research.default_max_elapsed_seconds,
    )

    def _service_factory() -> HarnessSessionService:
        tool_specs = _all_tool_specs()
        root = Path(runtime.root_path) if runtime.root_path is not None else None
        enabled_tools = set(runtime.enabled_tools)
        llm_config = _llm_config_for_runtime(config, runtime)
        api_key = _current_api_key(llm_config)
        policy = build_assistant_policy(
            enabled_tools=enabled_tools,
            tool_specs=tool_specs,
            require_approval_for=set(runtime.require_approval_for),
            allow_network=runtime.allow_network,
            allow_filesystem=runtime.allow_filesystem and root is not None,
            allow_subprocess=runtime.allow_subprocess and root is not None,
            redaction_config=config.policy.redaction,
        )
        registry, workflow_executor = build_assistant_executor(policy=policy)
        exposed_tool_names = _exposed_tool_names_for_runtime(
            tool_specs=tool_specs,
            runtime=runtime,
            root=root,
            env=dict(os.environ),
        )
        harness_provider = build_live_harness_provider(
            config=config,
            provider_config=llm_config,
            model_name=runtime.model_name,
            api_key=api_key,
            mode_strategy=runtime.provider_mode_strategy,
            tool_registry=registry,
            enabled_tool_names=exposed_tool_names,
            workspace_enabled=root is not None,
        )
        return HarnessSessionService(
            store=FileHarnessStateStore(_research_store_dir(config)),
            workflow_executor=workflow_executor,
            provider=harness_provider,
            workspace=str(root) if root is not None else None,
        )

    return AssistantResearchSessionController(
        service_factory=_service_factory,
        budget_policy=budget_policy,
        include_replay_by_default=config.research.include_replay_by_default,
        list_limit=config.research.max_recent_sessions,
    )


def _render_sidebar_session_controls(
    app_state: AssistantWorkspaceState,
    *,
    config: StreamlitAssistantConfig,
    root_path: Path | None,
    runtime: StreamlitRuntimeConfig,
) -> bool:  # pragma: no cover
    st = _streamlit_module()
    st.markdown("## Assistant")
    if st.button("New chat", key="assistant-new-chat", use_container_width=True):
        _create_session(
            app_state,
            template_runtime=runtime.model_copy(deep=True),
        )
        return True
    for session_id in list(app_state.session_order):
        record = app_state.sessions[session_id]
        label = record.summary.title or session_id
        if st.button(label, key=f"session:{session_id}", use_container_width=True):
            app_state.active_session_id = session_id
            _save_workspace_state(app_state)
        if session_id == app_state.active_session_id and st.button(
            "Delete active chat",
            key=f"delete:{session_id}",
            use_container_width=True,
        ):
            _delete_session(
                app_state,
                session_id=session_id,
                config=config,
                root_path=root_path,
            )
            return True
    st.markdown("---")
    return False


def _render_sidebar_runtime_settings(
    runtime: StreamlitRuntimeConfig,
    *,
    config: StreamlitAssistantConfig,
) -> Any:  # pragma: no cover
    st = _streamlit_module()
    provider_options = [preset.value for preset in type(runtime.provider)]
    provider_value = st.selectbox(
        "Provider",
        options=provider_options,
        index=provider_options.index(runtime.provider.value),
    )
    runtime.provider = type(runtime.provider)(provider_value)
    runtime.model_name = (
        st.text_input("Model", value=runtime.model_name).strip() or runtime.model_name
    )
    runtime.api_base_url = (
        st.text_input("API base URL", value=runtime.api_base_url or "").strip() or None
    )
    current_root = st.text_input(
        "Workspace root",
        value=runtime.root_path or "",
        placeholder="Optional local directory",
    ).strip()
    if current_root:
        candidate = Path(current_root).expanduser()
        if candidate.exists() and candidate.is_dir():
            runtime.root_path = str(candidate.resolve())
            runtime.allow_filesystem = True
        else:
            st.caption("Workspace root must point to an existing directory.")
    else:
        runtime.root_path = None
        runtime.allow_filesystem = False

    llm_config = _llm_config_for_runtime(config, runtime)
    metadata = llm_config.credential_prompt_metadata()
    if metadata.expects_api_key:
        env_var = llm_config.api_key_env_var or "OPENAI_API_KEY"
        secret = st.text_input(
            env_var,
            value="",
            type="password",
            placeholder="Optional session-only API key",
        ).strip()
        if secret:
            cache = st.session_state.setdefault(_SECRET_CACHE_STATE_SLOT, {})
            if isinstance(cache, dict):
                cache[env_var] = secret
    return llm_config


def _render_sidebar_permission_controls(
    runtime: StreamlitRuntimeConfig,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    st.markdown("### Permissions")
    runtime.allow_network = st.checkbox("Network access", value=runtime.allow_network)
    runtime.allow_filesystem = st.checkbox(
        "Filesystem access",
        value=runtime.allow_filesystem,
        disabled=runtime.root_path is None,
    )
    runtime.allow_subprocess = st.checkbox(
        "Subprocess access",
        value=runtime.allow_subprocess,
        disabled=runtime.root_path is None,
    )

    st.markdown("### Approval gates")
    for side_effect, label in (
        (SideEffectClass.LOCAL_READ, "Local reads"),
        (SideEffectClass.LOCAL_WRITE, "Local writes"),
        (SideEffectClass.EXTERNAL_READ, "External reads"),
        (SideEffectClass.EXTERNAL_WRITE, "External writes"),
    ):
        checked = side_effect in runtime.require_approval_for
        if st.checkbox(label, value=checked, key=f"approval:{side_effect.value}"):
            runtime.require_approval_for.add(side_effect)
        else:
            runtime.require_approval_for.discard(side_effect)


def _render_sidebar_tool_controls(
    runtime: StreamlitRuntimeConfig,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    st.markdown("### Tools")
    capability_groups = build_tool_capabilities(
        tool_specs=_all_tool_specs(),
        enabled_tools=set(runtime.enabled_tools),
        root_path=runtime.root_path,
        env=dict(os.environ),
        allow_network=runtime.allow_network,
        allow_filesystem=runtime.allow_filesystem and runtime.root_path is not None,
        allow_subprocess=runtime.allow_subprocess and runtime.root_path is not None,
        require_approval_for=set(runtime.require_approval_for),
    )
    enabled = set(runtime.enabled_tools)
    for group_name, items in capability_groups.items():
        st.markdown(f"**{group_name}**")
        for item in items:
            checked = st.checkbox(
                item.tool_name,
                value=item.tool_name in enabled,
                key=f"tool:{runtime.provider.value}:{item.tool_name}",
            )
            if checked:
                enabled.add(item.tool_name)
            else:
                enabled.discard(item.tool_name)
            status_bits = [item.status.replace("_", " ")]
            if item.approval_required:
                status_bits.append("approval gated")
            st.caption(
                " | ".join(status_bits) + (f" | {item.detail}" if item.detail else "")
            )
    runtime.enabled_tools = sorted(enabled)
    runtime.inspector_open = st.checkbox(
        "Show inspector",
        value=runtime.inspector_open,
    )


def _append_research_summary(
    active: StreamlitPersistedSessionRecord,
    inspection: HarnessSessionInspection,
    app_state: AssistantWorkspaceState,
) -> None:
    active.transcript.append(
        StreamlitTranscriptEntry(
            role="system",
            text=AssistantResearchSessionController.summary_text(inspection),
        )
    )
    _touch_record(active)
    _save_workspace_state(app_state)


def _render_sidebar_research_controls(
    app_state: AssistantWorkspaceState,
    *,
    config: StreamlitAssistantConfig,
    runtime: StreamlitRuntimeConfig,
    active: StreamlitPersistedSessionRecord,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    if not config.research.enabled:
        return
    st.markdown("### Research sessions")
    controller = _build_research_controller(config=config, runtime=runtime)
    research_prompt = st.text_area(
        "Launch durable research",
        value="",
        placeholder="Create a durable research task from a prompt",
        height=120,
        key="assistant-research-prompt",
    )
    if st.button("Launch research session", use_container_width=True):
        prompt = research_prompt.strip()
        if prompt:
            inspection = controller.launch(prompt=prompt)
            _append_research_summary(active, inspection, app_state)
        else:
            st.warning("Enter a research prompt first.")
    try:
        recent = controller.list_recent()
    except Exception as exc:
        st.caption(f"Research sessions unavailable: {exc}")
        return

    for item in recent.sessions:
        summary = item.summary
        st.markdown(f"`{summary.session_id}`")
        st.caption(
            f"stop={summary.stop_reason.value if summary.stop_reason else 'running'} | turns={summary.total_turns} | approvals={len(summary.pending_approval_ids)}"
        )
        if summary.pending_approval_ids:
            cols = st.columns(4)
            if cols[0].button(
                "Insert summary", key=f"research-insert:{summary.session_id}"
            ):
                inspection = controller.inspect(summary.session_id)
                _append_research_summary(active, inspection, app_state)
            if cols[1].button("Approve", key=f"research-approve:{summary.session_id}"):
                inspection = controller.resume(
                    summary.session_id,
                    approval_resolution=ApprovalResolution.APPROVE,
                )
                _append_research_summary(active, inspection, app_state)
            if cols[2].button("Deny", key=f"research-deny:{summary.session_id}"):
                inspection = controller.resume(
                    summary.session_id,
                    approval_resolution=ApprovalResolution.DENY,
                )
                _append_research_summary(active, inspection, app_state)
            if cols[3].button("Stop", key=f"research-stop:{summary.session_id}"):
                inspection = controller.stop(summary.session_id)
                _append_research_summary(active, inspection, app_state)
            continue
        cols = st.columns(3)
        if cols[0].button(
            "Insert summary", key=f"research-insert:{summary.session_id}"
        ):
            inspection = controller.inspect(summary.session_id)
            _append_research_summary(active, inspection, app_state)
        if cols[1].button("Resume", key=f"research-resume:{summary.session_id}"):
            inspection = controller.resume(summary.session_id)
            _append_research_summary(active, inspection, app_state)
        if cols[2].button("Stop", key=f"research-stop:{summary.session_id}"):
            inspection = controller.stop(summary.session_id)
            _append_research_summary(active, inspection, app_state)


def _render_sidebar(
    app_state: AssistantWorkspaceState,
    *,
    config: StreamlitAssistantConfig,
    root_path: Path | None,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    active = _active_session(app_state)
    runtime = active.runtime
    with st.sidebar:
        if _render_sidebar_session_controls(
            app_state,
            config=config,
            root_path=root_path,
            runtime=runtime,
        ):
            return
        _render_sidebar_runtime_settings(runtime, config=config)
        _render_sidebar_permission_controls(runtime)
        _render_sidebar_tool_controls(runtime)
        _render_sidebar_research_controls(
            app_state,
            config=config,
            runtime=runtime,
            active=active,
        )
        _remember_runtime_preferences(app_state.preferences, runtime)
        _touch_record(active)
        _save_workspace_state(app_state)


def _render_status_and_composer(
    app_state: AssistantWorkspaceState,
    *,
    session_id: str,
    config: StreamlitAssistantConfig,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    record = app_state.sessions[session_id]
    runtime = record.runtime
    turn_state = _turn_state_for(app_state, session_id)
    if turn_state.status_text:
        st.caption(f"Status: {turn_state.status_text}")
    if turn_state.pending_approval is not None:
        cols = st.columns(2)
        if cols[0].button("Approve", use_container_width=True):
            _resolve_active_approval(app_state, session_id=session_id, approved=True)
        if cols[1].button("Deny", use_container_width=True):
            _resolve_active_approval(app_state, session_id=session_id, approved=False)
    prompt = st.text_area(
        "Ask the assistant",
        value=app_state.drafts.get(session_id, ""),
        height=140,
        placeholder="Ask a normal question, or ask for help across local and remote sources.",
        key=f"composer:{session_id}",
    )
    app_state.drafts[session_id] = prompt
    cols = st.columns(2)
    if cols[0].button("Send", use_container_width=True, disabled=turn_state.busy):
        llm_config = _llm_config_for_runtime(config, runtime)
        metadata = llm_config.credential_prompt_metadata()
        if metadata.expects_api_key and _current_api_key(llm_config) is None:
            st.warning(_missing_api_key_text(llm_config))
        else:
            _submit_streamlit_prompt(
                app_state=app_state,
                session_id=session_id,
                config=config,
                prompt=prompt,
            )
            app_state.drafts[session_id] = ""
    if cols[1].button(
        "Stop active turn",
        use_container_width=True,
        disabled=not turn_state.busy,
    ):
        _cancel_active_turn(app_state, session_id=session_id)
        _save_workspace_state(app_state)


def _render_inspector(
    record: StreamlitPersistedSessionRecord,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    if not record.runtime.inspector_open:
        return
    inspector = record.inspector_state
    st.markdown("### Inspector")
    for label, entries in (
        ("Provider messages", inspector.provider_messages),
        ("Parsed responses", inspector.parsed_responses),
        ("Tool executions", inspector.tool_executions),
    ):
        if not entries:
            continue
        st.markdown(f"**{label}**")
        for entry in entries[-5:]:
            st.caption(entry.label)
            st.code(pretty_json(entry.payload))


def run_streamlit_assistant_app(
    *, root_path: Path | None, config: StreamlitAssistantConfig
) -> None:  # pragma: no cover
    """Render the Streamlit assistant UI."""
    st = _streamlit_module()
    st.set_page_config(**_page_config())
    if _APP_STATE_SLOT not in st.session_state:
        st.session_state[_APP_STATE_SLOT] = _load_workspace_state(
            root_path=root_path,
            config=config,
        )
    st.session_state.setdefault(_ACTIVE_TURN_STATE_SLOT, None)
    st.session_state.setdefault(_SECRET_CACHE_STATE_SLOT, {})
    app_state: AssistantWorkspaceState = st.session_state[_APP_STATE_SLOT]

    _render_theme(app_state.preferences)
    pending_prompt = _drain_active_turn_events(app_state)
    _render_sidebar(app_state, config=config, root_path=root_path)
    active_record = _active_session(app_state)
    for notice in app_state.startup_notices:
        st.warning(notice)
    app_state.startup_notices.clear()

    _render_summary_chips(active_record)

    if pending_prompt is not None and pending_prompt.strip():
        _submit_streamlit_prompt(
            app_state=app_state,
            session_id=app_state.active_session_id,
            config=config,
            prompt=pending_prompt,
        )

    visible_entries = _visible_transcript_entries(active_record.transcript)
    if not visible_entries:
        _render_empty_state(active_record)
    else:
        for entry in visible_entries:
            _render_transcript_entry(entry)
    _render_status_and_composer(
        app_state,
        session_id=app_state.active_session_id,
        config=config,
    )
    _render_inspector(active_record)

    if _turn_state_for(app_state, app_state.active_session_id).busy:
        time.sleep(_POLL_INTERVAL_SECONDS)
        st.rerun()


def _launch_streamlit_app(script_args: Sequence[str]) -> int:  # pragma: no cover
    from streamlit.web import cli as streamlit_cli

    previous_argv = list(sys.argv)
    try:
        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            _STREAMLIT_BROWSER_USAGE_STATS_FLAG,
            _STREAMLIT_TOOLBAR_MODE_FLAG,
            "--",
            *list(script_args),
        ]
        return int(streamlit_cli.main())
    finally:
        sys.argv = previous_argv


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Launch the Streamlit assistant app through the Streamlit CLI."""
    script_args = list(argv) if argv is not None else list(sys.argv[1:])
    return _launch_streamlit_app(script_args)


def _run_streamlit_script(
    argv: Sequence[str] | None = None,
) -> None:  # pragma: no cover
    args = build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    config = _resolve_assistant_config(args)
    try:
        run_streamlit_assistant_app(
            root_path=_resolve_root_argument(args, config),
            config=config,
        )
    except Exception as exc:
        _render_fatal_error(exc)


if __name__ == "__main__":
    _run_streamlit_script()

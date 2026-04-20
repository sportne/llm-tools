"""Streamlit app shell for the assistant-focused chat experience."""

from __future__ import annotations

import argparse
import hashlib
import os
import queue
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import yaml  # type: ignore[import-untyped]

from llm_tools.apps.assistant_config import (
    StreamlitAssistantConfig,
    load_streamlit_assistant_config,
)
from llm_tools.apps.assistant_prompts import build_assistant_system_prompt
from llm_tools.apps.assistant_runtime import (
    AssistantToolCapability,
    AssistantToolCapabilityReason,
    AssistantToolCapabilityReasonCode,
    build_assistant_available_tool_specs,
    build_assistant_context,
    build_assistant_executor,
    build_assistant_policy,
    build_live_harness_provider,
    build_tool_capabilities,
    build_tool_group_capability_summaries,
    resolve_assistant_default_enabled_tools,
)
from llm_tools.apps.chat_presentation import format_citation, pretty_json
from llm_tools.apps.chat_runtime import create_provider
from llm_tools.apps.protection_runtime import (
    build_protection_controller,
    build_protection_environment,
)
from llm_tools.apps.streamlit_models import (
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
    HarnessInvocationTrace,
    HarnessSessionCreateRequest,
    HarnessSessionInspection,
    HarnessSessionInspectRequest,
    HarnessSessionListRequest,
    HarnessSessionListResult,
    HarnessSessionResumeRequest,
    HarnessSessionRunRequest,
    HarnessSessionService,
    HarnessSessionStopRequest,
    HarnessTurnTrace,
    ResumedHarnessSession,
    ResumeDisposition,
    resume_session,
)
from llm_tools.harness_api.context import (
    DefaultHarnessContextBuilder,
    TurnContextBundle,
)
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderPreflightResult
from llm_tools.tool_api import SideEffectClass, ToolSpec
from llm_tools.workflow_api import (
    ChatSessionState,
    ChatSessionTurnRunner,
    ChatTokenUsage,
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowApprovalState,
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
    ModelTurnProvider,
    ProtectionConfig,
    ProtectionCorpusLoadReport,
    inspect_protection_corpus,
    run_interactive_chat_session_turn,
)

_APP_STATE_SLOT = "llm_tools_streamlit_assistant_app_state"
_ACTIVE_TURN_STATE_SLOT = "llm_tools_streamlit_assistant_active_turn"
_SELECTED_RESEARCH_SESSION_SLOT = "llm_tools_streamlit_assistant_research_selection"
_SECRET_CACHE_STATE_SLOT = "llm_tools_streamlit_assistant_secret_cache"  # noqa: S105
_CONNECTION_CHECK_STATE_SLOT = "llm_tools_streamlit_assistant_connection_check"
_EXPORTED_CONFIG_STATE_SLOT = "llm_tools_streamlit_assistant_exported_config"
_STREAMLIT_BROWSER_USAGE_STATS_FLAG = "--browser.gatherUsageStats=false"
_STREAMLIT_TOOLBAR_MODE_FLAG = "--client.toolbarMode=minimal"
_POLL_INTERVAL_SECONDS = 1.0
_DEFAULT_THEME_MODE: Literal["dark", "light"] = "light"
_STORAGE_ENV_VAR = "LLM_TOOLS_STREAMLIT_ASSISTANT_STATE_DIR"
_SESSION_STORAGE_DIR_NAME = "sessions"

_REMOTE_CREDENTIAL_FIELDS: dict[str, tuple[str, ...]] = {
    "GitLab": ("GITLAB_BASE_URL", "GITLAB_API_TOKEN"),
    "Jira": ("JIRA_BASE_URL", "JIRA_USERNAME", "JIRA_API_TOKEN"),
    "Confluence": (
        "CONFLUENCE_BASE_URL",
        "CONFLUENCE_USERNAME",
        "CONFLUENCE_API_TOKEN",
    ),
    "Bitbucket": (
        "BITBUCKET_BASE_URL",
        "BITBUCKET_USERNAME",
        "BITBUCKET_API_TOKEN",
    ),
}


@dataclass(slots=True)
class AssistantTurnState:
    """Mutable UI state for one assistant session."""

    busy: bool = False
    status_text: str = ""
    pending_approval: ChatWorkflowApprovalState | None = None
    approval_decision_in_flight: bool = False
    active_turn_number: int = 0
    queued_follow_up_prompt: str | None = None
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


@dataclass(slots=True)
class AssistantResearchSessionView:
    """Assistant-side view model for one durable research session."""

    session_id: str
    state_label: str
    state_detail: str
    summarized: bool
    can_resume: bool
    waiting_for_approval: bool
    is_stopped: bool
    issue_messages: list[str] = field(default_factory=list)


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
        return _research_summary_text(inspection.summary)


class AssistantHarnessContextBuilder:
    """App-local harness context builder that injects session env overrides."""

    def __init__(
        self,
        *,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        self._delegate = DefaultHarnessContextBuilder()
        self._env_overrides = dict(env_overrides or {})

    def build(
        self,
        *,
        state: Any,
        selected_task_ids: Sequence[str],
        turn_index: int,
        workspace: str | None = None,
    ) -> TurnContextBundle:
        bundle = self._delegate.build(
            state=state,
            selected_task_ids=selected_task_ids,
            turn_index=turn_index,
            workspace=workspace,
        )
        env = _merged_runtime_env(self._env_overrides)
        metadata = dict(bundle.tool_context.metadata)
        metadata["assistant_mode"] = "streamlit_assistant_research"
        return bundle.model_copy(
            update={
                "tool_context": bundle.tool_context.model_copy(
                    update={
                        "env": env,
                        "metadata": metadata,
                    }
                )
            }
        )


def _secret_cache() -> dict[str, str]:
    st = _streamlit_module()
    cache = st.session_state.setdefault(_SECRET_CACHE_STATE_SLOT, {})
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[_SECRET_CACHE_STATE_SLOT] = cache
    return cache


def _get_secret_value(name: str) -> str | None:
    cached_value = str(_secret_cache().get(name, "")).strip()
    if cached_value:
        return cached_value
    env_value = os.getenv(name)
    return env_value or None


def _set_secret_value(name: str, value: str) -> None:
    cleaned = value.strip()
    cache = _secret_cache()
    if cleaned:
        cache[name] = cleaned
    else:
        cache.pop(name, None)


def _remote_env_overrides() -> dict[str, str]:
    overrides: dict[str, str] = {}
    for fields in _REMOTE_CREDENTIAL_FIELDS.values():
        for field_name in fields:
            value = str(_secret_cache().get(field_name, "")).strip()
            if value:
                overrides[field_name] = value
    return overrides


def _merged_runtime_env(env_overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in (env_overrides or {}).items():
        cleaned = value.strip()
        if cleaned:
            env[key] = cleaned
    return env


def _runtime_env() -> dict[str, str]:
    return _merged_runtime_env(_remote_env_overrides())


def _provider_signature(
    *,
    runtime: StreamlitRuntimeConfig,
    api_key: str | None,
) -> tuple[str, ...]:
    api_key_fingerprint = "missing"
    cleaned_api_key = (api_key or "").strip()
    if cleaned_api_key:
        api_key_fingerprint = hashlib.sha256(
            cleaned_api_key.encode("utf-8")
        ).hexdigest()[:12]
    return (
        runtime.provider.value,
        runtime.provider_mode_strategy.value,
        runtime.model_name,
        runtime.api_base_url or "",
        api_key_fingerprint,
    )


def _stored_connection_report(
    session_id: str,
    *,
    signature: tuple[str, ...],
) -> ProviderPreflightResult | None:
    st = _streamlit_module()
    raw = st.session_state.setdefault(_CONNECTION_CHECK_STATE_SLOT, {})
    if not isinstance(raw, dict):
        raw = {}
        st.session_state[_CONNECTION_CHECK_STATE_SLOT] = raw
    candidate = raw.get(session_id)
    if not isinstance(candidate, dict):
        return None
    if tuple(candidate.get("signature", ())) != signature:
        return None
    payload = candidate.get("report")
    if not isinstance(payload, dict):
        return None
    return ProviderPreflightResult.model_validate(payload)


def _store_connection_report(
    session_id: str,
    *,
    signature: tuple[str, ...],
    report: ProviderPreflightResult,
) -> None:
    st = _streamlit_module()
    raw = st.session_state.setdefault(_CONNECTION_CHECK_STATE_SLOT, {})
    if not isinstance(raw, dict):
        raw = {}
        st.session_state[_CONNECTION_CHECK_STATE_SLOT] = raw
    raw[session_id] = {
        "signature": list(signature),
        "report": report.model_dump(mode="json"),
    }


def _validate_model_connection(
    *,
    session_id: str,
    llm_config: Any,
    runtime: StreamlitRuntimeConfig,
) -> ProviderPreflightResult:
    api_key = _current_api_key(llm_config)
    signature = _provider_signature(runtime=runtime, api_key=api_key)
    provider = _create_provider_for_runtime(
        llm_config,
        runtime,
        api_key=api_key,
        model_name=runtime.model_name,
    )
    if not hasattr(provider, "preflight"):
        report = ProviderPreflightResult(
            ok=True,
            connection_succeeded=True,
            model_accepted=True,
            selected_mode_supported=True,
            model_listing_supported=False,
            resolved_mode=(
                None
                if runtime.provider_mode_strategy.value == "auto"
                else runtime.provider_mode_strategy
            ),
            actionable_message=(
                "Model connection is ready for this session. "
                "Provider preflight is unavailable for this provider instance."
            ),
        )
        _store_connection_report(session_id, signature=signature, report=report)
        return report
    report = provider.preflight(request_params={"temperature": 0.0})
    _store_connection_report(session_id, signature=signature, report=report)
    return report


def _ensure_model_connection_ready(
    *,
    session_id: str,
    llm_config: Any,
    runtime: StreamlitRuntimeConfig,
) -> ProviderPreflightResult:
    api_key = _current_api_key(llm_config)
    signature = _provider_signature(runtime=runtime, api_key=api_key)
    cached = _stored_connection_report(session_id, signature=signature)
    if cached is not None and cached.ok:
        return cached
    return _validate_model_connection(
        session_id=session_id,
        llm_config=llm_config,
        runtime=runtime,
    )


def _protection_report(protection: ProtectionConfig) -> ProtectionCorpusLoadReport:
    return inspect_protection_corpus(protection)


def _protection_next_steps(report: ProtectionCorpusLoadReport) -> str:
    if report.usable_document_count > 0 and not report.issues:
        return "Protection corpus is ready for this session."
    if report.usable_document_count > 0:
        return "Protection corpus is usable, but some configured paths need attention."
    return (
        "Protection is enabled but not ready yet. Add at least one readable document "
        "file or directory in Step 6."
    )


def _execution_blocker(
    *,
    session_id: str,
    config: StreamlitAssistantConfig,
    runtime: StreamlitRuntimeConfig,
    require_tool_readiness: bool = False,
) -> str | None:
    llm_config = _llm_config_for_runtime(config, runtime)
    metadata = llm_config.credential_prompt_metadata()
    if metadata.expects_api_key and _current_api_key(llm_config) is None:
        return _missing_api_key_text(llm_config)
    report = _ensure_model_connection_ready(
        session_id=session_id,
        llm_config=llm_config,
        runtime=runtime,
    )
    if not report.ok:
        return report.actionable_message
    if runtime.protection.enabled:
        protection_report = _protection_report(runtime.protection)
        if protection_report.usable_document_count == 0:
            return _protection_next_steps(protection_report)
    if require_tool_readiness:
        summaries = build_tool_group_capability_summaries(
            build_tool_capabilities(
                tool_specs=_all_tool_specs(),
                enabled_tools=set(runtime.enabled_tools),
                root_path=runtime.root_path,
                env=_runtime_env(),
                allow_network=runtime.allow_network,
                allow_filesystem=(
                    runtime.allow_filesystem and runtime.root_path is not None
                ),
                allow_subprocess=(
                    runtime.allow_subprocess and runtime.root_path is not None
                ),
                require_approval_for=set(runtime.require_approval_for),
            )
        )
        if any(summary.missing_credentials_tools for summary in summaries.values()):
            return (
                "Research is not ready yet. Add credentials for the enabled remote "
                "sources in Step 2."
            )
    return None


def _runtime_to_export_config(
    *,
    config: StreamlitAssistantConfig,
    runtime: StreamlitRuntimeConfig,
) -> StreamlitAssistantConfig:
    return config.model_copy(
        deep=True,
        update={
            "llm": config.llm.model_copy(
                update={
                    "provider": runtime.provider,
                    "provider_mode_strategy": runtime.provider_mode_strategy,
                    "model_name": runtime.model_name,
                    "api_base_url": runtime.api_base_url,
                }
            ),
            "policy": config.policy.model_copy(
                update={
                    "enabled_tools": list(runtime.enabled_tools),
                    "require_approval_for": set(runtime.require_approval_for),
                }
            ),
            "workspace": config.workspace.model_copy(
                update={"default_root": config.workspace.default_root}
            ),
            "ui": config.ui.model_copy(
                update={"inspector_open_by_default": runtime.inspector_open}
            ),
            "protection": runtime.protection.model_copy(deep=True),
        },
    )


def _rendered_exported_config(
    *,
    config: StreamlitAssistantConfig,
    runtime: StreamlitRuntimeConfig,
) -> str:
    exported = _runtime_to_export_config(config=config, runtime=runtime)
    payload = exported.model_dump(mode="json")
    return cast(str, yaml.safe_dump(payload, sort_keys=False))


def _research_summary_text(summary: Any) -> str:
    """Return the assistant transcript summary for one research session."""
    lines = [
        f"Research session: {summary.session_id}",
        f"Stop reason: {summary.stop_reason.value if summary.stop_reason else 'running'}",
        f"Turns: {summary.total_turns}",
        ("Completed tasks: " + (", ".join(summary.completed_task_ids) or "none")),
        f"Active tasks: {', '.join(summary.active_task_ids) or 'none'}",
    ]
    if summary.pending_approval_ids:
        lines.append("Pending approvals: " + ", ".join(summary.pending_approval_ids))
    if summary.latest_decision_summary:
        lines.append(f"Latest decision: {summary.latest_decision_summary}")
    return "\n".join(lines)


def _research_prompt_seed(
    app_state: AssistantWorkspaceState,
    *,
    active: StreamlitPersistedSessionRecord,
) -> str:
    session_id = active.summary.session_id
    draft = app_state.drafts.get(session_id, "").strip()
    if draft:
        return draft
    turn_state = _turn_state_for(app_state, session_id)
    queued = (turn_state.queued_follow_up_prompt or "").strip()
    if queued:
        return queued
    return ""


def _research_transition_copy(
    app_state: AssistantWorkspaceState,
    *,
    active: StreamlitPersistedSessionRecord,
) -> str:
    session_id = active.summary.session_id
    if app_state.drafts.get(session_id, "").strip():
        return "Stay in chat for quick back-and-forth. Start research when the task needs multiple steps, may pause for approval, or should be resumable later. Your current chat draft is loaded below so you can continue it as research."
    turn_state = _turn_state_for(app_state, session_id)
    if (turn_state.queued_follow_up_prompt or "").strip():
        return "Stay in chat for quick back-and-forth. Start research when the task needs multiple steps, may pause for approval, or should be resumable later. Your queued follow-up is loaded below so you can continue it as research."
    return "Stay in chat for quick back-and-forth. Start research when the task needs multiple steps, may pause for approval, or should be resumable later. Research summaries come back into this chat so you can keep the conversation continuous."


def _append_research_launch_note(
    active: StreamlitPersistedSessionRecord,
    *,
    prompt: str,
    inspection: HarnessSessionInspection,
    app_state: AssistantWorkspaceState,
) -> None:
    active.transcript.append(
        StreamlitTranscriptEntry(
            role="system",
            text=(
                "Started research task from this chat.\n"
                f"Prompt: {prompt}\n"
                f"Research session: {inspection.summary.session_id}"
            ),
        )
    )
    _touch_record(active)
    _save_workspace_state(app_state)


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
    parser.add_argument("--provider-mode-strategy", type=str)
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
    if args.provider_mode_strategy is not None:
        raw["llm"]["provider_mode_strategy"] = args.provider_mode_strategy
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
        provider_mode_strategy=config.llm.provider_mode_strategy,
        model_name=config.llm.model_name,
        api_base_url=config.llm.api_base_url,
        root_path=str(root_path) if root_path is not None else None,
        enabled_tools=sorted(resolve_assistant_default_enabled_tools(config)),
        require_approval_for=default_approvals,
        allow_network=False,
        allow_filesystem=False,
        allow_subprocess=False,
        inspector_open=config.ui.inspector_open_by_default,
        protection=config.protection.model_copy(deep=True),
    )


def _new_session_record(
    session_id: str,
    runtime: StreamlitRuntimeConfig,
) -> StreamlitPersistedSessionRecord:
    now = _now_iso()
    summary = StreamlitSessionSummary(
        session_id=session_id,
        title="New assistant session",
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


def _is_default_assistant_session_title(title: str) -> bool:
    return title in {"New assistant session", "New assistant chat"}


def _assistant_status_copy(status_text: str) -> str:
    status_map = {
        "thinking": "Assistant is drafting a response.",
        "approval required": "Approval is needed to continue.",
        "approving": "Applying approval and continuing.",
        "denying": "Skipping the requested tool and continuing.",
        "resuming turn": "Assistant is continuing with the current turn.",
        "continuing without approval": "Assistant is continuing without that tool.",
        "approval timed out": "Approval window expired for this step.",
        "stopping": "Stopping the current turn.",
    }
    cleaned = status_text.strip()
    if not cleaned:
        return ""
    return status_map.get(cleaned, cleaned.replace("_", " ").capitalize() + ".")


def _approval_request_copy(approval: ChatWorkflowApprovalState) -> str:
    reason = approval.policy_reason.strip().rstrip(".")
    if reason:
        return f"Approval needed before using {approval.tool_name}. {reason}."
    return f"Approval needed before using {approval.tool_name}."


def _approval_resolution_copy(resolution: str) -> str:
    return {
        "approved": "Approved the requested tool run.",
        "denied": "Skipped the requested tool run.",
        "timed_out": "The approval request timed out.",
        "cancelled": "The approval request was cancelled.",
    }[resolution]


def _tool_status_copy(status: str) -> str:
    return {
        "available": "Ready for this session.",
        "disabled": "Off for this session.",
        "missing_workspace": "Not ready yet.",
        "missing_credentials": "Not ready yet.",
        "permission_blocked": "Not ready yet.",
    }[status]


def _tool_reason_copy(reason: AssistantToolCapabilityReason) -> str:
    if reason.code is AssistantToolCapabilityReasonCode.WORKSPACE_REQUIRED:
        return "Choose a workspace root in Step 3."
    if reason.code is AssistantToolCapabilityReasonCode.MISSING_CREDENTIALS:
        if reason.missing_secrets:
            missing = ", ".join(reason.missing_secrets)
            return f"Add credentials: {missing}."
        return "Add the required credentials for this source."
    if reason.code is AssistantToolCapabilityReasonCode.NETWORK_PERMISSION_BLOCKED:
        return "Turn on network access in Step 4."
    if reason.code is AssistantToolCapabilityReasonCode.FILESYSTEM_PERMISSION_BLOCKED:
        return "Turn on filesystem access in Step 4."
    if reason.code is AssistantToolCapabilityReasonCode.SUBPROCESS_PERMISSION_BLOCKED:
        return "Turn on subprocess access in Step 4."
    if reason.code is AssistantToolCapabilityReasonCode.APPROVAL_REQUIRED:
        return "This tool pauses for approval before it runs."
    return reason.message


def _tool_reason_copies(item: AssistantToolCapability) -> list[str]:
    reason_copies: list[str] = []
    for reason in item.reasons:
        if item.status == "missing_workspace" and reason.code in {
            AssistantToolCapabilityReasonCode.FILESYSTEM_PERMISSION_BLOCKED,
            AssistantToolCapabilityReasonCode.SUBPROCESS_PERMISSION_BLOCKED,
        }:
            continue
        reason_copies.append(_tool_reason_copy(reason))
    if item.enabled and item.approval_required:
        reason_copies.append("This tool pauses for approval before it runs.")
    return reason_copies


def _tool_capability_caption(item: AssistantToolCapability) -> str:
    parts = [_tool_status_copy(item.status), *_tool_reason_copies(item)]
    return " ".join(parts)


def _group_readiness_copy(group_name: str, summary: Any) -> str:
    if summary.enabled_tools == 0:
        return f"{group_name} is off for this session."
    parts = [f"{summary.enabled_tools} enabled"]
    if summary.available_tools:
        parts.append(f"{summary.available_tools} ready")
    if summary.missing_workspace_tools:
        parts.append("needs workspace")
    if summary.missing_credentials_tools:
        parts.append("needs credentials")
    if summary.permission_blocked_tools:
        parts.append("blocked by permissions")
    if summary.approval_gated_tools:
        parts.append("approval on use")
    return " | ".join(parts)


def _source_setup_summary_copy(summaries: dict[str, Any]) -> str:
    enabled_summaries = [
        f"{group_name}: {_group_readiness_copy(group_name, summary)}"
        for group_name, summary in summaries.items()
        if summary.enabled_tools
    ]
    if not enabled_summaries:
        return "No sources are turned on yet. Chat-only replies still work."
    return "Enabled sources: " + "; ".join(enabled_summaries)


def _source_setup_next_steps_copy(
    runtime: StreamlitRuntimeConfig,
    summaries: dict[str, Any],
) -> str:
    if not any(summary.enabled_tools for summary in summaries.values()):
        return "Next: turn on only the sources you need in Step 4."

    next_steps: list[str] = []
    if any(summary.missing_workspace_tools for summary in summaries.values()):
        next_steps.append(
            "Next: choose a workspace root in Step 3 for local file and git sources."
        )
    if any(summary.missing_credentials_tools for summary in summaries.values()):
        next_steps.append(
            "Next: provide the required service credentials in Step 2 for the enabled remote sources."
        )
    if any(summary.permission_blocked_tools for summary in summaries.values()):
        next_steps.append(
            "Next: turn on the required session permissions in Step 4 for the sources you enabled."
        )
    if not next_steps:
        next_steps.append("Enabled sources are ready for this session.")
    if any(summary.approval_gated_tools for summary in summaries.values()):
        next_steps.append("Some enabled sources will pause for approval before use.")
    if runtime.root_path is None and runtime.allow_filesystem:
        next_steps.append(
            "Filesystem access stays unavailable until a workspace root is selected."
        )
    return " ".join(next_steps)


def _source_readiness_tokens(runtime: StreamlitRuntimeConfig) -> list[str]:
    capability_groups = build_tool_capabilities(
        tool_specs=_all_tool_specs(),
        enabled_tools=set(runtime.enabled_tools),
        root_path=runtime.root_path,
        env=_runtime_env(),
        allow_network=runtime.allow_network,
        allow_filesystem=runtime.allow_filesystem and runtime.root_path is not None,
        allow_subprocess=runtime.allow_subprocess and runtime.root_path is not None,
        require_approval_for=set(runtime.require_approval_for),
    )
    summaries = build_tool_group_capability_summaries(capability_groups)
    if not summaries or not any(
        summary.enabled_tools for summary in summaries.values()
    ):
        return ["sources: chat only"]

    tokens: list[str] = []
    for group_name, summary in summaries.items():
        if summary.enabled_tools == 0:
            continue
        readiness = "ready"
        if summary.missing_workspace_tools:
            readiness = "needs workspace"
        elif summary.missing_credentials_tools:
            readiness = "needs credentials"
        elif summary.permission_blocked_tools:
            readiness = "blocked"
        elif summary.approval_gated_tools:
            readiness = "approval on use"
        tokens.append(f"{group_name}: {readiness}")
    return tokens or ["sources: chat only"]


def _session_meta_copy(
    record: StreamlitPersistedSessionRecord,
    *,
    turn_state: AssistantTurnState,
    draft: str,
    is_active: bool,
) -> str:
    parts: list[str] = []
    if is_active:
        parts.append("current")
    parts.append(f"{record.summary.message_count} msgs")
    if turn_state.busy:
        parts.append("working")
    if turn_state.queued_follow_up_prompt:
        parts.append("follow-up queued")
    if draft.strip():
        parts.append("draft saved")
    if record.runtime.root_path:
        parts.append("workspace on")
    return " | ".join(parts)


def _llm_config_for_runtime(
    config: StreamlitAssistantConfig,
    runtime: StreamlitRuntimeConfig,
) -> Any:
    return config.llm.model_copy(
        update={
            "provider": runtime.provider,
            "provider_mode_strategy": runtime.provider_mode_strategy,
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
    metadata = llm_config.credential_prompt_metadata()
    if not metadata.expects_api_key:
        return None
    env_var = llm_config.api_key_env_var or "OPENAI_API_KEY"
    return _get_secret_value(env_var)


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
    env_overrides: dict[str, str] | None = None,
) -> ChatSessionTurnRunner:
    tool_specs = _all_tool_specs()
    root = Path(runtime.root_path) if runtime.root_path is not None else None
    enabled_tools = set(runtime.enabled_tools)
    effective_env = _merged_runtime_env(env_overrides)
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
        env=effective_env,
    )
    protection_controller = build_protection_controller(
        config=runtime.protection,
        provider=provider,
        environment=build_protection_environment(
            app_name="streamlit_assistant",
            model_name=runtime.model_name,
            workspace=runtime.root_path,
            enabled_tools=sorted(exposed_tool_names),
            allow_network=runtime.allow_network,
            allow_filesystem=runtime.allow_filesystem and root is not None,
            allow_subprocess=runtime.allow_subprocess and root is not None,
        ),
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
            env_overrides=env_overrides,
        ),
        session_config=config.session,
        tool_limits=config.tool_limits,
        redaction_config=config.policy.redaction,
        temperature=config.llm.temperature,
        protection_controller=protection_controller,
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
    pending_prompt = turn_state.queued_follow_up_prompt
    turn_state.queued_follow_up_prompt = None
    _touch_record(record)
    return pending_prompt


def _append_inspector_entry(
    entries: list[StreamlitInspectorEntry], *, label: str, payload: object
) -> None:
    entries.append(StreamlitInspectorEntry(label=label, payload=payload))


def _user_facing_turn_error_message(error_message: str) -> str:
    if "All provider mode attempts failed." in error_message:
        return (
            "Provider compatibility error. "
            "The endpoint did not return a usable structured response in any fallback mode. "
            + error_message
        )
    return error_message


def _apply_turn_error(
    app_state: AssistantWorkspaceState,
    *,
    session_id: str,
    error_message: str,
) -> str | None:
    record = app_state.sessions[session_id]
    turn_state = _turn_state_for(app_state, session_id)
    record.transcript.append(
        StreamlitTranscriptEntry(
            role="error",
            text=_user_facing_turn_error_message(error_message),
        )
    )
    pending_prompt = turn_state.queued_follow_up_prompt
    turn_state.busy = False
    turn_state.status_text = ""
    turn_state.pending_approval = None
    turn_state.approval_decision_in_flight = False
    turn_state.queued_follow_up_prompt = None
    turn_state.cancelling = False
    _touch_record(record)
    return pending_prompt


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
        turn_state.status_text = "approval required"
        record.transcript.append(
            StreamlitTranscriptEntry(
                role="system",
                text=_approval_request_copy(approval_event.approval),
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
        resolution_text = _approval_resolution_copy(approval_resolved_event.resolution)
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
        return _apply_turn_error(
            app_state,
            session_id=queued_event.session_id,
            error_message=str(queued_event.payload),
        )
    if queued_event.kind == "complete":
        if turn_state.busy and turn_state.cancelling:
            pending_prompt = turn_state.queued_follow_up_prompt
            turn_state.busy = False
            turn_state.status_text = ""
            turn_state.pending_approval = None
            turn_state.approval_decision_in_flight = False
            turn_state.queued_follow_up_prompt = None
            turn_state.cancelling = False
            record.transcript.append(
                StreamlitTranscriptEntry(role="system", text="Stopped the active turn.")
            )
            _touch_record(record)
            return pending_prompt
        return None
    raise ValueError(f"Unsupported queued event kind: {queued_event.kind}")


def _drain_active_turn_events(
    app_state: AssistantWorkspaceState,
) -> tuple[str, str] | None:
    st = _streamlit_module()
    handle = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
    if not isinstance(handle, AssistantActiveTurnHandle):
        return None
    pending_prompt: tuple[str, str] | None = None
    while True:
        try:
            queued_event = handle.event_queue.get_nowait()
        except queue.Empty:
            break
        next_prompt = _apply_queued_event(app_state, queued_event)
        if next_prompt is not None:
            pending_prompt = (queued_event.session_id, next_prompt)
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
    env_overrides: dict[str, str] | None = None,
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
        env_overrides=env_overrides,
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
    turn_state.queued_follow_up_prompt = None
    turn_state.cancelling = False
    if _is_default_assistant_session_title(record.summary.title) and not any(
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
) -> bool:
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        return False
    record = app_state.sessions[session_id]
    turn_state = _turn_state_for(app_state, session_id)
    if turn_state.busy:
        turn_state.queued_follow_up_prompt = cleaned_prompt
        _save_workspace_state(app_state)
        return True
    blocker = _execution_blocker(
        session_id=session_id,
        config=config,
        runtime=record.runtime,
    )
    if blocker is not None:
        with suppress(ModuleNotFoundError):
            _streamlit_module().warning(blocker)
        return False
    llm_config = _llm_config_for_runtime(config, record.runtime)
    api_key = _current_api_key(llm_config)
    env_overrides = _remote_env_overrides()
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
        env_overrides=env_overrides,
    )
    return True


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
        if turn_state.busy:
            turn_state.pending_approval = None
            turn_state.approval_decision_in_flight = False
            turn_state.cancelling = False
            turn_state.status_text = ""
            turn_state.busy = False
            app_state.sessions[session_id].transcript.append(
                StreamlitTranscriptEntry(role="system", text="Stopped the active turn.")
            )
        return
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
    env_overrides: dict[str, str] | None = None,
) -> StreamlitAssistantTurnOutcome:
    """Execute one assistant turn using the shared reducers."""
    runtime = runtime_config or _default_runtime_config(config, root_path=root_path)
    if runtime.protection.enabled:
        protection_report = _protection_report(runtime.protection)
        if protection_report.usable_document_count == 0:
            raise ValueError(_protection_next_steps(protection_report))
    transcript_entries: list[StreamlitTranscriptEntry] = []
    token_usage: ChatTokenUsage | None = None
    runner = _build_assistant_runner(
        session_id="test-session",
        config=config,
        runtime=runtime,
        provider=provider,
        session_state=session_state,
        user_message=user_message,
        env_overrides=env_overrides,
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
                    text=_approval_request_copy(approval_event.approval),
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
                    text=_approval_resolution_copy(approval_resolved_event.resolution),
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
                st.markdown("**Source notes**")
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
    root_text = record.runtime.root_path or "chat only"
    readiness = "".join(
        f"<span class='assistant-chip'>{token}</span>"
        for token in _source_readiness_tokens(record.runtime)
    )
    st.markdown(
        "<div class='assistant-panel'>"
        "<div><strong>Assistant session</strong></div>"
        f"<span class='assistant-chip'>model: {record.runtime.model_name}</span>"
        f"<span class='assistant-chip'>provider: {record.runtime.provider.value}</span>"
        f"<span class='assistant-chip'>workspace: {root_text}</span>"
        f"{readiness}"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_empty_state(
    record: StreamlitPersistedSessionRecord,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    root_text = record.runtime.root_path or "No workspace selected"
    st.markdown("### Start with a normal question")
    st.caption(f"Workspace: {root_text}")
    st.markdown(
        "This assistant can answer directly without tools. Turn on local or connected sources in the sidebar only when you want it to pull from workspace or external systems."
    )
    st.markdown(
        "Use the research panel for durable multi-turn work when you need something longer-running than a normal assistant reply."
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
        env_overrides = _remote_env_overrides()
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
            env=_runtime_env(),
        )
        harness_provider = build_live_harness_provider(
            config=config.model_copy(
                update={"protection": runtime.protection}, deep=True
            ),
            provider_config=llm_config,
            model_name=runtime.model_name,
            api_key=api_key,
            mode_strategy=runtime.provider_mode_strategy,
            tool_registry=registry,
            enabled_tool_names=exposed_tool_names,
            workspace_enabled=root is not None,
            workspace=str(root) if root is not None else None,
            allow_network=runtime.allow_network,
            allow_filesystem=runtime.allow_filesystem and root is not None,
            allow_subprocess=runtime.allow_subprocess and root is not None,
        )
        return HarnessSessionService(
            store=FileHarnessStateStore(_research_store_dir(config)),
            workflow_executor=workflow_executor,
            provider=harness_provider,
            context_builder=AssistantHarnessContextBuilder(
                env_overrides=env_overrides,
            ),
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
    st.markdown("## Assistant sessions")
    st.caption(
        "New sessions reuse the current model, permissions, and enabled sources."
    )
    if st.button(
        "New session from current setup",
        key="assistant-new-chat",
        use_container_width=True,
    ):
        _create_session(
            app_state,
            template_runtime=runtime.model_copy(deep=True),
        )
        return True
    for session_id in list(app_state.session_order):
        record = app_state.sessions[session_id]
        turn_state = _turn_state_for(app_state, session_id)
        is_active = session_id == app_state.active_session_id
        row = st.columns([5, 1])
        label = record.summary.title or session_id
        if row[0].button(label, key=f"session:{session_id}", use_container_width=True):
            app_state.active_session_id = session_id
            _save_workspace_state(app_state)
        if is_active and row[1].button(
            "Delete",
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
        st.caption(
            _session_meta_copy(
                record,
                turn_state=turn_state,
                draft=app_state.drafts.get(session_id, ""),
                is_active=is_active,
            )
        )
    st.markdown("---")
    return False


def _render_sidebar_runtime_settings(
    runtime: StreamlitRuntimeConfig,
    *,
    config: StreamlitAssistantConfig,
    session_id: str,
) -> Any:  # pragma: no cover
    st = _streamlit_module()
    st.markdown("### 1. Connect model")
    st.caption(
        "Start here. Pick the endpoint, response mode, and model you want the assistant to use for this session."
    )
    provider_options = [preset.value for preset in type(runtime.provider)]
    provider_value = st.selectbox(
        "Provider",
        options=provider_options,
        index=provider_options.index(runtime.provider.value),
    )
    runtime.provider = type(runtime.provider)(provider_value)
    mode_options = [mode.value for mode in type(runtime.provider_mode_strategy)]
    mode_value = st.selectbox(
        "Provider mode",
        options=mode_options,
        index=mode_options.index(runtime.provider_mode_strategy.value),
    )
    runtime.provider_mode_strategy = type(runtime.provider_mode_strategy)(mode_value)
    runtime.model_name = (
        st.text_input("Model", value=runtime.model_name).strip() or runtime.model_name
    )
    runtime.api_base_url = (
        st.text_input("API base URL", value=runtime.api_base_url or "").strip() or None
    )

    llm_config = _llm_config_for_runtime(config, runtime)
    metadata = llm_config.credential_prompt_metadata()
    if metadata.expects_api_key:
        env_var = llm_config.api_key_env_var or "OPENAI_API_KEY"
        secret = st.text_input(
            env_var,
            value=str(_secret_cache().get(env_var, "")),
            type="password",
            placeholder="Optional session-only API key",
        ).strip()
        _set_secret_value(env_var, secret)
        if _current_api_key(llm_config) is None:
            st.caption(
                f"This provider needs {env_var}. Use an environment variable or enter a session-only key here before you validate or run a turn."
            )
        else:
            st.caption("Model credentials are available for this session.")
    else:
        st.caption("This provider does not require an API key for the current session.")

    if st.button("Validate model connection", key=f"validate-connection:{session_id}"):
        report = _validate_model_connection(
            session_id=session_id,
            llm_config=llm_config,
            runtime=runtime,
        )
        if report.ok:
            st.caption(report.actionable_message)
        else:
            st.warning(report.actionable_message)
    else:
        cached_report = _stored_connection_report(
            session_id,
            signature=_provider_signature(
                runtime=runtime,
                api_key=_current_api_key(llm_config),
            ),
        )
        if cached_report is None:
            st.caption(
                "Validate the current model settings before you run chat or research."
            )
        elif cached_report.ok:
            st.caption(cached_report.actionable_message)
        else:
            st.warning(cached_report.actionable_message)

    st.markdown("### 2. Remote credentials")
    st.caption(
        "Add session-only credentials for the connected sources you want to use. These values are kept in memory only and are never written to session storage or exported YAML."
    )
    for group_name, field_names in _REMOTE_CREDENTIAL_FIELDS.items():
        st.markdown(f"**{group_name}**")
        for field_name in field_names:
            secret = st.text_input(
                field_name,
                value=str(_secret_cache().get(field_name, "")),
                key=f"remote-secret:{field_name}",
                type="password",
                placeholder="Optional session-only credential",
            ).strip()
            _set_secret_value(field_name, secret)

    st.markdown("### 3. Choose workspace")
    current_root = st.text_input(
        "Workspace root",
        value=runtime.root_path or "",
        placeholder="Optional local directory",
    ).strip()
    if current_root:
        candidate = Path(current_root).expanduser()
        if candidate.exists() and candidate.is_dir():
            runtime.root_path = str(candidate.resolve())
        else:
            st.caption("Workspace root must point to an existing directory.")
    else:
        runtime.root_path = None
        runtime.allow_filesystem = False
        runtime.allow_subprocess = False
    if runtime.root_path is None:
        st.caption(
            "Chat-only works now. Add a workspace root when you want local files or git."
        )
    else:
        st.caption(
            "Workspace selected. Local tools are scoped now, and you can allow filesystem or subprocess access in Step 4."
        )
    return llm_config


def _render_sidebar_permission_controls(
    runtime: StreamlitRuntimeConfig,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    st.markdown("### 4. Allow access")
    st.caption(
        "Turn on only what you need: network unlocks connected sources, filesystem unlocks local reads, and subprocess unlocks local command tools."
    )
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
    if runtime.root_path is None:
        st.caption(
            "Select a workspace in Step 3 before filesystem or subprocess access becomes available."
        )
    else:
        st.caption(
            "Filesystem and subprocess access stay off until you enable them here, even after a workspace is selected."
        )

    st.markdown("### Approval gates")
    st.caption(
        "Use approval gates when you want the assistant to pause before reads or writes that matter for this session."
    )
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
    st.markdown("### 5. Choose sources")
    st.caption(
        "Turn on the sources you want after Steps 1-4 are configured. Each source shows what is ready and what is still missing."
    )
    capability_groups = build_tool_capabilities(
        tool_specs=_all_tool_specs(),
        enabled_tools=set(runtime.enabled_tools),
        root_path=runtime.root_path,
        env=_runtime_env(),
        allow_network=runtime.allow_network,
        allow_filesystem=runtime.allow_filesystem and runtime.root_path is not None,
        allow_subprocess=runtime.allow_subprocess and runtime.root_path is not None,
        require_approval_for=set(runtime.require_approval_for),
    )
    summaries = build_tool_group_capability_summaries(capability_groups)
    st.markdown("**Current source readiness**")
    st.caption(_source_setup_summary_copy(summaries))
    st.caption(_source_setup_next_steps_copy(runtime, summaries))

    enabled = set(runtime.enabled_tools)
    for group_name, items in capability_groups.items():
        st.markdown(f"**{group_name}**")
        summary = summaries[group_name]
        st.caption(_group_readiness_copy(group_name, summary))
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
            st.caption(_tool_capability_caption(item))
    runtime.enabled_tools = sorted(enabled)
    runtime.inspector_open = st.checkbox(
        "Show inspector details",
        value=runtime.inspector_open,
    )


def _render_sidebar_protection_controls(
    runtime: StreamlitRuntimeConfig,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    st.markdown("### 6. Protection")
    runtime.protection.enabled = st.checkbox(
        "Enable proprietary protection",
        value=runtime.protection.enabled,
    )
    if not runtime.protection.enabled:
        st.caption("Protection is off for this session.")
        return
    raw_paths = st.text_area(
        "Protection corpus paths",
        value="\n".join(runtime.protection.document_paths),
        placeholder="One file or directory path per line",
        key="assistant-protection-paths",
    )
    runtime.protection.document_paths = [
        entry.strip() for entry in raw_paths.splitlines() if entry.strip()
    ]
    runtime.protection.corrections_path = (
        st.text_input(
            "Protection corrections path",
            value=runtime.protection.corrections_path or "",
            placeholder="Optional JSON or YAML corrections file",
        ).strip()
        or None
    )
    report = _protection_report(runtime.protection)
    st.caption(_protection_next_steps(report))
    if report.usable_document_count:
        st.caption(f"Loaded protection documents: {report.usable_document_count}")
    for issue in report.issues[:10]:
        st.caption(f"{issue.path}: {issue.message}")


def _render_sidebar_config_export(
    *,
    config: StreamlitAssistantConfig,
    runtime: StreamlitRuntimeConfig,
    session_id: str,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    st.markdown("### 7. Save configuration")
    st.caption(
        "Config YAML is optional. Generate a reusable preset from the current session setup when you want to save or share the non-secret defaults."
    )
    if st.button("Generate config YAML", key=f"generate-config:{session_id}"):
        st.session_state[_EXPORTED_CONFIG_STATE_SLOT] = _rendered_exported_config(
            config=config,
            runtime=runtime,
        )
    exported = st.session_state.get(_EXPORTED_CONFIG_STATE_SLOT, "")
    if isinstance(exported, str) and exported.strip():
        st.text_area(
            "Exported config YAML",
            value=exported,
            key=f"exported-config:{session_id}",
            height=220,
        )
        save_path = st.text_input(
            "Save config path",
            value="",
            key=f"save-config-path:{session_id}",
            placeholder="Optional file path to save this YAML",
        ).strip()
        if st.button("Save config YAML", key=f"save-config:{session_id}"):
            if not save_path:
                st.warning("Enter a config path first.")
            else:
                target = Path(save_path).expanduser()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(exported, encoding="utf-8")
                st.caption(f"Saved config YAML to {target.resolve()}.")


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


def _selected_research_session_id() -> str | None:
    st = _streamlit_module()
    value = st.session_state.get(_SELECTED_RESEARCH_SESSION_SLOT)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _select_research_session(session_id: str | None) -> None:
    st = _streamlit_module()
    st.session_state[_SELECTED_RESEARCH_SESSION_SLOT] = session_id


def _research_summary_inserted(
    record: StreamlitPersistedSessionRecord,
    *,
    session_id: str,
) -> bool:
    prefix = f"Research session: {session_id}\n"
    return any(
        entry.role == "system" and entry.text.startswith(prefix)
        for entry in record.transcript
    )


def _research_issue_messages(resumed: ResumedHarnessSession) -> list[str]:
    return [issue.message for issue in resumed.issues]


def _build_research_session_view(
    summary: Any,
    *,
    resumed: ResumedHarnessSession,
    summarized: bool,
) -> AssistantResearchSessionView:
    waiting_for_approval = resumed.disposition is ResumeDisposition.WAITING_FOR_APPROVAL
    can_resume = resumed.disposition is ResumeDisposition.RUNNABLE
    issue_messages = _research_issue_messages(resumed)
    is_stopped = summary.stop_reason is not None or resumed.disposition in {
        ResumeDisposition.TERMINAL,
        ResumeDisposition.APPROVAL_EXPIRED,
        ResumeDisposition.CORRUPT,
        ResumeDisposition.INCOMPATIBLE_SCHEMA,
    }
    if waiting_for_approval:
        state_label = "awaiting approval"
    elif can_resume:
        state_label = "running" if summary.total_turns == 0 else "resumable"
    else:
        state_label = "stopped"
    detail_bits = [
        f"turns={summary.total_turns}",
        f"active={len(summary.active_task_ids)}",
    ]
    if summary.pending_approval_ids:
        detail_bits.append(f"approvals={len(summary.pending_approval_ids)}")
    if summary.stop_reason is not None:
        detail_bits.append(f"reason={summary.stop_reason.value}")
    if summarized:
        detail_bits.append("summarized")
    if issue_messages:
        detail_bits.append("needs attention")
    return AssistantResearchSessionView(
        session_id=summary.session_id,
        state_label=state_label,
        state_detail=" | ".join(detail_bits),
        summarized=summarized,
        can_resume=can_resume,
        waiting_for_approval=waiting_for_approval,
        is_stopped=is_stopped,
        issue_messages=issue_messages,
    )


def _payload_for_display(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _research_session_title(inspection: HarnessSessionInspection) -> str | None:
    state = inspection.snapshot.state
    root_task_id = state.session.root_task_id
    for task in state.tasks:
        if task.task_id == root_task_id:
            return task.title
    return None


def _research_verification_copy(summary: Any) -> str:
    if not summary.verification_status_counts:
        return "No verification recorded yet."
    ordered = sorted(summary.verification_status_counts.items())
    return " | ".join(f"{name}={count}" for name, count in ordered)


def _render_research_invocation_trace(
    invocation: HarnessInvocationTrace,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    status = invocation.status.value
    label = (
        f"{invocation.tool_name} [{status}] | invocation={invocation.invocation_index}"
    )
    st.caption(label)
    extra_bits: list[str] = []
    if invocation.approval_id is not None:
        extra_bits.append(f"approval={invocation.approval_id}")
    if invocation.error_code is not None:
        extra_bits.append(f"error={invocation.error_code.value}")
    if invocation.policy_snapshot is not None:
        extra_bits.append(f"policy={invocation.policy_snapshot.reason}")
        if invocation.policy_snapshot.requires_approval:
            extra_bits.append("approval required")
    if extra_bits:
        st.caption(" | ".join(extra_bits))
    if invocation.logs:
        st.markdown("Logs")
        for item in invocation.logs:
            st.code(item)
    if invocation.artifacts:
        st.caption("Artifacts: " + ", ".join(invocation.artifacts))
    st.code(pretty_json(_payload_for_display(invocation)))


def _render_research_turn_trace(turn: HarnessTurnTrace) -> None:  # pragma: no cover
    st = _streamlit_module()
    summary_bits: list[str] = []
    if turn.selected_task_ids:
        summary_bits.append("tasks=" + ", ".join(turn.selected_task_ids))
    if turn.pending_approval_id is not None:
        summary_bits.append(f"pending approval={turn.pending_approval_id}")
    if turn.decision_action is not None:
        summary_bits.append(f"decision={turn.decision_action.value}")
    with st.expander(f"Turn {turn.turn_index}", expanded=False):
        if summary_bits:
            st.caption(" | ".join(summary_bits))
        if turn.planner_selected_task_ids:
            st.caption("Planner selected: " + ", ".join(turn.planner_selected_task_ids))
        if turn.replanning_triggers:
            st.caption("Replanning: " + ", ".join(turn.replanning_triggers))
        if turn.workflow_outcome_statuses:
            st.caption(
                "Outcomes: "
                + ", ".join(status.value for status in turn.workflow_outcome_statuses)
            )
        if turn.verification_status_by_task_id:
            st.caption(
                "Verification: "
                + ", ".join(
                    f"{task_id}={status}"
                    for task_id, status in sorted(
                        turn.verification_status_by_task_id.items()
                    )
                )
            )
        if turn.no_progress_signals:
            st.caption("No-progress signals: " + ", ".join(turn.no_progress_signals))
        if turn.decision_summary:
            st.markdown(turn.decision_summary)
        for invocation in turn.invocation_traces:
            _render_research_invocation_trace(invocation)
        st.code(pretty_json(_payload_for_display(turn)))


def _run_research_action(
    session_id: str,
    *,
    action: Callable[[], HarnessSessionInspection],
    failure_prefix: str,
) -> HarnessSessionInspection | None:
    st = _streamlit_module()
    try:
        inspection = action()
    except Exception as exc:
        st.warning(f"{failure_prefix} for {session_id}: {exc}")
        return None
    _select_research_session(inspection.summary.session_id)
    return inspection


def _run_and_append_research_action(
    *,
    active: StreamlitPersistedSessionRecord,
    inspection_action: Callable[[], HarnessSessionInspection],
    app_state: AssistantWorkspaceState,
    session_id: str,
    failure_prefix: str,
    config: StreamlitAssistantConfig | None = None,
    require_ready: bool = False,
) -> HarnessSessionInspection | None:
    if require_ready and config is not None:
        blocker = _execution_blocker(
            session_id=active.summary.session_id,
            config=config,
            runtime=active.runtime,
            require_tool_readiness=True,
        )
        if blocker is not None:
            _streamlit_module().warning(blocker)
            return None
    inspection = _run_research_action(
        session_id,
        action=inspection_action,
        failure_prefix=failure_prefix,
    )
    if inspection is not None:
        _append_research_summary(active, inspection, app_state)
    return inspection


def _conditional_button(
    container: Any,
    *,
    visible: bool,
    label: str,
    key: str,
) -> bool:
    if not visible:
        return False
    return bool(container.button(label, key=key, use_container_width=True))


def _updated_research_view(
    inspection: HarnessSessionInspection,
    *,
    active: StreamlitPersistedSessionRecord,
) -> AssistantResearchSessionView:
    return _build_research_session_view(
        inspection.summary,
        resumed=inspection.resumed,
        summarized=_research_summary_inserted(
            active,
            session_id=inspection.summary.session_id,
        ),
    )


def _apply_research_action_result(
    inspection: HarnessSessionInspection,
    *,
    updated: HarnessSessionInspection | None,
) -> HarnessSessionInspection:
    if updated is None:
        return inspection
    return updated


def _render_research_detail_header(
    inspection: HarnessSessionInspection,
    *,
    view: AssistantResearchSessionView,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    title = _research_session_title(inspection)
    if title:
        st.markdown(f"### {title}")
    st.markdown(f"`{inspection.summary.session_id}`")
    st.caption(
        f"{view.state_label} | {view.state_detail}"
        f" | revision={inspection.snapshot.revision}"
        f" | saved={inspection.snapshot.saved_at}"
    )
    for message in view.issue_messages:
        st.warning(message)


def _render_research_detail_actions(
    inspection: HarnessSessionInspection,
    *,
    controller: AssistantResearchSessionController,
    active: StreamlitPersistedSessionRecord,
    app_state: AssistantWorkspaceState,
    view: AssistantResearchSessionView,
    config: StreamlitAssistantConfig,
) -> HarnessSessionInspection:  # pragma: no cover
    st = _streamlit_module()
    session_id = inspection.summary.session_id
    action_cols = st.columns(5)
    if action_cols[0].button(
        "Insert summary",
        key=f"research-detail-insert:{session_id}",
        use_container_width=True,
    ):
        _append_research_summary(active, inspection, app_state)
    inspection = _apply_research_action_result(
        inspection,
        updated=(
            _run_and_append_research_action(
                active=active,
                inspection_action=lambda: controller.resume(
                    session_id,
                    approval_resolution=ApprovalResolution.APPROVE,
                ),
                app_state=app_state,
                session_id=session_id,
                failure_prefix="Research approval failed",
                config=config,
                require_ready=True,
            )
            if _conditional_button(
                action_cols[1],
                visible=view.waiting_for_approval,
                label="Approve",
                key=f"research-detail-approve:{session_id}",
            )
            else None
        ),
    )
    inspection = _apply_research_action_result(
        inspection,
        updated=(
            _run_and_append_research_action(
                active=active,
                inspection_action=lambda: controller.resume(
                    session_id,
                    approval_resolution=ApprovalResolution.DENY,
                ),
                app_state=app_state,
                session_id=session_id,
                failure_prefix="Research denial failed",
                config=config,
                require_ready=True,
            )
            if _conditional_button(
                action_cols[2],
                visible=view.waiting_for_approval,
                label="Deny",
                key=f"research-detail-deny:{session_id}",
            )
            else None
        ),
    )
    inspection = _apply_research_action_result(
        inspection,
        updated=(
            _run_and_append_research_action(
                active=active,
                inspection_action=lambda: controller.resume(session_id),
                app_state=app_state,
                session_id=session_id,
                failure_prefix="Research resume failed",
                config=config,
                require_ready=True,
            )
            if _conditional_button(
                action_cols[3],
                visible=view.can_resume and not view.waiting_for_approval,
                label="Resume",
                key=f"research-detail-resume:{session_id}",
            )
            else None
        ),
    )
    inspection = _apply_research_action_result(
        inspection,
        updated=(
            _run_and_append_research_action(
                active=active,
                inspection_action=lambda: controller.stop(session_id),
                app_state=app_state,
                session_id=session_id,
                failure_prefix="Research stop failed",
            )
            if _conditional_button(
                action_cols[4],
                visible=not view.is_stopped,
                label="Stop",
                key=f"research-detail-stop:{session_id}",
            )
            else None
        ),
    )
    return inspection


def _render_research_overview(
    inspection: HarnessSessionInspection,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    st.markdown("**Overview**")
    st.caption(f"Turns: {inspection.summary.total_turns}")
    st.caption(
        "Completed tasks: "
        + (", ".join(inspection.summary.completed_task_ids) or "none")
    )
    st.caption(
        "Active tasks: " + (", ".join(inspection.summary.active_task_ids) or "none")
    )
    if inspection.summary.pending_approval_ids:
        st.caption(
            "Pending approvals: " + ", ".join(inspection.summary.pending_approval_ids)
        )
    if inspection.summary.latest_decision_summary:
        st.markdown(inspection.summary.latest_decision_summary)
    st.caption(_research_verification_copy(inspection.summary))


def _render_research_approval_state(
    inspection: HarnessSessionInspection,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    if inspection.resumed.pending_approval is None:
        return
    approval_request = inspection.resumed.pending_approval.approval_request
    st.markdown("**Approval state**")
    st.caption(
        f"Approval {approval_request.approval_id}"
        f" | tool={approval_request.tool_name}"
        f" | expires={approval_request.expires_at}"
    )
    st.code(pretty_json(_payload_for_display(approval_request)))


def _render_research_replay(
    inspection: HarnessSessionInspection,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    if inspection.replay is None:
        return
    st.markdown("**Replay**")
    if inspection.replay.limitations:
        st.caption("Limitations: " + "; ".join(inspection.replay.limitations))
    for step in inspection.replay.steps:
        step_bits = [
            f"Turn {step.turn_index}",
            f"tasks={', '.join(step.selected_task_ids) or 'none'}",
        ]
        if step.decision_action is not None:
            step_bits.append(f"decision={step.decision_action.value}")
        if step.decision_stop_reason is not None:
            step_bits.append(f"stop={step.decision_stop_reason.value}")
        st.caption(" | ".join(step_bits))
        if step.decision_summary:
            st.markdown(step.decision_summary)
        st.code(pretty_json(_payload_for_display(step)))


def _render_research_trace(
    inspection: HarnessSessionInspection,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    trace = inspection.snapshot.artifacts.trace
    if trace is None or not trace.turns:
        return
    st.markdown("**Trace**")
    if trace.final_stop_reason is not None:
        st.caption(f"Final stop reason: {trace.final_stop_reason.value}")
    for turn in trace.turns:
        _render_research_turn_trace(turn)


def _render_research_raw_payload(
    inspection: HarnessSessionInspection,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    with st.expander("Raw inspection payload", expanded=False):
        st.code(pretty_json(_payload_for_display(inspection)))


def _render_research_session_details(
    app_state: AssistantWorkspaceState,
    *,
    config: StreamlitAssistantConfig,
    runtime: StreamlitRuntimeConfig,
    active: StreamlitPersistedSessionRecord,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    if not config.research.enabled:
        return
    session_id = _selected_research_session_id()
    if session_id is None:
        return
    controller = _build_research_controller(config=config, runtime=runtime)
    inspection = _run_research_action(
        session_id,
        action=lambda: controller.inspect(session_id),
        failure_prefix="Research inspection failed",
    )
    if inspection is None:
        return
    with st.expander("Research session details", expanded=True):
        view = _updated_research_view(inspection, active=active)
        _render_research_detail_header(inspection, view=view)
        inspection = _render_research_detail_actions(
            inspection,
            controller=controller,
            active=active,
            app_state=app_state,
            view=view,
            config=config,
        )
        _render_research_overview(inspection)
        _render_research_approval_state(inspection)
        _render_research_replay(inspection)
        _render_research_trace(inspection)
        _render_research_raw_payload(inspection)


def _launch_research_task(
    prompt: str,
    *,
    controller: AssistantResearchSessionController,
    active: StreamlitPersistedSessionRecord,
    app_state: AssistantWorkspaceState,
    config: StreamlitAssistantConfig | None = None,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    cleaned = prompt.strip()
    if config is not None:
        blocker = _execution_blocker(
            session_id=active.summary.session_id,
            config=config,
            runtime=active.runtime,
            require_tool_readiness=True,
        )
        if blocker is not None:
            st.warning(blocker)
            return
    if not cleaned:
        st.warning("Enter a research task first.")
        return
    inspection = _run_research_action(
        _title_from_prompt(cleaned),
        action=lambda: controller.launch(prompt=cleaned),
        failure_prefix="Research launch failed",
    )
    if inspection is None:
        return
    _append_research_launch_note(
        active,
        prompt=cleaned,
        inspection=inspection,
        app_state=app_state,
    )
    _append_research_summary(active, inspection, app_state)
    st.session_state["assistant-research-prompt"] = ""


def _render_sidebar_research_session_item(
    item: Any,
    *,
    controller: AssistantResearchSessionController,
    active: StreamlitPersistedSessionRecord,
    app_state: AssistantWorkspaceState,
    config: StreamlitAssistantConfig,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    summary = item.summary
    session_id = summary.session_id
    view = _build_research_session_view(
        summary,
        resumed=resume_session(item.snapshot),
        summarized=_research_summary_inserted(active, session_id=session_id),
    )
    st.markdown(f"`{session_id}`")
    st.caption(f"{view.state_label} | {view.state_detail}")
    for message in view.issue_messages:
        st.caption(message)
    columns = st.columns(5)
    if columns[0].button(
        "View details",
        key=f"research-view:{session_id}",
        use_container_width=True,
    ):
        _select_research_session(session_id)
    if columns[1].button(
        "Insert summary",
        key=f"research-insert:{session_id}",
        use_container_width=True,
    ):
        _run_and_append_research_action(
            active=active,
            inspection_action=lambda: controller.inspect(session_id),
            app_state=app_state,
            session_id=session_id,
            failure_prefix="Research inspection failed",
        )
    if _conditional_button(
        columns[2],
        visible=view.waiting_for_approval,
        label="Approve",
        key=f"research-approve:{session_id}",
    ):
        _run_and_append_research_action(
            active=active,
            inspection_action=lambda: controller.resume(
                session_id,
                approval_resolution=ApprovalResolution.APPROVE,
            ),
            app_state=app_state,
            session_id=session_id,
            failure_prefix="Research approval failed",
            config=config,
            require_ready=True,
        )
    if _conditional_button(
        columns[3],
        visible=view.waiting_for_approval,
        label="Deny",
        key=f"research-deny:{session_id}",
    ):
        _run_and_append_research_action(
            active=active,
            inspection_action=lambda: controller.resume(
                session_id,
                approval_resolution=ApprovalResolution.DENY,
            ),
            app_state=app_state,
            session_id=session_id,
            failure_prefix="Research denial failed",
            config=config,
            require_ready=True,
        )
    if _conditional_button(
        columns[2],
        visible=view.can_resume and not view.waiting_for_approval,
        label="Resume",
        key=f"research-resume:{session_id}",
    ):
        _run_and_append_research_action(
            active=active,
            inspection_action=lambda: controller.resume(session_id),
            app_state=app_state,
            session_id=session_id,
            failure_prefix="Research resume failed",
            config=config,
            require_ready=True,
        )
    if _conditional_button(
        columns[4],
        visible=not view.is_stopped,
        label="Stop",
        key=f"research-stop:{session_id}",
    ):
        _run_and_append_research_action(
            active=active,
            inspection_action=lambda: controller.stop(session_id),
            app_state=app_state,
            session_id=session_id,
            failure_prefix="Research stop failed",
        )


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
    st.markdown("### Research tasks")
    st.caption(_research_transition_copy(app_state, active=active))
    controller = _build_research_controller(config=config, runtime=runtime)
    research_prompt = st.text_area(
        "Create a research task",
        value=_research_prompt_seed(app_state, active=active),
        placeholder="Describe a longer-running research task",
        height=120,
        key="assistant-research-prompt",
    )
    if st.button("Start research task", use_container_width=True):
        _launch_research_task(
            research_prompt,
            controller=controller,
            active=active,
            app_state=app_state,
            config=config,
        )
    try:
        recent = controller.list_recent()
    except Exception as exc:
        st.caption(f"Research sessions unavailable: {exc}")
        return

    for item in recent.sessions:
        _render_sidebar_research_session_item(
            item,
            controller=controller,
            active=active,
            app_state=app_state,
            config=config,
        )


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
        st.caption(
            "Follow the setup order below so the assistant always shows what is ready, what is blocked, and what to configure next."
        )
        _render_sidebar_runtime_settings(
            runtime,
            config=config,
            session_id=active.summary.session_id,
        )
        _render_sidebar_permission_controls(runtime)
        _render_sidebar_tool_controls(runtime)
        _render_sidebar_protection_controls(runtime)
        _render_sidebar_config_export(
            config=config,
            runtime=runtime,
            session_id=active.summary.session_id,
        )
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
        st.caption(_assistant_status_copy(turn_state.status_text))
    if turn_state.pending_approval is not None:
        st.markdown(
            f"**Approval needed**: {_approval_request_copy(turn_state.pending_approval)}"
        )
        cols = st.columns(2)
        if cols[0].button(
            "Allow tool",
            use_container_width=True,
            disabled=turn_state.approval_decision_in_flight,
        ):
            _resolve_active_approval(app_state, session_id=session_id, approved=True)
        if cols[1].button(
            "Skip tool",
            use_container_width=True,
            disabled=turn_state.approval_decision_in_flight,
        ):
            _resolve_active_approval(app_state, session_id=session_id, approved=False)
    if turn_state.queued_follow_up_prompt:
        st.caption(f"Queued follow-up: {turn_state.queued_follow_up_prompt}")
    prompt = st.text_area(
        "Message the assistant",
        value=app_state.drafts.get(session_id, ""),
        height=140,
        placeholder=(
            "Ask directly, or turn on local and connected sources in the sidebar "
            "when you need workspace or external data."
        ),
        key=f"composer:{session_id}",
    )
    app_state.drafts[session_id] = prompt
    cols = st.columns(2)
    send_label = "Queue follow-up" if turn_state.busy else "Send"
    if cols[0].button(send_label, use_container_width=True):
        llm_config = _llm_config_for_runtime(config, runtime)
        metadata = llm_config.credential_prompt_metadata()
        if metadata.expects_api_key and _current_api_key(llm_config) is None:
            st.warning(_missing_api_key_text(llm_config))
        else:
            submitted = _submit_streamlit_prompt(
                app_state=app_state,
                session_id=session_id,
                config=config,
                prompt=prompt,
            )
            if submitted:
                app_state.drafts[session_id] = ""
    if cols[1].button(
        "Stop current turn",
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
    st.session_state.setdefault(_SELECTED_RESEARCH_SESSION_SLOT, None)
    st.session_state.setdefault(_SECRET_CACHE_STATE_SLOT, {})
    st.session_state.setdefault(_CONNECTION_CHECK_STATE_SLOT, {})
    st.session_state.setdefault(_EXPORTED_CONFIG_STATE_SLOT, "")
    app_state: AssistantWorkspaceState = st.session_state[_APP_STATE_SLOT]

    _render_theme(app_state.preferences)
    pending_prompt = _drain_active_turn_events(app_state)
    _render_sidebar(app_state, config=config, root_path=root_path)
    active_record = _active_session(app_state)
    for notice in app_state.startup_notices:
        st.warning(notice)
    app_state.startup_notices.clear()

    _render_summary_chips(active_record)
    _render_research_session_details(
        app_state,
        config=config,
        runtime=active_record.runtime,
        active=active_record,
    )

    if pending_prompt is not None:
        pending_session_id, pending_text = pending_prompt
        if pending_session_id in app_state.sessions and pending_text.strip():
            _submit_streamlit_prompt(
                app_state=app_state,
                session_id=pending_session_id,
                config=config,
                prompt=pending_text,
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

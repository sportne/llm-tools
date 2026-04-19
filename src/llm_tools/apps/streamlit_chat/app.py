"""Streamlit app shell for interactive chat."""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
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

from llm_tools.apps.chat_config import (
    ProviderPreset,
    TextualChatConfig,
    load_textual_chat_config,
)
from llm_tools.apps.chat_controls import (
    ChatCommandOutcome,
    ChatControlNotice,
    ChatControlState,
    ModelCatalogOutcome,
    ModelSwitchOutcome,
    handle_chat_command,
)
from llm_tools.apps.chat_presentation import (
    format_citation,
    format_final_response,
    format_transcript_text,
    pretty_json,
)
from llm_tools.apps.chat_prompts import build_chat_system_prompt
from llm_tools.apps.chat_runtime import (
    build_available_tool_specs,
    build_chat_context,
    build_chat_executor,
    create_provider,
)
from llm_tools.apps.streamlit_chat.models import (
    StreamlitInspectorEntry,
    StreamlitInspectorState,
    StreamlitPersistedSessionRecord,
    StreamlitPreferences,
    StreamlitRuntimeConfig,
    StreamlitSessionIndex,
    StreamlitSessionSummary,
    StreamlitTranscriptEntry,
)
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderModeStrategy
from llm_tools.tool_api import SideEffectClass, ToolPolicy, ToolSpec
from llm_tools.workflow_api import (
    ChatFinalResponse,
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
from llm_tools.workflow_api.chat_session import ChatSessionTurnRunner, ModelTurnProvider

_APP_STATE_SLOT = "llm_tools_streamlit_chat_app_state"
_ACTIVE_TURN_STATE_SLOT = "llm_tools_streamlit_chat_active_turn"
_SECRET_CACHE_STATE_SLOT = "llm_tools_streamlit_chat_secret_cache"  # noqa: S105
_WIDGET_OVERRIDES_STATE_SLOT = "llm_tools_streamlit_chat_widget_overrides"
_STREAMLIT_BROWSER_USAGE_STATS_FLAG = "--browser.gatherUsageStats=false"
_STREAMLIT_TOOLBAR_MODE_FLAG = "--client.toolbarMode=minimal"
_POLL_INTERVAL_SECONDS = 1.0
_DEFAULT_THEME_MODE: Literal["dark", "light"] = "dark"
_STORAGE_ENV_VAR = "LLM_TOOLS_STREAMLIT_STATE_DIR"
_SESSION_STORAGE_DIR_NAME = "sessions"
_ROOT_SENTINEL = "__none__"
_THEME_TOGGLE_KEY = "theme-mode"
_MODE_LABELS: dict[ProviderModeStrategy, str] = {
    ProviderModeStrategy.AUTO: "Auto (fallback)",
    ProviderModeStrategy.TOOLS: "Tools",
    ProviderModeStrategy.JSON: "Structured response",
    ProviderModeStrategy.MD_JSON: "Prompt text",
}


@dataclass(slots=True)
class StreamlitTurnState:
    """Mutable UI state for one chat session."""

    busy: bool = False
    status_text: str = ""
    pending_approval: ChatWorkflowApprovalState | None = None
    approval_decision_in_flight: bool = False
    active_turn_number: int = 0
    pending_interrupt_draft: str | None = None
    confidence: float | None = None
    cancelling: bool = False


@dataclass(slots=True)
class StreamlitQueuedEvent:
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
class StreamlitActiveTurnHandle:
    """Background runner handle stored in Streamlit session state."""

    session_id: str
    runner: ChatSessionTurnRunner
    event_queue: queue.Queue[StreamlitQueuedEvent]
    thread: threading.Thread
    turn_number: int


@dataclass(slots=True)
class StreamlitTurnOutcome:
    """One processed user turn plus any transcript-side system messages."""

    session_state: ChatSessionState
    transcript_entries: list[StreamlitTranscriptEntry]
    token_usage: ChatTokenUsage | None = None


@dataclass(slots=True)
class StreamlitWorkspaceState:
    """In-memory Streamlit app state layered over persisted session files."""

    sessions: dict[str, StreamlitPersistedSessionRecord]
    session_order: list[str]
    active_session_id: str
    preferences: StreamlitPreferences
    turn_states: dict[str, StreamlitTurnState] = field(default_factory=dict)
    drafts: dict[str, str] = field(default_factory=dict)
    clear_draft_for: set[str] = field(default_factory=set)
    show_export_for: set[str] = field(default_factory=set)
    startup_notices: list[str] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by the bootstrap and script entrypoints."""
    parser = argparse.ArgumentParser(
        prog="llm-tools-streamlit-chat",
        description="Streamlit chat over workspace tools.",
    )
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--directory", dest="directory_override", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--provider",
        choices=[preset.value for preset in ProviderPreset],
    )
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


def _resolve_chat_config(args: argparse.Namespace) -> TextualChatConfig:
    base_config = (
        load_textual_chat_config(args.config)
        if args.config is not None
        else TextualChatConfig()
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
    return TextualChatConfig.model_validate(raw)


def _resolve_root_argument(args: argparse.Namespace) -> Path | None:
    candidate = args.directory_override or args.directory
    if candidate is None:
        return None
    resolved_candidate = (
        candidate if isinstance(candidate, Path) else Path(str(candidate))
    )
    return resolved_candidate.expanduser().resolve()


def _streamlit_module() -> Any:  # pragma: no cover
    import streamlit as streamlit

    return streamlit


def _page_config() -> dict[str, object]:  # pragma: no cover
    return {
        "page_title": "llm-tools chat",
        "page_icon": "💬",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
        "menu_items": {
            "Get help": None,
            "Report a bug": None,
            "About": None,
        },
    }


def _all_tool_specs() -> dict[str, ToolSpec]:
    return build_available_tool_specs()


def _workspace_tool_names() -> set[str]:
    return {
        name
        for name, spec in _all_tool_specs().items()
        if spec.requires_filesystem or spec.requires_subprocess
    }


def _safe_local_read_tool_names() -> set[str]:
    return {
        name
        for name, spec in _all_tool_specs().items()
        if spec.side_effects is SideEffectClass.LOCAL_READ
        and spec.requires_filesystem
        and not spec.requires_subprocess
        and not spec.requires_network
    }


def _tool_group(spec: ToolSpec) -> str:
    tags = set(spec.tags)
    if "git" in tags:
        return "Git"
    if "atlassian" in tags or "jira" in tags:
        return "Atlassian"
    if "filesystem" in tags:
        return "Filesystem"
    if "text" in tags:
        return "Text"
    return "Other"


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


def _default_model_for_provider(
    base_config: TextualChatConfig,
    provider: ProviderPreset,
) -> str:
    if base_config.llm.provider is provider:
        return base_config.llm.model_name
    if provider is ProviderPreset.OPENAI:
        return "gpt-4.1-mini"
    if provider is ProviderPreset.OLLAMA:
        return "gemma4:26b"
    return "gpt-4.1-mini"


def _default_base_url_for_provider(
    base_config: TextualChatConfig,
    provider: ProviderPreset,
) -> str | None:
    if base_config.llm.provider is provider:
        return base_config.llm.api_base_url
    if provider is ProviderPreset.OLLAMA:
        return "http://127.0.0.1:11434/v1"
    return None


def _filter_enabled_tools_for_root(
    enabled_tools: set[str],
    *,
    root_path: str | None,
) -> set[str]:
    if root_path is not None:
        return set(enabled_tools)
    return enabled_tools.difference(_workspace_tool_names())


def _default_enabled_tool_names(
    config: TextualChatConfig,
    *,
    root_path: Path | None,
) -> set[str]:
    configured = config.policy.enabled_tools
    if configured is not None:
        return _filter_enabled_tools_for_root(
            set(configured).intersection(_all_tool_specs()),
            root_path=str(root_path) if root_path is not None else None,
        )
    default_tools = _safe_local_read_tool_names() if root_path is not None else set()
    return _filter_enabled_tools_for_root(
        default_tools,
        root_path=str(root_path) if root_path is not None else None,
    )


def resolve_enabled_tool_names(
    config: TextualChatConfig,
    *,
    root_path: Path | None = None,
) -> set[str]:
    """Return the default enabled tool names for a new Streamlit session."""
    return _default_enabled_tool_names(config, root_path=root_path)


def _default_runtime_config(
    config: TextualChatConfig,
    *,
    root_path: Path | None,
) -> StreamlitRuntimeConfig:
    return StreamlitRuntimeConfig(
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        api_base_url=config.llm.api_base_url,
        root_path=str(root_path) if root_path is not None else None,
        enabled_tools=sorted(_default_enabled_tool_names(config, root_path=root_path)),
        require_approval_for=set(config.policy.require_approval_for),
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
        inspector_open=config.ui.inspector_open_by_default,
    )


def _llm_config_for_runtime(
    config: TextualChatConfig,
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


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _storage_root() -> Path:
    override = os.getenv(_STORAGE_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".llm-tools" / "chat" / "streamlit").resolve()


def _sessions_dir() -> Path:
    return _storage_root() / _SESSION_STORAGE_DIR_NAME


def _preferences_path() -> Path:
    return _storage_root() / "preferences.json"


def _index_path() -> Path:
    return _storage_root() / "index.json"


def _session_path(session_id: str) -> Path:
    return _sessions_dir() / f"{session_id}.json"


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


def _new_session_record(
    session_id: str,
    runtime: StreamlitRuntimeConfig,
) -> StreamlitPersistedSessionRecord:
    now = _now_iso()
    summary = StreamlitSessionSummary(
        session_id=session_id,
        title="New chat",
        created_at=now,
        updated_at=now,
        root_path=runtime.root_path,
        provider=runtime.provider,
        model_name=runtime.model_name,
        message_count=0,
    )
    return StreamlitPersistedSessionRecord(
        summary=summary,
        runtime=runtime,
    )


def _load_workspace_state(
    *,
    root_path: Path | None,
    config: TextualChatConfig,
) -> StreamlitWorkspaceState:
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
                _session_path(session_id),
                StreamlitPersistedSessionRecord,
            )
        except Exception as exc:
            startup_notices.append(
                f"Skipped unreadable chat session {session_id}: {exc}"
            )
            continue
        if record is None:
            startup_notices.append(f"Skipped missing chat session {session_id}.")
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

    turn_states = {session_id: StreamlitTurnState() for session_id in session_order}
    return StreamlitWorkspaceState(
        sessions=sessions,
        session_order=session_order,
        active_session_id=active_session_id,
        preferences=preferences,
        turn_states=turn_states,
        startup_notices=startup_notices,
    )


def _save_workspace_state(app_state: StreamlitWorkspaceState) -> None:
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
    app_state: StreamlitWorkspaceState,
) -> StreamlitPersistedSessionRecord:
    return app_state.sessions[app_state.active_session_id]


def _turn_state_for(
    app_state: StreamlitWorkspaceState,
    session_id: str,
) -> StreamlitTurnState:
    return app_state.turn_states.setdefault(session_id, StreamlitTurnState())


def _widget_overrides_state() -> dict[str, object]:
    st = _streamlit_module()
    raw = st.session_state.setdefault(_WIDGET_OVERRIDES_STATE_SLOT, {})
    if isinstance(raw, dict):
        return cast(dict[str, object], raw)
    overrides: dict[str, object] = {}
    st.session_state[_WIDGET_OVERRIDES_STATE_SLOT] = overrides
    return overrides


def _prime_widget_value(key: str, value: object) -> None:
    st = _streamlit_module()
    overrides = _widget_overrides_state()
    if key in overrides:
        st.session_state[key] = overrides.pop(key)
        return
    st.session_state.setdefault(key, value)


def _queue_widget_override(key: str, value: object) -> None:
    _widget_overrides_state()[key] = value


def _delete_session(
    app_state: StreamlitWorkspaceState,
    *,
    session_id: str,
    config: TextualChatConfig,
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
        isinstance(handle, StreamlitActiveTurnHandle)
        and handle.session_id == session_id
    ):
        handle.runner.cancel()
        if st is not None:
            st.session_state[_ACTIVE_TURN_STATE_SLOT] = None
    app_state.sessions.pop(session_id, None)
    app_state.turn_states.pop(session_id, None)
    app_state.drafts.pop(session_id, None)
    app_state.show_export_for.discard(session_id)
    app_state.session_order = [
        item for item in app_state.session_order if item != session_id
    ]
    _session_path(session_id).unlink(missing_ok=True)
    if not app_state.session_order:
        template_runtime = _default_runtime_config(config, root_path=root_path)
        new_id = f"session-{uuid4().hex[:12]}"
        app_state.sessions[new_id] = _new_session_record(new_id, template_runtime)
        app_state.session_order = [new_id]
        app_state.turn_states[new_id] = StreamlitTurnState()
    if app_state.active_session_id not in app_state.sessions:
        app_state.active_session_id = app_state.session_order[0]
    _save_workspace_state(app_state)


def _create_session(
    app_state: StreamlitWorkspaceState,
    *,
    template_runtime: StreamlitRuntimeConfig,
) -> str:
    session_id = f"session-{uuid4().hex[:12]}"
    runtime = template_runtime.model_copy(deep=True)
    record = _new_session_record(session_id, runtime)
    app_state.sessions[session_id] = record
    app_state.session_order.insert(0, session_id)
    app_state.turn_states[session_id] = StreamlitTurnState()
    app_state.active_session_id = session_id
    _remember_runtime_preferences(app_state.preferences, runtime)
    _save_workspace_state(app_state)
    return session_id


def _streamlit_theme_css(
    theme_mode: Literal["dark", "light"],
) -> str:  # pragma: no cover
    palette = {
        "dark": {
            "page": "#07111f",
            "sidebar": "#091423",
            "surface": "#101c2f",
            "surface_alt": "#15253d",
            "border": "#31435e",
            "text": "#e2e8f0",
            "muted": "#94a3b8",
            "accent": "#7dd3fc",
            "accent_soft": "rgba(125, 211, 252, 0.16)",
            "shadow": "rgba(2, 6, 23, 0.45)",
            "gradient": "radial-gradient(circle at top left, rgba(125, 211, 252, 0.22), transparent 38%), radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 28%), linear-gradient(180deg, #08121f 0%, #07111f 100%)",
        },
        "light": {
            "page": "#edf3fb",
            "sidebar": "#dfe9f6",
            "surface": "#ffffff",
            "surface_alt": "#f7fbff",
            "border": "#c8d6ea",
            "text": "#0f172a",
            "muted": "#475569",
            "accent": "#0369a1",
            "accent_soft": "rgba(3, 105, 161, 0.10)",
            "shadow": "rgba(15, 23, 42, 0.10)",
            "gradient": "radial-gradient(circle at top left, rgba(3, 105, 161, 0.12), transparent 36%), radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 26%), linear-gradient(180deg, #f5f8fd 0%, #edf3fb 100%)",
        },
    }[theme_mode]
    return f"""
<style>
:root {{
  --llm-tools-page: {palette["page"]};
  --llm-tools-sidebar: {palette["sidebar"]};
  --llm-tools-surface: {palette["surface"]};
  --llm-tools-surface-alt: {palette["surface_alt"]};
  --llm-tools-border: {palette["border"]};
  --llm-tools-text: {palette["text"]};
  --llm-tools-muted: {palette["muted"]};
  --llm-tools-accent: {palette["accent"]};
  --llm-tools-accent-soft: {palette["accent_soft"]};
  --llm-tools-shadow: {palette["shadow"]};
}}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
  background: {palette["gradient"]};
  color: var(--llm-tools-text);
}}
section[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {{
  background: linear-gradient(180deg, var(--llm-tools-sidebar) 0%, var(--llm-tools-page) 100%);
  color: var(--llm-tools-text);
}}
[data-testid="stAppViewBlockContainer"] {{
  padding-top: 0.75rem;
}}
.stApp p,
.stApp label,
.stApp .stMarkdown,
.stApp .stCaption,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-baseweb="select"] span,
.stApp [role="tab"],
.stApp [data-testid="stExpander"] summary {{
  color: var(--llm-tools-text);
}}
.stApp [data-testid="stChatMessage"],
.stApp [data-testid="stExpander"],
.stApp [data-testid="stVerticalBlockBorderWrapper"],
.stApp [data-testid="stTabs"],
.stApp [data-baseweb="popover"],
.stApp [data-baseweb="tab-list"] {{
  background: color-mix(in srgb, var(--llm-tools-surface) 96%, transparent);
  border-color: var(--llm-tools-border);
  color: var(--llm-tools-text);
}}
.stApp [data-testid="stChatMessage"] {{
  border: 1px solid var(--llm-tools-border);
  border-radius: 1rem;
  padding: 0.9rem 1rem;
  box-shadow: 0 18px 36px var(--llm-tools-shadow);
}}
.stApp textarea,
.stApp input,
.stApp select,
.stApp div[data-baseweb="input"] > div,
.stApp div[data-baseweb="base-input"] > div,
.stApp div[data-baseweb="textarea"] > div,
.stApp div[data-baseweb="select"] > div,
.stApp [data-baseweb="tab"],
.stApp .stButton > button,
.stApp [data-testid="stDownloadButton"] > button {{
  background: var(--llm-tools-surface);
  color: var(--llm-tools-text);
  border: 1px solid var(--llm-tools-border);
  border-radius: 0.85rem;
}}
.stApp textarea::placeholder,
.stApp input::placeholder {{
  color: var(--llm-tools-muted);
}}
.stApp .stButton > button:hover,
.stApp [data-testid="stDownloadButton"] > button:hover,
.stApp [data-baseweb="tab"]:hover {{
  border-color: var(--llm-tools-accent);
  color: var(--llm-tools-accent);
}}
.stApp [data-baseweb="tab"][aria-selected="true"] {{
  background: var(--llm-tools-accent-soft);
  color: var(--llm-tools-text);
}}
.stApp [data-testid="stExpander"] details,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpanderDetails"] {{
  background: transparent;
  color: var(--llm-tools-text);
}}
.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stToggle"] label {{
  color: var(--llm-tools-text);
}}
.llm-tools-hero,
.llm-tools-summary-bar,
.llm-tools-status-line,
.llm-tools-empty {{
  border: 1px solid var(--llm-tools-border);
  background: color-mix(in srgb, var(--llm-tools-surface) 96%, transparent);
  border-radius: 1rem;
  box-shadow: 0 18px 40px var(--llm-tools-shadow);
}}
.llm-tools-settings-shell {{
  border: 1px solid var(--llm-tools-border);
  background: var(--llm-tools-sidebar);
  border-radius: 1rem;
  box-shadow: 0 18px 40px var(--llm-tools-shadow);
  padding: 0.5rem 0.45rem 0.65rem 0.45rem;
  min-height: 8rem;
}}
.llm-tools-hero {{ padding: 1rem 1.1rem; margin-bottom: 0.9rem; }}
.llm-tools-summary-bar {{ padding: 0.65rem 0.85rem; margin-bottom: 1rem; }}
.llm-tools-status-line {{ padding: 0.55rem 0.8rem; margin: 1rem 0 0.55rem 0; }}
.llm-tools-empty {{ padding: 1.4rem; margin: 1rem 0; }}
.llm-tools-brand {{ font-size: 1.45rem; font-weight: 700; letter-spacing: -0.02em; }}
.llm-tools-subtitle {{ color: var(--llm-tools-muted); font-size: 0.95rem; }}
.llm-tools-chip {{ display: inline-block; margin: 0.12rem 0.35rem 0.12rem 0; padding: 0.2rem 0.55rem; border-radius: 999px; background: var(--llm-tools-accent-soft); color: var(--llm-tools-text); border: 1px solid var(--llm-tools-border); font-size: 0.82rem; }}
.llm-tools-session-meta {{ color: var(--llm-tools-muted); font-size: 0.78rem; margin-top: 0.2rem; }}
.llm-tools-sidebar-title {{ font-weight: 700; letter-spacing: -0.02em; font-size: 1.2rem; margin-bottom: 0.25rem; }}
.llm-tools-status-text {{ font-size: 0.92rem; font-weight: 600; }}
.llm-tools-status-label {{ color: var(--llm-tools-muted); margin-right: 0.35rem; }}
.llm-tools-settings-title {{ font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; color: var(--llm-tools-text); }}
.llm-tools-settings-meta {{ color: var(--llm-tools-muted); font-size: 0.78rem; margin-bottom: 0.5rem; }}
.llm-tools-settings-rail-label {{ font-size: 0.78rem; color: var(--llm-tools-muted); margin: 0.25rem 0 0.55rem 0; text-transform: uppercase; letter-spacing: 0.06em; text-align: center; }}
.llm-tools-chevron-button button {{ min-width: 2.25rem; font-weight: 700; }}
.stApp [data-testid="InputInstructions"] {{ pointer-events: none; }}
</style>
"""


def _theme_mode_for_session(
    app_state: StreamlitWorkspaceState,
) -> Literal["dark", "light"]:
    st = _streamlit_module()
    theme_enabled = bool(
        st.session_state.setdefault(
            _THEME_TOGGLE_KEY,
            app_state.preferences.theme_mode == "dark",
        )
    )
    return "dark" if theme_enabled else "light"


def _render_theme(preferences: StreamlitPreferences) -> None:  # pragma: no cover
    st = _streamlit_module()
    st.markdown(
        _streamlit_theme_css(preferences.theme_mode),
        unsafe_allow_html=True,
    )


def _render_brand_header() -> None:  # pragma: no cover
    st = _streamlit_module()
    st.markdown(
        "<div class='llm-tools-hero'><div class='llm-tools-brand'>llm-tools chat</div><div class='llm-tools-subtitle'>Multi-session workspace chat with runtime model, provider, root, and tool controls.</div></div>",
        unsafe_allow_html=True,
    )


def _render_fatal_error(exc: Exception) -> None:  # pragma: no cover
    st = _streamlit_module()
    traceback_text = "".join(traceback.format_exception(exc)).rstrip()
    st.markdown(
        "<div class='llm-tools-hero'><div class='llm-tools-brand'>llm-tools chat</div><div class='llm-tools-subtitle'>The app hit an unexpected error. Streamlit's default exception screen is suppressed for this app.</div></div>",
        unsafe_allow_html=True,
    )
    st.error(f"{type(exc).__name__}: {exc}")
    with st.expander("Traceback", expanded=True):
        st.text_area(
            "Traceback text",
            value=traceback_text,
            height=320,
            key="fatal-error-traceback",
        )


def _transcript_export_text(entries: list[StreamlitTranscriptEntry]) -> str:
    parts = []
    for entry in entries:
        if entry.final_response is not None:
            text = format_final_response(entry.final_response)
            parts.append(format_transcript_text("assistant", text))
            continue
        parts.append(
            format_transcript_text(
                entry.role,
                entry.text,
                assistant_completion_state=entry.assistant_completion_state,
            )
        )
    return "\n\n".join(part for part in parts if part).rstrip()


def _render_final_response(response: ChatFinalResponse) -> None:  # pragma: no cover
    st = _streamlit_module()
    st.markdown(response.answer)
    if response.confidence is not None:
        st.caption(f"Confidence: {response.confidence:.2f}")
    if response.citations:
        st.markdown("**Citations**")
        for citation in response.citations:
            st.markdown(f"- `{format_citation(citation)}`")
            if citation.excerpt:
                st.code(citation.excerpt)
    if response.uncertainty:
        st.markdown("**Uncertainty**")
        for item in response.uncertainty:
            st.markdown(f"- {item}")
    if response.missing_information:
        st.markdown("**Missing Information**")
        for item in response.missing_information:
            st.markdown(f"- {item}")
    if response.follow_up_suggestions:
        st.markdown("**Follow-up Suggestions**")
        for item in response.follow_up_suggestions:
            st.markdown(f"- {item}")


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
            _render_final_response(entry.final_response)
            return
        st.markdown(entry.text)


def _render_empty_state(
    record: StreamlitPersistedSessionRecord,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    root_text = record.runtime.root_path or "No workspace selected"
    with st.container():
        st.markdown("### Start a new conversation")
        st.caption(f"Current root: {root_text}")
        st.markdown(
            "Choose a provider and model in the settings rail, optionally select a root directory, then ask a grounded question or enable external tools from the settings panel."
        )


def _serialize_workflow_event(
    event: object,
    *,
    turn_number: int,
    session_id: str,
) -> StreamlitQueuedEvent:
    if isinstance(event, ChatWorkflowStatusEvent):
        return StreamlitQueuedEvent(
            kind="status",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowApprovalEvent):
        return StreamlitQueuedEvent(
            kind="approval_requested",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowApprovalResolvedEvent):
        return StreamlitQueuedEvent(
            kind="approval_resolved",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowInspectorEvent):
        return StreamlitQueuedEvent(
            kind="inspector",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowResultEvent):
        return StreamlitQueuedEvent(
            kind="result",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    raise TypeError(f"Unsupported workflow event type: {type(event)!r}")


def _worker_run_turn(handle: StreamlitActiveTurnHandle) -> None:  # pragma: no cover
    try:
        for event in handle.runner:
            handle.event_queue.put(
                _serialize_workflow_event(
                    event,
                    turn_number=handle.turn_number,
                    session_id=handle.session_id,
                )
            )
    except Exception as exc:  # pragma: no cover - surfaced through queued errors
        handle.event_queue.put(
            StreamlitQueuedEvent(
                kind="error",
                payload=str(exc),
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )
    finally:
        handle.event_queue.put(
            StreamlitQueuedEvent(
                kind="complete",
                payload=None,
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )


def _title_from_prompt(prompt: str) -> str:
    cleaned = " ".join(prompt.strip().split())
    if len(cleaned) <= 60:
        return cleaned
    return f"{cleaned[:57].rstrip()}..."


def _effective_enabled_tools(runtime: StreamlitRuntimeConfig) -> set[str]:
    return _filter_enabled_tools_for_root(
        set(runtime.enabled_tools),
        root_path=runtime.root_path,
    )


def _build_chat_runner(
    *,
    session_id: str,
    config: TextualChatConfig,
    runtime: StreamlitRuntimeConfig,
    provider: ModelTurnProvider,
    session_state: ChatSessionState,
    user_message: str,
) -> ChatSessionTurnRunner:
    available_tool_specs = _all_tool_specs()
    enabled_tools = _effective_enabled_tools(runtime)
    allowed_side_effects = {SideEffectClass.NONE}
    for tool_name in enabled_tools:
        spec = available_tool_specs.get(tool_name)
        if spec is not None:
            allowed_side_effects.add(spec.side_effects)
    policy = ToolPolicy(
        allowed_tools=enabled_tools,
        allowed_side_effects=allowed_side_effects,
        require_approval_for=set(runtime.require_approval_for),
        allow_network=runtime.allow_network,
        allow_filesystem=runtime.allow_filesystem and runtime.root_path is not None,
        allow_subprocess=runtime.allow_subprocess and runtime.root_path is not None,
        redaction=config.policy.redaction.model_copy(deep=True),
    )
    registry, executor = build_chat_executor(policy=policy)
    root = Path(runtime.root_path) if runtime.root_path is not None else None
    return run_interactive_chat_session_turn(
        user_message=user_message,
        session_state=session_state,
        executor=executor,
        provider=provider,
        system_prompt=build_chat_system_prompt(
            tool_registry=registry,
            tool_limits=config.tool_limits,
            enabled_tool_names=enabled_tools,
            workspace_enabled=root is not None,
        ),
        base_context=build_chat_context(
            root_path=root,
            config=config,
            app_name=f"streamlit-chat-{session_id}",
        ),
        session_config=config.session,
        tool_limits=config.tool_limits,
        redaction_config=config.policy.redaction,
        temperature=config.llm.temperature,
    )


def _append_notice(
    record: StreamlitPersistedSessionRecord,
    *,
    role: Literal["system", "error"],
    text: str,
) -> None:
    record.transcript.append(StreamlitTranscriptEntry(role=role, text=text))
    _touch_record(record)


def _apply_turn_error(
    app_state: StreamlitWorkspaceState,
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


def _apply_turn_result(
    app_state: StreamlitWorkspaceState,
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
        turn_state.confidence = result.final_response.confidence
        record.confidence = result.final_response.confidence
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
        turn_state.confidence = None
        record.confidence = None
    else:
        turn_state.confidence = None
        record.confidence = None
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


def _apply_queued_event(
    app_state: StreamlitWorkspaceState,
    queued_event: StreamlitQueuedEvent,
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


def _drain_active_turn_events(app_state: StreamlitWorkspaceState) -> str | None:
    st = _streamlit_module()
    handle = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
    if not isinstance(handle, StreamlitActiveTurnHandle):
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


def _start_streamlit_turn(  # pragma: no cover
    *,
    app_state: StreamlitWorkspaceState,
    session_id: str,
    config: TextualChatConfig,
    provider: ModelTurnProvider,
    user_message: str,
) -> None:
    st = _streamlit_module()
    record = app_state.sessions[session_id]
    turn_state = _turn_state_for(app_state, session_id)
    turn_number = turn_state.active_turn_number + 1
    runner = _build_chat_runner(
        session_id=session_id,
        config=config,
        runtime=record.runtime,
        provider=provider,
        session_state=record.workflow_session_state,
        user_message=user_message,
    )
    event_queue: queue.Queue[StreamlitQueuedEvent] = queue.Queue()
    handle = StreamlitActiveTurnHandle(
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
    if record.summary.title == "New chat" and not any(
        entry.role == "user" for entry in record.transcript
    ):
        record.summary.title = _title_from_prompt(user_message)
    record.transcript.append(StreamlitTranscriptEntry(role="user", text=user_message))
    _touch_record(record)
    st.session_state[_ACTIVE_TURN_STATE_SLOT] = handle
    thread.start()
    _save_workspace_state(app_state)


def _current_api_key(  # pragma: no cover
    llm_config: Any,
) -> str | None:
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


def _available_model_options(  # pragma: no cover
    runtime: StreamlitRuntimeConfig,
    preferences: StreamlitPreferences,
    config: TextualChatConfig,
) -> tuple[list[str], str | None]:
    llm_config = _llm_config_for_runtime(config, runtime)
    current_api_key = _current_api_key(llm_config)
    fallback = _dedupe_preserve(
        [
            runtime.model_name,
            _default_model_for_provider(config, runtime.provider),
            *preferences.recent_models.get(runtime.provider.value, []),
        ]
    )
    metadata = llm_config.credential_prompt_metadata()
    if metadata.expects_api_key and current_api_key is None:
        return fallback, _missing_api_key_text(llm_config)
    try:
        provider = _create_provider_for_runtime(
            llm_config,
            runtime,
            api_key=current_api_key,
            model_name=runtime.model_name,
        )
        options = _dedupe_preserve([*provider.list_available_models(), *fallback])
        if options:
            return options, None
        return fallback, "No models were returned by the provider."
    except Exception as exc:
        return fallback, f"Unable to list available models: {exc}"


def _missing_api_key_text(llm_config: Any) -> str:
    env_var = llm_config.api_key_env_var or "OPENAI_API_KEY"
    return f"Set {env_var} or enter it in the header controls to use this provider."


def _submit_streamlit_prompt(
    *,
    app_state: StreamlitWorkspaceState,
    session_id: str,
    config: TextualChatConfig,
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
            app_state, session_id=session_id, preserve_pending_prompt=True
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


def _resolve_active_approval(  # pragma: no cover
    app_state: StreamlitWorkspaceState,
    *,
    session_id: str,
    approved: bool,
) -> None:
    st = _streamlit_module()
    handle = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
    turn_state = _turn_state_for(app_state, session_id)
    if not isinstance(handle, StreamlitActiveTurnHandle):
        return
    if handle.session_id != session_id or turn_state.pending_approval is None:
        return
    if not handle.runner.resolve_pending_approval(approved):
        return
    turn_state.approval_decision_in_flight = True
    turn_state.status_text = "approving" if approved else "denying"
    _save_workspace_state(app_state)


def _cancel_active_turn(  # pragma: no cover
    app_state: StreamlitWorkspaceState,
    *,
    session_id: str,
    preserve_pending_prompt: bool = False,
) -> None:
    st = _streamlit_module()
    handle = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
    turn_state = _turn_state_for(app_state, session_id)
    if (
        not isinstance(handle, StreamlitActiveTurnHandle)
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
            _append_notice(
                app_state.sessions[session_id],
                role="system",
                text="Stopped active turn.",
            )
        return
    if not preserve_pending_prompt:
        turn_state.pending_interrupt_draft = None
    turn_state.pending_approval = None
    turn_state.approval_decision_in_flight = False
    turn_state.cancelling = True
    turn_state.status_text = "stopping"
    handle.runner.cancel()


def _runtime_to_control_state(
    config: TextualChatConfig,
    runtime: StreamlitRuntimeConfig,
) -> ChatControlState:
    default_enabled = _default_enabled_tool_names(
        config,
        root_path=Path(runtime.root_path) if runtime.root_path else None,
    )
    return ChatControlState(
        active_model_name=runtime.model_name,
        default_enabled_tools=set(default_enabled),
        enabled_tools=set(runtime.enabled_tools),
        require_approval_for=set(runtime.require_approval_for),
        inspector_open=runtime.inspector_open,
    )


def _apply_control_state(
    runtime: StreamlitRuntimeConfig,
    state: ChatControlState,
) -> None:
    runtime.model_name = state.active_model_name
    runtime.enabled_tools = sorted(
        _filter_enabled_tools_for_root(
            set(state.enabled_tools),
            root_path=runtime.root_path,
        )
    )
    runtime.require_approval_for = set(state.require_approval_for)
    runtime.inspector_open = state.inspector_open


def _run_streamlit_command(  # pragma: no cover
    *,
    config: TextualChatConfig,
    app_state: StreamlitWorkspaceState,
    session_id: str,
) -> ChatCommandOutcome[ModelTurnProvider]:
    record = app_state.sessions[session_id]
    runtime = record.runtime
    llm_config = _llm_config_for_runtime(config, runtime)
    state = _runtime_to_control_state(config, runtime)

    def _list_models() -> ModelCatalogOutcome:
        options, notice = _available_model_options(
            runtime, app_state.preferences, config
        )
        if notice is not None:
            return ModelCatalogOutcome(
                notice=ChatControlNotice(
                    role="system",
                    text=f"Current model: {runtime.model_name}\n{notice}",
                )
            )
        return ModelCatalogOutcome(model_ids=options)

    def _switch_model(new_model_name: str) -> ModelSwitchOutcome[ModelTurnProvider]:
        api_key = _current_api_key(llm_config)
        metadata = llm_config.credential_prompt_metadata()
        if metadata.expects_api_key and api_key is None:
            return ModelSwitchOutcome(
                notice=ChatControlNotice(
                    role="system",
                    text=_missing_api_key_text(llm_config),
                )
            )
        try:
            provider = _create_provider_for_runtime(
                llm_config,
                runtime,
                api_key=api_key,
                model_name=new_model_name,
            )
        except Exception as exc:
            return ModelSwitchOutcome(
                notice=ChatControlNotice(
                    role="error",
                    text=f"Unable to switch model to {new_model_name}: {exc}",
                )
            )
        return ModelSwitchOutcome(provider=provider)

    outcome = handle_chat_command(
        app_state.drafts.get(session_id, ""),
        state=state,
        available_tool_specs=_all_tool_specs(),
        busy=_turn_state_for(app_state, session_id).busy,
        list_models=_list_models,
        switch_model=_switch_model,
        exit_mode="notice",
        exit_notice="Streamlit chat keeps running. Use the session rail or close the browser tab to leave.",
    )
    _apply_control_state(runtime, state)
    return outcome


def _apply_streamlit_command(  # pragma: no cover
    app_state: StreamlitWorkspaceState,
    *,
    session_id: str,
    outcome: ChatCommandOutcome[ModelTurnProvider],
) -> None:
    record = app_state.sessions[session_id]
    for notice in outcome.notices:
        _append_notice(record, role=notice.role, text=notice.text)
    if outcome.request_copy:
        app_state.show_export_for.add(session_id)
    _touch_record(record)
    _save_workspace_state(app_state)


def _is_command_prompt(prompt: str) -> bool:
    cleaned = prompt.strip().lower()
    return cleaned.startswith("/") or cleaned in {"quit", "exit"}


def process_streamlit_chat_turn(  # pragma: no cover
    *,
    root_path: Path | None,
    config: TextualChatConfig,
    provider: ModelTurnProvider,
    session_state: ChatSessionState,
    user_message: str,
    approval_resolver: Callable[[ChatWorkflowApprovalState], bool] | None = None,
    runtime_config: StreamlitRuntimeConfig | None = None,
) -> StreamlitTurnOutcome:
    """Execute one chat turn using the shared Streamlit reducers."""
    runtime = runtime_config or _default_runtime_config(config, root_path=root_path)
    transcript_entries: list[StreamlitTranscriptEntry] = []
    token_usage: ChatTokenUsage | None = None
    inspector_state = StreamlitInspectorState()
    turn_state = StreamlitTurnState(busy=True, active_turn_number=1)
    runner = _build_chat_runner(
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
            turn_state.pending_approval = approval_event.approval
            turn_state.status_text = (
                f"approval required for {approval_event.approval.tool_name}"
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
            turn_state.pending_approval = None
            continue
        if isinstance(event, ChatWorkflowInspectorEvent):
            inspector_event = ChatWorkflowInspectorEvent.model_validate(
                event.model_dump(mode="json")
            )
            label = (
                f"Turn 1 Round {inspector_event.round_index} "
                f"{inspector_event.kind.replace('_', ' ')}"
            )
            target = {
                "provider_messages": inspector_state.provider_messages,
                "parsed_response": inspector_state.parsed_responses,
                "tool_execution": inspector_state.tool_executions,
            }[inspector_event.kind]
            _append_inspector_entry(
                target, label=label, payload=inspector_event.payload
            )
            continue
        if isinstance(event, ChatWorkflowStatusEvent):
            turn_state.status_text = event.status
            continue
        if not isinstance(event, ChatWorkflowResultEvent):
            continue
        result = ChatWorkflowTurnResult.model_validate(event.result)
        updated_session_state = result.session_state or updated_session_state
        token_usage = result.token_usage
        if result.context_warning:
            transcript_entries.append(
                StreamlitTranscriptEntry(role="system", text=result.context_warning)
            )
        if result.status == "needs_continuation" and result.continuation_reason:
            transcript_entries.append(
                StreamlitTranscriptEntry(role="system", text=result.continuation_reason)
            )
        if result.final_response is not None:
            transcript_entries.append(
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
                transcript_entries.append(
                    StreamlitTranscriptEntry(
                        role="assistant",
                        text=interrupted_message.content,
                        assistant_completion_state="interrupted",
                    )
                )
            elif result.interruption_reason:
                transcript_entries.append(
                    StreamlitTranscriptEntry(
                        role="system", text=result.interruption_reason
                    )
                )
        turn_state.busy = False
    return StreamlitTurnOutcome(
        session_state=updated_session_state,
        transcript_entries=transcript_entries,
        token_usage=token_usage,
    )


def _running_in_wsl() -> bool:
    return bool(os.getenv("WSL_DISTRO_NAME") or os.getenv("WSL_INTEROP"))


def _pick_root_directory_via_powershell() -> tuple[str | None, str | None]:
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        return None, "Native directory picker is not available in this environment."
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
        "$dialog.ShowNewFolderButton = $false; "
        "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
        "{ [Console]::Out.Write($dialog.SelectedPath) }"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return None, f"Unable to open native directory picker: {exc}"
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "picker failed"
        return None, f"Unable to open native directory picker: {message}"
    selected = result.stdout.strip()
    if selected == "":
        return None, None
    wslpath = shutil.which("wslpath")
    if wslpath is not None:
        translated = subprocess.run(
            [wslpath, "-u", selected],
            capture_output=True,
            text=True,
            check=False,
        )
        if translated.returncode == 0 and translated.stdout.strip():
            selected = translated.stdout.strip()
    return _resolve_root_text(selected)


def _pick_root_directory_via_tk() -> tuple[str | None, str | None]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None, "Native directory picker is not available in this environment."
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        with suppress(Exception):
            root.attributes("-topmost", True)
        selected = filedialog.askdirectory(mustexist=True)
    except Exception as exc:
        return None, f"Unable to open native directory picker: {exc}"
    finally:
        if root is not None:
            root.destroy()
    if not selected:
        return None, None
    return _resolve_root_text(selected)


def _pick_root_directory() -> tuple[str | None, str | None]:
    if _running_in_wsl():
        selected_root, error_message = _pick_root_directory_via_powershell()
        if selected_root is not None or error_message is None:
            return selected_root, error_message
    return _pick_root_directory_via_tk()


def _apply_runtime_root(
    *,
    config: TextualChatConfig,
    preferences: StreamlitPreferences,
    record: StreamlitPersistedSessionRecord,
    runtime: StreamlitRuntimeConfig,
    resolved_root: str | None,
) -> None:
    previous_root = runtime.root_path
    runtime.root_path = resolved_root
    if resolved_root is None:
        runtime.enabled_tools = sorted(
            _filter_enabled_tools_for_root(
                set(runtime.enabled_tools),
                root_path=None,
            )
        )
        _append_notice(record, role="system", text="Workspace root cleared.")
        _touch_record(record)
        return
    if previous_root is None and not runtime.enabled_tools:
        runtime.enabled_tools = sorted(
            _default_enabled_tool_names(config, root_path=Path(resolved_root))
        )
    else:
        runtime.enabled_tools = sorted(
            _filter_enabled_tools_for_root(
                set(runtime.enabled_tools),
                root_path=resolved_root,
            )
        )
    _remember_runtime_preferences(preferences, runtime)
    _append_notice(
        record,
        role="system",
        text=f"Workspace root updated to {runtime.root_path}.",
    )
    _touch_record(record)


def _provider_control_strip(  # pragma: no cover  # noqa: C901
    app_state: StreamlitWorkspaceState,
    *,
    config: TextualChatConfig,
    session_id: str,
) -> None:
    st = _streamlit_module()
    record = app_state.sessions[session_id]
    runtime = record.runtime
    turn_state = _turn_state_for(app_state, session_id)
    busy = turn_state.busy
    provider_key = f"provider:{session_id}"
    mode_key = f"provider-mode:{session_id}"
    model_key = f"model:{session_id}"
    root_input_key = f"root-input:{session_id}"
    recent_root_key = f"recent-root:{session_id}"
    base_url_key = f"base-url:{session_id}"

    with st.expander("Session configuration", expanded=True):
        provider_values = [preset.value for preset in ProviderPreset]
        _prime_widget_value(provider_key, runtime.provider.value)
        if st.session_state.get(provider_key) not in provider_values:
            st.session_state[provider_key] = runtime.provider.value
        selected_provider = st.selectbox(
            "Provider",
            options=provider_values,
            index=provider_values.index(str(st.session_state[provider_key])),
            key=provider_key,
            disabled=busy,
        )
        if selected_provider != runtime.provider.value:
            runtime.provider = ProviderPreset(selected_provider)
            runtime.api_base_url = _default_base_url_for_provider(
                config, runtime.provider
            )
            runtime.model_name = _default_model_for_provider(config, runtime.provider)
            _queue_widget_override(provider_key, runtime.provider.value)
            _queue_widget_override(model_key, runtime.model_name)
            _queue_widget_override(base_url_key, runtime.api_base_url or "")
            _remember_runtime_preferences(app_state.preferences, runtime)
            _touch_record(record)
            _save_workspace_state(app_state)
            st.rerun()

        mode_values = [strategy.value for strategy in ProviderModeStrategy]
        _prime_widget_value(mode_key, runtime.provider_mode_strategy.value)
        if st.session_state.get(mode_key) not in mode_values:
            st.session_state[mode_key] = runtime.provider_mode_strategy.value
        selected_mode = st.selectbox(
            "Instructor mode",
            options=mode_values,
            index=mode_values.index(str(st.session_state[mode_key])),
            key=mode_key,
            disabled=busy,
            format_func=lambda value: _MODE_LABELS[ProviderModeStrategy(str(value))],
        )
        if selected_mode != runtime.provider_mode_strategy.value:
            runtime.provider_mode_strategy = ProviderModeStrategy(selected_mode)
            _queue_widget_override(mode_key, runtime.provider_mode_strategy.value)
            _touch_record(record)
            _save_workspace_state(app_state)
            st.rerun()
        if runtime.provider_mode_strategy is ProviderModeStrategy.AUTO:
            st.caption(
                "Auto mode falls back in order: tools, structured response, prompt text."
            )

        model_options, model_notice = _available_model_options(
            runtime, app_state.preferences, config
        )
        if runtime.model_name not in model_options:
            model_options = _dedupe_preserve([runtime.model_name, *model_options])
        _prime_widget_value(model_key, runtime.model_name)
        if st.session_state.get(model_key) not in model_options:
            st.session_state[model_key] = runtime.model_name
        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(str(st.session_state[model_key])),
            key=model_key,
            disabled=busy,
        )
        if selected_model != runtime.model_name:
            runtime.model_name = selected_model
            _queue_widget_override(model_key, runtime.model_name)
            _remember_runtime_preferences(app_state.preferences, runtime)
            _touch_record(record)
            _save_workspace_state(app_state)
            st.rerun()

        _prime_widget_value(root_input_key, runtime.root_path or "")
        root_value = st.text_input(
            "Root directory",
            value=str(st.session_state[root_input_key]),
            key=root_input_key,
            disabled=busy,
            placeholder="No root selected",
        )
        root_action_cols = st.columns([1, 1, 1])
        if root_action_cols[0].button(
            "Apply", key=f"root-apply:{session_id}", disabled=busy
        ):
            resolved_root, error_message = _resolve_root_text(root_value)
            if error_message is not None:
                _append_notice(record, role="error", text=error_message)
                _touch_record(record)
            else:
                _queue_widget_override(root_input_key, resolved_root or "")
                _apply_runtime_root(
                    config=config,
                    preferences=app_state.preferences,
                    record=record,
                    runtime=runtime,
                    resolved_root=resolved_root,
                )
            _save_workspace_state(app_state)
            st.rerun()
        if root_action_cols[1].button(
            "Browse", key=f"root-browse:{session_id}", disabled=busy
        ):
            selected_root, error_message = _pick_root_directory()
            if error_message is not None:
                _append_notice(record, role="error", text=error_message)
                _touch_record(record)
            elif selected_root is not None:
                _queue_widget_override(root_input_key, selected_root)
                _apply_runtime_root(
                    config=config,
                    preferences=app_state.preferences,
                    record=record,
                    runtime=runtime,
                    resolved_root=selected_root,
                )
            if error_message is not None or selected_root is not None:
                _save_workspace_state(app_state)
                st.rerun()
        if root_action_cols[2].button(
            "Clear", key=f"root-clear:{session_id}", disabled=busy
        ):
            _queue_widget_override(root_input_key, "")
            _apply_runtime_root(
                config=config,
                preferences=app_state.preferences,
                record=record,
                runtime=runtime,
                resolved_root=None,
            )
            _save_workspace_state(app_state)
            st.rerun()

        recent_roots = [_ROOT_SENTINEL, *app_state.preferences.recent_roots]
        _prime_widget_value(recent_root_key, _ROOT_SENTINEL)
        if st.session_state.get(recent_root_key) not in recent_roots:
            st.session_state[recent_root_key] = _ROOT_SENTINEL
        selected_recent_root = st.selectbox(
            "Recent roots",
            options=recent_roots,
            index=recent_roots.index(str(st.session_state[recent_root_key])),
            format_func=lambda value: (
                "Choose a recent root" if value == _ROOT_SENTINEL else value
            ),
            key=recent_root_key,
            disabled=busy,
        )
        if selected_recent_root != _ROOT_SENTINEL:
            resolved_root, error_message = _resolve_root_text(selected_recent_root)
            _queue_widget_override(recent_root_key, _ROOT_SENTINEL)
            if error_message is not None:
                _append_notice(record, role="error", text=error_message)
                _touch_record(record)
            elif resolved_root is not None:
                _queue_widget_override(root_input_key, resolved_root)
                _apply_runtime_root(
                    config=config,
                    preferences=app_state.preferences,
                    record=record,
                    runtime=runtime,
                    resolved_root=resolved_root,
                )
            _save_workspace_state(app_state)
            st.rerun()

        show_base_url = runtime.provider is not ProviderPreset.OPENAI
        _prime_widget_value(base_url_key, runtime.api_base_url or "")
        base_url_value = st.text_input(
            "Base URL",
            value=str(st.session_state[base_url_key]),
            key=base_url_key,
            disabled=busy or not show_base_url,
            placeholder="Provider default",
        )
        if show_base_url and (base_url_value.strip() or None) != runtime.api_base_url:
            runtime.api_base_url = base_url_value.strip() or None
            _remember_runtime_preferences(app_state.preferences, runtime)
            _touch_record(record)
            _save_workspace_state(app_state)

        llm_config = _llm_config_for_runtime(config, runtime)
        metadata = llm_config.credential_prompt_metadata()
        if metadata.expects_api_key:
            env_var = llm_config.api_key_env_var or "OPENAI_API_KEY"
            cached_api_key = _current_api_key(llm_config)
            st.caption(
                f"API key: {env_var} {'available' if cached_api_key else 'required'}"
            )
            if os.getenv(env_var) is None:
                entered_key = st.text_input(
                    "API key",
                    value="",
                    type="password",
                    key=f"api-key:{env_var}",
                    disabled=busy,
                )
                if entered_key.strip():
                    cache = st.session_state.setdefault(_SECRET_CACHE_STATE_SLOT, {})
                    cache[env_var] = entered_key.strip()
        else:
            st.caption("API key: not required for this provider")

        if model_notice is not None:
            st.caption(model_notice)


def _resolve_root_text(raw_value: str) -> tuple[str | None, str | None]:
    cleaned = raw_value.strip()
    if cleaned == "":
        return None, None
    candidate = Path(cleaned).expanduser().resolve()
    if not candidate.exists():
        return None, f"Root directory does not exist: {candidate}"
    if not candidate.is_dir():
        return None, f"Root path is not a directory: {candidate}"
    return str(candidate), None


def _render_summary_chips(
    record: StreamlitPersistedSessionRecord,
) -> None:  # pragma: no cover
    st = _streamlit_module()
    token_usage = record.token_usage
    chips = [
        f"<span class='llm-tools-chip'>root: {record.runtime.root_path or 'none'}</span>",
        f"<span class='llm-tools-chip'>provider: {record.runtime.provider.value}</span>",
        f"<span class='llm-tools-chip'>model: {record.runtime.model_name}</span>",
    ]
    if isinstance(token_usage, ChatTokenUsage):
        chips.append(
            f"<span class='llm-tools-chip'>tokens: {token_usage.session_tokens or '-'}</span>"
        )
    if record.confidence is not None:
        chips.append(
            f"<span class='llm-tools-chip'>confidence: {record.confidence:.2f}</span>"
        )
    st.markdown(
        f"<div class='llm-tools-summary-bar'>{''.join(chips)}</div>",
        unsafe_allow_html=True,
    )


def _render_tools_popover(  # pragma: no cover
    app_state: StreamlitWorkspaceState,
    *,
    session_id: str,
    config: TextualChatConfig,
) -> None:
    st = _streamlit_module()
    record = app_state.sessions[session_id]
    runtime = record.runtime
    busy = _turn_state_for(app_state, session_id).busy
    tool_specs = _all_tool_specs()
    workspace_tool_names = _workspace_tool_names()
    with st.expander("Tools and approvals", expanded=False):
        st.caption("Enable built-in tools and the capabilities they require.")
        preset_cols = st.columns(3)
        if preset_cols[0].button(
            "Default", key=f"tools-default:{session_id}", disabled=busy
        ):
            runtime.enabled_tools = sorted(
                _default_enabled_tool_names(
                    config,
                    root_path=Path(runtime.root_path) if runtime.root_path else None,
                )
            )
            _touch_record(record)
            _save_workspace_state(app_state)
            st.rerun()
        if preset_cols[1].button(
            "Workspace + Git",
            key=f"tools-git:{session_id}",
            disabled=busy or runtime.root_path is None,
        ):
            runtime.enabled_tools = sorted(
                _filter_enabled_tools_for_root(
                    _safe_local_read_tool_names().union(
                        {
                            name
                            for name, spec in tool_specs.items()
                            if "git" in spec.tags
                        }
                    ),
                    root_path=runtime.root_path,
                )
            )
            _touch_record(record)
            _save_workspace_state(app_state)
            st.rerun()
        if preset_cols[2].button(
            "All built-ins", key=f"tools-all:{session_id}", disabled=busy
        ):
            runtime.enabled_tools = sorted(
                _filter_enabled_tools_for_root(
                    set(tool_specs),
                    root_path=runtime.root_path,
                )
            )
            runtime.allow_network = True
            runtime.allow_subprocess = runtime.root_path is not None
            _touch_record(record)
            _save_workspace_state(app_state)
            st.rerun()

        capability_cols = st.columns(3)
        allow_network = capability_cols[0].toggle(
            "Allow network",
            value=runtime.allow_network,
            key=f"allow-network:{session_id}",
            disabled=busy,
        )
        allow_filesystem = capability_cols[1].toggle(
            "Allow filesystem",
            value=runtime.allow_filesystem,
            key=f"allow-filesystem:{session_id}",
            disabled=busy,
        )
        allow_subprocess = capability_cols[2].toggle(
            "Allow subprocess",
            value=runtime.allow_subprocess,
            key=f"allow-subprocess:{session_id}",
            disabled=busy or runtime.root_path is None,
        )
        if (
            allow_network != runtime.allow_network
            or allow_filesystem != runtime.allow_filesystem
            or allow_subprocess != runtime.allow_subprocess
        ):
            runtime.allow_network = allow_network
            runtime.allow_filesystem = allow_filesystem
            runtime.allow_subprocess = allow_subprocess
            _touch_record(record)
            _save_workspace_state(app_state)

        approval_cols = st.columns(4)
        for side_effect, label, col in (
            (SideEffectClass.LOCAL_READ, "Approve local read", approval_cols[0]),
            (SideEffectClass.LOCAL_WRITE, "Approve local write", approval_cols[1]),
            (SideEffectClass.EXTERNAL_READ, "Approve external read", approval_cols[2]),
            (
                SideEffectClass.EXTERNAL_WRITE,
                "Approve external write",
                approval_cols[3],
            ),
        ):
            approved = side_effect in runtime.require_approval_for
            toggled = col.toggle(
                label,
                value=approved,
                key=f"approval:{session_id}:{side_effect.value}",
                disabled=busy,
            )
            if toggled != approved:
                if toggled:
                    runtime.require_approval_for.add(side_effect)
                else:
                    runtime.require_approval_for.discard(side_effect)
                _touch_record(record)
                _save_workspace_state(app_state)

        grouped_tools: dict[str, list[tuple[str, ToolSpec]]] = {}
        for name, spec in sorted(tool_specs.items()):
            grouped_tools.setdefault(_tool_group(spec), []).append((name, spec))
        for group_name, entries in grouped_tools.items():
            st.markdown(f"**{group_name}**")
            for tool_name, spec in entries:
                enabled = tool_name in runtime.enabled_tools
                needs_workspace = tool_name in workspace_tool_names
                workspace_missing = needs_workspace and runtime.root_path is None
                disabled = busy or workspace_missing
                label = f"{tool_name} ({spec.side_effects.value})"
                checked = st.checkbox(
                    label,
                    value=enabled,
                    key=f"tool:{session_id}:{tool_name}",
                    disabled=disabled,
                )
                if checked != enabled:
                    if checked:
                        runtime.enabled_tools = sorted(
                            _filter_enabled_tools_for_root(
                                set(runtime.enabled_tools).union({tool_name}),
                                root_path=runtime.root_path,
                            )
                        )
                    else:
                        runtime.enabled_tools = sorted(
                            set(runtime.enabled_tools).difference({tool_name})
                        )
                    _touch_record(record)
                    _save_workspace_state(app_state)
                if workspace_missing:
                    st.caption("Select a root directory to enable this tool.")


def _render_session_details(  # pragma: no cover
    app_state: StreamlitWorkspaceState,
    *,
    session_id: str,
) -> None:
    st = _streamlit_module()
    record = app_state.sessions[session_id]
    turn_state = _turn_state_for(app_state, session_id)
    expanded = record.runtime.inspector_open or session_id in app_state.show_export_for
    with st.expander("Session details", expanded=expanded):
        inspector_tab, export_tab, advanced_tab = st.tabs(
            ["Inspector", "Export", "Advanced"]
        )
        with inspector_tab:
            tool_state = {
                "enabled_tools": record.runtime.enabled_tools,
                "disabled_tools": sorted(
                    set(_all_tool_specs()).difference(record.runtime.enabled_tools)
                ),
                "require_approval_for": sorted(
                    side_effect.value
                    for side_effect in record.runtime.require_approval_for
                ),
                "allow_network": record.runtime.allow_network,
                "allow_filesystem": record.runtime.allow_filesystem,
                "allow_subprocess": record.runtime.allow_subprocess,
            }
            st.code(pretty_json(tool_state))
            for label, entries in (
                ("Model Messages", record.inspector_state.provider_messages),
                ("Parsed Responses", record.inspector_state.parsed_responses),
                ("Tool Execution Records", record.inspector_state.tool_executions),
            ):
                st.caption(label)
                for entry in entries:
                    st.code(f"{entry.label}\n{pretty_json(entry.payload)}")
        with export_tab:
            export_text = _transcript_export_text(record.transcript)
            st.text_area(
                "Transcript export", export_text, height=240, key=f"export:{session_id}"
            )
            st.download_button(
                "Download transcript",
                data=export_text,
                file_name="llm-tools-chat-transcript.txt",
                use_container_width=True,
            )
        with advanced_tab:
            st.code(pretty_json(record.model_dump(mode="json")))
            if turn_state.pending_approval is not None:
                st.code(
                    pretty_json(turn_state.pending_approval.model_dump(mode="json"))
                )


def _render_status_and_composer(  # pragma: no cover
    app_state: StreamlitWorkspaceState,
    *,
    session_id: str,
    config: TextualChatConfig,
) -> None:
    st = _streamlit_module()
    turn_state = _turn_state_for(app_state, session_id)
    status_cols = st.columns([6, 1, 1, 1])
    status_text = turn_state.status_text or "idle"
    status_cols[0].markdown(
        (
            "<div class='llm-tools-status-line'>"
            "<span class='llm-tools-status-label'>Status:</span>"
            f"<span class='llm-tools-status-text'>{status_text}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if turn_state.pending_approval is not None:
        if status_cols[1].button(
            "Approve",
            key=f"approve:{session_id}",
            use_container_width=True,
            disabled=turn_state.approval_decision_in_flight,
        ):
            _resolve_active_approval(app_state, session_id=session_id, approved=True)
            st.rerun()
        if status_cols[2].button(
            "Deny",
            key=f"deny:{session_id}",
            use_container_width=True,
            disabled=turn_state.approval_decision_in_flight,
        ):
            _resolve_active_approval(app_state, session_id=session_id, approved=False)
            st.rerun()
    elif turn_state.busy:
        if status_cols[1].button(
            "Stop",
            key=f"stop:{session_id}",
            use_container_width=True,
        ):
            _cancel_active_turn(app_state, session_id=session_id)
            _save_workspace_state(app_state)
            st.rerun()
    draft_key = f"composer:{session_id}"
    if session_id in app_state.clear_draft_for:
        st.session_state[draft_key] = ""
        app_state.clear_draft_for.discard(session_id)
    with st.form(
        key=f"composer-form:{session_id}",
        clear_on_submit=False,
        enter_to_submit=True,
        border=False,
    ):
        composer_cols = st.columns([6, 1.2])
        composer_cols[0].text_area(
            "Message",
            key=draft_key,
            value=app_state.drafts.get(session_id, ""),
            height=90,
            placeholder="Ask a grounded question or use slash commands like /help. Press Ctrl+Enter or Cmd+Enter to send.",
            label_visibility="collapsed",
        )
        send_clicked = composer_cols[1].form_submit_button(
            "Send",
            key=f"send:{session_id}",
            use_container_width=True,
        )
    app_state.drafts[session_id] = str(st.session_state.get(draft_key, ""))
    if not send_clicked:
        return
    prompt = app_state.drafts.get(session_id, "").strip()
    if not prompt:
        return
    app_state.drafts[session_id] = ""
    app_state.clear_draft_for.add(session_id)
    if _is_command_prompt(prompt):
        app_state.drafts[session_id] = prompt
        outcome = _run_streamlit_command(
            config=config,
            app_state=app_state,
            session_id=session_id,
        )
        _apply_streamlit_command(
            app_state,
            session_id=session_id,
            outcome=outcome,
        )
        app_state.drafts[session_id] = ""
        st.rerun()
    _submit_streamlit_prompt(
        app_state=app_state,
        session_id=session_id,
        config=config,
        prompt=prompt,
    )
    st.rerun()


def _render_settings_panel(  # pragma: no cover
    app_state: StreamlitWorkspaceState,
    *,
    session_id: str,
    config: TextualChatConfig,
) -> None:
    st = _streamlit_module()
    record = app_state.sessions[session_id]
    st.markdown("<div class='llm-tools-settings-shell'>", unsafe_allow_html=True)
    if not app_state.preferences.settings_panel_open:
        st.markdown(
            "<div class='llm-tools-settings-rail-label'>Settings</div>",
            unsafe_allow_html=True,
        )
        with st.container():
            st.markdown(
                "<div class='llm-tools-chevron-button'>", unsafe_allow_html=True
            )
            if st.button(
                "«",
                key="settings-panel-toggle",
                use_container_width=True,
            ):
                app_state.preferences.settings_panel_open = True
                _save_workspace_state(app_state)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    header_cols = st.columns([4.2, 1])
    header_cols[0].markdown(
        "<div class='llm-tools-settings-title'>Settings</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            "<div class='llm-tools-settings-meta'>"
            f"{record.runtime.provider.value} | {record.runtime.model_name}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    with header_cols[1]:
        st.markdown("<div class='llm-tools-chevron-button'>", unsafe_allow_html=True)
        if st.button(
            "»",
            key="settings-panel-toggle",
            use_container_width=True,
        ):
            app_state.preferences.settings_panel_open = False
            _save_workspace_state(app_state)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    _provider_control_strip(app_state, config=config, session_id=session_id)
    _render_tools_popover(app_state, session_id=session_id, config=config)
    _render_session_details(app_state, session_id=session_id)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_sidebar(  # pragma: no cover
    app_state: StreamlitWorkspaceState,
    *,
    config: TextualChatConfig,
    root_path: Path | None,
) -> None:
    st = _streamlit_module()
    with st.sidebar:
        st.markdown(
            "<div class='llm-tools-sidebar-title'>llm-tools chat</div>",
            unsafe_allow_html=True,
        )
        st.caption("Session rail")
        active_record = _active_session(app_state)
        if st.button("New chat", use_container_width=True):
            _create_session(
                app_state,
                template_runtime=active_record.runtime,
            )
            st.rerun()
        theme_mode = st.toggle(
            "Dark mode",
            key=_THEME_TOGGLE_KEY,
        )
        resolved_theme = "dark" if theme_mode else "light"
        if resolved_theme != app_state.preferences.theme_mode:
            app_state.preferences.theme_mode = cast(
                Literal["dark", "light"], resolved_theme
            )
            _save_workspace_state(app_state)
        search_value = st.text_input("Search sessions", value="", key="session-search")
        filtered_ids = [
            session_id
            for session_id in app_state.session_order
            if _session_matches(app_state.sessions[session_id], search_value)
        ]
        for session_id in filtered_ids:
            record = app_state.sessions[session_id]
            is_active = session_id == app_state.active_session_id
            row_cols = st.columns([5, 1])
            button_label = f"{'● ' if is_active else ''}{record.summary.title}"
            if row_cols[0].button(
                button_label,
                key=f"session-select:{session_id}",
                use_container_width=True,
            ):
                app_state.active_session_id = session_id
                _save_workspace_state(app_state)
                st.rerun()
            if row_cols[1].button(
                "×", key=f"session-delete:{session_id}", use_container_width=True
            ):
                _delete_session(
                    app_state,
                    session_id=session_id,
                    config=config,
                    root_path=root_path,
                )
                st.rerun()
            meta_bits = [record.runtime.provider.value, record.runtime.model_name]
            if record.runtime.root_path:
                meta_bits.append(record.runtime.root_path)
            st.markdown(
                f"<div class='llm-tools-session-meta'>{' | '.join(meta_bits)}</div>",
                unsafe_allow_html=True,
            )


def _session_matches(record: StreamlitPersistedSessionRecord, query: str) -> bool:
    cleaned = query.strip().lower()
    if cleaned == "":
        return True
    haystack = " ".join(
        [
            record.summary.title,
            record.runtime.provider.value,
            record.runtime.model_name,
            record.runtime.root_path or "",
        ]
    ).lower()
    return cleaned in haystack


def run_streamlit_chat_app(  # pragma: no cover
    *, root_path: Path | None, config: TextualChatConfig
) -> None:
    """Render the Streamlit chat UI."""
    st = _streamlit_module()
    st.set_page_config(**_page_config())
    if _APP_STATE_SLOT not in st.session_state:
        st.session_state[_APP_STATE_SLOT] = _load_workspace_state(
            root_path=root_path,
            config=config,
        )
    st.session_state.setdefault(_ACTIVE_TURN_STATE_SLOT, None)
    st.session_state.setdefault(_SECRET_CACHE_STATE_SLOT, {})
    st.session_state.setdefault(_WIDGET_OVERRIDES_STATE_SLOT, {})
    app_state: StreamlitWorkspaceState = st.session_state[_APP_STATE_SLOT]

    theme_mode = _theme_mode_for_session(app_state)
    if theme_mode != app_state.preferences.theme_mode:
        app_state.preferences.theme_mode = theme_mode
        _save_workspace_state(app_state)

    pending_prompt = _drain_active_turn_events(app_state)
    _render_theme(app_state.preferences)
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

    if app_state.preferences.settings_panel_open:
        main_col, settings_col = st.columns([4.45, 1.75])
    else:
        main_col, settings_col = st.columns([5.85, 0.35])

    with main_col:
        if not active_record.transcript:
            _render_empty_state(active_record)
        else:
            for entry in active_record.transcript:
                _render_transcript_entry(entry)
        _render_status_and_composer(
            app_state,
            session_id=app_state.active_session_id,
            config=config,
        )

    with settings_col:
        _render_settings_panel(
            app_state,
            session_id=app_state.active_session_id,
            config=config,
        )

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
    """Launch the Streamlit chat app through the Streamlit CLI."""
    script_args = list(argv) if argv is not None else list(sys.argv[1:])
    return _launch_streamlit_app(script_args)


def _run_streamlit_script(
    argv: Sequence[str] | None = None,
) -> None:  # pragma: no cover
    args = build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    try:
        run_streamlit_chat_app(
            root_path=_resolve_root_argument(args),
            config=_resolve_chat_config(args),
        )
    except Exception as exc:
        _render_fatal_error(exc)


if __name__ == "__main__":
    _run_streamlit_script()

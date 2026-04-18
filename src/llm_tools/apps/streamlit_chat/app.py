"""Streamlit app shell for interactive repository chat."""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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
    build_chat_control_state,
    build_startup_message,
    build_tool_state_payload,
    handle_chat_command,
    resolve_default_enabled_tools,
)
from llm_tools.apps.chat_presentation import (
    format_citation,
    format_final_response,
    format_transcript_text,
    pretty_json,
)
from llm_tools.apps.chat_prompts import build_chat_system_prompt
from llm_tools.apps.chat_runtime import (
    build_available_tool_names,
    build_available_tool_specs,
    build_chat_context,
    build_chat_executor,
    create_provider,
)
from llm_tools.tool_api import SideEffectClass, ToolPolicy
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

_TRANSCRIPT_STATE_SLOT = "llm_tools_streamlit_chat_transcript"  # noqa: S105
_SESSION_STATE_SLOT = "llm_tools_streamlit_chat_session_state"  # noqa: S105
_TOKEN_USAGE_STATE_SLOT = "llm_tools_streamlit_chat_token_usage"  # noqa: S105
_API_KEY_STATE_SLOT = "llm_tools_streamlit_chat_api_key"  # noqa: S105
_API_SECRET_STATE_SLOT = "llm_tools_streamlit_chat_api_secret"  # noqa: S105
_CONTROL_STATE_SLOT = "llm_tools_streamlit_chat_control_state"  # noqa: S105
_TURN_STATE_SLOT = "llm_tools_streamlit_chat_turn_state"  # noqa: S105
_INSPECTOR_STATE_SLOT = "llm_tools_streamlit_chat_inspector_state"  # noqa: S105
_ACTIVE_TURN_STATE_SLOT = "llm_tools_streamlit_chat_active_turn"  # noqa: S105
_TRANSCRIPT_EXPORT_STATE_SLOT = "llm_tools_streamlit_chat_transcript_export"  # noqa: S105
_THEME_MODE_STATE_SLOT = "llm_tools_streamlit_chat_theme_mode"  # noqa: S105

_POLL_INTERVAL_SECONDS = 0.05
_DEFAULT_THEME_MODE: Literal["dark", "light"] = "dark"
_STREAMLIT_BROWSER_USAGE_STATS_FLAG = "--browser.gatherUsageStats=false"
_STREAMLIT_TOOLBAR_MODE_FLAG = "--client.toolbarMode=minimal"


@dataclass(slots=True)
class StreamlitTranscriptEntry:
    """One rendered transcript entry persisted in Streamlit session state."""

    role: Literal["user", "assistant", "system", "error"]
    text: str
    final_response: ChatFinalResponse | None = None
    assistant_completion_state: Literal["complete", "interrupted"] = "complete"

    @property
    def transcript_text(self) -> str:
        if self.final_response is not None:
            return format_transcript_text(
                "assistant", format_final_response(self.final_response)
            )
        return format_transcript_text(
            self.role,
            self.text,
            assistant_completion_state=self.assistant_completion_state,
        )


@dataclass(slots=True)
class StreamlitInspectorEntry:
    """One persisted inspector log entry."""

    label: str
    payload: object


@dataclass(slots=True)
class StreamlitInspectorState:
    """Inspector/debug state for the Streamlit chat session."""

    provider_messages: list[StreamlitInspectorEntry] = field(default_factory=list)
    parsed_responses: list[StreamlitInspectorEntry] = field(default_factory=list)
    tool_executions: list[StreamlitInspectorEntry] = field(default_factory=list)


@dataclass(slots=True)
class StreamlitTurnState:
    """Mutable UI state for the active or last-completed turn."""

    busy: bool = False
    status_text: str = ""
    pending_approval: ChatWorkflowApprovalState | None = None
    approval_decision_in_flight: bool = False
    active_turn_number: int = 0
    pending_interrupt_draft: str | None = None
    confidence: float | None = None


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


@dataclass(slots=True)
class StreamlitActiveTurnHandle:
    """Background runner handle stored in Streamlit session state."""

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


@dataclass(frozen=True, slots=True)
class StreamlitThemeTokens:
    """Semantic colors and surfaces for Streamlit chat chrome."""

    mode: Literal["dark", "light"]
    page_background: str
    sidebar_background: str
    surface_background: str
    surface_elevated_background: str
    widget_background: str
    widget_background_hover: str
    widget_border: str
    widget_border_focus: str
    text_color: str
    muted_text_color: str
    accent_color: str
    accent_color_hover: str
    icon_color: str
    icon_muted_color: str
    composer_shell_background: str
    composer_input_background: str
    shadow_color: str


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by the bootstrap and script entrypoints."""
    parser = argparse.ArgumentParser(
        prog="llm-tools-streamlit-chat",
        description="Streamlit directory-scoped chat over repository files.",
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", required=True, type=Path)
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
    base_config = load_textual_chat_config(args.config)
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


def resolve_enabled_tool_names(config: TextualChatConfig) -> set[str]:
    """Return the session-visible fixed read-only tool set."""
    return resolve_default_enabled_tools(
        config,
        available_tool_names=build_available_tool_names(),
    )


def _streamlit_module() -> Any:
    import streamlit as streamlit

    return streamlit


def _streamlit_page_config() -> dict[str, object]:
    return {
        "page_title": "llm-tools Streamlit Chat",
        "layout": "wide",
        "menu_items": {
            "Get help": None,
            "Report a bug": None,
            "About": None,
        },
    }


def _streamlit_theme_mode() -> Literal["dark", "light"]:
    st = _streamlit_module()
    if st.session_state.get(_THEME_MODE_STATE_SLOT) == "light":
        return "light"
    return "dark"


def _streamlit_theme_tokens(
    theme_mode: Literal["dark", "light"],
) -> StreamlitThemeTokens:
    if theme_mode == "light":
        return StreamlitThemeTokens(
            mode="light",
            page_background="#f3f6fb",
            sidebar_background="#e8edf5",
            surface_background="#ffffff",
            surface_elevated_background="#ffffff",
            widget_background="#ffffff",
            widget_background_hover="#eef4ff",
            widget_border="#cbd5e1",
            widget_border_focus="#1d4ed8",
            text_color="#0f172a",
            muted_text_color="#475569",
            accent_color="#1d4ed8",
            accent_color_hover="#1e40af",
            icon_color="#0f172a",
            icon_muted_color="#64748b",
            composer_shell_background="rgba(255, 255, 255, 0.96)",
            composer_input_background="#ffffff",
            shadow_color="rgba(15, 23, 42, 0.16)",
        )
    return StreamlitThemeTokens(
        mode="dark",
        page_background="#0f172a",
        sidebar_background="#020617",
        surface_background="#111827",
        surface_elevated_background="#162033",
        widget_background="#111827",
        widget_background_hover="#172033",
        widget_border="#334155",
        widget_border_focus="#60a5fa",
        text_color="#e2e8f0",
        muted_text_color="#94a3b8",
        accent_color="#60a5fa",
        accent_color_hover="#93c5fd",
        icon_color="#e2e8f0",
        icon_muted_color="#94a3b8",
        composer_shell_background="rgba(8, 15, 30, 0.94)",
        composer_input_background="#0b1220",
        shadow_color="rgba(2, 6, 23, 0.52)",
    )


def _streamlit_theme_css_variables(tokens: StreamlitThemeTokens) -> str:
    return "\n".join(
        (
            f"    --llm-tools-page-background: {tokens.page_background};",
            f"    --llm-tools-sidebar-background: {tokens.sidebar_background};",
            f"    --llm-tools-surface-background: {tokens.surface_background};",
            (
                "    --llm-tools-surface-elevated-background: "
                f"{tokens.surface_elevated_background};"
            ),
            f"    --llm-tools-widget-background: {tokens.widget_background};",
            (
                "    --llm-tools-widget-background-hover: "
                f"{tokens.widget_background_hover};"
            ),
            f"    --llm-tools-widget-border: {tokens.widget_border};",
            (f"    --llm-tools-widget-border-focus: {tokens.widget_border_focus};"),
            f"    --llm-tools-text-color: {tokens.text_color};",
            f"    --llm-tools-muted-text-color: {tokens.muted_text_color};",
            f"    --llm-tools-accent-color: {tokens.accent_color};",
            f"    --llm-tools-accent-color-hover: {tokens.accent_color_hover};",
            f"    --llm-tools-icon-color: {tokens.icon_color};",
            f"    --llm-tools-icon-muted-color: {tokens.icon_muted_color};",
            (
                "    --llm-tools-composer-shell-background: "
                f"{tokens.composer_shell_background};"
            ),
            (
                "    --llm-tools-composer-input-background: "
                f"{tokens.composer_input_background};"
            ),
            f"    --llm-tools-shadow-color: {tokens.shadow_color};",
        )
    )


def _streamlit_theme_css(theme_mode: Literal["dark", "light"]) -> str:
    tokens = _streamlit_theme_tokens(theme_mode)
    css_variables = _streamlit_theme_css_variables(tokens)
    return f"""
<style>
:root {{
    color-scheme: {tokens.mode};
{css_variables}
}}
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {{
    background: var(--llm-tools-page-background);
    color: var(--llm-tools-text-color);
}}
[data-testid="stHeader"] {{
    border-bottom: 1px solid transparent;
}}
section[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {{
    background: var(--llm-tools-sidebar-background);
    color: var(--llm-tools-text-color);
}}
.stApp p,
.stApp label,
.stApp .stMarkdown,
.stApp .stCaption {{
    color: var(--llm-tools-text-color);
}}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] p {{
    color: var(--llm-tools-muted-text-color);
}}
.stApp a {{
    color: var(--llm-tools-accent-color);
}}
.stApp a:hover {{
    color: var(--llm-tools-accent-color-hover);
}}
[data-testid="stChatMessage"] {{
    background: var(--llm-tools-surface-background);
    border: 1px solid var(--llm-tools-widget-border);
    border-radius: 0.75rem;
    padding: 0.75rem 1rem;
}}
.stApp pre,
.stApp code,
.stApp [data-testid="stCodeBlock"],
.stApp [data-testid="stCode"] {{
    background: var(--llm-tools-surface-elevated-background);
    color: var(--llm-tools-text-color);
    border-color: var(--llm-tools-widget-border);
}}
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > div,
div[data-baseweb="textarea"] > div,
.stApp textarea,
.stApp input,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="input"] [data-testid="stWidgetLabel"],
div[data-baseweb="base-input"] [data-testid="stWidgetLabel"] {{
    background: var(--llm-tools-widget-background);
    color: var(--llm-tools-text-color);
    border: 1px solid var(--llm-tools-widget-border);
    -webkit-text-fill-color: var(--llm-tools-text-color);
    caret-color: var(--llm-tools-text-color);
}}
.stApp input::placeholder,
.stApp textarea::placeholder,
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="base-input"] input::placeholder {{
    color: var(--llm-tools-muted-text-color);
    opacity: 1;
}}
div[data-baseweb="input"] > div:hover,
div[data-baseweb="base-input"] > div:hover,
div[data-baseweb="textarea"] > div:hover {{
    background: var(--llm-tools-widget-background-hover);
}}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="base-input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {{
    border-color: var(--llm-tools-widget-border-focus);
    box-shadow: 0 0 0 1px var(--llm-tools-widget-border-focus);
}}
.stApp button[kind],
.stApp button[data-testid^="stBaseButton"] {{
    background: var(--llm-tools-widget-background);
    color: var(--llm-tools-text-color);
    border: 1px solid var(--llm-tools-widget-border);
}}
.stApp button[kind]:hover,
.stApp button[data-testid^="stBaseButton"]:hover {{
    background: var(--llm-tools-widget-background-hover);
    color: var(--llm-tools-accent-color);
    border-color: var(--llm-tools-widget-border-focus);
}}
.stApp button[kind]:focus-visible,
.stApp button[data-testid^="stBaseButton"]:focus-visible {{
    box-shadow: 0 0 0 1px var(--llm-tools-widget-border-focus);
    outline: none;
}}
[data-testid="stBottomBlockContainer"] {{
    background: var(--llm-tools-page-background);
}}
[data-testid="stChatInput"] {{
    background: var(--llm-tools-composer-shell-background);
    border: 1px solid var(--llm-tools-widget-border);
    border-radius: 1rem;
    box-shadow: 0 18px 40px var(--llm-tools-shadow-color);
    padding: 0.35rem;
    backdrop-filter: blur(10px);
}}
[data-testid="stChatInput"] > div {{
    background: transparent;
}}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input,
[data-testid="stChatInput"] [data-baseweb="textarea"] > div,
[data-testid="stChatInput"] [data-baseweb="input"] > div {{
    background: var(--llm-tools-composer-input-background);
    color: var(--llm-tools-text-color);
    border-color: transparent;
    border-width: 0;
    box-shadow: none;
    -webkit-text-fill-color: var(--llm-tools-text-color);
    caret-color: var(--llm-tools-text-color);
}}
[data-testid="stChatInput"] [data-baseweb="textarea"] > div {{
    border-radius: 0.85rem;
}}
[data-testid="stChatInput"] [data-baseweb="textarea"] > div:focus-within {{
    border-color: transparent;
    box-shadow: none;
}}
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {{
    color: var(--llm-tools-muted-text-color);
    opacity: 1;
}}
[data-testid="stChatInput"] button {{
    background: var(--llm-tools-widget-background);
    color: var(--llm-tools-accent-color);
    border: 1px solid var(--llm-tools-widget-border);
}}
[data-testid="stChatInput"] button:hover {{
    background: var(--llm-tools-widget-background-hover);
    color: var(--llm-tools-accent-color-hover);
    border-color: var(--llm-tools-widget-border-focus);
}}
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {{
    background: transparent;
}}
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button,
button[data-testid="stBaseButton-headerNoPadding"] {{
    background: var(--llm-tools-surface-elevated-background);
    color: var(--llm-tools-icon-color);
    border: 1px solid var(--llm-tools-widget-border);
    border-radius: 0.85rem;
    box-shadow: 0 10px 24px var(--llm-tools-shadow-color);
}}
[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover,
button[data-testid="stBaseButton-headerNoPadding"]:hover {{
    background: var(--llm-tools-widget-background-hover);
    color: var(--llm-tools-accent-color);
    border-color: var(--llm-tools-widget-border-focus);
}}
[data-testid="collapsedControl"] button svg,
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="stSidebarCollapseButton"] button svg,
button[data-testid="stBaseButton-headerNoPadding"] svg {{
    color: currentColor;
    fill: currentColor;
    stroke: currentColor;
}}
div[data-baseweb="switch"] > div {{
    background: var(--llm-tools-icon-muted-color);
    transition: background-color 180ms ease;
}}
div[data-baseweb="switch"] input:checked + div {{
    background: var(--llm-tools-accent-color);
}}
div[data-baseweb="switch"] div[aria-hidden="true"] {{
    background: var(--llm-tools-surface-background);
}}
</style>
"""


def _render_streamlit_theme() -> None:
    st = _streamlit_module()
    st.markdown(
        _streamlit_theme_css(_streamlit_theme_mode()),
        unsafe_allow_html=True,
    )


def _exit_notice() -> str:
    return "Streamlit chat keeps running. Use Clear chat or close the browser tab to leave."


def _missing_api_key_text(config: TextualChatConfig) -> str:
    env_var = config.llm.api_key_env_var or "OPENAI_API_KEY"
    return f"Set {env_var} or enter it in the sidebar to start chatting."


def _current_api_key(config: TextualChatConfig) -> str | None:
    st = _streamlit_module()
    metadata = config.llm.credential_prompt_metadata()
    if not metadata.expects_api_key:
        return None
    env_var = config.llm.api_key_env_var or "OPENAI_API_KEY"
    env_value = os.getenv(env_var)
    if env_value:
        return env_value
    cached = str(st.session_state.get(_API_SECRET_STATE_SLOT, "")).strip()
    return cached or None


def _render_chat_notice(
    transcript: list[StreamlitTranscriptEntry],
    notice: ChatControlNotice,
) -> None:
    transcript.append(StreamlitTranscriptEntry(role=notice.role, text=notice.text))


def _build_startup_entry(
    root_path: Path,
    control_state: ChatControlState,
) -> StreamlitTranscriptEntry:
    return StreamlitTranscriptEntry(
        role="system",
        text=build_startup_message(
            root_path=root_path,
            model_name=control_state.active_model_name,
            exit_hint="Type quit or exit for close instructions.",
        ),
    )


def _ensure_session_state(root_path: Path, config: TextualChatConfig) -> None:
    st = _streamlit_module()
    if _CONTROL_STATE_SLOT not in st.session_state:
        st.session_state[_CONTROL_STATE_SLOT] = build_chat_control_state(
            config,
            available_tool_names=set(build_available_tool_specs()),
        )
    control_state = st.session_state[_CONTROL_STATE_SLOT]
    if _TRANSCRIPT_STATE_SLOT not in st.session_state:
        st.session_state[_TRANSCRIPT_STATE_SLOT] = [
            _build_startup_entry(root_path, control_state)
        ]
    if _SESSION_STATE_SLOT not in st.session_state:
        st.session_state[_SESSION_STATE_SLOT] = ChatSessionState()
    if _TOKEN_USAGE_STATE_SLOT not in st.session_state:
        st.session_state[_TOKEN_USAGE_STATE_SLOT] = None
    if _TURN_STATE_SLOT not in st.session_state:
        st.session_state[_TURN_STATE_SLOT] = StreamlitTurnState()
    if _INSPECTOR_STATE_SLOT not in st.session_state:
        st.session_state[_INSPECTOR_STATE_SLOT] = StreamlitInspectorState()
    st.session_state.setdefault(_ACTIVE_TURN_STATE_SLOT, None)
    st.session_state.setdefault(_TRANSCRIPT_EXPORT_STATE_SLOT, False)
    st.session_state.setdefault(_API_KEY_STATE_SLOT, "")
    st.session_state.setdefault(_API_SECRET_STATE_SLOT, "")
    st.session_state.setdefault(_THEME_MODE_STATE_SLOT, _DEFAULT_THEME_MODE)


def _reset_session_state(root_path: Path, config: TextualChatConfig) -> None:
    st = _streamlit_module()
    active_turn = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
    if isinstance(active_turn, StreamlitActiveTurnHandle):
        active_turn.runner.cancel()
    control_state = build_chat_control_state(
        config,
        available_tool_names=set(build_available_tool_specs()),
    )
    st.session_state[_CONTROL_STATE_SLOT] = control_state
    st.session_state[_TRANSCRIPT_STATE_SLOT] = [
        _build_startup_entry(root_path, control_state)
    ]
    st.session_state[_SESSION_STATE_SLOT] = ChatSessionState()
    st.session_state[_TOKEN_USAGE_STATE_SLOT] = None
    st.session_state[_TURN_STATE_SLOT] = StreamlitTurnState()
    st.session_state[_INSPECTOR_STATE_SLOT] = StreamlitInspectorState()
    st.session_state[_ACTIVE_TURN_STATE_SLOT] = None
    st.session_state[_TRANSCRIPT_EXPORT_STATE_SLOT] = False
    st.session_state[_API_KEY_STATE_SLOT] = ""
    st.session_state[_API_SECRET_STATE_SLOT] = ""


def _resolve_api_key(config: TextualChatConfig) -> str | None:
    st = _streamlit_module()
    metadata = config.llm.credential_prompt_metadata()
    if not metadata.expects_api_key:
        return None
    env_var = config.llm.api_key_env_var or "OPENAI_API_KEY"
    env_value = os.getenv(env_var)
    if env_value:
        return env_value
    cached_secret = str(st.session_state[_API_SECRET_STATE_SLOT]).strip()
    if cached_secret:
        return cached_secret
    st.sidebar.subheader("Credentials")
    api_key = st.sidebar.text_input(
        f"Enter {env_var}",
        value=st.session_state[_API_KEY_STATE_SLOT],
        type="password",
    )
    api_key_text = str(api_key)
    st.session_state[_API_KEY_STATE_SLOT] = api_key_text
    cleaned_api_key = api_key_text.strip()
    if cleaned_api_key:
        st.session_state[_API_SECRET_STATE_SLOT] = cleaned_api_key
        st.session_state[_API_KEY_STATE_SLOT] = ""
        return cleaned_api_key
    st.info(_missing_api_key_text(config))
    return None


def _build_chat_runner(
    *,
    root_path: Path,
    config: TextualChatConfig,
    provider: ModelTurnProvider,
    session_state: ChatSessionState,
    control_state: ChatControlState,
    user_message: str,
) -> ChatSessionTurnRunner:
    policy = ToolPolicy(
        allowed_tools=set(control_state.enabled_tools),
        allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
        require_approval_for=set(control_state.require_approval_for),
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
        redaction=config.policy.redaction.model_copy(deep=True),
    )
    registry, executor = build_chat_executor(policy=policy)
    return run_interactive_chat_session_turn(
        user_message=user_message,
        session_state=session_state,
        executor=executor,
        provider=provider,
        system_prompt=build_chat_system_prompt(
            tool_registry=registry,
            tool_limits=config.tool_limits,
            enabled_tool_names=control_state.enabled_tools,
        ),
        base_context=build_chat_context(
            root_path=root_path,
            config=config,
            app_name="streamlit-chat",
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
) -> StreamlitQueuedEvent:
    if isinstance(event, ChatWorkflowStatusEvent):
        return StreamlitQueuedEvent(
            kind="status",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
        )
    if isinstance(event, ChatWorkflowApprovalEvent):
        return StreamlitQueuedEvent(
            kind="approval_requested",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
        )
    if isinstance(event, ChatWorkflowApprovalResolvedEvent):
        return StreamlitQueuedEvent(
            kind="approval_resolved",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
        )
    if isinstance(event, ChatWorkflowInspectorEvent):
        return StreamlitQueuedEvent(
            kind="inspector",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
        )
    if isinstance(event, ChatWorkflowResultEvent):
        return StreamlitQueuedEvent(
            kind="result",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
        )
    raise TypeError(f"Unsupported workflow event type: {type(event)!r}")


def _append_inspector_entry(
    entries: list[StreamlitInspectorEntry],
    *,
    label: str,
    payload: object,
) -> None:
    entries.append(StreamlitInspectorEntry(label=label, payload=payload))


def _apply_turn_error(error_message: str) -> None:
    st = _streamlit_module()
    transcript = st.session_state[_TRANSCRIPT_STATE_SLOT]
    turn_state = st.session_state[_TURN_STATE_SLOT]
    transcript.append(StreamlitTranscriptEntry(role="error", text=error_message))
    turn_state.busy = False
    turn_state.status_text = ""
    turn_state.pending_approval = None
    turn_state.approval_decision_in_flight = False
    turn_state.pending_interrupt_draft = None
    st.session_state[_ACTIVE_TURN_STATE_SLOT] = None


def _apply_turn_result(
    event: ChatWorkflowResultEvent,
) -> str | None:
    st = _streamlit_module()
    result = ChatWorkflowTurnResult.model_validate(event.result)
    transcript = st.session_state[_TRANSCRIPT_STATE_SLOT]
    turn_state: StreamlitTurnState = st.session_state[_TURN_STATE_SLOT]
    st.session_state[_SESSION_STATE_SLOT] = (
        result.session_state or st.session_state[_SESSION_STATE_SLOT]
    )
    st.session_state[_TOKEN_USAGE_STATE_SLOT] = result.token_usage
    turn_state.pending_approval = None
    turn_state.approval_decision_in_flight = False
    if result.context_warning:
        transcript.append(
            StreamlitTranscriptEntry(role="system", text=result.context_warning)
        )
    if result.status == "needs_continuation" and result.continuation_reason:
        transcript.append(
            StreamlitTranscriptEntry(role="system", text=result.continuation_reason)
        )
    if result.final_response is not None:
        transcript.append(
            StreamlitTranscriptEntry(
                role="assistant",
                text=result.final_response.answer,
                final_response=result.final_response,
            )
        )
        turn_state.confidence = result.final_response.confidence
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
            transcript.append(
                StreamlitTranscriptEntry(
                    role="assistant",
                    text=interrupted_message.content,
                    assistant_completion_state="interrupted",
                )
            )
        elif result.interruption_reason:
            transcript.append(
                StreamlitTranscriptEntry(role="system", text=result.interruption_reason)
            )
        turn_state.confidence = None
    else:
        turn_state.confidence = None
    turn_state.status_text = ""
    turn_state.busy = False
    pending_prompt = turn_state.pending_interrupt_draft
    turn_state.pending_interrupt_draft = None
    st.session_state[_ACTIVE_TURN_STATE_SLOT] = None
    return pending_prompt


def _apply_queued_event(queued_event: StreamlitQueuedEvent) -> str | None:
    st = _streamlit_module()
    transcript = st.session_state[_TRANSCRIPT_STATE_SLOT]
    turn_state = st.session_state[_TURN_STATE_SLOT]
    inspector_state = st.session_state[_INSPECTOR_STATE_SLOT]
    if queued_event.kind == "status":
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
        transcript.append(
            StreamlitTranscriptEntry(
                role="system",
                text=(
                    f"Approval requested for {approval_event.approval.tool_name}: "
                    f"{approval_event.approval.policy_reason}"
                ),
            )
        )
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
        transcript.append(StreamlitTranscriptEntry(role="system", text=resolution_text))
        if approval_resolved_event.resolution == "approved":
            turn_state.status_text = "resuming turn"
        elif approval_resolved_event.resolution == "denied":
            turn_state.status_text = "continuing without approval"
        elif approval_resolved_event.resolution == "timed_out":
            turn_state.status_text = "approval timed out"
        else:
            turn_state.status_text = ""
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
        return None
    if queued_event.kind == "result":
        result_event = ChatWorkflowResultEvent.model_validate(queued_event.payload)
        return _apply_turn_result(result_event)
    if queued_event.kind == "error":
        _apply_turn_error(str(queued_event.payload))
        return None
    if queued_event.kind == "complete":
        handle = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
        if (
            isinstance(handle, StreamlitActiveTurnHandle)
            and handle.turn_number == queued_event.turn_number
            and not st.session_state[_TURN_STATE_SLOT].busy
        ):
            st.session_state[_ACTIVE_TURN_STATE_SLOT] = None
        return None
    raise ValueError(f"Unsupported queued event kind: {queued_event.kind}")


def _drain_active_turn_events() -> str | None:
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
        next_prompt = _apply_queued_event(queued_event)
        if next_prompt is not None:
            pending_prompt = next_prompt
    if (
        not handle.thread.is_alive()
        and handle.event_queue.empty()
        and not st.session_state[_TURN_STATE_SLOT].busy
    ):
        st.session_state[_ACTIVE_TURN_STATE_SLOT] = None
    return pending_prompt


def _worker_run_turn(handle: StreamlitActiveTurnHandle) -> None:
    try:
        for event in handle.runner:
            handle.event_queue.put(
                _serialize_workflow_event(event, turn_number=handle.turn_number)
            )
    except Exception as exc:  # pragma: no cover - exercised through queue result
        handle.event_queue.put(
            StreamlitQueuedEvent(
                kind="error",
                payload=str(exc),
                turn_number=handle.turn_number,
            )
        )
    finally:
        handle.event_queue.put(
            StreamlitQueuedEvent(
                kind="complete",
                payload=None,
                turn_number=handle.turn_number,
            )
        )


def _start_streamlit_turn(
    *,
    root_path: Path,
    config: TextualChatConfig,
    provider: ModelTurnProvider,
    user_message: str,
) -> None:
    st = _streamlit_module()
    control_state: ChatControlState = st.session_state[_CONTROL_STATE_SLOT]
    turn_state: StreamlitTurnState = st.session_state[_TURN_STATE_SLOT]
    turn_number = turn_state.active_turn_number + 1
    runner = _build_chat_runner(
        root_path=root_path,
        config=config,
        provider=provider,
        session_state=st.session_state[_SESSION_STATE_SLOT],
        control_state=control_state,
        user_message=user_message,
    )
    event_queue: queue.Queue[StreamlitQueuedEvent] = queue.Queue()
    handle = StreamlitActiveTurnHandle(
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
    st.session_state[_TRANSCRIPT_STATE_SLOT].append(
        StreamlitTranscriptEntry(role="user", text=user_message)
    )
    st.session_state[_ACTIVE_TURN_STATE_SLOT] = handle
    thread.start()


def _resolve_active_approval(*, approved: bool) -> None:
    st = _streamlit_module()
    handle = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
    turn_state = st.session_state[_TURN_STATE_SLOT]
    if not isinstance(handle, StreamlitActiveTurnHandle):
        return
    if turn_state.pending_approval is None:
        return
    if not handle.runner.resolve_pending_approval(approved):
        return
    turn_state.approval_decision_in_flight = True
    turn_state.status_text = "approving" if approved else "denying"


def _cancel_active_turn(*, preserve_pending_prompt: bool = False) -> None:
    st = _streamlit_module()
    handle = st.session_state.get(_ACTIVE_TURN_STATE_SLOT)
    turn_state = st.session_state[_TURN_STATE_SLOT]
    if not isinstance(handle, StreamlitActiveTurnHandle):
        return
    if not preserve_pending_prompt:
        turn_state.pending_interrupt_draft = None
    turn_state.status_text = "stopping"
    handle.runner.cancel()


def _submit_streamlit_prompt(
    *,
    root_path: Path,
    config: TextualChatConfig,
    prompt: str,
) -> None:
    st = _streamlit_module()
    turn_state: StreamlitTurnState = st.session_state[_TURN_STATE_SLOT]
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        return
    if turn_state.busy:
        turn_state.pending_interrupt_draft = cleaned_prompt
        _cancel_active_turn(preserve_pending_prompt=True)
        return
    api_key = _current_api_key(config)
    provider = create_provider(
        config.llm,
        api_key=api_key,
        model_name=st.session_state[_CONTROL_STATE_SLOT].active_model_name,
    )
    _start_streamlit_turn(
        root_path=root_path,
        config=config,
        provider=provider,
        user_message=cleaned_prompt,
    )


def _run_streamlit_command(
    *,
    config: TextualChatConfig,
    raw_command: str,
) -> ChatCommandOutcome[ModelTurnProvider]:
    st = _streamlit_module()
    control_state: ChatControlState = st.session_state[_CONTROL_STATE_SLOT]

    def _list_models() -> ModelCatalogOutcome:
        api_key = _current_api_key(config)
        metadata = config.llm.credential_prompt_metadata()
        if metadata.expects_api_key and api_key is None:
            return ModelCatalogOutcome(
                notice=ChatControlNotice(
                    role="system", text=_missing_api_key_text(config)
                )
            )
        try:
            provider = create_provider(
                config.llm,
                api_key=api_key,
                model_name=control_state.active_model_name,
            )
        except Exception as exc:
            return ModelCatalogOutcome(
                notice=ChatControlNotice(
                    role="error",
                    text=f"Unable to create provider for {control_state.active_model_name}: {exc}",
                )
            )
        try:
            return ModelCatalogOutcome(model_ids=provider.list_available_models())
        except Exception as exc:
            return ModelCatalogOutcome(
                notice=ChatControlNotice(
                    role="system",
                    text=(
                        f"Current model: {control_state.active_model_name}\n"
                        f"Unable to list available models: {exc}"
                    ),
                )
            )

    def _switch_model(new_model_name: str) -> ModelSwitchOutcome[ModelTurnProvider]:
        api_key = _current_api_key(config)
        metadata = config.llm.credential_prompt_metadata()
        if metadata.expects_api_key and api_key is None:
            return ModelSwitchOutcome(
                notice=ChatControlNotice(
                    role="system", text=_missing_api_key_text(config)
                )
            )
        try:
            provider = create_provider(
                config.llm,
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

    return handle_chat_command(
        raw_command,
        state=control_state,
        available_tool_specs=build_available_tool_specs(),
        busy=st.session_state[_TURN_STATE_SLOT].busy,
        list_models=_list_models,
        switch_model=_switch_model,
        exit_mode="notice",
        exit_notice=_exit_notice(),
    )


def _apply_streamlit_command(outcome: ChatCommandOutcome[ModelTurnProvider]) -> None:
    st = _streamlit_module()
    transcript = st.session_state[_TRANSCRIPT_STATE_SLOT]
    for notice in outcome.notices:
        _render_chat_notice(transcript, notice)
    if outcome.request_copy:
        st.session_state[_TRANSCRIPT_EXPORT_STATE_SLOT] = True


def _is_command_prompt(prompt: str) -> bool:
    cleaned = prompt.strip().lower()
    return cleaned.startswith("/") or cleaned in {"quit", "exit"}


def process_streamlit_chat_turn(
    *,
    root_path: Path,
    config: TextualChatConfig,
    provider: ModelTurnProvider,
    session_state: ChatSessionState,
    user_message: str,
    approval_resolver: Callable[[ChatWorkflowApprovalState], bool] | None = None,
) -> StreamlitTurnOutcome:
    """Execute one repository-chat turn using the Streamlit event reducers."""
    control_state = build_chat_control_state(
        config,
        available_tool_names=set(build_available_tool_specs()),
    )
    turn_state = StreamlitTurnState(busy=True, active_turn_number=1)
    inspector_state = StreamlitInspectorState()
    transcript_entries: list[StreamlitTranscriptEntry] = []
    token_usage: ChatTokenUsage | None = None
    updated_session_state = session_state
    runner = _build_chat_runner(
        root_path=root_path,
        config=config,
        provider=provider,
        session_state=session_state,
        control_state=control_state,
        user_message=user_message,
    )
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
                target,
                label=label,
                payload=inspector_event.payload,
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


def _render_final_response(response: ChatFinalResponse) -> None:
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


def _render_transcript_entry(entry: StreamlitTranscriptEntry) -> None:
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


def _transcript_export_text() -> str:
    st = _streamlit_module()
    entries: list[StreamlitTranscriptEntry] = st.session_state[_TRANSCRIPT_STATE_SLOT]
    parts = [entry.transcript_text for entry in entries if entry.transcript_text]
    return "\n\n".join(parts).rstrip()


def _render_sidebar_session(
    *,
    root_path: Path,
    config: TextualChatConfig,
    control_state: ChatControlState,
    turn_state: StreamlitTurnState,
) -> None:
    st = _streamlit_module()
    token_usage = st.session_state[_TOKEN_USAGE_STATE_SLOT]
    st.subheader("Session")
    st.markdown(f"**Root**: `{root_path}`")
    st.markdown(f"**Model**: `{control_state.active_model_name}`")
    st.caption(f"Status: {turn_state.status_text or 'idle'}")
    if config.ui.show_token_usage and isinstance(token_usage, ChatTokenUsage):
        st.caption(
            "Session tokens: "
            f"{token_usage.session_tokens or '-'} | "
            f"Active context: {token_usage.active_context_tokens or '-'}"
        )
    if turn_state.confidence is not None:
        st.caption(f"Confidence: {turn_state.confidence:.2f}")
    if st.button("Clear chat", use_container_width=True):
        _reset_session_state(root_path, config)
        st.rerun()
    if turn_state.busy and st.button("Stop active turn", use_container_width=True):
        _cancel_active_turn()
        st.rerun()
    if turn_state.busy and st.button("Refresh active turn", use_container_width=True):
        st.rerun()


def _render_sidebar_appearance_controls() -> None:
    st = _streamlit_module()
    st.subheader("Appearance")
    use_dark_mode = st.toggle(
        "Dark mode",
        value=_streamlit_theme_mode() == "dark",
        key="appearance:dark_mode",
    )
    st.session_state[_THEME_MODE_STATE_SLOT] = "dark" if use_dark_mode else "light"


def _render_sidebar_model_controls(
    *,
    config: TextualChatConfig,
    control_state: ChatControlState,
) -> None:
    st = _streamlit_module()
    st.subheader("Model Controls")
    switch_target = st.text_input(
        "Switch model",
        value=control_state.active_model_name,
    )
    if st.button("List available models", use_container_width=True):
        _apply_streamlit_command(
            _run_streamlit_command(config=config, raw_command="/model")
        )
        st.rerun()
    if st.button("Switch model", use_container_width=True):
        _apply_streamlit_command(
            _run_streamlit_command(
                config=config,
                raw_command=f"/model {switch_target}".rstrip(),
            )
        )
        st.rerun()


def _render_sidebar_tool_controls(control_state: ChatControlState) -> None:
    st = _streamlit_module()
    available_tool_specs = build_available_tool_specs()
    st.subheader("Tool Controls")
    for tool_name in sorted(available_tool_specs):
        enabled = tool_name in control_state.enabled_tools
        checked = st.checkbox(
            f"Enable {tool_name}",
            value=enabled,
            key=f"tool:{tool_name}",
        )
        if checked != enabled:
            if checked:
                control_state.enabled_tools.add(tool_name)
            else:
                control_state.enabled_tools.discard(tool_name)
    if st.button("Reset tools to defaults", use_container_width=True):
        control_state.enabled_tools = set(control_state.default_enabled_tools)
        st.rerun()


def _render_sidebar_approval_controls(
    *,
    control_state: ChatControlState,
    turn_state: StreamlitTurnState,
) -> None:
    st = _streamlit_module()
    st.subheader("Approval Controls")
    approval_enabled = SideEffectClass.LOCAL_READ in control_state.require_approval_for
    require_approval = st.checkbox(
        "Require approval for local_read",
        value=approval_enabled,
        key="approvals:local_read",
    )
    if require_approval != approval_enabled:
        if require_approval:
            control_state.require_approval_for.add(SideEffectClass.LOCAL_READ)
        else:
            control_state.require_approval_for.discard(SideEffectClass.LOCAL_READ)
    if turn_state.pending_approval is None:
        return
    st.code(pretty_json(turn_state.pending_approval.model_dump(mode="json")))
    if st.button(
        "Approve pending request",
        use_container_width=True,
        disabled=turn_state.approval_decision_in_flight,
    ):
        _resolve_active_approval(approved=True)
        st.rerun()
    if st.button(
        "Deny pending request",
        use_container_width=True,
        disabled=turn_state.approval_decision_in_flight,
    ):
        _resolve_active_approval(approved=False)
        st.rerun()


def _render_sidebar_transcript_export() -> None:
    st = _streamlit_module()
    transcript_export = _transcript_export_text()
    st.subheader("Transcript Export")
    show_export = st.checkbox(
        "Show transcript export",
        value=bool(st.session_state[_TRANSCRIPT_EXPORT_STATE_SLOT]),
        key="transcript_export",
    )
    st.session_state[_TRANSCRIPT_EXPORT_STATE_SLOT] = show_export
    if not show_export:
        return
    st.text_area("Transcript export", transcript_export, height=220)
    st.download_button(
        "Download transcript",
        data=transcript_export,
        file_name="llm-tools-chat-transcript.txt",
        use_container_width=True,
    )


def _render_sidebar_inspector(
    *,
    control_state: ChatControlState,
    turn_state: StreamlitTurnState,
) -> None:
    st = _streamlit_module()
    available_tool_specs = build_available_tool_specs()
    st.subheader("Inspector")
    show_inspector = st.checkbox(
        "Show inspector",
        value=control_state.inspector_open,
        key="show_inspector",
    )
    control_state.inspector_open = show_inspector
    if not show_inspector:
        return
    st.code(
        pretty_json(
            build_tool_state_payload(
                control_state,
                available_tool_specs=available_tool_specs,
            )
        )
    )
    if turn_state.pending_approval is not None:
        st.code(pretty_json(turn_state.pending_approval.model_dump(mode="json")))
    inspector_state = st.session_state[_INSPECTOR_STATE_SLOT]
    for label, entries in (
        ("Model Messages", inspector_state.provider_messages),
        ("Parsed Responses", inspector_state.parsed_responses),
        ("Tool Execution Records", inspector_state.tool_executions),
    ):
        st.caption(label)
        for entry in entries:
            st.code(f"{entry.label}\n{pretty_json(entry.payload)}")


def _render_sidebar(root_path: Path, config: TextualChatConfig) -> None:
    st = _streamlit_module()
    control_state: ChatControlState = st.session_state[_CONTROL_STATE_SLOT]
    turn_state: StreamlitTurnState = st.session_state[_TURN_STATE_SLOT]
    with st.sidebar:
        _render_sidebar_session(
            root_path=root_path,
            config=config,
            control_state=control_state,
            turn_state=turn_state,
        )
        _render_sidebar_appearance_controls()
        _render_sidebar_model_controls(
            config=config,
            control_state=control_state,
        )
        _render_sidebar_tool_controls(control_state)
        _render_sidebar_approval_controls(
            control_state=control_state,
            turn_state=turn_state,
        )
        _render_sidebar_transcript_export()
        _render_sidebar_inspector(
            control_state=control_state,
            turn_state=turn_state,
        )


def run_streamlit_chat_app(*, root_path: Path, config: TextualChatConfig) -> None:
    """Render the Streamlit repository chat UI."""
    st = _streamlit_module()
    st.set_page_config(**_streamlit_page_config())
    st.title("llm-tools Streamlit Chat")
    _ensure_session_state(root_path, config)

    pending_prompt = _drain_active_turn_events()
    _render_sidebar(root_path, config)
    _render_streamlit_theme()

    api_key = _resolve_api_key(config)
    if config.llm.credential_prompt_metadata().expects_api_key and api_key is None:
        for entry in st.session_state[_TRANSCRIPT_STATE_SLOT]:
            _render_transcript_entry(entry)
        return

    if pending_prompt is not None and pending_prompt.strip():
        _submit_streamlit_prompt(
            root_path=root_path,
            config=config,
            prompt=pending_prompt,
        )

    for entry in st.session_state[_TRANSCRIPT_STATE_SLOT]:
        _render_transcript_entry(entry)

    if config.ui.show_footer_help:
        st.caption(
            "Use /help for controls. Native session, model, tool, approval, transcript, and inspector controls are in the sidebar."
        )

    prompt = st.chat_input("Ask about this repository")
    if prompt and prompt.strip():
        if _is_command_prompt(prompt):
            outcome = _run_streamlit_command(config=config, raw_command=prompt.strip())
            _apply_streamlit_command(outcome)
        else:
            _submit_streamlit_prompt(
                root_path=root_path,
                config=config,
                prompt=prompt.strip(),
            )
        st.rerun()

    turn_state: StreamlitTurnState = st.session_state[_TURN_STATE_SLOT]
    if turn_state.busy and turn_state.pending_approval is None:
        time.sleep(_POLL_INTERVAL_SECONDS)
        st.rerun()


def _launch_streamlit_app(script_args: Sequence[str]) -> int:
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


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the Streamlit chat app through the Streamlit CLI."""
    script_args = list(argv) if argv is not None else list(sys.argv[1:])
    return _launch_streamlit_app(script_args)


def _run_streamlit_script(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    run_streamlit_chat_app(
        root_path=args.directory.resolve(),
        config=_resolve_chat_config(args),
    )


if __name__ == "__main__":
    _run_streamlit_script()

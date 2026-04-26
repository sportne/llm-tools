"""NiceGUI chat client for llm-tools."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from llm_tools.apps.assistant_config import (
    StreamlitAssistantConfig,
    load_streamlit_assistant_config,
)
from llm_tools.apps.assistant_tool_registry import build_assistant_available_tool_specs
from llm_tools.apps.chat_config import ProviderPreset
from llm_tools.apps.nicegui_chat.controller import (
    NiceGUIChatController,
    default_runtime_config,
)
from llm_tools.apps.nicegui_chat.models import NiceGUIRuntimeConfig
from llm_tools.apps.nicegui_chat.store import SQLiteNiceGUIChatStore, default_db_path
from llm_tools.llm_providers import ProviderModeStrategy


def _first_nonempty_text(*values: object) -> str:
    """Return the first non-empty text value after trimming outer whitespace."""
    for value in values:
        text = str(value or "")
        if text.strip():
            return text
    return ""


def _event_payload_text(event: object) -> str:
    """Extract text emitted by a NiceGUI JavaScript event handler."""
    args = getattr(event, "args", None)
    if isinstance(args, str):
        return args
    if isinstance(args, dict):
        value = args.get("value")
        if value is not None:
            return str(value)
        target = args.get("target")
        if isinstance(target, dict):
            return str(target.get("value") or "")
    return ""


def _sidebar_container_classes(*, collapsed: bool) -> str:
    """Return sidebar container classes for the current collapsed state."""
    classes = "llmt-sidebar"
    if collapsed:
        classes += " collapsed"
    return classes


def _workbench_container_classes(*, open: bool = True) -> str:
    """Return workbench container classes."""
    classes = "llmt-workbench"
    if not open:
        classes += " closed"
    return classes


def _runtime_summary_parts(runtime: NiceGUIRuntimeConfig) -> list[tuple[str, str]]:
    """Return compact clickable runtime metadata labels."""
    root = runtime.root_path or "No workspace"
    return [
        ("provider", runtime.provider.value),
        ("model", runtime.model_name),
        ("mode", runtime.provider_mode_strategy.value),
        ("workspace", root),
    ]


def _runtime_summary_text(runtime: NiceGUIRuntimeConfig) -> str:
    """Return compact runtime metadata for tests and text fallbacks."""
    return " | ".join(value for _label, value in _runtime_summary_parts(runtime))


def _composer_action_icon(*, busy: bool) -> str:
    """Return the icon for the primary composer action."""
    return "stop" if busy else "send"


def _models_endpoint_url(base_url: str) -> str:
    """Return the OpenAI-compatible model listing URL for a base URL."""
    trimmed = base_url.strip().rstrip("/")
    return f"{trimmed}/models" if trimmed else ""


def _extract_model_names_from_models_payload(payload: object) -> list[str]:
    """Extract model IDs from OpenAI-compatible and Ollama-style payloads."""
    if not isinstance(payload, dict):
        return []
    raw_entries = payload.get("data")
    if raw_entries is None:
        raw_entries = payload.get("models")
    if not isinstance(raw_entries, list):
        return []
    names: list[str] = []
    for entry in raw_entries:
        raw_name: object
        if isinstance(entry, dict):
            raw_name = entry.get("id") or entry.get("name") or ""
        else:
            raw_name = entry
        name = str(raw_name or "").strip()
        if name.startswith("models/"):
            name = name.removeprefix("models/")
        if name and name not in names:
            names.append(name)
    return sorted(names)


def _discover_model_names(
    *,
    provider: ProviderPreset,
    base_url: str | None,
    timeout: float = 5.0,
) -> list[str]:
    """Discover available models from an OpenAI-compatible provider endpoint."""
    if not base_url:
        return []
    url = _models_endpoint_url(base_url)
    if not url:
        return []
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        return []
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY")
    if provider is not ProviderPreset.OLLAMA and api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers, method="GET")  # noqa: S310
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []
    return _extract_model_names_from_models_payload(payload)


def build_parser() -> argparse.ArgumentParser:
    """Build the NiceGUI chat CLI parser."""
    parser = argparse.ArgumentParser(
        prog="llm-tools-nicegui-chat",
        description="NiceGUI chat client backed by SQLite persistence.",
    )
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--directory", dest="directory_override", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--provider")
    parser.add_argument("--model", type=str)
    parser.add_argument("--provider-mode-strategy", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--api-base-url", type=str)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
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


def resolve_assistant_config(args: argparse.Namespace) -> StreamlitAssistantConfig:
    """Resolve config file and CLI overrides."""
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


def resolve_root_argument(
    args: argparse.Namespace,
    config: StreamlitAssistantConfig,
) -> Path | None:
    """Resolve the workspace root from CLI or config."""
    candidate = args.directory_override or args.directory
    if candidate is None:
        default_root = config.workspace.default_root
        if default_root is None:
            return None
        return Path(default_root).expanduser().resolve()
    return Path(candidate).expanduser().resolve()


def run_nicegui_chat_app(
    *,
    root_path: Path | None,
    config: StreamlitAssistantConfig,
    db_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    show: bool = True,
) -> None:  # pragma: no cover
    """Render and run the NiceGUI chat app."""
    from nicegui import ui

    store = SQLiteNiceGUIChatStore(db_path or default_db_path())
    store.initialize()
    controller = NiceGUIChatController(
        store=store,
        config=config,
        root_path=root_path,
    )
    build_nicegui_chat_ui(controller)
    ui.run(
        host=host,
        port=port,
        reload=False,
        show=show,
        title="llm-tools chat",
    )


def build_nicegui_chat_ui(  # noqa: C901
    controller: NiceGUIChatController,
) -> None:  # pragma: no cover
    """Build the NiceGUI component tree for the chat app."""
    from nicegui import ui

    ui.add_head_html(
        """
        <style>
        *, *::before, *::after { box-sizing: border-box; }
        html, body, #app {
            height: 100%; width: 100%; margin: 0; overflow: hidden;
        }
        .nicegui-layout, .q-page-container, .q-page {
            height: 100% !important; min-height: 100% !important;
            overflow: hidden;
        }
        .nicegui-content {
            width: 100% !important; height: 100% !important;
            padding: 0 !important; overflow: hidden;
        }
        body {
            --llmt-accent: #3f4142;
            --llmt-accent-contrast: #ffffff;
            --q-primary: #3f4142;
            --q-negative: #3f4142;
            background: #f6f6f3; color: #20211f;
        }
        body.body--dark {
            --llmt-accent: #d2d2ce;
            --llmt-accent-contrast: #161715;
            --q-primary: #d2d2ce;
            --q-negative: #d2d2ce;
            background: #161715; color: #e8e6de;
        }
        body.desktop {
            --q-primary: #3f4142 !important;
            --q-negative: #3f4142 !important;
        }
        body.desktop.body--dark {
            --q-primary: #d2d2ce !important;
            --q-negative: #d2d2ce !important;
        }
        .llmt-shell {
            height: calc(100dvh - 8px); width: 100%; max-width: 100%;
            overflow: hidden; display: flex; flex-wrap: nowrap;
        }
        .llmt-shell .q-btn.bg-primary {
            color: var(--llmt-accent-contrast) !important;
        }
        .llmt-shell .text-primary,
        .q-dialog .text-primary,
        .q-menu .text-primary,
        .llmt-shell .text-negative,
        .q-dialog .text-negative,
        .q-menu .text-negative {
            color: var(--llmt-accent) !important;
        }
        .llmt-shell .bg-primary,
        .q-dialog .bg-primary,
        .q-menu .bg-primary,
        .llmt-shell .bg-negative,
        .q-dialog .bg-negative,
        .q-menu .bg-negative {
            background: var(--llmt-accent) !important;
            color: var(--llmt-accent-contrast) !important;
        }
        .llmt-sidebar {
            flex: 0 0 284px; width: 284px; min-width: 284px; max-width: 284px;
            height: 100%; overflow: hidden; background: #ecebe6;
            border-right: 1px solid #d8d6ce; color: var(--llmt-accent);
            display: flex; flex-direction: column;
        }
        body.body--dark .llmt-sidebar {
            background: #20221f; border-right-color: #373a34;
            color: var(--llmt-accent);
        }
        .llmt-sidebar.collapsed {
            flex-basis: 74px; width: 74px; min-width: 74px; max-width: 74px;
        }
        .llmt-main {
            flex: 1 1 auto; min-width: 0; height: 100%; overflow: hidden;
            background: #fbfbf8; display: flex; flex-direction: column;
        }
        body.body--dark .llmt-main { background: #171816; }
        .llmt-header {
            flex: 0 0 58px; min-height: 58px; max-height: 58px;
            border-bottom: 1px solid #dedcd4; overflow: hidden;
        }
        body.body--dark .llmt-header { border-bottom-color: #373a34; }
        .llmt-header-title { min-width: 0; overflow: hidden; }
        .llmt-header-title .nicegui-label {
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            max-width: 100%;
        }
        .llmt-header-actions {
            flex: 0 0 auto; flex-wrap: nowrap; min-width: max-content;
        }
        .llmt-sidebar-body {
            flex: 1 1 auto; min-height: 0; width: 100%; overflow: hidden;
        }
        .llmt-sidebar-footer {
            flex: 0 0 auto; border-top: 1px solid #d8d6ce;
            padding: 6px 8px 10px !important;
        }
        body.body--dark .llmt-sidebar-footer { border-top-color: #373a34; }
        .llmt-transcript {
            flex: 1 1 auto; min-height: 0; width: 100%; overflow: hidden;
        }
        .llmt-composer {
            flex: 0 0 auto; border-top: 1px solid #dedcd4;
            background: #fbfbf8; padding: 10px 16px 10px !important;
        }
        body.body--dark .llmt-composer {
            background: #171816; border-top-color: #373a34;
        }
        .llmt-composer-row { flex-wrap: nowrap; }
        .llmt-composer-input { flex: 1 1 auto; min-width: 0; }
        .llmt-composer-meta {
            flex-wrap: nowrap; min-width: 0; overflow: hidden;
            max-width: 100%; min-height: 22px;
        }
        .llmt-composer-meta .q-btn {
            min-height: 20px; max-width: 36%;
            padding: 0 2px; font-size: 12px;
        }
        .llmt-composer-meta .q-btn__content {
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            display: block; min-width: 0;
        }
        .llmt-composer-disclaimer {
            width: 100%; text-align: center; font-size: 12px; line-height: 16px;
            min-height: 16px;
        }
        .llmt-selected-tools {
            min-height: 0; max-height: 64px; overflow: hidden; flex-wrap: wrap;
        }
        .llmt-selected-tools .q-btn {
            min-height: 22px; padding: 0 6px; font-size: 12px;
        }
        .llmt-workbench {
            flex: 0 0 340px; width: 340px; min-width: 340px; max-width: 340px;
            height: 100%; min-height: 0; overflow: hidden; background: #f1f0eb;
            border-left: 1px solid #d8d6ce; display: flex; flex-direction: column;
        }
        .llmt-workbench.closed {
            flex-basis: 0; width: 0; min-width: 0; max-width: 0;
            border-left: 0; overflow: hidden;
        }
        body.body--dark .llmt-workbench {
            background: #1d1f1c; border-left-color: #373a34;
        }
        .llmt-workbench-body {
            flex: 1 1 auto; min-height: 0; width: 100%; overflow: hidden;
        }
        .llmt-json-editor {
            width: 100%; max-width: 100%; height: min(420px, 55vh);
            overflow: hidden;
        }
        .llmt-json-editor .jse-main { min-width: 0; }
        .llmt-json-editor .jse-contents { overflow: auto; }
        .llmt-action-button .q-btn__content { flex-wrap: nowrap; }
        .llmt-action-button .q-btn__content span {
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
        .llmt-message {
            border-radius: 8px; padding: 12px 14px; max-width: 820px;
            white-space: normal; line-height: 1.45;
        }
        .llmt-message .q-markdown, .llmt-message .nicegui-markdown {
            line-height: 1.45;
        }
        .llmt-message .q-markdown > :first-child,
        .llmt-message .nicegui-markdown > :first-child { margin-top: 0; }
        .llmt-message .q-markdown > :last-child,
        .llmt-message .nicegui-markdown > :last-child { margin-bottom: 0; }
        .llmt-message h1 {
            font-size: 1.6rem; line-height: 1.2; margin: 0.9rem 0 0.55rem;
        }
        .llmt-message h2 {
            font-size: 1.25rem; line-height: 1.25; margin: 0.8rem 0 0.45rem;
        }
        .llmt-message h3 {
            font-size: 1.05rem; line-height: 1.3; margin: 0.7rem 0 0.35rem;
        }
        .llmt-message p { margin: 0.55rem 0; }
        .llmt-message ul, .llmt-message ol {
            margin: 0.45rem 0; padding-left: 1.4rem;
        }
        .llmt-message li { margin: 0.15rem 0; }
        .llmt-message pre { white-space: pre-wrap; overflow-x: auto; }
        .llmt-user { background: #e8f2ef; margin-left: auto; }
        .llmt-assistant { background: #ffffff; border: 1px solid #e0ded7; }
        .llmt-system { background: #fff7df; border: 1px solid #ead797; }
        .llmt-error { background: #fff0ef; border: 1px solid #e9b7b1; }
        body.body--dark .llmt-user { background: #223832; }
        body.body--dark .llmt-assistant {
            background: #20221f; border-color: #3a3d37;
        }
        body.body--dark .llmt-system {
            background: #3b321b; border-color: #6b5b28;
        }
        body.body--dark .llmt-error {
            background: #3b2221; border-color: #6e3b36;
        }
        .llmt-session { border-radius: 8px; min-height: 42px; }
        .llmt-session.active { background: #dfe8e4; }
        body.body--dark .llmt-session.active { background: #293732; }
        .llmt-muted { color: #6e6d67; }
        body.body--dark .llmt-muted { color: #aaa79e; }
        .llmt-code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
        </style>
        """
    )

    dark_mode_control = ui.dark_mode(value=controller.preferences.theme_mode == "dark")
    session_filter = {"value": ""}
    model_options_state: dict[str, list[str]] = {"values": []}
    workspace_browser_path: dict[str, Path] = {
        "value": Path(controller.active_record.runtime.root_path or Path.cwd())
    }
    workspace_browser_target: dict[str, str] = {"value": "settings"}
    transcript_column: Any = None
    session_column: Any = None
    workbench_column: Any = None
    header_title: Any = None
    status_chip: Any = None
    selected_tools_row: Any = None
    composer_meta_row: Any = None
    composer_action_button: Any = None
    provider_select: Any = None
    model_input: Any = None
    base_url_input: Any = None
    mode_select: Any = None
    workspace_input: Any = None
    provider_quick_select: Any = None
    base_url_quick_input: Any = None
    model_quick_select: Any = None
    mode_quick_select: Any = None
    workspace_quick_input: Any = None
    dark_mode_switch: Any = None
    workspace_browser_label: Any = None
    workspace_browser_column: Any = None
    composer_input: Any = None
    composer_state: dict[str, str] = {"text": ""}
    approval_tool_label: Any = None
    approval_args: Any = None
    dialogs: dict[str, Any] = {}

    def refresh_all() -> None:
        apply_layout_state()
        render_sessions()
        render_header()
        render_transcript()
        render_workbench()
        render_approval()

    def active_runtime() -> NiceGUIRuntimeConfig:
        return controller.active_record.runtime

    def apply_layout_state() -> None:
        dark_mode_control.value = controller.preferences.theme_mode == "dark"
        if session_column is not None:
            session_column.classes(
                replace=_sidebar_container_classes(
                    collapsed=controller.preferences.sidebar_collapsed
                )
            )
        if workbench_column is not None:
            workbench_column.classes(
                replace=_workbench_container_classes(
                    open=controller.preferences.workbench_open
                )
            )
            workbench_column.set_visibility(controller.preferences.workbench_open)

    def settings_button_props(*, collapsed: bool) -> str:
        props = "flat color=primary"
        if collapsed:
            props += " round"
        return props

    def render_settings_footer(*, collapsed: bool) -> None:
        with ui.row().classes(
            "llmt-sidebar-footer w-full q-px-sm q-pt-sm items-center "
            + ("justify-center" if collapsed else "")
        ):
            if collapsed:
                ui.button(
                    icon="settings",
                    on_click=open_settings_dialog,
                ).props(settings_button_props(collapsed=True))
            else:
                ui.button(
                    "Settings",
                    icon="settings",
                    on_click=open_settings_dialog,
                ).props(settings_button_props(collapsed=False))

    def set_model_options(values: Sequence[str], *, selected: str) -> None:
        options = [value for value in values if value]
        if selected and selected not in options:
            options.insert(0, selected)
        model_options_state["values"] = options
        if model_input is not None:
            model_input.set_options(options, value=selected or None)
        if model_quick_select is not None:
            model_quick_select.set_options(options, value=selected or None)

    def open_settings_dialog() -> None:
        dialogs["settings"].open()

    def open_provider_dialog() -> None:
        runtime = active_runtime()
        provider_quick_select.value = runtime.provider.value
        base_url_quick_input.value = runtime.api_base_url or ""
        dialogs["provider_settings"].open()

    def open_model_dialog() -> None:
        runtime = active_runtime()
        set_model_options(model_options_state["values"], selected=runtime.model_name)
        model_quick_select.value = runtime.model_name
        dialogs["model_settings"].open()

    def open_mode_dialog() -> None:
        mode_quick_select.value = active_runtime().provider_mode_strategy.value
        dialogs["mode_settings"].open()

    def open_workspace_dialog() -> None:
        workspace_quick_input.value = active_runtime().root_path or ""
        dialogs["workspace_settings"].open()

    def open_runtime_part_dialog(part_name: str) -> None:
        if part_name == "provider":
            open_provider_dialog()
        elif part_name == "model":
            open_model_dialog()
        elif part_name == "mode":
            open_mode_dialog()
        elif part_name == "workspace":
            open_workspace_dialog()
        else:
            open_settings_dialog()

    def render_runtime_summary(runtime: NiceGUIRuntimeConfig) -> None:
        composer_meta_row.clear()
        with composer_meta_row:
            for index, (label, value) in enumerate(_runtime_summary_parts(runtime)):
                if index:
                    ui.label("|").classes("text-xs llmt-muted")
                ui.button(
                    value,
                    on_click=lambda _event, part=label: open_runtime_part_dialog(part),
                ).props("flat dense no-caps color=primary").classes("llmt-runtime-chip")

    def available_tool_names() -> list[str]:
        return sorted(build_assistant_available_tool_specs())

    def render_selected_tools() -> None:
        if selected_tools_row is None:
            return
        selected_tools_row.clear()
        selected = sorted(set(active_runtime().enabled_tools))
        if not selected:
            return
        with selected_tools_row:
            ui.label("Tools").classes("text-xs llmt-muted")
            for tool_name in selected:
                ui.button(
                    tool_name,
                    icon="check",
                    on_click=lambda _event, name=tool_name: toggle_runtime_tool(name),
                ).props("flat dense no-caps color=primary")

    def toggle_runtime_tool(tool_name: str) -> None:
        runtime = active_runtime()
        enabled = set(runtime.enabled_tools)
        if tool_name in enabled:
            enabled.remove(tool_name)
        else:
            enabled.add(tool_name)
        runtime.enabled_tools = sorted(enabled)
        controller.save_active_session()
        render_selected_tools()

    def set_workspace_browser_target_value(path: str) -> None:
        if workspace_browser_target["value"] == "quick":
            workspace_quick_input.value = path
        else:
            workspace_input.value = path

    def render_sessions() -> None:
        nonlocal session_column
        session_column.clear()
        with session_column:
            collapsed = controller.preferences.sidebar_collapsed
            with ui.row().classes("w-full items-center gap-2 q-pa-sm"):
                ui.button(icon="add", on_click=lambda: new_chat(False)).props(
                    "flat round color=primary"
                )
                if not collapsed:
                    ui.button(
                        "Temporary",
                        icon="lock_clock",
                        on_click=lambda: new_chat(True),
                    ).props("flat color=primary")
                    ui.button(
                        icon="chevron_left",
                        on_click=toggle_sidebar,
                    ).props("flat round color=primary")
                else:
                    ui.button(icon="chevron_right", on_click=toggle_sidebar).props(
                        "flat round color=primary"
                    )
            if collapsed:
                ui.element("div").classes("llmt-sidebar-body")
                render_settings_footer(collapsed=True)
                return
            ui.input(
                "Search chats",
                value=session_filter["value"],
                on_change=lambda event: update_filter(event.value),
            ).props("dense outlined clearable").classes("w-full q-px-sm q-pb-sm")
            with ui.scroll_area().classes("llmt-sidebar-body"):
                for summary in controller.list_session_summaries(
                    query=session_filter["value"]
                ):
                    is_active = summary.session_id == controller.active_session_id
                    classes = "llmt-session w-full items-center justify-between q-px-sm"
                    if is_active:
                        classes += " active"
                    with ui.row().classes(classes):
                        with (
                            ui.column()
                            .classes("gap-0")
                            .on(
                                "click",
                                lambda _event, sid=summary.session_id: select_session(
                                    sid
                                ),
                            )
                        ):
                            ui.label(summary.title).classes("text-sm")
                            ui.label(
                                f"{summary.model_name} | {summary.message_count} msgs"
                            ).classes("text-xs llmt-muted")
                        with ui.row().classes("gap-0"):
                            ui.button(
                                icon="edit",
                                on_click=lambda _event, sid=summary.session_id: (
                                    rename_chat(sid)
                                ),
                            ).props("flat round dense")
                            ui.button(
                                icon="delete",
                                on_click=lambda _event, sid=summary.session_id: (
                                    delete_chat(sid)
                                ),
                            ).props("flat round dense color=negative")
            render_settings_footer(collapsed=False)

    def render_header() -> None:
        runtime = active_runtime()
        turn_state = controller.active_turn_state
        header_title.set_text(controller.active_record.summary.title)
        status_chip.set_text(turn_state.status_text or "ready")
        render_selected_tools()
        render_runtime_summary(runtime)
        composer_action_button.props(
            "icon="
            + _composer_action_icon(busy=turn_state.busy)
            + " color="
            + ("negative" if turn_state.busy else "primary")
        )
        provider_select.value = runtime.provider.value
        set_model_options(model_options_state["values"], selected=runtime.model_name)
        mode_select.value = runtime.provider_mode_strategy.value
        base_url_input.value = runtime.api_base_url or ""
        workspace_input.value = runtime.root_path or ""
        dark_mode_switch.value = controller.preferences.theme_mode == "dark"

    def render_transcript() -> None:
        transcript_column.clear()
        record = controller.active_record
        with transcript_column:
            if not record.transcript:
                with ui.column().classes("items-center justify-center w-full h-full"):
                    ui.label(
                        "Ask about code, files, repositories, or general work."
                    ).classes("text-lg")
                    ui.label(
                        "Tool use, approvals, and inspector payloads appear as the turn runs."
                    ).classes("llmt-muted")
                return
            for entry in record.transcript:
                if not entry.show_in_transcript:
                    continue
                message_class = {
                    "user": "llmt-message llmt-user",
                    "assistant": "llmt-message llmt-assistant",
                    "system": "llmt-message llmt-system",
                    "error": "llmt-message llmt-error",
                }[entry.role]
                with (
                    ui.row().classes("w-full q-px-xl q-py-sm"),
                    ui.column().classes(message_class),
                ):
                    ui.label(entry.role.title()).classes("text-xs llmt-muted")
                    ui.markdown(entry.text)
                    if entry.final_response and entry.final_response.citations:
                        citations = ", ".join(
                            citation.source_path
                            for citation in entry.final_response.citations
                        )
                        ui.label(f"Citations: {citations}").classes(
                            "text-xs llmt-muted"
                        )
                if entry.role == "assistant":
                    with ui.row().classes("q-px-xl q-pb-xs gap-1"):
                        ui.button(
                            icon="content_copy",
                            on_click=lambda _event, text=entry.text: copy_text(text),
                        ).props("flat round dense")
                        ui.button(
                            icon="refresh",
                            on_click=regenerate_last,
                        ).props("flat round dense")

    def render_workbench() -> None:
        workbench_column.clear()
        if not controller.preferences.workbench_open:
            return
        record = controller.active_record
        with workbench_column:
            with ui.row().classes("w-full items-center justify-between q-pa-sm"):
                ui.label("Workbench").classes("text-base")
                ui.button(icon="close", on_click=toggle_workbench).props("flat round")
            with ui.scroll_area().classes("llmt-workbench-body"):
                if not record.workbench_items:
                    ui.label(
                        "Inspector and artifact records will appear here."
                    ).classes("q-pa-md llmt-muted")
                    return
                for item in reversed(record.workbench_items[-20:]):
                    with ui.expansion(item.title, icon="fact_check").classes("w-full"):
                        ui.label(f"{item.kind} v{item.version}").classes(
                            "text-xs llmt-muted"
                        )
                        (
                            ui.json_editor({"content": {"json": item.payload}})
                            .classes("llmt-json-editor")
                            .props("read-only")
                        )

    def render_approval() -> None:
        turn_state = controller.active_turn_state
        approval = turn_state.pending_approval
        if approval is None:
            dialogs["approval"].close()
            return
        approval_tool_label.set_text(f"{approval.tool_name}: {approval.policy_reason}")
        approval_args.set_content(approval.redacted_arguments)
        dialogs["approval"].open()

    def new_chat(temporary: bool) -> None:
        controller.create_session(temporary=temporary)
        refresh_all()

    def select_session(session_id: str) -> None:
        controller.select_session(session_id)
        refresh_all()

    def delete_chat(session_id: str) -> None:
        controller.delete_session(session_id)
        refresh_all()

    def rename_chat(session_id: str) -> None:
        current = controller.sessions[session_id].summary.title

        def apply_rename() -> None:
            controller.rename_session(session_id, title_input.value)
            dialog.close()
            refresh_all()

        with ui.dialog() as dialog, ui.card().classes("w-96"):
            title_input = ui.input("Title", value=current).classes("w-full")
            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Rename", on_click=apply_rename)
        dialog.open()

    def update_filter(value: object) -> None:
        session_filter["value"] = str(value or "")
        render_sessions()

    def toggle_sidebar() -> None:
        controller.preferences.sidebar_collapsed = (
            not controller.preferences.sidebar_collapsed
        )
        controller.store.save_preferences(controller.preferences)
        refresh_all()

    def toggle_workbench() -> None:
        controller.preferences.workbench_open = (
            not controller.preferences.workbench_open
        )
        controller.store.save_preferences(controller.preferences)
        refresh_all()

    def copy_text(text: str) -> None:
        ui.run_javascript(f"navigator.clipboard.writeText({text!r})")
        ui.notify("Copied")

    def regenerate_last() -> None:
        users = [
            entry
            for entry in controller.active_record.transcript
            if entry.role == "user"
        ]
        if not users:
            return
        controller.submit_prompt(users[-1].text)
        refresh_all()

    def update_composer_text(value: object) -> None:
        composer_state["text"] = str(value or "")

    def send_prompt() -> None:
        text = _first_nonempty_text(composer_state["text"], composer_input.value)
        error = controller.submit_prompt(text)
        if error:
            ui.notify(error, type="negative")
        else:
            composer_state["text"] = ""
            composer_input.value = ""
        refresh_all()

    def send_prompt_from_key(event: object) -> None:
        emitted_text = _event_payload_text(event)
        if emitted_text:
            update_composer_text(emitted_text)
        send_prompt()

    def stop_turn() -> None:
        controller.cancel_active_turn()
        refresh_all()

    def submit_or_stop() -> None:
        if controller.active_turn_state.busy:
            stop_turn()
        else:
            send_prompt()

    def apply_settings() -> None:
        runtime = active_runtime()
        runtime.provider = ProviderPreset(str(provider_select.value))
        runtime.model_name = str(model_input.value or runtime.model_name)
        runtime.provider_mode_strategy = ProviderModeStrategy(str(mode_select.value))
        runtime.api_base_url = str(base_url_input.value or "").strip() or None
        runtime.root_path = str(workspace_input.value or "").strip() or None
        controller.preferences.theme_mode = (
            "dark" if bool(dark_mode_switch.value) else "light"
        )
        controller.save_active_session()
        controller.store.save_preferences(controller.preferences)
        refresh_all()

    def apply_settings_and_close() -> None:
        apply_settings()
        dialogs["settings"].close()

    def approve_current() -> None:
        controller.resolve_approval(approved=True)
        dialogs["approval"].close()
        refresh_all()

    def deny_current() -> None:
        controller.resolve_approval(approved=False)
        dialogs["approval"].close()
        refresh_all()

    def refresh_model_options(*, notify: bool = True) -> None:
        provider = ProviderPreset(
            str(provider_select.value or active_runtime().provider)
        )
        base_url = str(base_url_input.value or active_runtime().api_base_url or "")
        current_model = str(model_input.value or active_runtime().model_name)
        models = _discover_model_names(provider=provider, base_url=base_url)
        if not models:
            set_model_options(model_options_state["values"], selected=current_model)
            if notify:
                ui.notify("No models were discovered from the configured endpoint.")
            return
        set_model_options(models, selected=current_model)
        if notify:
            ui.notify(f"Loaded {len(models)} models")

    def refresh_quick_model_options(*, notify: bool = True) -> None:
        runtime = active_runtime()
        current_model = str(model_quick_select.value or runtime.model_name)
        models = _discover_model_names(
            provider=runtime.provider,
            base_url=runtime.api_base_url,
        )
        if not models:
            set_model_options(model_options_state["values"], selected=current_model)
            if notify:
                ui.notify("No models were discovered from the configured endpoint.")
            return
        set_model_options(models, selected=current_model)
        if notify:
            ui.notify(f"Loaded {len(models)} models")

    def apply_provider_settings_and_close() -> None:
        runtime = active_runtime()
        runtime.provider = ProviderPreset(str(provider_quick_select.value))
        runtime.api_base_url = str(base_url_quick_input.value or "").strip() or None
        controller.save_active_session()
        dialogs["provider_settings"].close()
        refresh_all()

    def apply_model_settings_and_close() -> None:
        runtime = active_runtime()
        runtime.model_name = str(model_quick_select.value or runtime.model_name)
        controller.save_active_session()
        dialogs["model_settings"].close()
        refresh_all()

    def apply_mode_settings_and_close() -> None:
        runtime = active_runtime()
        runtime.provider_mode_strategy = ProviderModeStrategy(
            str(mode_quick_select.value)
        )
        controller.save_active_session()
        dialogs["mode_settings"].close()
        refresh_all()

    def apply_workspace_settings_and_close() -> None:
        runtime = active_runtime()
        runtime.root_path = str(workspace_quick_input.value or "").strip() or None
        controller.save_active_session()
        dialogs["workspace_settings"].close()
        refresh_all()

    def workspace_start_path() -> Path:
        source_input = (
            workspace_quick_input
            if workspace_browser_target["value"] == "quick"
            else workspace_input
        )
        raw = str(source_input.value or active_runtime().root_path or Path.cwd())
        try:
            candidate = Path(raw).expanduser().resolve()
        except OSError:
            return Path.cwd().resolve()
        if candidate.is_file():
            return candidate.parent
        if candidate.is_dir():
            return candidate
        return Path.cwd().resolve()

    def render_workspace_browser(path: Path) -> None:
        workspace_browser_path["value"] = path
        workspace_browser_label.set_text(str(path))
        workspace_browser_column.clear()
        with workspace_browser_column:
            parent = path.parent
            ui.button(
                "Parent folder",
                icon="arrow_upward",
                on_click=lambda _event, target=parent: render_workspace_browser(target),
            ).props("flat no-caps")
            try:
                children = sorted(
                    [entry for entry in path.iterdir() if entry.is_dir()],
                    key=lambda entry: entry.name.lower(),
                )
            except OSError as exc:
                ui.label(f"Cannot read folder: {exc}").classes("text-negative")
                return
            if not children:
                ui.label("No child folders").classes("llmt-muted q-pa-sm")
            for child in children[:200]:
                ui.button(
                    child.name,
                    icon="folder",
                    on_click=lambda _event, target=child: render_workspace_browser(
                        target
                    ),
                ).props("flat no-caps align=left").classes("w-full justify-start")
            if len(children) > 200:
                ui.label("Showing first 200 folders").classes("llmt-muted q-pa-sm")

    def open_workspace_browser(*, target: str = "settings") -> None:
        workspace_browser_target["value"] = target
        render_workspace_browser(workspace_start_path())
        dialogs["workspace_browser"].open()

    def select_workspace_browser_path() -> None:
        set_workspace_browser_target_value(str(workspace_browser_path["value"]))
        dialogs["workspace_browser"].close()

    def drain_timer() -> None:
        events = controller.drain_events()
        if events:
            refresh_all()

    with ui.row().classes("llmt-shell no-wrap"):
        with ui.column() as session_column:
            pass

        with ui.column().classes("llmt-main"):
            with ui.row().classes(
                "llmt-header w-full items-center justify-between q-px-md"
            ):
                with ui.column().classes("llmt-header-title gap-0"):
                    header_title = ui.label("").classes("text-base")
                with ui.row().classes("llmt-header-actions items-center gap-2"):
                    status_chip = ui.badge("ready").props("outline")
                    ui.button(
                        icon="view_sidebar",
                        on_click=toggle_workbench,
                    ).props("flat round")

            with (
                ui.scroll_area().classes("llmt-transcript") as _scroll,
                ui.column().classes("w-full q-py-md") as transcript_column,
            ):
                pass

            with ui.column().classes("llmt-composer w-full"):
                with ui.row().classes("llmt-composer-row w-full items-end gap-2"):
                    with (
                        ui.button(icon="add").props("flat round color=primary"),
                        ui.menu(),
                    ):
                        for tool_name in available_tool_names():
                            ui.menu_item(
                                tool_name,
                                on_click=lambda _event, name=tool_name: (
                                    toggle_runtime_tool(name)
                                ),
                            )
                    composer_input = (
                        ui.textarea(
                            placeholder="Message llm-tools",
                            value=composer_state["text"],
                            on_change=lambda event: update_composer_text(event.value),
                        )
                        .props("autogrow outlined debounce=0")
                        .classes("llmt-composer-input")
                    )
                    composer_input.on(
                        "keydown.enter",
                        send_prompt_from_key,
                        js_handler=(
                            "(event) => {"
                            " if (!event.shiftKey) {"
                            " event.preventDefault();"
                            " emit(event.target.value);"
                            " }"
                            "}"
                        ),
                    )
                    composer_action_button = ui.button(
                        icon=_composer_action_icon(
                            busy=controller.active_turn_state.busy
                        ),
                        on_click=submit_or_stop,
                    ).props("round color=primary")
                with ui.row().classes(
                    "llmt-selected-tools w-full items-center gap-1"
                ) as selected_tools_row:
                    pass
                with ui.row().classes(
                    "llmt-composer-meta w-full items-center gap-1"
                ) as composer_meta_row:
                    pass
                ui.label("AI can make mistakes. Check important info.").classes(
                    "llmt-composer-disclaimer llmt-muted"
                )

        with ui.column() as workbench_column:
            pass

    with ui.dialog() as settings_dialog, ui.card().classes("w-[520px]"):
        dialogs["settings"] = settings_dialog
        ui.label("Settings").classes("text-lg")
        dark_mode_switch = ui.switch(
            "Dark mode",
            value=controller.preferences.theme_mode == "dark",
        ).classes("w-full")
        provider_select = ui.select(
            ["ollama", "openai", "custom_openai_compatible"],
            label="Provider",
        ).classes("w-full")
        with ui.row().classes("w-full items-end gap-2 no-wrap"):
            model_input = (
                ui.select(
                    [controller.active_record.runtime.model_name],
                    label="Model",
                    value=controller.active_record.runtime.model_name,
                    with_input=True,
                    new_value_mode="add-unique",
                )
                .classes("grow")
                .props("clearable")
            )
            ui.button(
                icon="refresh",
                on_click=lambda: refresh_model_options(notify=True),
            ).props("flat round color=primary")
        mode_select = ui.select(
            [strategy.value for strategy in ProviderModeStrategy],
            label="Provider mode",
        ).classes("w-full")
        base_url_input = ui.input("Base URL").classes("w-full")
        with ui.row().classes("w-full items-end gap-2 no-wrap"):
            workspace_input = ui.input("Workspace root").classes("grow")
            ui.button(
                icon="folder_open",
                on_click=lambda: open_workspace_browser(target="settings"),
            ).props("flat round color=primary")
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=settings_dialog.close).props("flat")
            ui.button("Apply", on_click=apply_settings_and_close)

    with ui.dialog() as provider_settings_dialog, ui.card().classes("w-[460px]"):
        dialogs["provider_settings"] = provider_settings_dialog
        ui.label("Provider").classes("text-lg")
        provider_quick_select = ui.select(
            ["ollama", "openai", "custom_openai_compatible"],
            label="Provider",
        ).classes("w-full")
        base_url_quick_input = ui.input("Base URL").classes("w-full")
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=provider_settings_dialog.close).props("flat")
            ui.button("Apply", on_click=apply_provider_settings_and_close)

    with ui.dialog() as model_settings_dialog, ui.card().classes("w-[460px]"):
        dialogs["model_settings"] = model_settings_dialog
        ui.label("Model").classes("text-lg")
        with ui.row().classes("w-full items-end gap-2 no-wrap"):
            model_quick_select = (
                ui.select(
                    [controller.active_record.runtime.model_name],
                    label="Model",
                    value=controller.active_record.runtime.model_name,
                    with_input=True,
                    new_value_mode="add-unique",
                )
                .classes("grow")
                .props("clearable")
            )
            ui.button(
                icon="refresh",
                on_click=lambda: refresh_quick_model_options(notify=True),
            ).props("flat round color=primary")
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=model_settings_dialog.close).props("flat")
            ui.button("Apply", on_click=apply_model_settings_and_close)

    with ui.dialog() as mode_settings_dialog, ui.card().classes("w-[420px]"):
        dialogs["mode_settings"] = mode_settings_dialog
        ui.label("Provider mode").classes("text-lg")
        mode_quick_select = ui.select(
            [strategy.value for strategy in ProviderModeStrategy],
            label="Provider mode",
        ).classes("w-full")
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=mode_settings_dialog.close).props("flat")
            ui.button("Apply", on_click=apply_mode_settings_and_close)

    with ui.dialog() as workspace_settings_dialog, ui.card().classes("w-[520px]"):
        dialogs["workspace_settings"] = workspace_settings_dialog
        ui.label("Workspace").classes("text-lg")
        with ui.row().classes("w-full items-end gap-2 no-wrap"):
            workspace_quick_input = ui.input("Workspace root").classes("grow")
            ui.button(
                icon="folder_open",
                on_click=lambda: open_workspace_browser(target="quick"),
            ).props("flat round color=primary")
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=workspace_settings_dialog.close).props("flat")
            ui.button("Apply", on_click=apply_workspace_settings_and_close)

    with ui.dialog() as workspace_dialog, ui.card().classes("w-[560px]"):
        dialogs["workspace_browser"] = workspace_dialog
        ui.label("Select Workspace Root").classes("text-lg")
        workspace_browser_label = ui.label("").classes("text-sm llmt-muted")
        with (
            ui.scroll_area().classes("w-full h-[360px]"),
            ui.column().classes("w-full") as workspace_browser_column,
        ):
            pass
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=workspace_dialog.close).props("flat")
            ui.button("Select", on_click=select_workspace_browser_path)

    with ui.dialog() as approval_dialog, ui.card().classes("w-[520px]"):
        dialogs["approval"] = approval_dialog
        ui.label("Tool Approval").classes("text-lg")
        approval_tool_label = ui.label("")
        approval_args = ui.json_editor({"content": {"json": {}}}).classes("w-full")
        with ui.row().classes("justify-end w-full"):
            ui.button("Deny", on_click=deny_current).props("flat color=negative")
            ui.button("Approve", on_click=approve_current).props("color=primary")

    refresh_all()
    ui.timer(0.25, drain_timer)


def main(argv: Sequence[str] | None = None) -> int:
    """Console entrypoint for the NiceGUI chat app."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_assistant_config(args)
    root_path = resolve_root_argument(args, config)
    if root_path is not None:
        default_runtime_config(config, root_path=root_path)
    run_nicegui_chat_app(
        root_path=root_path,
        config=config,
        db_path=args.db_path,
        host=args.host,
        port=args.port,
        show=not args.no_browser,
    )
    return 0


__all__ = [
    "build_nicegui_chat_ui",
    "build_parser",
    "main",
    "resolve_assistant_config",
    "resolve_root_argument",
    "run_nicegui_chat_app",
]

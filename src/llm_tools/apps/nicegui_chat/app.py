"""NiceGUI chat client for llm-tools."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_tools.apps.assistant_config import (
    StreamlitAssistantConfig,
    load_streamlit_assistant_config,
)
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
        body { background: #f6f6f3; color: #20211f; }
        .llmt-shell { height: 100vh; width: 100vw; overflow: hidden; }
        .llmt-sidebar {
            width: 284px; min-width: 284px; background: #ecebe6;
            border-right: 1px solid #d8d6ce;
        }
        .llmt-sidebar.collapsed { width: 74px; min-width: 74px; }
        .llmt-main { min-width: 0; background: #fbfbf8; }
        .llmt-header { height: 58px; border-bottom: 1px solid #dedcd4; }
        .llmt-transcript { overflow-y: auto; }
        .llmt-composer { border-top: 1px solid #dedcd4; background: #fbfbf8; }
        .llmt-workbench {
            width: 340px; min-width: 340px; background: #f1f0eb;
            border-left: 1px solid #d8d6ce;
        }
        .llmt-message {
            border-radius: 8px; padding: 12px 14px; max-width: 820px;
            white-space: pre-wrap; line-height: 1.45;
        }
        .llmt-user { background: #e8f2ef; margin-left: auto; }
        .llmt-assistant { background: #ffffff; border: 1px solid #e0ded7; }
        .llmt-system { background: #fff7df; border: 1px solid #ead797; }
        .llmt-error { background: #fff0ef; border: 1px solid #e9b7b1; }
        .llmt-session { border-radius: 8px; min-height: 42px; }
        .llmt-session.active { background: #dfe8e4; }
        .llmt-muted { color: #6e6d67; }
        .llmt-code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
        </style>
        """
    )

    session_filter = {"value": ""}
    transcript_column: Any = None
    session_column: Any = None
    workbench_column: Any = None
    header_title: Any = None
    header_meta: Any = None
    status_chip: Any = None
    provider_select: Any = None
    model_input: Any = None
    base_url_input: Any = None
    mode_select: Any = None
    workspace_input: Any = None
    composer_input: Any = None
    composer_state: dict[str, str] = {"text": ""}
    approval_tool_label: Any = None
    approval_args: Any = None
    dialogs: dict[str, Any] = {}

    def refresh_all() -> None:
        render_sessions()
        render_header()
        render_transcript()
        render_workbench()
        render_approval()

    def active_runtime() -> NiceGUIRuntimeConfig:
        return controller.active_record.runtime

    def render_sessions() -> None:
        nonlocal session_column
        session_column.clear()
        with session_column:
            collapsed = controller.preferences.sidebar_collapsed
            with ui.row().classes("w-full items-center gap-2 q-pa-sm"):
                ui.button(icon="add", on_click=lambda: new_chat(False)).props(
                    "flat round"
                )
                if not collapsed:
                    ui.button(
                        "Temporary",
                        icon="lock_clock",
                        on_click=lambda: new_chat(True),
                    ).props("flat")
                    ui.button(
                        icon="left_panel_close",
                        on_click=toggle_sidebar,
                    ).props("flat round")
                else:
                    ui.button(icon="left_panel_open", on_click=toggle_sidebar).props(
                        "flat round"
                    )
            if collapsed:
                return
            ui.input(
                "Search chats",
                value=session_filter["value"],
                on_change=lambda event: update_filter(event.value),
            ).props("dense outlined clearable").classes("w-full q-px-sm q-pb-sm")
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
                            lambda _event, sid=summary.session_id: select_session(sid),
                        )
                    ):
                        ui.label(summary.title).classes("text-sm")
                        ui.label(
                            f"{summary.model_name} | {summary.message_count} msgs"
                        ).classes("text-xs llmt-muted")
                    with ui.row().classes("gap-0"):
                        ui.button(
                            icon="edit",
                            on_click=lambda _event, sid=summary.session_id: rename_chat(
                                sid
                            ),
                        ).props("flat round dense")
                        ui.button(
                            icon="delete",
                            on_click=lambda _event, sid=summary.session_id: delete_chat(
                                sid
                            ),
                        ).props("flat round dense color=negative")

    def render_header() -> None:
        runtime = active_runtime()
        turn_state = controller.active_turn_state
        header_title.set_text(controller.active_record.summary.title)
        root = runtime.root_path or "No workspace"
        header_meta.set_text(
            f"{runtime.provider.value} | {runtime.model_name} | {root}"
        )
        status_chip.set_text(turn_state.status_text or "ready")
        provider_select.value = runtime.provider.value
        model_input.value = runtime.model_name
        mode_select.value = runtime.provider_mode_strategy.value
        base_url_input.value = runtime.api_base_url or ""
        workspace_input.value = runtime.root_path or ""

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
            if not record.workbench_items:
                ui.label("Inspector and artifact records will appear here.").classes(
                    "q-pa-md llmt-muted"
                )
                return
            for item in reversed(record.workbench_items[-20:]):
                with ui.expansion(item.title, icon="fact_check").classes("w-full"):
                    ui.label(f"{item.kind} v{item.version}").classes(
                        "text-xs llmt-muted"
                    )
                    ui.json_editor({"content": {"json": item.payload}}).classes(
                        "w-full"
                    ).props("read-only")

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

    def apply_settings() -> None:
        runtime = active_runtime()
        runtime.provider = ProviderPreset(str(provider_select.value))
        runtime.model_name = str(model_input.value or runtime.model_name)
        runtime.provider_mode_strategy = ProviderModeStrategy(str(mode_select.value))
        runtime.api_base_url = str(base_url_input.value or "").strip() or None
        runtime.root_path = str(workspace_input.value or "").strip() or None
        controller.save_active_session()
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

    def drain_timer() -> None:
        events = controller.drain_events()
        if events:
            refresh_all()

    with ui.row().classes("llmt-shell no-wrap"):
        sidebar_classes = "llmt-sidebar"
        if controller.preferences.sidebar_collapsed:
            sidebar_classes += " collapsed"
        with ui.column().classes(sidebar_classes) as session_column:
            pass

        with ui.column().classes("llmt-main fit"):
            with ui.row().classes(
                "llmt-header w-full items-center justify-between q-px-md"
            ):
                with ui.column().classes("gap-0"):
                    header_title = ui.label("").classes("text-base")
                    header_meta = ui.label("").classes("text-xs llmt-muted")
                with ui.row().classes("items-center gap-2"):
                    status_chip = ui.badge("ready").props("outline")
                    ui.button(
                        icon="tune", on_click=lambda: dialogs["settings"].open()
                    ).props("flat round")
                    ui.button(icon="dock_to_right", on_click=toggle_workbench).props(
                        "flat round"
                    )

            with (
                ui.scroll_area().classes("llmt-transcript w-full fit") as _scroll,
                ui.column().classes("w-full q-py-md") as transcript_column,
            ):
                pass

            with ui.column().classes("llmt-composer w-full q-pa-md"):
                with ui.row().classes("w-full items-end gap-2"):
                    ui.button(
                        icon="add",
                        on_click=lambda: ui.notify(
                            "Attachments are reserved for a future version."
                        ),
                    ).props("flat round")
                    composer_input = (
                        ui.textarea(
                            placeholder="Message llm-tools",
                            value=composer_state["text"],
                            on_change=lambda event: update_composer_text(event.value),
                        )
                        .props("autogrow outlined debounce=0")
                        .classes("fit")
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
                    ui.button("Send", icon="send", on_click=send_prompt).props(
                        "unelevated color=primary"
                    )
                    ui.button("Stop", icon="stop", on_click=stop_turn).props(
                        "outline color=negative"
                    )
                ui.label(
                    "Provider mode: " + active_runtime().provider_mode_strategy.value
                ).classes("text-xs llmt-muted")

        if controller.preferences.workbench_open:
            with ui.column().classes("llmt-workbench") as workbench_column:
                pass
        else:
            workbench_column = ui.column().classes("hidden")

    with ui.dialog() as settings_dialog, ui.card().classes("w-[520px]"):
        dialogs["settings"] = settings_dialog
        ui.label("Settings").classes("text-lg")
        provider_select = ui.select(
            ["ollama", "openai", "custom_openai_compatible"],
            label="Provider",
        ).classes("w-full")
        model_input = ui.input("Model").classes("w-full")
        mode_select = ui.select(
            [strategy.value for strategy in ProviderModeStrategy],
            label="Provider mode",
        ).classes("w-full")
        base_url_input = ui.input("Base URL").classes("w-full")
        workspace_input = ui.input("Workspace root").classes("w-full")
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=settings_dialog.close).props("flat")
            ui.button("Apply", on_click=apply_settings_and_close)

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

"""NiceGUI chat client for llm-tools."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from nicegui import app as nicegui_app

from llm_tools.apps.assistant_config import (
    StreamlitAssistantConfig,
    load_streamlit_assistant_config,
)
from llm_tools.apps.assistant_tool_capabilities import (
    AssistantToolCapability,
    build_tool_capabilities,
    build_tool_group_capability_summaries,
)
from llm_tools.apps.assistant_tool_registry import build_assistant_available_tool_specs
from llm_tools.apps.chat_config import ProviderPreset
from llm_tools.apps.nicegui_chat.auth import (
    LocalAuthProvider,
    ensure_secret_file,
    validate_hosted_startup,
)
from llm_tools.apps.nicegui_chat.controller import (
    PROVIDER_API_KEY_FIELD,
    NiceGUIChatController,
    default_runtime_config,
)
from llm_tools.apps.nicegui_chat.models import (
    NiceGUIHostedConfig,
    NiceGUIInteractionMode,
    NiceGUIRuntimeConfig,
    NiceGUIUser,
)
from llm_tools.apps.nicegui_chat.store import SQLiteNiceGUIChatStore, default_db_path
from llm_tools.llm_providers import ProviderModeStrategy
from llm_tools.tool_api import SideEffectClass
from llm_tools.workflow_api import (
    ProtectionConfig,
    ProtectionPendingPrompt,
    inspect_protection_corpus,
)

NICEGUI_PROVIDER_OPTIONS = [
    ProviderPreset.OLLAMA.value,
    ProviderPreset.CUSTOM_OPENAI_COMPATIBLE.value,
]
NICEGUI_APPROVAL_OPTIONS = [
    SideEffectClass.LOCAL_READ,
    SideEffectClass.LOCAL_WRITE,
    SideEffectClass.EXTERNAL_READ,
    SideEffectClass.EXTERNAL_WRITE,
]
NICEGUI_APPROVAL_LABELS = {
    SideEffectClass.LOCAL_READ.value: "Local read",
    SideEffectClass.LOCAL_WRITE.value: "Local write",
    SideEffectClass.EXTERNAL_READ.value: "External read",
    SideEffectClass.EXTERNAL_WRITE.value: "External write",
}
AUTH_DISABLED_WARNING = (
    "WARNING: NiceGUI auth is disabled. Use --auth-mode local for normal local "
    "or hosted use."
)
PROTECTION_UNDEFINED_LABEL = "Undefined"
PROTECTION_CORRECTIONS_FILENAME = ".llm-tools-protection-corrections.json"
INTERACTION_MODE_LABELS: dict[NiceGUIInteractionMode, str] = {
    "chat": "Chat",
    "deep_task": "Deep Task",
}
INTERACTION_MODE_TOOLTIPS: dict[NiceGUIInteractionMode, str] = {
    "chat": "Fast conversational replies for quick back-and-forth.",
    "deep_task": (
        "Durable multi-step work with task state, approvals, trace, and summary."
    ),
}


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


def _is_admin_user(user: NiceGUIUser | None) -> bool:
    """Return whether a user may open admin controls."""
    return user is not None and user.role == "admin"


def _can_admin_disable_user(
    current_user: NiceGUIUser | None,
    target_user: NiceGUIUser,
) -> bool:
    """Return whether the current admin may disable the target user."""
    if current_user is None or current_user.role != "admin":
        return False
    return current_user.user_id != target_user.user_id


def _format_workbench_duration(seconds: float | None) -> str:
    """Return a compact workbench duration label."""
    if seconds is None:
        return "duration unknown"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:.0f}s"


def _parse_information_security_categories(value: str) -> list[str]:
    """Parse user-entered sensitivity labels while preserving declared order."""
    labels: list[str] = []
    seen: set[str] = set()
    for raw_entry in value.replace(",", "\n").splitlines():
        label = raw_entry.strip()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def _format_information_security_level(config: ProtectionConfig) -> str:
    """Return the compact header level for the active protection config."""
    if (
        not config.enabled
        or not config.allowed_sensitivity_labels
        or not config.document_paths
    ):
        return PROTECTION_UNDEFINED_LABEL
    report = inspect_protection_corpus(config)
    if not report.usable_document_count:
        return PROTECTION_UNDEFINED_LABEL
    return "/".join(config.allowed_sensitivity_labels)


def _format_information_security_label(config: ProtectionConfig) -> str:
    """Return the visible information-security header label."""
    return (
        f"Information Security Settings: {_format_information_security_level(config)}"
    )


def _default_protection_corrections_path(corpus_directory: str) -> str:
    """Return the hidden corrections sidecar path for a corpus directory."""
    cleaned = corpus_directory.strip()
    if not cleaned:
        return ""
    return str(Path(cleaned).expanduser() / PROTECTION_CORRECTIONS_FILENAME)


def _protection_corpus_readiness_text(config: ProtectionConfig) -> str:
    """Return a compact corpus readiness summary."""
    if not config.enabled:
        return "Protection is disabled."
    if not config.document_paths:
        return "Choose a corpus directory before protection can engage."
    if not config.allowed_sensitivity_labels:
        return "Add at least one allowed category before protection can engage."
    report = inspect_protection_corpus(config)
    if report.usable_document_count:
        return (
            f"Ready: {report.usable_document_count} document(s), "
            f"{len(report.corpus.feedback_entries)} correction(s)."
        )
    return "Not ready: no readable protection documents were found."


def _referenced_document_text(prompt: ProtectionPendingPrompt) -> str:
    """Return referenced document IDs for a challenge dialog."""
    if not prompt.referenced_document_ids:
        return "No referenced document IDs."
    return ", ".join(prompt.referenced_document_ids)


def _provider_api_key_env_var(
    provider: ProviderPreset,
    configured_env_var: str | None,
) -> str:
    """Return the configured provider API-key env var without exposing a secret."""
    if provider is ProviderPreset.OLLAMA:
        return configured_env_var or ""
    return configured_env_var or "OPENAI_API_KEY"


def _tool_capability_tooltip(capability: AssistantToolCapability) -> str:
    """Return compact hover text for a tool capability."""
    lines = [capability.tool_name]
    if capability.detail:
        lines.append(capability.detail)
    elif capability.exposed_to_model:
        lines.append("Available to the model.")
    elif capability.enabled:
        lines.append("Selected but blocked from the model.")
    else:
        lines.append("Not selected.")
    if capability.approval_required:
        lines.append("Approval required before execution.")
    if capability.required_secrets:
        lines.append("Credentials: " + ", ".join(capability.required_secrets))
    return "\n".join(lines)


def _selected_tool_chip_classes(capability: AssistantToolCapability) -> str:
    """Return selected tool chip classes."""
    classes = "llmt-tool-chip"
    if capability.exposed_to_model:
        classes += " llmt-tool-chip-enabled"
    else:
        classes += " llmt-tool-chip-blocked"
    return classes


def _selected_tool_icon(capability: AssistantToolCapability) -> str:
    """Return a compact icon for selected tool availability."""
    return "check" if capability.exposed_to_model else "block"


def _is_tool_url_setting(name: str) -> bool:
    """Return whether a required tool value is a non-secret URL."""
    normalized = name.strip().upper()
    return normalized.endswith("_BASE_URL") or normalized.endswith("_URL")


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
    api_key_env_var: str | None = None,
    api_key: str | None = None,
    allow_env_api_key: bool = True,
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
    api_key_env_var = _provider_api_key_env_var(provider, api_key_env_var)
    effective_api_key = api_key or (
        os.environ.get(api_key_env_var)
        if api_key_env_var and allow_env_api_key
        else None
    )
    if provider is not ProviderPreset.OLLAMA and effective_api_key:
        headers["Authorization"] = f"Bearer {effective_api_key}"
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
    parser.add_argument("--api-key-env-var", type=str)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--db-key-file", type=Path, default=None)
    parser.add_argument("--user-key-file", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--auth-mode", choices=["none", "local"], default="local")
    parser.add_argument("--public-base-url", type=str)
    parser.add_argument("--tls-certfile", type=Path)
    parser.add_argument("--tls-keyfile", type=Path)
    parser.add_argument("--allow-insecure-hosted-secrets", action="store_true")
    parser.add_argument("--hosted-secret-key-path", type=Path)
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
    if args.api_key_env_var is not None:
        raw["llm"]["api_key_env_var"] = args.api_key_env_var
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
    db_key_file: Path | None = None,
    user_key_file: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    show: bool = True,
    auth_mode: str = "local",
    public_base_url: str | None = None,
    tls_certfile: Path | None = None,
    tls_keyfile: Path | None = None,
    allow_insecure_hosted_secrets: bool = False,
    hosted_secret_key_path: Path | None = None,
) -> None:  # pragma: no cover
    """Render and run the NiceGUI chat app."""
    from nicegui import ui

    store = SQLiteNiceGUIChatStore(
        db_path or default_db_path(),
        db_key_file=db_key_file,
        user_key_file=user_key_file,
    )
    store.initialize()
    startup = validate_hosted_startup(
        auth_mode=auth_mode,
        host=host,
        public_base_url=public_base_url,
        tls_certfile=str(tls_certfile) if tls_certfile is not None else None,
        tls_keyfile=str(tls_keyfile) if tls_keyfile is not None else None,
        allow_insecure_hosted_secrets=allow_insecure_hosted_secrets,
        secret_key_path=hosted_secret_key_path,
    )
    if startup.config.auth_mode == "none":
        print(AUTH_DISABLED_WARNING, file=sys.stderr)
    storage_secret: str | None = None
    auth_provider: LocalAuthProvider | None = None
    if startup.config.auth_mode == "local":
        storage_secret = ensure_secret_file(Path(str(startup.config.secret_key_path)))
        auth_provider = LocalAuthProvider(store)

        @ui.page("/")
        def hosted_page() -> None:
            if auth_provider is None:
                raise RuntimeError("Hosted NiceGUI auth was not initialized.")
            render_hosted_nicegui_page(
                store=store,
                config=config,
                root_path=root_path,
                hosted_config=startup.config,
                auth_provider=auth_provider,
            )

    else:
        controller = NiceGUIChatController(
            store=store,
            config=config,
            root_path=root_path,
            hosted_config=startup.config,
        )
        build_nicegui_chat_ui(controller)
    base_run_kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "reload": False,
        "show": show,
        "title": "llm-tools chat",
        "storage_secret": storage_secret,
        "session_middleware_kwargs": {
            "https_only": bool(startup.tls_enabled),
            "same_site": "lax",
        },
    }
    if tls_certfile is not None and tls_keyfile is not None:
        ui.run(
            **base_run_kwargs,
            ssl_certfile=str(tls_certfile),
            ssl_keyfile=str(tls_keyfile),
        )
    else:
        ui.run(**base_run_kwargs)


def _hosted_storage_values() -> tuple[str | None, str | None]:
    session_id = nicegui_app.storage.user.get("nicegui_chat_session_id")
    token = nicegui_app.storage.user.get("nicegui_chat_session_token")
    return (
        str(session_id) if session_id else None,
        str(token) if token else None,
    )


def _set_hosted_storage_values(session_id: str, token: str) -> None:
    nicegui_app.storage.user["nicegui_chat_session_id"] = session_id
    nicegui_app.storage.user["nicegui_chat_session_token"] = token


def _clear_hosted_storage_values() -> None:
    nicegui_app.storage.user.pop("nicegui_chat_session_id", None)
    nicegui_app.storage.user.pop("nicegui_chat_session_token", None)


def clear_hosted_session(auth_provider: LocalAuthProvider | None) -> None:
    """Revoke and clear the current browser auth session."""
    session_id, _token = _hosted_storage_values()
    if session_id and auth_provider is not None:
        auth_provider.revoke_session(session_id)
    _clear_hosted_storage_values()


def render_hosted_nicegui_page(
    *,
    store: SQLiteNiceGUIChatStore,
    config: StreamlitAssistantConfig,
    root_path: Path | None,
    hosted_config: NiceGUIHostedConfig,
    auth_provider: LocalAuthProvider,
) -> None:
    """Render first-run admin, login, or authenticated chat UI."""
    if not auth_provider.has_users():
        render_first_admin_page(auth_provider)
        return
    session_id, token = _hosted_storage_values()
    current_user = (
        None
        if session_id is None or token is None
        else auth_provider.user_for_session(session_id, token)
    )
    if current_user is None:
        render_login_page(auth_provider)
        return
    controller = NiceGUIChatController(
        store=store,
        config=config,
        root_path=root_path,
        hosted_config=hosted_config,
        current_user=current_user,
        auth_provider=auth_provider,
    )
    build_nicegui_chat_ui(controller)


def render_first_admin_page(
    auth_provider: LocalAuthProvider,
) -> None:  # pragma: no cover
    """Render first-run hosted admin creation."""
    from nicegui import ui

    ui.dark_mode(value=True)
    _add_auth_page_styles()
    username_input: Any = None
    password_input: Any = None

    def create_admin() -> None:
        username = str(username_input.value or "").strip()
        password = str(password_input.value or "")
        try:
            user = auth_provider.create_user(
                username=username,
                password=password,
                role="admin",
            )
            session_id, token = auth_provider.create_session(user.user_id)
            _set_hosted_storage_values(session_id, token)
            ui.notify("Admin user created.")
            ui.navigate.reload()
        except Exception as exc:  # pragma: no cover - UI guard
            ui.notify(f"Could not create admin user: {exc}", type="negative")

    with (
        ui.column().classes("w-screen h-screen items-center justify-center q-pa-md"),
        ui.card().classes("w-[420px]"),
    ):
        ui.label("Create Admin").classes("text-lg")
        ui.label("Hosted mode needs one local admin user.").classes(
            "text-sm llmt-muted"
        )
        username_input = ui.input("Username").classes("w-full")
        password_input = ui.input("Password").props("type=password").classes("w-full")
        ui.button("Create admin", on_click=create_admin).classes("w-full")


def render_login_page(auth_provider: LocalAuthProvider) -> None:  # pragma: no cover
    """Render hosted-mode login."""
    from nicegui import ui

    ui.dark_mode(value=True)
    _add_auth_page_styles()
    username_input: Any = None
    password_input: Any = None

    def login() -> None:
        username = str(username_input.value or "").strip()
        password = str(password_input.value or "")
        user = auth_provider.authenticate(username=username, password=password)
        if user is None:
            ui.notify("Invalid username or password.", type="negative")
            return
        session_id, token = auth_provider.create_session(user.user_id)
        _set_hosted_storage_values(session_id, token)
        ui.navigate.reload()

    with (
        ui.column().classes("w-screen h-screen items-center justify-center q-pa-md"),
        ui.card().classes("w-[420px]"),
    ):
        ui.label("Sign In").classes("text-lg")
        username_input = ui.input("Username").classes("w-full")
        password_input = ui.input("Password").props("type=password").classes("w-full")
        password_input.on("keydown.enter", lambda: login())
        ui.button("Sign in", on_click=login).classes("w-full")


def _add_auth_page_styles() -> None:
    """Apply the default unauthenticated NiceGUI dark theme."""
    from nicegui import ui

    ui.add_head_html(
        """
        <style>
        html, body, #app {
            min-height: 100%; margin: 0;
            background: #161715; color: #e8e6de;
        }
        body.body--dark {
            --q-primary: #d2d2ce;
            --q-negative: #d2d2ce;
        }
        body.body--dark .q-card {
            background: #20221f; color: #e8e6de;
            border: 1px solid #3a3d37;
        }
        body.body--dark .q-field__native,
        body.body--dark .q-field__label {
            color: #e8e6de;
        }
        body.body--dark .q-btn.bg-primary {
            background: #d2d2ce !important; color: #161715 !important;
        }
        .llmt-muted { color: #aaa79e; }
        </style>
        """
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
        .llmt-shell .bg-primary .q-icon,
        .q-dialog .bg-primary .q-icon,
        .q-menu .bg-primary .q-icon,
        .llmt-shell .bg-primary .q-btn__content,
        .q-dialog .bg-primary .q-btn__content,
        .q-menu .bg-primary .q-btn__content,
        .llmt-shell .bg-negative .q-icon,
        .q-dialog .bg-negative .q-icon,
        .q-menu .bg-negative .q-icon,
        .llmt-shell .bg-negative .q-btn__content,
        .q-dialog .bg-negative .q-btn__content,
        .q-menu .bg-negative .q-btn__content {
            color: var(--llmt-accent-contrast) !important;
        }
        body.body--dark .llmt-shell .bg-primary,
        body.body--dark .q-dialog .bg-primary,
        body.body--dark .q-menu .bg-primary,
        body.body--dark .llmt-shell .bg-negative,
        body.body--dark .q-dialog .bg-negative,
        body.body--dark .q-menu .bg-negative {
            background: #d7d7d2 !important;
            color: #11120f !important;
        }
        body.body--dark .llmt-shell .bg-primary .q-icon,
        body.body--dark .q-dialog .bg-primary .q-icon,
        body.body--dark .q-menu .bg-primary .q-icon,
        body.body--dark .llmt-shell .bg-primary .q-btn__content,
        body.body--dark .q-dialog .bg-primary .q-btn__content,
        body.body--dark .q-menu .bg-primary .q-btn__content,
        body.body--dark .llmt-shell .bg-negative .q-icon,
        body.body--dark .q-dialog .bg-negative .q-icon,
        body.body--dark .q-menu .bg-negative .q-icon,
        body.body--dark .llmt-shell .bg-negative .q-btn__content,
        body.body--dark .q-dialog .bg-negative .q-btn__content,
        body.body--dark .q-menu .bg-negative .q-btn__content {
            color: #11120f !important;
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
        .llmt-header-security {
            flex: 1 1 auto; justify-content: center; min-width: 160px;
            overflow: hidden;
        }
        .llmt-info-security-button {
            max-width: 100%; min-height: 28px; padding: 0 8px;
            border: 1px solid #d4d2ca;
        }
        body.body--dark .llmt-info-security-button { border-color: #4b4d47; }
        .llmt-info-security-button .q-btn__content {
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            display: block; min-width: 0;
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
        .llmt-selected-tools .q-btn,
        .llmt-tool-menu .q-btn {
            min-height: 22px; padding: 0 6px; font-size: 12px;
        }
        .llmt-tool-menu {
            min-width: 280px; max-width: min(420px, 92vw);
            max-height: min(520px, 80vh); overflow-y: auto;
            padding: 8px;
        }
        .llmt-account-menu {
            min-width: 180px; padding: 6px;
        }
        .llmt-tool-menu-note {
            font-size: 12px; line-height: 16px; padding: 2px 4px 8px;
        }
        .llmt-tool-group-header {
            border-top: 1px solid #dedcd4; padding-top: 6px; margin-top: 4px;
        }
        body.body--dark .llmt-tool-group-header { border-top-color: #373a34; }
        .llmt-tool-name {
            justify-content: flex-start; width: 100%; min-width: 0;
        }
        .llmt-tool-name .q-btn__content,
        .llmt-tool-chip .q-btn__content {
            min-width: 0; overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap; display: flex; flex-wrap: nowrap;
        }
        .llmt-tool-chip-enabled {
            color: var(--llmt-accent) !important;
        }
        .llmt-tool-chip-blocked {
            text-decoration: line-through; opacity: 0.72;
            color: var(--llmt-accent) !important;
        }
        .llmt-credential-row {
            border-top: 1px solid #dedcd4; padding: 6px 0;
        }
        body.body--dark .llmt-credential-row { border-top-color: #373a34; }
        .llmt-settings-body {
            width: 100%; max-height: min(70vh, 560px);
        }
        .llmt-settings-section {
            border-top: 1px solid #dedcd4; padding-top: 10px; margin-top: 4px;
        }
        body.body--dark .llmt-settings-section { border-top-color: #373a34; }
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
    information_security_button: Any = None
    status_chip: Any = None
    selected_tools_row: Any = None
    tool_menu_column: Any = None
    composer_meta_row: Any = None
    composer_action_button: Any = None
    provider_select: Any = None
    model_input: Any = None
    base_url_input: Any = None
    provider_api_key_input: Any = None
    mode_select: Any = None
    workspace_input: Any = None
    database_path_input: Any = None
    provider_quick_select: Any = None
    base_url_quick_input: Any = None
    provider_api_key_quick_input: Any = None
    model_quick_select: Any = None
    mode_quick_select: Any = None
    workspace_quick_input: Any = None
    dark_mode_switch: Any = None
    admin_user_list_column: Any = None
    admin_create_username_input: Any = None
    admin_create_password_input: Any = None
    admin_create_role_select: Any = None
    admin_deep_task_switch: Any = None
    allow_network_switch: Any = None
    allow_filesystem_switch: Any = None
    allow_subprocess_switch: Any = None
    approval_select: Any = None
    tool_credentials_column: Any = None
    tool_credential_inputs: dict[str, Any] = {}
    tool_url_inputs: dict[str, Any] = {}
    protection_enabled_switch: Any = None
    protection_categories_input: Any = None
    protection_corpus_input: Any = None
    protection_corrections_input: Any = None
    protection_readiness_label: Any = None
    protection_issues_column: Any = None
    protection_predicted_label: Any = None
    protection_reasoning_label: Any = None
    protection_refs_label: Any = None
    protection_original_request: Any = None
    protection_overrule_category_input: Any = None
    protection_overrule_rationale_input: Any = None
    protection_dialog_state: dict[str, str | None] = {"challenge_key": None}
    workspace_browser_label: Any = None
    workspace_browser_title: Any = None
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
        render_protection_challenge()

    def active_runtime() -> NiceGUIRuntimeConfig:
        return controller.active_record.runtime

    def active_interaction_mode() -> NiceGUIInteractionMode:
        runtime = active_runtime()
        if (
            runtime.interaction_mode == "deep_task"
            and not controller.deep_task_mode_enabled()
        ):
            return "chat"
        return runtime.interaction_mode

    def secret_entry_enabled() -> bool:
        return controller.hosted_config.secret_entry_enabled

    def clear_provider_api_key() -> None:
        controller.clear_session_secret(PROVIDER_API_KEY_FIELD)
        refresh_all()

    def clear_tool_credential(name: str) -> None:
        controller.clear_session_secret(name)
        refresh_all()

    def clear_tool_url(name: str) -> None:
        active_runtime().tool_urls.pop(name, None)
        controller.save_active_session()
        refresh_all()

    def remember_recent_runtime_values(runtime: NiceGUIRuntimeConfig) -> None:
        """Persist non-secret runtime values for future sessions."""
        provider_key = runtime.provider.value
        if runtime.model_name:
            models = controller.preferences.recent_models.setdefault(provider_key, [])
            if runtime.model_name in models:
                models.remove(runtime.model_name)
            models.insert(0, runtime.model_name)
            del models[10:]
        if runtime.api_base_url:
            urls = controller.preferences.recent_base_urls.setdefault(provider_key, [])
            if runtime.api_base_url in urls:
                urls.remove(runtime.api_base_url)
            urls.insert(0, runtime.api_base_url)
            del urls[10:]
        if runtime.root_path:
            roots = controller.preferences.recent_roots
            if runtime.root_path in roots:
                roots.remove(runtime.root_path)
            roots.insert(0, runtime.root_path)
            del roots[10:]

    def base_url_options(runtime: NiceGUIRuntimeConfig) -> list[str]:
        options = list(
            controller.preferences.recent_base_urls.get(runtime.provider.value, [])
        )
        if runtime.api_base_url and runtime.api_base_url not in options:
            options.insert(0, runtime.api_base_url)
        return options

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

    def sync_settings_inputs() -> None:
        runtime = active_runtime()
        if provider_select is not None:
            provider_select.value = runtime.provider.value
        if model_input is not None:
            set_model_options(
                model_options_state["values"], selected=runtime.model_name
            )
        if mode_select is not None:
            mode_select.value = runtime.provider_mode_strategy.value
        if base_url_input is not None:
            base_url_input.set_options(
                base_url_options(runtime), value=runtime.api_base_url or None
            )
            base_url_input.value = runtime.api_base_url or ""
        if provider_api_key_input is not None:
            provider_api_key_input.value = ""
        if workspace_input is not None:
            workspace_input.value = runtime.root_path or ""
        if database_path_input is not None:
            database_path_input.value = str(controller.store.db_path)
        if dark_mode_switch is not None:
            dark_mode_switch.value = controller.preferences.theme_mode == "dark"
        if allow_network_switch is not None:
            allow_network_switch.value = runtime.allow_network
        if allow_filesystem_switch is not None:
            allow_filesystem_switch.value = runtime.allow_filesystem
        if allow_subprocess_switch is not None:
            allow_subprocess_switch.value = runtime.allow_subprocess
        if approval_select is not None:
            approval_select.value = [
                side_effect.value
                for side_effect in NICEGUI_APPROVAL_OPTIONS
                if side_effect in runtime.require_approval_for
            ]
        if tool_credentials_column is not None:
            render_tool_credentials()

    def open_settings_dialog() -> None:
        sync_settings_inputs()
        dialogs["settings"].open()

    def sync_information_security_inputs() -> None:
        config = active_runtime().protection
        if protection_enabled_switch is not None:
            protection_enabled_switch.value = config.enabled
        if protection_categories_input is not None:
            protection_categories_input.value = "\n".join(
                config.allowed_sensitivity_labels
            )
        corpus_directory = config.document_paths[0] if config.document_paths else ""
        if protection_corpus_input is not None:
            protection_corpus_input.value = corpus_directory
        if protection_corrections_input is not None:
            protection_corrections_input.value = (
                config.corrections_path
                or _default_protection_corrections_path(corpus_directory)
            )
        render_information_security_readiness(config)

    def open_information_security_dialog() -> None:
        sync_information_security_inputs()
        dialogs["information_security"].open()

    def render_information_security_readiness(config: ProtectionConfig) -> None:
        if protection_readiness_label is not None:
            protection_readiness_label.set_text(
                _protection_corpus_readiness_text(config)
            )
        if protection_issues_column is None:
            return
        protection_issues_column.clear()
        if not config.enabled or not config.document_paths:
            return
        report = inspect_protection_corpus(config)
        if not report.issues:
            return
        with protection_issues_column:
            for issue in report.issues[:5]:
                ui.label(issue.message).classes("text-xs llmt-muted")
            if len(report.issues) > 5:
                ui.label(f"{len(report.issues) - 5} more issue(s)").classes(
                    "text-xs llmt-muted"
                )

    def open_provider_dialog() -> None:
        runtime = active_runtime()
        provider_quick_select.value = runtime.provider.value
        base_url_quick_input.set_options(
            base_url_options(runtime), value=runtime.api_base_url or None
        )
        base_url_quick_input.value = runtime.api_base_url or ""
        provider_api_key_quick_input.value = ""
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

    def current_tool_capability_groups() -> dict[str, list[AssistantToolCapability]]:
        runtime = active_runtime()
        return build_tool_capabilities(
            tool_specs=build_assistant_available_tool_specs(),
            enabled_tools=set(runtime.enabled_tools),
            root_path=runtime.root_path,
            env=controller.effective_tool_env(
                runtime=runtime,
                session_id=controller.active_session_id,
            ),
            allow_network=runtime.allow_network,
            allow_filesystem=runtime.allow_filesystem,
            allow_subprocess=runtime.allow_subprocess,
            require_approval_for=set(runtime.require_approval_for),
        )

    def current_tool_capabilities_by_name() -> dict[str, AssistantToolCapability]:
        return {
            capability.tool_name: capability
            for capabilities in current_tool_capability_groups().values()
            for capability in capabilities
        }

    def tool_required_value_entries() -> list[tuple[str, list[str], bool, bool]]:
        entries: dict[str, set[str]] = {}
        for tool_name, spec in build_assistant_available_tool_specs().items():
            for secret_name in spec.required_secrets:
                entries.setdefault(secret_name, set()).add(tool_name)
        effective_env = controller.effective_tool_env(
            runtime=active_runtime(),
            session_id=controller.active_session_id,
        )
        return [
            (
                secret_name,
                sorted(tool_names),
                (
                    bool(active_runtime().tool_urls.get(secret_name))
                    if _is_tool_url_setting(secret_name)
                    else controller.has_session_secret(secret_name)
                )
                or bool(effective_env.get(secret_name)),
                _is_tool_url_setting(secret_name),
            )
            for secret_name, tool_names in sorted(entries.items())
        ]

    def render_tool_credentials() -> None:
        if tool_credentials_column is None:
            return
        tool_credentials_column.clear()
        tool_credential_inputs.clear()
        tool_url_inputs.clear()
        entries = tool_required_value_entries()
        with tool_credentials_column:
            if not entries:
                ui.label("No tool access settings are required.").classes(
                    "text-xs llmt-muted"
                )
                return
            if not secret_entry_enabled():
                ui.label(
                    controller.hosted_config.insecure_hosted_warning
                    or "Secret entry is disabled for this hosted session."
                ).classes("text-xs text-negative")
            for secret_name, tool_names, is_set, is_url in entries:
                with ui.row().classes("llmt-credential-row w-full items-end gap-2"):
                    with ui.column().classes("gap-0 min-w-0"):
                        ui.label(secret_name).classes("text-sm llmt-code")
                        tooltip_text = "Required by: " + ", ".join(tool_names)
                        ui.label(f"{len(tool_names)} tool(s)").classes(
                            "text-xs llmt-muted"
                        )
                    if is_url:
                        url_input = ui.input(
                            "URL",
                            value=active_runtime().tool_urls.get(secret_name, ""),
                        ).classes("grow")
                        tool_url_inputs[secret_name] = url_input
                    else:
                        credential_input = (
                            ui.input(
                                "Credential",
                                placeholder=(
                                    "Stored for this session"
                                    if is_set
                                    else "Paste value"
                                ),
                            )
                            .props("type=password autocomplete=off")
                            .classes("grow")
                        )
                        if not secret_entry_enabled():
                            credential_input.disable()
                        tool_credential_inputs[secret_name] = credential_input
                    badge = ui.icon("check" if is_set else "block").classes("q-pb-sm")
                    if not is_set:
                        badge.classes("llmt-tool-chip-blocked")
                    ui.tooltip(tooltip_text).props("delay=700")
                    ui.button(
                        icon="clear",
                        on_click=lambda _event, name=secret_name, url=is_url: (
                            clear_tool_url(name) if url else clear_tool_credential(name)
                        ),
                    ).props("flat round dense")

    def render_tool_menu() -> None:
        if tool_menu_column is None:
            return
        tool_menu_column.clear()
        groups = current_tool_capability_groups()
        summaries = build_tool_group_capability_summaries(groups)
        mode_locked = controller.interaction_mode_locked()
        with tool_menu_column:
            ui.label(
                "Choose mode before the first message. Tool changes apply to the next message."
            ).classes("llmt-tool-menu-note llmt-muted")
            with ui.expansion("Mode", icon="tune", value=True).classes("w-full"):
                modes: list[NiceGUIInteractionMode] = ["chat"]
                if controller.deep_task_mode_enabled():
                    modes.append("deep_task")
                for mode in modes:
                    label = INTERACTION_MODE_LABELS[mode]
                    selected = active_interaction_mode() == mode
                    button = ui.button(
                        label,
                        icon="radio_button_checked"
                        if selected
                        else "radio_button_unchecked",
                        on_click=lambda _event, selected_mode=mode: (
                            set_interaction_mode(selected_mode)
                        ),
                    ).props("flat dense no-caps align=left")
                    button.classes("llmt-tool-name")
                    if mode_locked:
                        button.disable()
                    with button:
                        tooltip = INTERACTION_MODE_TOOLTIPS[mode]
                        if mode_locked:
                            tooltip += " Mode is locked after the first message."
                        ui.tooltip(tooltip).props("delay=700")
            with ui.expansion("Tools", icon="construction", value=True).classes(
                "w-full"
            ):
                for group_name, capabilities in groups.items():
                    summary = summaries[group_name]
                    all_enabled = bool(capabilities) and all(
                        capability.enabled for capability in capabilities
                    )
                    some_enabled = any(
                        capability.enabled for capability in capabilities
                    )
                    icon = (
                        "check_box"
                        if all_enabled
                        else "indeterminate_check_box"
                        if some_enabled
                        else "check_box_outline_blank"
                    )
                    with ui.column().classes("llmt-tool-group-header w-full gap-1"):
                        with ui.row().classes(
                            "w-full items-center justify-between no-wrap"
                        ):
                            ui.button(
                                group_name,
                                icon=icon,
                                on_click=lambda _event, name=group_name: (
                                    toggle_tool_group(name)
                                ),
                            ).props("flat dense no-caps align=left").classes("grow")
                            ui.label(
                                f"{summary.exposed_tools}/{summary.total_tools}"
                            ).classes("text-xs llmt-muted")
                        for capability in capabilities:
                            icon_name = (
                                _selected_tool_icon(capability)
                                if capability.enabled
                                else None
                            )
                            button = ui.button(
                                capability.tool_name,
                                icon=icon_name,
                                on_click=lambda _event, name=capability.tool_name: (
                                    toggle_runtime_tool(name)
                                ),
                            ).props("flat dense no-caps align=left")
                            button.classes(
                                "llmt-tool-name "
                                + (
                                    _selected_tool_chip_classes(capability)
                                    if capability.enabled
                                    else ""
                                )
                            )
                            with button:
                                ui.tooltip(_tool_capability_tooltip(capability)).props(
                                    "delay=700"
                                )

    def render_selected_tools() -> None:
        if selected_tools_row is None:
            return
        selected_tools_row.clear()
        runtime = active_runtime()
        capabilities_by_name = current_tool_capabilities_by_name()
        selected = [
            capabilities_by_name[tool_name]
            for tool_name in sorted(set(runtime.enabled_tools))
            if tool_name in capabilities_by_name
        ]
        with selected_tools_row:
            ui.label("Mode").classes("text-xs llmt-muted")
            interaction_mode = active_interaction_mode()
            mode_label = INTERACTION_MODE_LABELS[interaction_mode]
            mode_button = ui.button(
                mode_label,
                icon=(
                    "manage_search"
                    if interaction_mode == "deep_task"
                    else "chat_bubble_outline"
                ),
                on_click=lambda _event: render_tool_menu(),
            ).props("flat dense no-caps")
            mode_button.classes("llmt-tool-chip llmt-tool-chip-enabled")
            if controller.interaction_mode_locked():
                mode_button.disable()
            with mode_button:
                tooltip = INTERACTION_MODE_TOOLTIPS[interaction_mode]
                if controller.interaction_mode_locked():
                    tooltip += " Mode is locked after the first message."
                ui.tooltip(tooltip).props("delay=700")
            if selected:
                ui.label("Tools").classes("text-xs llmt-muted")
                for capability in selected:
                    button = ui.button(
                        capability.tool_name,
                        icon=_selected_tool_icon(capability),
                        on_click=lambda _event, name=capability.tool_name: (
                            toggle_runtime_tool(name)
                        ),
                    ).props("flat dense no-caps")
                    button.classes(_selected_tool_chip_classes(capability))
                    with button:
                        ui.tooltip(_tool_capability_tooltip(capability)).props(
                            "delay=700"
                        )

    def set_interaction_mode(mode: NiceGUIInteractionMode) -> None:
        if controller.set_interaction_mode(mode):
            render_selected_tools()
            render_tool_menu()
        elif mode == "deep_task" and not controller.deep_task_mode_enabled():
            ui.notify(
                "Deep Task mode is disabled by the administrator.", type="warning"
            )
        else:
            ui.notify("Mode is locked after the first message.", type="warning")

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
        render_tool_menu()

    def toggle_tool_group(group_name: str) -> None:
        runtime = active_runtime()
        groups = current_tool_capability_groups()
        group_tool_names = {
            capability.tool_name for capability in groups.get(group_name, [])
        }
        if not group_tool_names:
            return
        enabled = set(runtime.enabled_tools)
        if group_tool_names.issubset(enabled):
            enabled.difference_update(group_tool_names)
        else:
            enabled.update(group_tool_names)
        runtime.enabled_tools = sorted(enabled)
        controller.save_active_session()
        render_selected_tools()
        render_tool_menu()

    def set_workspace_browser_target_value(path: str) -> None:
        if workspace_browser_target["value"] == "database":
            current = Path(str(database_path_input.value or controller.store.db_path))
            filename = current.name or "chat.sqlite3"
            database_path_input.value = str(Path(path) / filename)
        elif workspace_browser_target["value"] == "quick":
            workspace_quick_input.value = path
        elif workspace_browser_target["value"] == "protection_corpus":
            protection_corpus_input.value = path
            if not str(protection_corrections_input.value or "").strip():
                protection_corrections_input.value = (
                    _default_protection_corrections_path(path)
                )
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
        information_security_button.set_text(
            _format_information_security_label(runtime.protection)
        )
        status_chip.set_text(turn_state.status_text or "ready")
        render_selected_tools()
        render_tool_menu()
        render_runtime_summary(runtime)
        composer_action_button.props(
            "icon="
            + _composer_action_icon(busy=turn_state.busy)
            + " color="
            + ("negative" if turn_state.busy else "primary")
        )
        sync_settings_inputs()
        render_admin_user_list()

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
                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label(f"{item.kind} v{item.version}").classes(
                                "text-xs llmt-muted"
                            )
                            ui.label(
                                _format_workbench_duration(item.duration_seconds)
                            ).classes("text-xs llmt-muted")
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

    def render_protection_challenge() -> None:
        prompt = controller.pending_protection_prompt()
        if prompt is None:
            protection_dialog_state["challenge_key"] = None
            dialogs["protection_challenge"].close()
            return
        challenge_key = f"{prompt.created_at}:{prompt.original_user_message}"
        protection_predicted_label.set_text(
            prompt.predicted_sensitivity_label or "Unspecified"
        )
        protection_reasoning_label.set_text(prompt.reasoning)
        protection_refs_label.set_text(_referenced_document_text(prompt))
        protection_original_request.set_text(prompt.original_user_message)
        if protection_dialog_state["challenge_key"] != challenge_key:
            protection_overrule_category_input.value = (
                prompt.predicted_sensitivity_label or ""
            )
            protection_overrule_rationale_input.value = ""
            protection_dialog_state["challenge_key"] = challenge_key
        dialogs["protection_challenge"].open()

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
        controller.save_preferences()
        refresh_all()

    def toggle_workbench() -> None:
        controller.preferences.workbench_open = (
            not controller.preferences.workbench_open
        )
        controller.save_preferences()
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

    def logout_hosted_user() -> None:
        clear_hosted_session(controller.auth_provider)
        ui.navigate.reload()

    def open_admin_dialog() -> None:
        if not _is_admin_user(controller.current_user):
            ui.notify("Admin access is required.", type="negative")
            return
        if admin_deep_task_switch is not None:
            admin_deep_task_switch.value = controller.deep_task_mode_enabled()
        render_admin_user_list()
        dialogs["admin"].open()

    def set_admin_deep_task_enabled(event: Any) -> None:
        if not _is_admin_user(controller.current_user):
            ui.notify("Admin access is required.", type="negative")
            return
        controller.set_deep_task_mode_enabled(bool(event.value))
        render_selected_tools()
        render_tool_menu()
        ui.notify("Deep Task mode updated.")

    def render_admin_user_list() -> None:
        if admin_user_list_column is None:
            return
        admin_user_list_column.clear()
        if controller.auth_provider is None or not _is_admin_user(
            controller.current_user
        ):
            return
        with admin_user_list_column:
            for user in controller.store.list_users():
                with ui.column().classes("w-full gap-1 llmt-credential-row"):
                    with ui.row().classes(
                        "w-full items-center justify-between no-wrap"
                    ):
                        with ui.column().classes("gap-0"):
                            ui.label(user.username).classes("text-sm")
                            status = str(user.role)
                            if user.disabled:
                                status += " | disabled"
                            ui.label(status).classes("text-xs llmt-muted")
                        if _can_admin_disable_user(controller.current_user, user):
                            ui.button(
                                icon="block" if not user.disabled else "check",
                                on_click=lambda _event, target=user: (
                                    toggle_user_disabled(
                                        target.user_id, not target.disabled
                                    )
                                ),
                            ).props("flat round dense")
                        else:
                            ui.icon("shield").classes("llmt-muted")
                    with ui.row().classes("w-full items-end gap-2 no-wrap"):
                        password_input = (
                            ui.input("New password")
                            .props("type=password autocomplete=off")
                            .classes("grow")
                        )
                        ui.button(
                            icon="key",
                            on_click=lambda _event, target=user, field=password_input: (
                                reset_hosted_user_password(target.user_id, field)
                            ),
                        ).props("flat round dense")

    def toggle_user_disabled(user_id: str, disabled: bool) -> None:
        if not _is_admin_user(controller.current_user):
            ui.notify("Admin access is required.", type="negative")
            return
        if user_id == getattr(controller.current_user, "user_id", None):
            ui.notify("You cannot disable your own account.", type="negative")
            return
        controller.store.set_user_disabled(user_id, disabled)
        render_admin_user_list()

    def reset_hosted_user_password(user_id: str, password_input: Any) -> None:
        if controller.auth_provider is None or not _is_admin_user(
            controller.current_user
        ):
            ui.notify("Admin access is required.", type="negative")
            return
        try:
            controller.auth_provider.reset_password(
                user_id=user_id,
                password=str(password_input.value or ""),
            )
            password_input.value = ""
            ui.notify("Password reset.")
        except Exception as exc:  # pragma: no cover - UI guard
            ui.notify(f"Could not reset password: {exc}", type="negative")

    def create_hosted_user() -> None:
        if controller.auth_provider is None or not _is_admin_user(
            controller.current_user
        ):
            ui.notify("Admin access is required.", type="negative")
            return
        try:
            controller.auth_provider.create_user(
                username=str(admin_create_username_input.value or ""),
                password=str(admin_create_password_input.value or ""),
                role=str(admin_create_role_select.value or "user"),
            )
            admin_create_username_input.value = ""
            admin_create_password_input.value = ""
            render_admin_user_list()
            ui.notify("User created.")
        except Exception as exc:  # pragma: no cover - UI guard
            ui.notify(f"Could not create user: {exc}", type="negative")

    def apply_settings() -> None:
        runtime = active_runtime()
        runtime.provider = ProviderPreset(str(provider_select.value))
        runtime.model_name = str(model_input.value or runtime.model_name)
        runtime.provider_mode_strategy = ProviderModeStrategy(str(mode_select.value))
        runtime.api_base_url = str(base_url_input.value or "").strip() or None
        provider_api_key = str(provider_api_key_input.value or "").strip()
        if provider_api_key and secret_entry_enabled():
            controller.set_session_secret(PROVIDER_API_KEY_FIELD, provider_api_key)
            provider_api_key_input.value = ""
        for secret_name, credential_input in tool_credential_inputs.items():
            secret_value = str(credential_input.value or "").strip()
            if secret_value and secret_entry_enabled():
                controller.set_session_secret(secret_name, secret_value)
                credential_input.value = ""
        runtime.tool_urls = {
            name: str(url_input.value or "").strip()
            for name, url_input in tool_url_inputs.items()
            if str(url_input.value or "").strip()
        }
        runtime.root_path = str(workspace_input.value or "").strip() or None
        requested_db_path = str(database_path_input.value or "").strip()
        runtime.allow_network = bool(allow_network_switch.value)
        runtime.allow_filesystem = bool(allow_filesystem_switch.value)
        runtime.allow_subprocess = bool(allow_subprocess_switch.value)
        runtime.require_approval_for = {
            SideEffectClass(value) for value in (approval_select.value or [])
        }
        controller.preferences.theme_mode = (
            "dark" if bool(dark_mode_switch.value) else "light"
        )
        remember_recent_runtime_values(runtime)
        if (
            controller.current_user is None
            and requested_db_path
            and Path(requested_db_path).expanduser() != controller.store.db_path
        ):
            controller.switch_database(Path(requested_db_path).expanduser())
        else:
            controller.save_active_session()
        controller.save_preferences()
        refresh_all()

    def apply_settings_and_close() -> None:
        apply_settings()
        dialogs["settings"].close()

    def apply_information_security_settings() -> None:
        protection = active_runtime().protection
        corpus_directory = str(protection_corpus_input.value or "").strip()
        corrections_path = str(protection_corrections_input.value or "").strip()
        protection.enabled = bool(protection_enabled_switch.value)
        protection.allowed_sensitivity_labels = _parse_information_security_categories(
            str(protection_categories_input.value or "")
        )
        protection.document_paths = [corpus_directory] if corpus_directory else []
        protection.corrections_path = (
            corrections_path
            or _default_protection_corrections_path(corpus_directory)
            or None
        )
        controller.save_active_session()
        refresh_all()

    def apply_information_security_settings_and_close() -> None:
        apply_information_security_settings()
        dialogs["information_security"].close()

    def accept_protection_ruling() -> None:
        error = controller.submit_protection_accept()
        if error:
            ui.notify(error, type="negative")
            return
        dialogs["protection_challenge"].close()
        refresh_all()

    def overrule_protection_ruling() -> None:
        error = controller.submit_protection_overrule(
            expected_sensitivity_label=str(
                protection_overrule_category_input.value or ""
            ),
            rationale=str(protection_overrule_rationale_input.value or ""),
        )
        if error:
            ui.notify(error, type="negative")
            return
        dialogs["protection_challenge"].close()
        refresh_all()

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
        provider_api_key = str(provider_api_key_input.value or "").strip()
        current_model = str(model_input.value or active_runtime().model_name)
        models = _discover_model_names(
            provider=provider,
            base_url=base_url,
            api_key_env_var=active_runtime().api_key_env_var,
            api_key=provider_api_key
            or controller.provider_api_key(session_id=controller.active_session_id),
            allow_env_api_key=False,
        )
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
            api_key_env_var=runtime.api_key_env_var,
            api_key=controller.provider_api_key(
                session_id=controller.active_session_id
            ),
            allow_env_api_key=False,
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
        provider_api_key = str(provider_api_key_quick_input.value or "").strip()
        if provider_api_key and secret_entry_enabled():
            controller.set_session_secret(PROVIDER_API_KEY_FIELD, provider_api_key)
            provider_api_key_quick_input.value = ""
        remember_recent_runtime_values(runtime)
        controller.save_active_session()
        dialogs["provider_settings"].close()
        refresh_all()

    def apply_model_settings_and_close() -> None:
        runtime = active_runtime()
        runtime.model_name = str(model_quick_select.value or runtime.model_name)
        remember_recent_runtime_values(runtime)
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
        remember_recent_runtime_values(runtime)
        controller.save_active_session()
        dialogs["workspace_settings"].close()
        refresh_all()

    def workspace_start_path() -> Path:
        if workspace_browser_target["value"] == "database":
            source_input = database_path_input
        elif workspace_browser_target["value"] == "quick":
            source_input = workspace_quick_input
        elif workspace_browser_target["value"] == "protection_corpus":
            source_input = protection_corpus_input
        else:
            source_input = workspace_input
        fallback = (
            str(controller.store.db_path)
            if workspace_browser_target["value"] == "database"
            else (
                active_runtime().protection.document_paths[0]
                if workspace_browser_target["value"] == "protection_corpus"
                and active_runtime().protection.document_paths
                else active_runtime().root_path
            )
        )
        raw = str(source_input.value or fallback or Path.cwd())
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
        if workspace_browser_title is not None:
            title = "Select Workspace Root"
            if target == "database":
                title = "Select Database Folder"
            elif target == "protection_corpus":
                title = "Select Protection Corpus"
            workspace_browser_title.set_text(title)
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
                with ui.row().classes("llmt-header-security items-center"):
                    information_security_button = (
                        ui.button(
                            _format_information_security_label(
                                controller.active_record.runtime.protection
                            ),
                            on_click=open_information_security_dialog,
                        )
                        .props("flat no-caps")
                        .classes("llmt-info-security-button")
                    )
                with ui.row().classes("llmt-header-actions items-center gap-2"):
                    status_chip = ui.badge("ready").props("outline")
                    if controller.current_user is not None:
                        with (
                            ui.button(icon="account_circle").props("flat round"),
                            ui.menu().classes("llmt-account-menu"),
                        ):
                            ui.label(controller.current_user.username).classes(
                                "text-sm q-px-sm q-pt-sm"
                            )
                            ui.label(controller.current_user.role).classes(
                                "text-xs llmt-muted q-px-sm q-pb-sm"
                            )
                            if _is_admin_user(controller.current_user):
                                ui.menu_item("Admin", on_click=open_admin_dialog)
                            ui.menu_item("Log out", on_click=logout_hosted_user)
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
                        ui.menu().classes("llmt-tool-menu"),
                        ui.column().classes("w-full gap-1") as tool_menu_column,
                    ):
                        pass
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

    with ui.dialog() as admin_dialog, ui.card().classes("w-[560px] max-h-[88vh]"):
        dialogs["admin"] = admin_dialog
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Admin").classes("text-lg")
            ui.button(icon="close", on_click=admin_dialog.close).props("flat round")
        ui.label("Manage local NiceGUI users.").classes("text-sm llmt-muted")
        with ui.column().classes("llmt-settings-section w-full gap-2"):
            ui.label("Features").classes("text-base")
            admin_deep_task_switch = ui.switch(
                "Deep Task mode",
                value=controller.deep_task_mode_enabled(),
                on_change=set_admin_deep_task_enabled,
            ).classes("w-full")
            with admin_deep_task_switch:
                ui.tooltip(
                    "Shows Deep Task as a selectable chat mode before the first message."
                ).props("delay=700")
        with ui.column().classes("llmt-settings-section w-full gap-2"):
            ui.label("Create user").classes("text-base")
            with ui.row().classes("w-full items-end gap-2 no-wrap"):
                admin_create_username_input = ui.input("Username").classes("grow")
                admin_create_role_select = ui.select(
                    ["user", "admin"], value="user", label="Role"
                ).classes("w-[120px]")
            admin_create_password_input = (
                ui.input("Password")
                .props("type=password autocomplete=off")
                .classes("w-full")
            )
            ui.button("Create user", on_click=create_hosted_user).props("flat")
        with ui.column().classes("llmt-settings-section w-full gap-2"):
            ui.label("Users").classes("text-base")
            with (
                ui.scroll_area().classes("w-full max-h-[360px]"),
                ui.column().classes("w-full gap-1") as admin_user_list_column,
            ):
                pass

    with ui.dialog() as settings_dialog, ui.card().classes("w-[520px] max-h-[88vh]"):
        dialogs["settings"] = settings_dialog
        ui.label("Settings").classes("text-lg")
        with (
            ui.scroll_area().classes("llmt-settings-body"),
            ui.column().classes("w-full gap-2 q-pr-sm"),
        ):
            dark_mode_switch = ui.switch(
                "Dark mode",
                value=controller.preferences.theme_mode == "dark",
            ).classes("w-full")
            provider_select = ui.select(
                NICEGUI_PROVIDER_OPTIONS,
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
            base_url_input = (
                ui.select(
                    base_url_options(controller.active_record.runtime),
                    label="Base URL",
                    with_input=True,
                    new_value_mode="add-unique",
                )
                .props("clearable")
                .classes("w-full")
            )
            with ui.row().classes("w-full items-end gap-2 no-wrap"):
                provider_api_key_input = (
                    ui.input(
                        "Provider API key",
                        placeholder="Paste for this session; leave blank to keep",
                    )
                    .props("type=password autocomplete=off")
                    .classes("grow")
                )
                if not secret_entry_enabled():
                    provider_api_key_input.disable()
                ui.button(
                    icon="clear",
                    on_click=clear_provider_api_key,
                ).props("flat round")
            with ui.row().classes("w-full items-end gap-2 no-wrap"):
                workspace_input = ui.input("Workspace root").classes("grow")
                ui.button(
                    icon="folder_open",
                    on_click=lambda: open_workspace_browser(target="settings"),
                ).props("flat round color=primary")
            with ui.column().classes("llmt-settings-section w-full gap-1"):
                ui.label("Persistence").classes("text-base")
                with ui.row().classes("w-full items-end gap-2 no-wrap"):
                    database_path_input = ui.input("SQLite database").classes("grow")
                    if controller.current_user is not None:
                        database_path_input.disable()
                    ui.button(
                        icon="folder_open",
                        on_click=lambda: open_workspace_browser(target="database"),
                    ).props("flat round color=primary")
            with ui.column().classes("llmt-settings-section w-full gap-1"):
                ui.label("Session permissions").classes("text-base")
                allow_network_switch = ui.switch(
                    "Allow network tools",
                    value=controller.active_record.runtime.allow_network,
                ).classes("w-full")
                allow_filesystem_switch = ui.switch(
                    "Allow workspace file tools",
                    value=controller.active_record.runtime.allow_filesystem,
                ).classes("w-full")
                allow_subprocess_switch = ui.switch(
                    "Allow subprocess tools",
                    value=controller.active_record.runtime.allow_subprocess,
                ).classes("w-full")
                approval_select = (
                    ui.select(
                        NICEGUI_APPROVAL_LABELS,
                        label="Require approval for",
                        multiple=True,
                        value=[
                            side_effect.value
                            for side_effect in controller.active_record.runtime.require_approval_for
                            if side_effect in NICEGUI_APPROVAL_OPTIONS
                        ],
                    )
                    .props("use-chips")
                    .classes("w-full")
                )
            with ui.column().classes("llmt-settings-section w-full gap-1"):
                ui.label("Tool access").classes("text-base")
                with ui.column().classes("w-full gap-0") as tool_credentials_column:
                    pass
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=settings_dialog.close).props("flat")
            ui.button("Apply", on_click=apply_settings_and_close)

    with (
        ui.dialog() as information_security_dialog,
        ui.card().classes("w-[560px] max-h-[88vh]"),
    ):
        dialogs["information_security"] = information_security_dialog
        ui.label("Information Security Settings").classes("text-lg")
        with (
            ui.scroll_area().classes("llmt-settings-body"),
            ui.column().classes("w-full gap-2 q-pr-sm"),
        ):
            protection_enabled_switch = ui.switch(
                "Enable protection for this chat",
                value=controller.active_record.runtime.protection.enabled,
            ).classes("w-full")
            protection_categories_input = (
                ui.textarea(
                    "Allowed categories",
                    placeholder="TRIVIAL\nMINOR",
                    value="\n".join(
                        controller.active_record.runtime.protection.allowed_sensitivity_labels
                    ),
                )
                .props("autogrow outlined")
                .classes("w-full")
            )
            with ui.row().classes("w-full items-end gap-2 no-wrap"):
                protection_corpus_input = ui.input(
                    "Protection corpus directory"
                ).classes("grow")
                ui.button(
                    icon="folder_open",
                    on_click=lambda: open_workspace_browser(target="protection_corpus"),
                ).props("flat round color=primary")
            protection_corrections_input = ui.input(
                "Corrections sidecar path",
                placeholder="<corpus>/.llm-tools-protection-corrections.json",
            ).classes("w-full")
            protection_readiness_label = ui.label("").classes("text-sm")
            with ui.column().classes("w-full gap-1") as protection_issues_column:
                pass
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=information_security_dialog.close).props(
                "flat"
            )
            ui.button("Apply", on_click=apply_information_security_settings_and_close)

    with ui.dialog() as provider_settings_dialog, ui.card().classes("w-[460px]"):
        dialogs["provider_settings"] = provider_settings_dialog
        ui.label("Provider").classes("text-lg")
        provider_quick_select = ui.select(
            NICEGUI_PROVIDER_OPTIONS,
            label="Provider",
        ).classes("w-full")
        base_url_quick_input = (
            ui.select(
                base_url_options(controller.active_record.runtime),
                label="Base URL",
                with_input=True,
                new_value_mode="add-unique",
            )
            .props("clearable")
            .classes("w-full")
        )
        with ui.row().classes("w-full items-end gap-2 no-wrap"):
            provider_api_key_quick_input = (
                ui.input(
                    "Provider API key",
                    placeholder="Paste for this session; leave blank to keep",
                )
                .props("type=password autocomplete=off")
                .classes("grow")
            )
            if not secret_entry_enabled():
                provider_api_key_quick_input.disable()
            ui.button(
                icon="clear",
                on_click=clear_provider_api_key,
            ).props("flat round")
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
        workspace_browser_title = ui.label("Select Workspace Root").classes("text-lg")
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

    with (
        ui.dialog() as protection_challenge_dialog,
        ui.card().classes("w-[620px] max-h-[88vh]"),
    ):
        dialogs["protection_challenge"] = protection_challenge_dialog
        ui.label("Information Security Review").classes("text-lg")
        with (
            ui.scroll_area().classes("llmt-settings-body"),
            ui.column().classes("w-full gap-3 q-pr-sm"),
        ):
            with ui.column().classes("w-full gap-1"):
                ui.label("Predicted category").classes("text-xs llmt-muted")
                protection_predicted_label = ui.label("").classes("text-base")
            with ui.column().classes("w-full gap-1"):
                ui.label("Justification").classes("text-xs llmt-muted")
                protection_reasoning_label = ui.label("").classes("text-sm")
            with ui.column().classes("w-full gap-1"):
                ui.label("Referenced documents").classes("text-xs llmt-muted")
                protection_refs_label = ui.label("").classes("text-sm")
            with ui.column().classes("w-full gap-1"):
                ui.label("Original request").classes("text-xs llmt-muted")
                protection_original_request = ui.label("").classes("text-sm")
            with ui.column().classes("llmt-settings-section w-full gap-2"):
                ui.label("Overrule ruling").classes("text-base")
                protection_overrule_category_input = ui.input(
                    "Expected category"
                ).classes("w-full")
                protection_overrule_rationale_input = (
                    ui.textarea("Explanation")
                    .props("autogrow outlined")
                    .classes("w-full")
                )
        with ui.row().classes("justify-end w-full"):
            ui.button("Accept ruling", on_click=accept_protection_ruling).props("flat")
            ui.button("Overrule", on_click=overrule_protection_ruling).props(
                "color=primary"
            )

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
        db_key_file=args.db_key_file,
        user_key_file=args.user_key_file,
        host=args.host,
        port=args.port,
        show=not args.no_browser,
        auth_mode=args.auth_mode,
        public_base_url=args.public_base_url,
        tls_certfile=args.tls_certfile,
        tls_keyfile=args.tls_keyfile,
        allow_insecure_hosted_secrets=args.allow_insecure_hosted_secrets,
        hosted_secret_key_path=args.hosted_secret_key_path,
    )
    return 0


__all__ = [
    "build_nicegui_chat_ui",
    "build_parser",
    "clear_hosted_session",
    "main",
    "resolve_assistant_config",
    "resolve_root_argument",
    "run_nicegui_chat_app",
]

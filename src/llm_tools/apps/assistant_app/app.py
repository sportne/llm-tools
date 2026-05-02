"""Assistant app CLI and startup wiring for llm-tools."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from urllib.request import urlopen

from nicegui import app as nicegui_app

from llm_tools.apps.assistant_app import ui as _ui_module
from llm_tools.apps.assistant_app.auth import (
    LocalAuthProvider,
    ensure_secret_file,
    validate_hosted_startup,
)
from llm_tools.apps.assistant_app.controller import (
    NiceGUIChatController,
    default_runtime_config,
)
from llm_tools.apps.assistant_app.paths import expand_app_path
from llm_tools.apps.assistant_app.store import SQLiteNiceGUIChatStore, default_db_path
from llm_tools.apps.assistant_app.ui import (
    NICEGUI_APPROVAL_LABELS,
    NICEGUI_APPROVAL_OPTIONS,
    NICEGUI_PROVIDER_OPTIONS,
    AssistantUIRenderer,
    _account_menu_action_labels,
    _account_menu_identity_labels,
    _apply_branding_page_metadata,
    _available_windows_drive_roots,
    _branding_favicon_href,
    _branding_favicon_javascript,
    _branding_icon_uses_material,
    _can_admin_disable_user,
    _composer_action_icon,
    _context_capacity_meter_state,
    _default_protection_corrections_path,
    _ensure_information_security_category_catalog,
    _event_payload_text,
    _extract_model_names_from_models_payload,
    _first_nonempty_text,
    _format_information_security_category_catalog,
    _format_information_security_label,
    _format_information_security_level,
    _format_transcript_time,
    _format_workbench_duration,
    _is_admin_user,
    _is_tool_url_setting,
    _models_endpoint_url,
    _parse_information_security_categories,
    _parse_information_security_category_catalog,
    _parse_optional_positive_int_setting,
    _parse_positive_float_setting,
    _parse_positive_int_setting,
    _parse_temperature_setting,
    _protection_corpus_readiness_text,
    _provider_base_url_help_text,
    _provider_endpoint_menu_rows,
    _referenced_document_text,
    _runtime_summary_parts,
    _runtime_summary_text,
    _runtime_with_settings_values,
    _selected_tool_chip_classes,
    _selected_tool_groups,
    _selected_tool_icon,
    _session_token_estimate_text,
    _set_composer_draft,
    _settings_section_default_open,
    _sidebar_container_classes,
    _tool_capability_tooltip,
    _workbench_container_classes,
    build_assistant_ui,
    render_first_admin_page,
    render_login_page,
)
from llm_tools.apps.assistant_config import AssistantConfig, load_assistant_config

AUTH_DISABLED_WARNING = (
    "WARNING: Assistant app auth is disabled. Use --auth-mode local for normal local "
    "or hosted use."
)


def _discover_model_names(*args: Any, **kwargs: Any) -> list[str]:
    """Compatibility wrapper for tests that monkeypatch app.urlopen."""
    cast(Any, _ui_module).urlopen = urlopen
    return _ui_module._discover_model_names(*args, **kwargs)


def _sync_ui_compatibility_globals() -> None:
    """Sync facade-level monkeypatches into the extracted UI module."""
    ui_module = cast(Any, _ui_module)
    ui_module.nicegui_app = nicegui_app
    ui_module._apply_branding_page_metadata = _apply_branding_page_metadata
    ui_module.render_first_admin_page = render_first_admin_page
    ui_module.render_login_page = render_login_page
    ui_module.build_assistant_ui = build_assistant_ui


def _hosted_storage_values() -> tuple[str | None, str | None]:
    """Return hosted session storage values via the extracted UI helper."""
    _sync_ui_compatibility_globals()
    return _ui_module._hosted_storage_values()


def _set_hosted_storage_values(session_id: str, token: str) -> None:
    """Set hosted session storage values via the extracted UI helper."""
    _sync_ui_compatibility_globals()
    _ui_module._set_hosted_storage_values(session_id, token)


def _clear_hosted_storage_values() -> None:
    """Clear hosted session storage values via the extracted UI helper."""
    _sync_ui_compatibility_globals()
    _ui_module._clear_hosted_storage_values()


def clear_hosted_session(auth_provider: LocalAuthProvider | None) -> None:
    """Revoke and clear the current browser auth session."""
    _sync_ui_compatibility_globals()
    _ui_module.clear_hosted_session(auth_provider)


def render_hosted_nicegui_page(*args: Any, **kwargs: Any) -> None:
    """Render a hosted NiceGUI page through the extracted UI module."""
    _sync_ui_compatibility_globals()
    _ui_module.render_hosted_nicegui_page(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    """Build the assistant app CLI parser."""
    parser = argparse.ArgumentParser(
        prog="llm-tools-assistant",
        description="LLM Tools Assistant backed by encrypted SQLite persistence.",
    )
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--directory", dest="directory_override", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--provider-protocol")
    parser.add_argument("--model", type=str)
    parser.add_argument("--response-mode-strategy", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--api-base-url", type=str)
    parser.add_argument("--requires-bearer-token", action="store_true")
    parser.add_argument("--no-bearer-token", action="store_true")
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


def resolve_assistant_config(args: argparse.Namespace) -> AssistantConfig:
    """Resolve config file and CLI overrides."""
    base_config = (
        load_assistant_config(args.config)
        if args.config is not None
        else AssistantConfig()
    )
    raw = base_config.model_dump(mode="python")
    raw.setdefault("llm", {})
    raw.setdefault("session", {})
    raw.setdefault("tool_limits", {})
    if args.provider_protocol is not None:
        raw["llm"]["provider_protocol"] = args.provider_protocol
    if args.model is not None:
        raw["llm"]["selected_model"] = args.model
    if args.response_mode_strategy is not None:
        raw["llm"]["response_mode_strategy"] = args.response_mode_strategy
    if args.temperature is not None:
        raw["llm"]["temperature"] = args.temperature
    provider_connection = dict(raw["llm"].get("provider_connection") or {})
    if args.api_base_url is not None:
        provider_connection["api_base_url"] = args.api_base_url
    if args.requires_bearer_token:
        provider_connection["requires_bearer_token"] = True
    if args.no_bearer_token:
        provider_connection["requires_bearer_token"] = False
    raw["llm"]["provider_connection"] = provider_connection
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
    return AssistantConfig.model_validate(raw)


def resolve_root_argument(
    args: argparse.Namespace,
    config: AssistantConfig,
) -> Path | None:
    """Resolve the workspace root from CLI or config."""
    candidate = args.directory_override or args.directory
    if candidate is None:
        default_root = config.workspace.default_root
        if default_root is None:
            return None
        return expand_app_path(default_root).resolve()
    return expand_app_path(candidate).resolve()


def run_assistant_app(
    *,
    root_path: Path | None,
    config: AssistantConfig,
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
    """Render and run the assistant app."""
    from nicegui import ui

    store = SQLiteNiceGUIChatStore(
        expand_app_path(db_path) if db_path is not None else default_db_path(),
        db_key_file=expand_app_path(db_key_file) if db_key_file is not None else None,
        user_key_file=(
            expand_app_path(user_key_file) if user_key_file is not None else None
        ),
    )
    store.initialize()
    branding = store.load_admin_settings().branding
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
        storage_secret = ensure_secret_file(
            expand_app_path(str(startup.config.secret_key_path))
        )
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
        build_assistant_ui(controller)
    base_run_kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "reload": False,
        "show": show,
        "title": branding.app_name,
        "favicon": branding.favicon_svg,
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


def main(argv: Sequence[str] | None = None) -> int:
    """Console entrypoint for the assistant app."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_assistant_config(args)
    root_path = resolve_root_argument(args, config)
    if root_path is not None:
        default_runtime_config(config, root_path=root_path)
    run_assistant_app(
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
    "AssistantUIRenderer",
    "NICEGUI_APPROVAL_LABELS",
    "NICEGUI_APPROVAL_OPTIONS",
    "NICEGUI_PROVIDER_OPTIONS",
    "_account_menu_action_labels",
    "_account_menu_identity_labels",
    "_apply_branding_page_metadata",
    "_available_windows_drive_roots",
    "_branding_favicon_href",
    "_branding_favicon_javascript",
    "_branding_icon_uses_material",
    "_can_admin_disable_user",
    "_clear_hosted_storage_values",
    "_composer_action_icon",
    "_context_capacity_meter_state",
    "_default_protection_corrections_path",
    "_discover_model_names",
    "_ensure_information_security_category_catalog",
    "_event_payload_text",
    "_extract_model_names_from_models_payload",
    "_first_nonempty_text",
    "_format_information_security_category_catalog",
    "_format_information_security_label",
    "_format_information_security_level",
    "_format_transcript_time",
    "_format_workbench_duration",
    "_hosted_storage_values",
    "_is_admin_user",
    "_is_tool_url_setting",
    "_models_endpoint_url",
    "_parse_information_security_categories",
    "_parse_information_security_category_catalog",
    "_parse_optional_positive_int_setting",
    "_parse_positive_float_setting",
    "_parse_positive_int_setting",
    "_parse_temperature_setting",
    "_protection_corpus_readiness_text",
    "_provider_base_url_help_text",
    "_provider_endpoint_menu_rows",
    "_referenced_document_text",
    "_runtime_summary_parts",
    "_runtime_summary_text",
    "_runtime_with_settings_values",
    "_selected_tool_chip_classes",
    "_selected_tool_groups",
    "_selected_tool_icon",
    "_session_token_estimate_text",
    "_set_composer_draft",
    "_set_hosted_storage_values",
    "_settings_section_default_open",
    "_sidebar_container_classes",
    "_tool_capability_tooltip",
    "_workbench_container_classes",
    "build_assistant_ui",
    "build_parser",
    "clear_hosted_session",
    "main",
    "render_first_admin_page",
    "render_hosted_nicegui_page",
    "render_login_page",
    "resolve_assistant_config",
    "resolve_root_argument",
    "run_assistant_app",
]

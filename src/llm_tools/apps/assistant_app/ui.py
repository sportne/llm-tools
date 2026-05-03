"""NiceGUI assistant UI rendering helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from nicegui import app as nicegui_app

from llm_tools.apps.assistant_app.auth import LocalAuthProvider
from llm_tools.apps.assistant_app.controller import (
    PROVIDER_API_KEY_FIELD,
    NiceGUIChatController,
    SessionSecretState,
)
from llm_tools.apps.assistant_app.models import (
    AssistantBranding,
    NiceGUIHostedConfig,
    NiceGUIInteractionMode,
    NiceGUIRuntimeConfig,
    NiceGUIUser,
)
from llm_tools.apps.assistant_app.paths import expand_app_path, expanded_path_text
from llm_tools.apps.assistant_app.provider_endpoints import (
    COMMON_PROVIDER_ENDPOINTS,
)
from llm_tools.apps.assistant_app.store import SQLiteNiceGUIChatStore
from llm_tools.apps.assistant_config import AssistantConfig
from llm_tools.apps.assistant_tool_capabilities import (
    AssistantToolCapability,
    build_tool_capabilities,
    build_tool_group_capability_summaries,
)
from llm_tools.apps.chat_config import ProviderAuthScheme, ProviderProtocol
from llm_tools.apps.chat_presentation import final_response_details
from llm_tools.llm_providers import ResponseModeStrategy
from llm_tools.tool_api import SideEffectClass
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import (
    ChatSessionConfig,
    ChatTokenUsage,
    ProtectionCategory,
    ProtectionConfig,
    ProtectionPendingPrompt,
    inspect_protection_corpus,
)

NICEGUI_PROVIDER_OPTIONS = [
    ProviderProtocol.OPENAI_API.value,
    ProviderProtocol.OLLAMA_NATIVE.value,
    ProviderProtocol.ASK_SAGE_NATIVE.value,
]
NICEGUI_PROVIDER_AUTH_OPTIONS = [
    ProviderAuthScheme.NONE.value,
    ProviderAuthScheme.BEARER.value,
    ProviderAuthScheme.X_ACCESS_TOKENS.value,
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
    "WARNING: Assistant app auth is disabled. Use --auth-mode local for normal local "
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


def _branding_favicon_href(branding: AssistantBranding) -> str:
    """Return a browser-safe data URI for the configured favicon SVG."""
    return "data:image/svg+xml," + quote(branding.favicon_svg, safe="")


def _branding_favicon_javascript(branding: AssistantBranding) -> str:
    """Return JavaScript that updates the current browser tab favicon."""
    href_json = json.dumps(_branding_favicon_href(branding))
    return (
        "(() => {"
        ' document.querySelectorAll(\'link[rel~="icon"], link[rel="shortcut icon"]\').forEach((link) => link.remove());'
        " const link = document.createElement('link');"
        " link.rel = 'icon';"
        " link.type = 'image/svg+xml';"
        f" link.href = {href_json};"
        " document.head.appendChild(link);"
        "})()"
    )


def _branding_icon_uses_material(icon_name: str) -> bool:
    """Return whether a branding icon value looks like a Material icon name."""
    return bool(icon_name) and all(
        character.isascii() and (character.isalnum() or character in {"_", "-", ":"})
        for character in icon_name
    )


def _render_brand_icon(icon_name: str, classes: str) -> None:  # pragma: no cover
    """Render the brand icon as either a Material icon or a literal symbol."""
    from nicegui import ui

    if _branding_icon_uses_material(icon_name):
        ui.icon(icon_name).classes(classes)
    else:
        ui.label(icon_name).classes(f"{classes} llmt-brand-symbol")


def _apply_branding_page_metadata(
    branding: AssistantBranding, *, update_favicon: bool = False
) -> None:
    """Apply browser tab metadata for one rendered page."""
    from nicegui import ui

    ui.page_title(branding.app_name)
    if update_favicon:
        ui.run_javascript(_branding_favicon_javascript(branding))


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


def _session_token_estimate_text(token_usage: ChatTokenUsage | None) -> str:
    """Return compact session token usage for the composer summary."""
    token_count = 0
    if token_usage is not None:
        usage_count = (
            token_usage.session_tokens
            if token_usage.session_tokens is not None
            else token_usage.total_tokens
        )
        if usage_count is None:
            token_count = token_usage.input_tokens + token_usage.output_tokens
        else:
            token_count = usage_count
    return f"~{token_count:,} tokens"


@dataclass(frozen=True)
class ContextCapacityMeterState:
    """UI state for the chat context capacity meter."""

    used_tokens: int
    limit_tokens: int
    remaining_tokens: int
    used_ratio: float
    used_percent: int
    compacting: bool
    track_classes: str
    fill_style: str
    tooltip: str


def _context_capacity_meter_state(
    runtime: NiceGUIRuntimeConfig,
    token_usage: ChatTokenUsage | None = None,
    *,
    status_text: str = "",
) -> ContextCapacityMeterState:
    """Return display state for the context capacity meter."""
    limit_tokens = max(1, runtime.session_config.max_context_tokens)
    used_tokens = 0
    if token_usage is not None and token_usage.active_context_tokens is not None:
        used_tokens = max(0, token_usage.active_context_tokens)
    used_ratio = min(used_tokens / limit_tokens, 1.0)
    used_percent = round(used_ratio * 100)
    remaining_tokens = max(limit_tokens - used_tokens, 0)
    compacting = status_text.strip().lower() == "compacting context"
    track_classes = "llmt-context-meter-track"
    if compacting:
        track_classes += " compacting"
    tooltip = (
        f"Context used: ~{used_tokens:,} of ~{limit_tokens:,} tokens "
        f"({used_percent}%). ~{remaining_tokens:,} tokens remain before compaction."
    )
    return ContextCapacityMeterState(
        used_tokens=used_tokens,
        limit_tokens=limit_tokens,
        remaining_tokens=remaining_tokens,
        used_ratio=used_ratio,
        used_percent=used_percent,
        compacting=compacting,
        track_classes=track_classes,
        fill_style=f"width: {used_ratio * 100:.1f}%;",
        tooltip=tooltip,
    )


def _runtime_summary_parts(
    runtime: NiceGUIRuntimeConfig,
    token_usage: ChatTokenUsage | None = None,
) -> list[tuple[str, str]]:
    """Return compact clickable runtime metadata labels."""
    root = runtime.root_path or "No workspace"
    endpoint = runtime.provider_connection.api_base_url or "No endpoint"
    model = runtime.selected_model or "No model"
    credential = (
        "credential needed"
        if runtime.provider_connection.auth_scheme.requires_secret()
        else "no credential"
    )
    parts = [
        ("provider_protocol", runtime.provider_protocol.value),
        ("endpoint", endpoint),
        ("credential", credential),
        ("model", model),
        ("mode", runtime.response_mode_strategy.value),
        ("workspace", root),
    ]
    if runtime.show_token_usage:
        parts.append(("tokens", _session_token_estimate_text(token_usage)))
    return parts


def _runtime_summary_text(
    runtime: NiceGUIRuntimeConfig,
    token_usage: ChatTokenUsage | None = None,
) -> str:
    """Return compact runtime metadata for tests and text fallbacks."""
    return " | ".join(
        value for _label, value in _runtime_summary_parts(runtime, token_usage)
    )


def _composer_action_icon(*, busy: bool) -> str:
    """Return the icon for the primary composer action."""
    return "stop" if busy else "send"


def _set_composer_draft(
    text: str,
    composer_state: dict[str, str],
    composer_input: Any | None,
) -> None:
    """Set composer draft text without submitting it."""
    composer_state["text"] = text
    if composer_input is None:
        return
    composer_input.value = text
    run_method = getattr(composer_input, "run_method", None)
    if callable(run_method):
        run_method("focus")


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


_SETTINGS_SECTION_DEFAULT_OPEN = {
    "Connection": True,
    "Workspace": True,
    "Display": False,
    "Persistence": False,
    "Session permissions": False,
    "Chat limits": False,
    "Tool limits": False,
    "Deep Task limits": False,
    "Tool credentials": False,
}


def _account_menu_action_labels(user: NiceGUIUser | None) -> list[str]:
    """Return account menu action labels in display order."""
    labels = ["Settings"]
    if _is_admin_user(user):
        labels.append("Admin")
    if user is not None:
        labels.append("Log out")
    return labels


def _account_menu_identity_labels(user: NiceGUIUser | None) -> tuple[str, str]:
    """Return account menu identity text for authenticated and no-auth modes."""
    if user is None:
        return ("Development mode", "no auth")
    return (user.username, user.role)


def _settings_section_default_open(section_name: str) -> bool:
    """Return whether a settings section should open by default."""
    return _SETTINGS_SECTION_DEFAULT_OPEN.get(section_name, False)


def _setting_text(value: object) -> str:
    return str(value or "").strip()


def _parse_positive_int_setting(label: str, value: object) -> int:
    """Parse a required positive integer setting."""
    raw = _setting_text(value)
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return parsed


def _parse_optional_positive_int_setting(label: str, value: object) -> int | None:
    """Parse an optional positive integer setting."""
    raw = _setting_text(value)
    if not raw:
        return None
    return _parse_positive_int_setting(label, raw)


def _parse_positive_float_setting(label: str, value: object) -> float:
    """Parse a required positive float setting."""
    raw = _setting_text(value)
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be a positive number.") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive number.")
    return parsed


def _parse_temperature_setting(value: object) -> float:
    """Parse a model temperature setting."""
    raw = _setting_text(value)
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise ValueError("Temperature must be a number between 0 and 1.") from exc
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError("Temperature must be a number between 0 and 1.")
    return parsed


def _runtime_with_settings_values(
    runtime: NiceGUIRuntimeConfig, values: Mapping[str, object]
) -> NiceGUIRuntimeConfig:
    """Return a runtime copy with advanced settings parsed from UI values."""
    session_config = ChatSessionConfig(
        max_context_tokens=_parse_positive_int_setting(
            "Max context tokens", values.get("max_context_tokens")
        ),
        max_tool_round_trips=_parse_positive_int_setting(
            "Max tool rounds", values.get("max_tool_round_trips")
        ),
        max_tool_calls_per_round=_parse_positive_int_setting(
            "Max tool calls per round", values.get("max_tool_calls_per_round")
        ),
        max_total_tool_calls_per_turn=_parse_positive_int_setting(
            "Max total tool calls per turn",
            values.get("max_total_tool_calls_per_turn"),
        ),
    )
    tool_limits = ToolLimits(
        max_entries_per_call=_parse_positive_int_setting(
            "Max entries per call", values.get("max_entries_per_call")
        ),
        max_recursive_depth=_parse_positive_int_setting(
            "Max recursive depth", values.get("max_recursive_depth")
        ),
        max_files_scanned=_parse_positive_int_setting(
            "Max files scanned", values.get("max_files_scanned")
        ),
        max_search_matches=_parse_positive_int_setting(
            "Max search matches", values.get("max_search_matches")
        ),
        max_read_lines=_parse_positive_int_setting(
            "Max read lines", values.get("max_read_lines")
        ),
        max_read_input_bytes=_parse_positive_int_setting(
            "Max read input bytes", values.get("max_read_input_bytes")
        ),
        max_file_size_characters=_parse_positive_int_setting(
            "Max file size characters", values.get("max_file_size_characters")
        ),
        max_read_file_chars=_parse_optional_positive_int_setting(
            "Max read file chars", values.get("max_read_file_chars")
        ),
        max_tool_result_chars=_parse_positive_int_setting(
            "Max tool result chars", values.get("max_tool_result_chars")
        ),
    )
    research = runtime.research.model_copy(
        update={
            "default_max_turns": _parse_positive_int_setting(
                "Deep Task max turns", values.get("deep_task_max_turns")
            ),
            "default_max_tool_invocations": _parse_optional_positive_int_setting(
                "Deep Task max tool invocations",
                values.get("deep_task_max_tool_invocations"),
            ),
            "default_max_elapsed_seconds": _parse_optional_positive_int_setting(
                "Deep Task max elapsed seconds",
                values.get("deep_task_max_elapsed_seconds"),
            ),
            "include_replay_by_default": bool(values.get("deep_task_include_replay")),
        }
    )
    return runtime.model_copy(
        update={
            "temperature": _parse_temperature_setting(values.get("temperature")),
            "timeout_seconds": _parse_positive_float_setting(
                "Provider timeout seconds", values.get("timeout_seconds")
            ),
            "show_token_usage": bool(values.get("show_token_usage")),
            "show_footer_help": bool(values.get("show_footer_help")),
            "inspector_open": bool(values.get("inspector_open")),
            "session_config": session_config,
            "tool_limits": tool_limits,
            "research": research,
        }
    )


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


def _format_transcript_time(created_at: str | None) -> str:
    """Return compact local time text for one transcript entry."""
    if not created_at:
        return ""
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.strftime("%H:%M")


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


def _format_information_security_category_catalog(
    categories: Sequence[ProtectionCategory],
) -> str:
    """Return compact editable category catalog text."""
    lines: list[str] = []
    for category in categories:
        line = category.label
        if category.aliases:
            line += ": " + ", ".join(category.aliases)
        if category.description:
            line += " | " + category.description
        if category.examples:
            line += " | " + "; ".join(category.examples)
        lines.append(line)
    return "\n".join(lines)


def _parse_information_security_category_catalog(
    value: str,
) -> list[ProtectionCategory]:
    """Parse compact category catalog text.

    Each non-empty line is:
    LABEL: alias 1, alias 2 | optional description | example 1; example 2
    """
    categories: list[ProtectionCategory] = []
    seen: set[str] = set()
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        label_and_aliases, *optional_parts = [part.strip() for part in line.split("|")]
        label, separator, raw_aliases = label_and_aliases.partition(":")
        label = label.strip()
        if not label:
            continue
        key = label.casefold()
        if key in seen:
            continue
        aliases = (
            [alias.strip() for alias in raw_aliases.split(",") if alias.strip()]
            if separator
            else []
        )
        description = optional_parts[0] if optional_parts else None
        examples = (
            [
                example.strip()
                for example in optional_parts[1].replace(",", ";").split(";")
                if example.strip()
            ]
            if len(optional_parts) > 1
            else []
        )
        categories.append(
            ProtectionCategory(
                label=label,
                aliases=aliases,
                description=description,
                examples=examples,
            )
        )
        seen.add(key)
    return categories


def _ensure_information_security_category_catalog(
    *,
    categories: list[ProtectionCategory],
    allowed_labels: list[str],
) -> list[ProtectionCategory]:
    """Ensure every allowed label has a canonical category entry."""
    existing = {category.label.casefold() for category in categories}
    normalized = list(categories)
    for label in allowed_labels:
        key = label.casefold()
        if key not in existing:
            normalized.append(ProtectionCategory(label=label))
            existing.add(key)
    return normalized


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
    return str(expand_app_path(cleaned) / PROTECTION_CORRECTIONS_FILENAME)


def _available_windows_drive_roots(
    path_exists: Callable[[Path], bool] | None = None,
) -> list[Path]:
    """Return currently available Windows drive roots."""
    exists = path_exists or (lambda path: path.exists())
    roots: list[Path] = []
    for codepoint in range(ord("A"), ord("Z") + 1):
        root = Path(f"{chr(codepoint)}:\\")
        if exists(root):
            roots.append(root)
    return roots


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
        parts = [f"Ready: {report.usable_document_count} document(s)"]
        if report.converted_document_count:
            parts.append(f"{report.converted_document_count} converted")
        if report.uncategorized_document_count:
            parts.append(f"{report.uncategorized_document_count} uncategorized")
        parts.append(f"{len(report.corpus.feedback_entries)} correction(s)")
        return ", ".join(parts) + "."
    return "Not ready: no readable protection documents were found."


def _referenced_document_text(prompt: ProtectionPendingPrompt) -> str:
    """Return referenced document IDs for a challenge dialog."""
    if not prompt.referenced_document_ids:
        return "No referenced document IDs."
    return ", ".join(prompt.referenced_document_ids)


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


def _skill_tooltip(skill: Any, *, enabled: bool) -> str:
    """Return compact hover text for a skill."""
    state = "Enabled" if enabled else "Disabled"
    return "\n".join(
        [
            f"{skill.name} ({skill.scope.value})",
            state,
            skill.description,
            str(skill.path),
        ]
    )


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


def _selected_tool_groups(
    capabilities: Sequence[AssistantToolCapability],
) -> list[tuple[str, list[AssistantToolCapability]]]:
    """Return selected tools grouped by assistant-facing source."""
    grouped: dict[str, list[AssistantToolCapability]] = {}
    for capability in capabilities:
        grouped.setdefault(capability.group, []).append(capability)
    return [
        (group_name, sorted(group_items, key=lambda item: item.tool_name))
        for group_name, group_items in sorted(grouped.items())
    ]


def _is_tool_url_setting(name: str) -> bool:
    """Return whether a required tool value is a non-secret URL."""
    normalized = name.strip().upper()
    return normalized.endswith("_BASE_URL") or normalized.endswith("_URL")


def _credential_display_name(name: str) -> str:
    """Return a user-facing label for one credential key."""
    if name == PROVIDER_API_KEY_FIELD:
        return "Provider API key"
    return name


def _tool_credential_placeholder(state: SessionSecretState) -> str:
    """Return the credential input placeholder for one state."""
    if state == "present":
        return "Stored for this session"
    if state == "expired":
        return "Expired; paste value"
    return "Paste value"


def _provider_endpoint_menu_rows(
    visible_protocols: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Return copyable common provider endpoint rows."""
    allowed = set(visible_protocols) if visible_protocols is not None else None
    return [
        (entry.name, entry.url)
        for entry in COMMON_PROVIDER_ENDPOINTS
        if entry.name.strip() and entry.url.strip()
        if allowed is None or entry.provider_protocol.value in allowed
    ]


def _provider_base_url_help_text() -> str:
    """Return concise helper text for provider Base URL inputs."""
    return "Use the provider's documented API base URL for the selected protocol."


def _models_endpoint_url(base_url: str) -> str:
    """Return the OpenAI-compatible model listing URL for a base URL."""
    trimmed = _normalized_provider_base_url(base_url) or ""
    return f"{trimmed}/models" if trimmed else ""


def _normalized_provider_base_url(base_url: str | None) -> str | None:
    """Return the normalized provider base URL used for connection identity."""
    if base_url is None:
        return None
    cleaned = base_url.strip().rstrip("/")
    return cleaned or None


def _provider_connection_identity(
    *,
    provider_protocol: ProviderProtocol,
    base_url: str | None,
    auth_scheme: ProviderAuthScheme,
) -> tuple[str, str | None, str]:
    """Return the non-secret identity for provider credential scoping."""
    return (
        provider_protocol.value,
        _normalized_provider_base_url(base_url),
        auth_scheme.value,
    )


def _selected_model_unavailable_message(
    *,
    selected_model: str | None,
    discovered_models: Sequence[str],
) -> str | None:
    """Return a blocking Apply message when discovery disproves the model."""
    if selected_model and discovered_models and selected_model not in discovered_models:
        return (
            f"Model '{selected_model}' was not found for the configured provider "
            "connection."
        )
    return None


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
            raw_name = entry.get("id") or entry.get("name") or entry.get("model") or ""
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
    provider_protocol: ProviderProtocol,
    base_url: str | None,
    auth_scheme: ProviderAuthScheme = ProviderAuthScheme.BEARER,
    api_key: str | None = None,
    timeout: float = 5.0,
) -> list[str]:
    """Discover available models from the configured provider endpoint."""
    if provider_protocol is ProviderProtocol.OLLAMA_NATIVE:
        if not base_url or auth_scheme is not ProviderAuthScheme.NONE:
            return []
        try:
            import ollama

            response = ollama.Client(host=base_url).list()
        except Exception:
            return []
        return _extract_model_names_from_models_payload(
            response.model_dump(mode="json")
        )
    if provider_protocol is ProviderProtocol.ASK_SAGE_NATIVE:
        if (
            not base_url
            or auth_scheme is not ProviderAuthScheme.X_ACCESS_TOKENS
            or not api_key
        ):
            return []
        try:
            from llm_tools.llm_providers import AskSageNativeProvider

            return AskSageNativeProvider(
                model="discovery",
                access_token=api_key,
                base_url=base_url,
                default_request_params={"timeout": timeout},
            ).list_available_models()
        except Exception:
            return []
    if provider_protocol.value != ProviderProtocol.OPENAI_API.value:
        return []
    if not base_url:
        return []
    url = _models_endpoint_url(base_url)
    if not url:
        return []
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        return []
    headers = {"Accept": "application/json"}
    if auth_scheme.requires_secret() and not api_key:
        return []
    if api_key and auth_scheme is ProviderAuthScheme.BEARER:
        headers["Authorization"] = f"Bearer {api_key}"
    if api_key and auth_scheme is ProviderAuthScheme.X_ACCESS_TOKENS:
        headers["x-access-tokens"] = api_key
    request = Request(url, headers=headers, method="GET")  # noqa: S310
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []
    return _extract_model_names_from_models_payload(payload)


def _hosted_storage_values() -> tuple[str | None, str | None]:
    session_id = nicegui_app.storage.user.get("assistant_app_session_id")
    token = nicegui_app.storage.user.get("assistant_app_session_token")
    return (
        str(session_id) if session_id else None,
        str(token) if token else None,
    )


def _set_hosted_storage_values(session_id: str, token: str) -> None:
    nicegui_app.storage.user["assistant_app_session_id"] = session_id
    nicegui_app.storage.user["assistant_app_session_token"] = token


def _clear_hosted_storage_values() -> None:
    nicegui_app.storage.user.pop("assistant_app_session_id", None)
    nicegui_app.storage.user.pop("assistant_app_session_token", None)


def clear_hosted_session(auth_provider: LocalAuthProvider | None) -> None:
    """Revoke and clear the current browser auth session."""
    session_id, _token = _hosted_storage_values()
    if session_id and auth_provider is not None:
        auth_provider.revoke_session(session_id)
    _clear_hosted_storage_values()


def render_hosted_nicegui_page(
    *,
    store: SQLiteNiceGUIChatStore,
    config: AssistantConfig,
    root_path: Path | None,
    hosted_config: NiceGUIHostedConfig,
    auth_provider: LocalAuthProvider,
) -> None:
    """Render first-run admin, login, or authenticated chat UI."""
    branding = store.load_admin_settings().branding
    _apply_branding_page_metadata(branding)
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
    build_assistant_ui(controller)


def render_first_admin_page(
    auth_provider: LocalAuthProvider,
    *,
    branding: AssistantBranding | None = None,
) -> None:  # pragma: no cover
    """Render first-run admin creation."""
    from nicegui import ui

    branding = branding or auth_provider.store.load_admin_settings().branding
    _apply_branding_page_metadata(branding)
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
        with ui.column().classes("w-full items-center gap-1 q-mb-md"):
            _render_brand_icon(branding.icon_name, "text-4xl text-primary")
            ui.label(branding.app_name).classes("text-xl")
        ui.label("Create Admin").classes("text-lg")
        ui.label("The assistant needs one local admin user.").classes(
            "text-sm llmt-muted"
        )
        username_input = ui.input("Username").classes("w-full")
        password_input = ui.input("Password").props("type=password").classes("w-full")
        ui.button("Create admin", on_click=create_admin).classes("w-full")


def render_login_page(
    auth_provider: LocalAuthProvider,
    *,
    branding: AssistantBranding | None = None,
) -> None:  # pragma: no cover
    """Render login."""
    from nicegui import ui

    branding = branding or auth_provider.store.load_admin_settings().branding
    _apply_branding_page_metadata(branding)
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
        with ui.column().classes("w-full items-center gap-1 q-mb-md"):
            _render_brand_icon(branding.icon_name, "text-4xl text-primary")
            ui.label(branding.app_name).classes("text-xl")
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


class AssistantUIRenderer:
    """Builds the NiceGUI assistant component tree."""

    def __init__(self, controller: NiceGUIChatController) -> None:
        self.controller = controller

    def build(self) -> None:
        build_assistant_ui(self.controller)


def build_assistant_ui(  # noqa: C901
    controller: NiceGUIChatController,
) -> None:  # pragma: no cover
    """Build the assistant app component tree."""
    from nicegui import ui

    _apply_branding_page_metadata(controller.admin_settings.branding)
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
        .llmt-brand {
            min-width: 0; color: var(--llmt-accent); line-height: 1;
        }
        .llmt-brand .nicegui-label {
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
        .llmt-brand-icon { font-size: 18px; }
        .llmt-brand-symbol {
            display: inline-flex; align-items: center; justify-content: center;
            line-height: 1;
        }
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
        .llmt-context-meter-shell {
            width: 100%; height: 3px; padding: 0; margin: 1px 0 0;
        }
        .llmt-context-meter-track {
            width: 100%; height: 3px; border-radius: 999px; overflow: hidden;
            background: #d5d3cb; position: relative;
        }
        body.body--dark .llmt-context-meter-track { background: #3c3e39; }
        .llmt-context-meter-fill {
            height: 100%; border-radius: inherit;
            background: linear-gradient(90deg, #74746e, #2f302c);
            transition: width 160ms ease-out;
        }
        body.body--dark .llmt-context-meter-fill {
            background: linear-gradient(90deg, #70726d, #f5f5f0);
        }
        .llmt-context-meter-track.compacting .llmt-context-meter-fill {
            animation: llmt-context-pulse 1.1s ease-in-out infinite;
        }
        @keyframes llmt-context-pulse {
            0% { opacity: 0.55; }
            50% { opacity: 1; }
            100% { opacity: 0.55; }
        }
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
            width: 100%; flex: 1 1 auto; min-height: 0;
        }
        .llmt-settings-card {
            width: min(760px, calc(100vw - 48px));
            height: min(820px, calc(100vh - 64px));
            max-width: calc(100vw - 48px);
            max-height: calc(100vh - 64px);
            display: flex; flex-direction: column; overflow: hidden;
        }
        .llmt-settings-card .q-card__section {
            flex: 0 0 auto;
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
        .llmt-response-details {
            border-top: 1px solid #e5e3dc; margin-top: 10px; padding-top: 8px;
        }
        body.body--dark .llmt-response-details { border-top-color: #383b35; }
        .llmt-response-chip {
            border: 1px solid #cfccc1; border-radius: 999px;
            padding: 2px 8px; font-size: 0.76rem;
        }
        body.body--dark .llmt-response-chip { border-color: #4a4d46; }
        .llmt-follow-up-button .q-btn__content {
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
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

    if controller.active_record.runtime.inspector_open:
        controller.preferences.workbench_open = True
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
    context_meter_row: Any = None
    composer_meta_row: Any = None
    composer_action_button: Any = None
    composer_disclaimer_label: Any = None
    provider_select: Any = None
    model_input: Any = None
    base_url_input: Any = None
    auth_scheme_select: Any = None
    provider_api_key_input: Any = None
    mode_select: Any = None
    workspace_input: Any = None
    database_path_input: Any = None
    provider_quick_select: Any = None
    base_url_quick_input: Any = None
    auth_scheme_quick_select: Any = None
    provider_api_key_quick_input: Any = None
    model_quick_select: Any = None
    mode_quick_select: Any = None
    workspace_quick_input: Any = None
    dark_mode_switch: Any = None
    temperature_input: Any = None
    timeout_seconds_input: Any = None
    show_token_usage_switch: Any = None
    show_footer_help_switch: Any = None
    inspector_open_switch: Any = None
    max_context_tokens_input: Any = None
    max_tool_round_trips_input: Any = None
    max_tool_calls_per_round_input: Any = None
    max_total_tool_calls_per_turn_input: Any = None
    max_entries_per_call_input: Any = None
    max_recursive_depth_input: Any = None
    max_files_scanned_input: Any = None
    max_search_matches_input: Any = None
    max_read_lines_input: Any = None
    max_read_input_bytes_input: Any = None
    max_file_size_characters_input: Any = None
    max_read_file_chars_input: Any = None
    max_tool_result_chars_input: Any = None
    deep_task_max_turns_input: Any = None
    deep_task_max_tool_invocations_input: Any = None
    deep_task_max_elapsed_seconds_input: Any = None
    deep_task_include_replay_switch: Any = None
    admin_user_list_column: Any = None
    admin_create_username_input: Any = None
    admin_create_password_input: Any = None
    admin_create_role_select: Any = None
    admin_deep_task_switch: Any = None
    admin_information_protection_switch: Any = None
    admin_skills_switch: Any = None
    admin_ollama_native_provider_switch: Any = None
    admin_ask_sage_native_provider_switch: Any = None
    admin_write_file_switch: Any = None
    admin_atlassian_switch: Any = None
    admin_gitlab_switch: Any = None
    admin_branding_app_name_input: Any = None
    admin_branding_short_name_input: Any = None
    admin_branding_icon_name_input: Any = None
    admin_branding_favicon_input: Any = None
    allow_network_switch: Any = None
    allow_filesystem_switch: Any = None
    allow_subprocess_switch: Any = None
    approval_select: Any = None
    tool_credentials_column: Any = None
    tool_credential_inputs: dict[str, Any] = {}
    tool_url_inputs: dict[str, Any] = {}
    protection_enabled_switch: Any = None
    protection_categories_input: Any = None
    protection_category_catalog_input: Any = None
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
    credential_reentry_title: Any = None
    credential_reentry_label: Any = None
    credential_reentry_input: Any = None
    credential_reentry_state: dict[str, str | Callable[[], None] | None] = {
        "name": None,
        "resume": None,
    }
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

    def open_credential_reentry_dialog(
        name: str, *, resume: Callable[[], None] | None = None
    ) -> None:
        if not secret_entry_enabled():
            ui.notify(
                controller.hosted_config.insecure_hosted_warning
                or "Secret entry is disabled for this hosted session.",
                type="negative",
            )
            return
        credential_reentry_state["name"] = name
        credential_reentry_state["resume"] = resume
        display_name = _credential_display_name(name)
        if credential_reentry_title is not None:
            credential_reentry_title.set_text(f"Re-enter {display_name}")
        if credential_reentry_label is not None:
            credential_reentry_label.set_text(
                f"{display_name} expired. Paste it again to continue."
            )
        if credential_reentry_input is not None:
            credential_reentry_input.value = ""
        dialogs["credential_reentry"].open()

    def submit_credential_reentry() -> None:
        name = credential_reentry_state.get("name")
        if not isinstance(name, str) or not name:
            dialogs["credential_reentry"].close()
            return
        value = str(credential_reentry_input.value or "").strip()
        if not value:
            ui.notify("Enter the credential to continue.", type="negative")
            return
        controller.set_session_secret(name, value)
        credential_reentry_input.value = ""
        resume = credential_reentry_state.get("resume")
        credential_reentry_state["name"] = None
        credential_reentry_state["resume"] = None
        dialogs["credential_reentry"].close()
        refresh_all()
        if callable(resume):
            resume()

    def cancel_credential_reentry() -> None:
        name = credential_reentry_state.get("name")
        if isinstance(name, str) and name:
            controller.clear_session_secret(name)
        if credential_reentry_input is not None:
            credential_reentry_input.value = ""
        credential_reentry_state["name"] = None
        credential_reentry_state["resume"] = None
        dialogs["credential_reentry"].close()
        refresh_all()

    def required_provider_credential_expired() -> bool:
        runtime = active_runtime()
        return (
            runtime.provider_connection.auth_scheme.requires_secret()
            and controller.session_secret_state(
                PROVIDER_API_KEY_FIELD, session_id=controller.active_session_id
            )
            == "expired"
        )

    def first_expired_tool_credential(tool_names: Sequence[str]) -> str | None:
        specs = controller.visible_tool_specs()
        for tool_name in sorted(set(tool_names)):
            spec = specs.get(tool_name)
            if spec is None:
                continue
            for secret_name in spec.required_secrets:
                if _is_tool_url_setting(secret_name):
                    continue
                if (
                    controller.session_secret_state(
                        secret_name, session_id=controller.active_session_id
                    )
                    == "expired"
                ):
                    return secret_name
        return None

    def block_for_expired_credential(
        name: str | None, *, resume: Callable[[], None]
    ) -> bool:
        if name is None:
            return False
        open_credential_reentry_dialog(name, resume=resume)
        return True

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
        provider_key = runtime.provider_protocol.value
        if runtime.selected_model:
            models = controller.preferences.recent_models.setdefault(provider_key, [])
            if runtime.selected_model in models:
                models.remove(runtime.selected_model)
            models.insert(0, runtime.selected_model)
            del models[10:]
        if runtime.provider_connection.api_base_url:
            urls = controller.preferences.recent_base_urls.setdefault(provider_key, [])
            if runtime.provider_connection.api_base_url in urls:
                urls.remove(runtime.provider_connection.api_base_url)
            urls.insert(0, runtime.provider_connection.api_base_url)
            del urls[10:]
        if runtime.root_path:
            roots = controller.preferences.recent_roots
            if runtime.root_path in roots:
                roots.remove(runtime.root_path)
            roots.insert(0, runtime.root_path)
            del roots[10:]

    def base_url_options(runtime: NiceGUIRuntimeConfig) -> list[str]:
        options = list(
            controller.preferences.recent_base_urls.get(
                runtime.provider_protocol.value, []
            )
        )
        if (
            runtime.provider_connection.api_base_url
            and runtime.provider_connection.api_base_url not in options
        ):
            options.insert(0, runtime.provider_connection.api_base_url)
        return options

    def provider_protocol_options() -> list[str]:
        return controller.visible_provider_protocol_options()

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

    def set_model_options(values: Sequence[str], *, selected: str | None) -> None:
        options = [value for value in values if value]
        if selected and selected not in options:
            options.insert(0, selected)
        model_options_state["values"] = options
        if model_input is not None:
            model_input.set_options(options, value=selected or None)
        if model_quick_select is not None:
            model_quick_select.set_options(options, value=selected or None)

    def sync_connection_settings(runtime: NiceGUIRuntimeConfig) -> None:
        if provider_select is not None:
            provider_select.set_options(
                provider_protocol_options(),
                value=runtime.provider_protocol.value,
            )
        if model_input is not None:
            set_model_options(
                model_options_state["values"], selected=runtime.selected_model
            )
        if mode_select is not None:
            mode_select.value = runtime.response_mode_strategy.value
        if base_url_input is not None:
            base_url_input.set_options(
                base_url_options(runtime),
                value=runtime.provider_connection.api_base_url or None,
            )
            base_url_input.value = runtime.provider_connection.api_base_url or ""
        if auth_scheme_select is not None:
            auth_scheme_select.value = runtime.provider_connection.auth_scheme.value
        if temperature_input is not None:
            temperature_input.value = str(runtime.temperature)
        if timeout_seconds_input is not None:
            timeout_seconds_input.value = str(runtime.timeout_seconds)
        if provider_api_key_input is not None:
            provider_api_key_input.value = ""

    def sync_display_settings(runtime: NiceGUIRuntimeConfig) -> None:
        if dark_mode_switch is not None:
            dark_mode_switch.value = controller.preferences.theme_mode == "dark"
        if show_token_usage_switch is not None:
            show_token_usage_switch.value = runtime.show_token_usage
        if show_footer_help_switch is not None:
            show_footer_help_switch.value = runtime.show_footer_help
        if inspector_open_switch is not None:
            inspector_open_switch.value = runtime.inspector_open

    def sync_path_settings(runtime: NiceGUIRuntimeConfig) -> None:
        if workspace_input is not None:
            workspace_input.value = runtime.root_path or ""
        if database_path_input is not None:
            database_path_input.value = str(controller.store.db_path)

    def sync_chat_limit_settings(runtime: NiceGUIRuntimeConfig) -> None:
        if max_context_tokens_input is not None:
            max_context_tokens_input.value = str(
                runtime.session_config.max_context_tokens
            )
        if max_tool_round_trips_input is not None:
            max_tool_round_trips_input.value = str(
                runtime.session_config.max_tool_round_trips
            )
        if max_tool_calls_per_round_input is not None:
            max_tool_calls_per_round_input.value = str(
                runtime.session_config.max_tool_calls_per_round
            )
        if max_total_tool_calls_per_turn_input is not None:
            max_total_tool_calls_per_turn_input.value = str(
                runtime.session_config.max_total_tool_calls_per_turn
            )

    def sync_tool_limit_settings(runtime: NiceGUIRuntimeConfig) -> None:
        if max_entries_per_call_input is not None:
            max_entries_per_call_input.value = str(
                runtime.tool_limits.max_entries_per_call
            )
        if max_recursive_depth_input is not None:
            max_recursive_depth_input.value = str(
                runtime.tool_limits.max_recursive_depth
            )
        if max_files_scanned_input is not None:
            max_files_scanned_input.value = str(runtime.tool_limits.max_files_scanned)
        if max_search_matches_input is not None:
            max_search_matches_input.value = str(runtime.tool_limits.max_search_matches)
        if max_read_lines_input is not None:
            max_read_lines_input.value = str(runtime.tool_limits.max_read_lines)
        if max_read_input_bytes_input is not None:
            max_read_input_bytes_input.value = str(
                runtime.tool_limits.max_read_input_bytes
            )
        if max_file_size_characters_input is not None:
            max_file_size_characters_input.value = str(
                runtime.tool_limits.max_file_size_characters
            )
        if max_read_file_chars_input is not None:
            max_read_file_chars_input.value = (
                ""
                if runtime.tool_limits.max_read_file_chars is None
                else str(runtime.tool_limits.max_read_file_chars)
            )
        if max_tool_result_chars_input is not None:
            max_tool_result_chars_input.value = str(
                runtime.tool_limits.max_tool_result_chars
            )

    def sync_deep_task_settings(runtime: NiceGUIRuntimeConfig) -> None:
        if deep_task_max_turns_input is not None:
            deep_task_max_turns_input.value = str(runtime.research.default_max_turns)
        if deep_task_max_tool_invocations_input is not None:
            deep_task_max_tool_invocations_input.value = (
                ""
                if runtime.research.default_max_tool_invocations is None
                else str(runtime.research.default_max_tool_invocations)
            )
        if deep_task_max_elapsed_seconds_input is not None:
            deep_task_max_elapsed_seconds_input.value = (
                ""
                if runtime.research.default_max_elapsed_seconds is None
                else str(runtime.research.default_max_elapsed_seconds)
            )
        if deep_task_include_replay_switch is not None:
            deep_task_include_replay_switch.value = (
                runtime.research.include_replay_by_default
            )

    def sync_permission_settings(runtime: NiceGUIRuntimeConfig) -> None:
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

    def sync_settings_inputs() -> None:
        runtime = active_runtime()
        sync_connection_settings(runtime)
        sync_display_settings(runtime)
        sync_path_settings(runtime)
        sync_chat_limit_settings(runtime)
        sync_tool_limit_settings(runtime)
        sync_deep_task_settings(runtime)
        sync_permission_settings(runtime)

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
        if protection_category_catalog_input is not None:
            protection_category_catalog_input.value = (
                _format_information_security_category_catalog(
                    config.sensitivity_categories
                )
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
        if not controller.information_protection_enabled():
            ui.notify(
                "Information Protection is disabled by the administrator.",
                type="warning",
            )
            return
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
        provider_quick_select.set_options(
            provider_protocol_options(),
            value=runtime.provider_protocol.value,
        )
        base_url_quick_input.set_options(
            base_url_options(runtime),
            value=runtime.provider_connection.api_base_url or None,
        )
        base_url_quick_input.value = runtime.provider_connection.api_base_url or ""
        auth_scheme_quick_select.value = runtime.provider_connection.auth_scheme.value
        provider_api_key_quick_input.value = ""
        dialogs["provider_settings"].open()

    def open_model_dialog() -> None:
        runtime = active_runtime()
        set_model_options(
            model_options_state["values"], selected=runtime.selected_model
        )
        model_quick_select.value = runtime.selected_model
        dialogs["model_settings"].open()

    def open_mode_dialog() -> None:
        mode_quick_select.value = active_runtime().response_mode_strategy.value
        dialogs["mode_settings"].open()

    def open_workspace_dialog() -> None:
        workspace_quick_input.value = active_runtime().root_path or ""
        dialogs["workspace_settings"].open()

    def open_runtime_part_dialog(part_name: str) -> None:
        if part_name in {"provider_protocol", "endpoint", "credential"}:
            open_provider_dialog()
        elif part_name == "model":
            open_model_dialog()
        elif part_name == "mode":
            open_mode_dialog()
        elif part_name == "workspace":
            open_workspace_dialog()
        else:
            open_settings_dialog()

    def render_runtime_summary(
        runtime: NiceGUIRuntimeConfig,
        token_usage: ChatTokenUsage | None,
    ) -> None:
        composer_meta_row.clear()
        with composer_meta_row:
            for index, (label, value) in enumerate(
                _runtime_summary_parts(runtime, token_usage)
            ):
                if index:
                    ui.label("|").classes("text-xs llmt-muted")
                if label == "tokens":
                    ui.label(value).classes("text-xs llmt-muted llmt-runtime-chip")
                    continue
                ui.button(
                    value,
                    on_click=lambda _event, part=label: open_runtime_part_dialog(part),
                ).props("flat dense no-caps color=primary").classes("llmt-runtime-chip")

    def render_context_meter(
        runtime: NiceGUIRuntimeConfig,
        token_usage: ChatTokenUsage | None,
        *,
        status_text: str,
    ) -> None:
        context_meter_row.clear()
        state = _context_capacity_meter_state(
            runtime,
            token_usage,
            status_text=status_text,
        )
        with (
            context_meter_row,
            ui.element("div").classes("llmt-context-meter-shell"),
            ui.element("div").classes(state.track_classes),
        ):
            ui.element("div").classes("llmt-context-meter-fill").style(state.fill_style)
            ui.tooltip(state.tooltip)

    def current_tool_capability_groups() -> dict[str, list[AssistantToolCapability]]:
        runtime = active_runtime()
        return build_tool_capabilities(
            tool_specs=controller.visible_tool_specs(),
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

    def tool_required_value_entries() -> list[
        tuple[str, list[str], SessionSecretState, bool]
    ]:
        entries: dict[str, set[str]] = {}
        for tool_name, spec in controller.visible_tool_specs().items():
            for secret_name in spec.required_secrets:
                entries.setdefault(secret_name, set()).add(tool_name)
        return [
            (
                secret_name,
                sorted(tool_names),
                (
                    "present"
                    if active_runtime().tool_urls.get(secret_name)
                    else "missing"
                )
                if _is_tool_url_setting(secret_name)
                else controller.session_secret_state(
                    secret_name, session_id=controller.active_session_id
                ),
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
            for secret_name, tool_names, state, is_url in entries:
                is_set = state == "present"
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
                                placeholder=_tool_credential_placeholder(state),
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
            with ui.expansion("Skills", icon="psychology", value=True).classes(
                "w-full"
            ):
                skills = controller.visible_skills()
                errors = controller.visible_skill_errors()
                if not skills and not errors:
                    ui.label("No local skills discovered.").classes(
                        "text-xs llmt-muted q-px-sm"
                    )
                for skill in skills:
                    enabled = controller.skill_enabled(skill)
                    icon = "check_box" if enabled else "check_box_outline_blank"
                    with ui.row().classes("w-full items-center gap-1 no-wrap"):
                        button = ui.button(
                            icon=icon,
                            on_click=lambda _event, target=skill, state=enabled: (
                                toggle_runtime_skill(target, not state)
                            ),
                        ).props("flat round dense")
                        button.classes(
                            "llmt-tool-name "
                            + ("llmt-tool-chip-enabled" if enabled else "")
                        )
                        with button:
                            ui.tooltip(_skill_tooltip(skill, enabled=enabled)).props(
                                "delay=700"
                            )
                        with ui.column().classes("grow min-w-0 gap-0"):
                            ui.label(skill.name).classes("text-sm")
                            ui.label(skill.description).classes("text-xs llmt-muted")
                            ui.label(f"{skill.scope.value} | {skill.path}").classes(
                                "text-xs llmt-muted break-all"
                            )
                        invoke_button = ui.button(
                            icon="add_comment",
                            on_click=lambda _event, name=skill.name: (
                                add_skill_invocation_to_composer(name)
                            ),
                        ).props("flat round dense")
                        if not enabled:
                            invoke_button.disable()
                        with invoke_button:
                            ui.tooltip(f"Insert ${skill.name}").props("delay=700")
                for error in errors[:5]:
                    ui.label(
                        f"{Path(str(error.path)).parent.name}: {error.message}"
                    ).classes("text-xs text-negative q-px-sm")
                if len(errors) > 5:
                    ui.label(f"{len(errors) - 5} more skill error(s)").classes(
                        "text-xs text-negative q-px-sm"
                    )
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
                for group_name, capabilities in _selected_tool_groups(selected):
                    ui.label(group_name).classes("text-xs llmt-muted")
                    for capability in capabilities:
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
            expired_credential = first_expired_tool_credential([tool_name])
            if block_for_expired_credential(
                expired_credential,
                resume=lambda: toggle_runtime_tool(tool_name),
            ):
                return
            enabled.add(tool_name)
        runtime.enabled_tools = sorted(enabled)
        controller.save_active_session()
        render_selected_tools()
        render_tool_menu()

    def toggle_runtime_skill(skill: Any, enabled: bool) -> None:
        controller.set_skill_enabled(skill, enabled)
        render_tool_menu()

    def add_skill_invocation_to_composer(skill_name: str) -> None:
        current = _first_nonempty_text(composer_state["text"], composer_input.value)
        invocation = f"${skill_name}"
        next_text = (
            current
            if invocation in current.split()
            else f"{invocation} {current}".strip()
        )
        _set_composer_draft(next_text, composer_state, composer_input)

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
            expired_credential = first_expired_tool_credential(sorted(group_tool_names))
            if block_for_expired_credential(
                expired_credential,
                resume=lambda: toggle_tool_group(group_name),
            ):
                return
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
                                f"{summary.selected_model or 'No model'} | "
                                f"{summary.message_count} msgs"
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

    def render_header() -> None:
        runtime = active_runtime()
        turn_state = controller.active_turn_state
        header_title.set_text(controller.active_record.summary.title)
        information_security_button.set_text(
            _format_information_security_label(runtime.protection)
        )
        information_security_button.set_visibility(
            controller.information_protection_enabled()
        )
        status_chip.set_text(turn_state.status_text or "ready")
        if composer_disclaimer_label is not None:
            composer_disclaimer_label.set_visibility(runtime.show_footer_help)
        render_selected_tools()
        render_tool_menu()
        render_context_meter(
            runtime,
            controller.active_record.token_usage,
            status_text=turn_state.status_text,
        )
        render_runtime_summary(runtime, controller.active_record.token_usage)
        composer_action_button.props(
            "icon="
            + _composer_action_icon(busy=turn_state.busy)
            + " color="
            + ("negative" if turn_state.busy else "primary")
        )
        sync_settings_inputs()
        render_admin_user_list()

    def set_follow_up_suggestion(text: str) -> None:
        _set_composer_draft(text, composer_state, composer_input)
        ui.notify("Suggestion added to composer")

    def render_response_details(entry: Any) -> None:
        if entry.final_response is None:
            return
        details = final_response_details(entry.final_response)
        if not details.has_content:
            return
        with ui.column().classes("llmt-response-details w-full gap-2"):
            if details.confidence_label or details.citations:
                with ui.row().classes("w-full items-center gap-1"):
                    if details.confidence_label:
                        ui.label(details.confidence_label).classes(
                            "llmt-response-chip llmt-muted"
                        )
                    for citation in details.citations:
                        citation_label = ui.label(citation.label).classes(
                            "llmt-response-chip llmt-muted"
                        )
                        if citation.excerpt:
                            with citation_label:
                                ui.tooltip(citation.excerpt).props("delay=700")
            if details.uncertainty:
                with (
                    ui.expansion("Uncertainty", value=False)
                    .props("dense expand-separator")
                    .classes("w-full text-xs"),
                    ui.column().classes("w-full gap-1 q-pt-xs"),
                ):
                    for item in details.uncertainty:
                        ui.label(f"- {item}").classes("text-xs llmt-muted")
            if details.missing_information:
                with (
                    ui.expansion("Missing information", value=False)
                    .props("dense expand-separator")
                    .classes("w-full text-xs"),
                    ui.column().classes("w-full gap-1 q-pt-xs"),
                ):
                    for item in details.missing_information:
                        ui.label(f"- {item}").classes("text-xs llmt-muted")
            if details.follow_up_suggestions:
                ui.label("Follow-up suggestions").classes("text-xs llmt-muted")
                with ui.row().classes("w-full items-center gap-1"):
                    for suggestion in details.follow_up_suggestions:
                        ui.button(
                            suggestion,
                            on_click=(
                                lambda _event, text=suggestion: (
                                    set_follow_up_suggestion(text)
                                )
                            ),
                        ).props("outline dense no-caps").classes(
                            "llmt-follow-up-button"
                        )

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
                    with ui.row().classes("items-baseline gap-2"):
                        ui.label(entry.role.title()).classes("text-xs llmt-muted")
                        posted_at = _format_transcript_time(entry.created_at)
                        if posted_at:
                            ui.label(posted_at).classes("text-xs llmt-muted")
                    ui.markdown(entry.text)
                    render_response_details(entry)
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
        if prompt is None or not controller.information_protection_enabled():
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
        if controller.active_record.runtime.inspector_open:
            controller.preferences.workbench_open = True
        refresh_all()

    def select_session(session_id: str) -> None:
        controller.select_session(session_id)
        if controller.active_record.runtime.inspector_open:
            controller.preferences.workbench_open = True
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

    def render_provider_endpoint_help_button() -> None:
        with ui.button(icon="help_outline").props("flat round"):
            ui.tooltip("Common OpenAI-compatible Base URLs").props("delay=700")
            with ui.menu().classes("w-[420px] max-w-[90vw]"):
                ui.label("Common OpenAI-compatible Base URLs").classes(
                    "text-sm llmt-muted q-px-md q-pt-sm"
                )
                for provider_name, endpoint_url in _provider_endpoint_menu_rows(
                    provider_protocol_options()
                ):
                    with ui.row().classes(
                        "w-full items-center justify-between gap-2 no-wrap q-px-sm"
                    ):
                        with ui.column().classes("gap-0 min-w-0 grow"):
                            ui.label(provider_name).classes("text-sm")
                            ui.label(endpoint_url).classes(
                                "text-xs llmt-muted ellipsis"
                            )
                        ui.button(
                            icon="content_copy",
                            on_click=lambda _event, url=endpoint_url: copy_text(url),
                        ).props("flat round")

    def update_composer_text(value: object) -> None:
        composer_state["text"] = str(value or "")

    def submit_text_with_credential_reentry(
        text: str,
        *,
        resume: Callable[[], None],
        clear_composer_on_success: bool,
    ) -> None:
        if not text.strip():
            error = controller.submit_prompt(text)
            if error:
                ui.notify(error, type="negative")
            refresh_all()
            return
        if required_provider_credential_expired():
            open_credential_reentry_dialog(
                PROVIDER_API_KEY_FIELD,
                resume=resume,
            )
            return
        expired_tool_credential = first_expired_tool_credential(
            active_runtime().enabled_tools
        )
        if block_for_expired_credential(
            expired_tool_credential,
            resume=resume,
        ):
            return
        error = controller.submit_prompt(text)
        if error:
            ui.notify(error, type="negative")
        elif clear_composer_on_success:
            composer_state["text"] = ""
            composer_input.value = ""
        refresh_all()

    def regenerate_last() -> None:
        users = [
            entry
            for entry in controller.active_record.transcript
            if entry.role == "user"
        ]
        if not users:
            return
        submit_text_with_credential_reentry(
            users[-1].text,
            resume=regenerate_last,
            clear_composer_on_success=False,
        )

    def send_prompt() -> None:
        text = _first_nonempty_text(composer_state["text"], composer_input.value)
        submit_text_with_credential_reentry(
            text,
            resume=send_prompt,
            clear_composer_on_success=True,
        )

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
        populate_admin_branding_inputs()
        populate_admin_feature_flags()
        render_admin_user_list()
        dialogs["admin"].open()

    def populate_admin_feature_flags() -> None:
        settings = controller.admin_settings
        if admin_deep_task_switch is not None:
            admin_deep_task_switch.value = settings.deep_task_mode_enabled
        if admin_information_protection_switch is not None:
            admin_information_protection_switch.value = (
                settings.information_protection_enabled
            )
        if admin_skills_switch is not None:
            admin_skills_switch.value = settings.skills_enabled
        if admin_ollama_native_provider_switch is not None:
            admin_ollama_native_provider_switch.value = (
                settings.ollama_native_provider_enabled
            )
        if admin_ask_sage_native_provider_switch is not None:
            admin_ask_sage_native_provider_switch.value = (
                settings.ask_sage_native_provider_enabled
            )
        if admin_write_file_switch is not None:
            admin_write_file_switch.value = settings.write_file_tool_enabled
        if admin_atlassian_switch is not None:
            admin_atlassian_switch.value = settings.atlassian_tools_enabled
        if admin_gitlab_switch is not None:
            admin_gitlab_switch.value = settings.gitlab_tools_enabled

    def populate_admin_branding_inputs() -> None:
        branding = controller.admin_settings.branding
        if admin_branding_app_name_input is not None:
            admin_branding_app_name_input.value = branding.app_name
        if admin_branding_short_name_input is not None:
            admin_branding_short_name_input.value = branding.short_name
        if admin_branding_icon_name_input is not None:
            admin_branding_icon_name_input.value = branding.icon_name
        if admin_branding_favicon_input is not None:
            admin_branding_favicon_input.value = branding.favicon_svg

    def save_admin_branding() -> None:
        if not _is_admin_user(controller.current_user):
            ui.notify("Admin access is required.", type="negative")
            return
        try:
            branding = AssistantBranding(
                app_name=str(admin_branding_app_name_input.value or ""),
                short_name=str(admin_branding_short_name_input.value or ""),
                icon_name=str(admin_branding_icon_name_input.value or ""),
                favicon_svg=str(admin_branding_favicon_input.value or ""),
            )
        except ValueError as exc:
            ui.notify(f"Could not save branding: {exc}", type="negative")
            return
        controller.set_branding(branding)
        _apply_branding_page_metadata(branding, update_favicon=True)
        refresh_all()
        ui.notify("Branding updated.")

    def reset_admin_branding() -> None:
        if not _is_admin_user(controller.current_user):
            ui.notify("Admin access is required.", type="negative")
            return
        controller.set_branding(AssistantBranding())
        populate_admin_branding_inputs()
        _apply_branding_page_metadata(
            controller.admin_settings.branding, update_favicon=True
        )
        refresh_all()
        ui.notify("Branding reset.")

    def set_admin_deep_task_enabled(event: Any) -> None:
        if not _is_admin_user(controller.current_user):
            ui.notify("Admin access is required.", type="negative")
            return
        controller.set_deep_task_mode_enabled(bool(event.value))
        render_selected_tools()
        render_tool_menu()
        ui.notify("Deep Task mode updated.")

    def set_admin_beta_feature_flag(name: str, value: bool) -> None:
        if not _is_admin_user(controller.current_user):
            ui.notify("Admin access is required.", type="negative")
            return
        if name == "information_protection_enabled":
            controller.set_beta_feature_flags(information_protection_enabled=value)
        elif name == "skills_enabled":
            controller.set_beta_feature_flags(skills_enabled=value)
        elif name == "ollama_native_provider_enabled":
            controller.set_beta_feature_flags(ollama_native_provider_enabled=value)
        elif name == "ask_sage_native_provider_enabled":
            controller.set_beta_feature_flags(ask_sage_native_provider_enabled=value)
        elif name == "write_file_tool_enabled":
            controller.set_beta_feature_flags(write_file_tool_enabled=value)
        elif name == "atlassian_tools_enabled":
            controller.set_beta_feature_flags(atlassian_tools_enabled=value)
        elif name == "gitlab_tools_enabled":
            controller.set_beta_feature_flags(gitlab_tools_enabled=value)
        else:
            raise ValueError(f"Unknown beta feature flag: {name}")
        render_header()
        render_tool_menu()
        ui.notify("Feature flag updated.")

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

    def apply_settings() -> bool:
        runtime = active_runtime()
        try:
            updated_runtime = _runtime_with_settings_values(
                runtime,
                {
                    "temperature": temperature_input.value,
                    "timeout_seconds": timeout_seconds_input.value,
                    "show_token_usage": show_token_usage_switch.value,
                    "show_footer_help": show_footer_help_switch.value,
                    "inspector_open": inspector_open_switch.value,
                    "max_context_tokens": max_context_tokens_input.value,
                    "max_tool_round_trips": max_tool_round_trips_input.value,
                    "max_tool_calls_per_round": max_tool_calls_per_round_input.value,
                    "max_total_tool_calls_per_turn": (
                        max_total_tool_calls_per_turn_input.value
                    ),
                    "max_entries_per_call": max_entries_per_call_input.value,
                    "max_recursive_depth": max_recursive_depth_input.value,
                    "max_files_scanned": max_files_scanned_input.value,
                    "max_search_matches": max_search_matches_input.value,
                    "max_read_lines": max_read_lines_input.value,
                    "max_read_input_bytes": max_read_input_bytes_input.value,
                    "max_file_size_characters": (max_file_size_characters_input.value),
                    "max_read_file_chars": max_read_file_chars_input.value,
                    "max_tool_result_chars": max_tool_result_chars_input.value,
                    "deep_task_max_turns": deep_task_max_turns_input.value,
                    "deep_task_max_tool_invocations": (
                        deep_task_max_tool_invocations_input.value
                    ),
                    "deep_task_max_elapsed_seconds": (
                        deep_task_max_elapsed_seconds_input.value
                    ),
                    "deep_task_include_replay": deep_task_include_replay_switch.value,
                },
            )
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
            return False
        old_provider_identity = _provider_connection_identity(
            provider_protocol=runtime.provider_protocol,
            base_url=runtime.provider_connection.api_base_url,
            auth_scheme=runtime.provider_connection.auth_scheme,
        )
        provider_protocol = ProviderProtocol(str(provider_select.value))
        selected_model = str(model_input.value or "").strip() or None
        response_mode_strategy = ResponseModeStrategy(str(mode_select.value))
        base_url = _normalized_provider_base_url(str(base_url_input.value or ""))
        auth_scheme = ProviderAuthScheme(str(auth_scheme_select.value))
        new_provider_identity = _provider_connection_identity(
            provider_protocol=provider_protocol,
            base_url=base_url,
            auth_scheme=auth_scheme,
        )
        provider_api_key = str(provider_api_key_input.value or "").strip()
        existing_provider_api_key = (
            controller.provider_api_key(session_id=controller.active_session_id)
            if old_provider_identity == new_provider_identity
            else None
        )
        if not validate_selected_model_for_apply(
            provider_protocol=provider_protocol,
            base_url=base_url,
            auth_scheme=auth_scheme,
            selected_model=selected_model,
            api_key=provider_api_key or existing_provider_api_key,
        ):
            return False
        runtime.provider_protocol = provider_protocol
        runtime.selected_model = selected_model
        runtime.response_mode_strategy = response_mode_strategy
        runtime.provider_connection.api_base_url = base_url
        runtime.provider_connection.auth_scheme = auth_scheme
        runtime.temperature = updated_runtime.temperature
        runtime.timeout_seconds = updated_runtime.timeout_seconds
        runtime.show_token_usage = updated_runtime.show_token_usage
        runtime.show_footer_help = updated_runtime.show_footer_help
        runtime.inspector_open = updated_runtime.inspector_open
        runtime.session_config = updated_runtime.session_config
        runtime.tool_limits = updated_runtime.tool_limits
        runtime.research = updated_runtime.research
        update_provider_secret_for_connection(
            old_identity=old_provider_identity,
            new_identity=new_provider_identity,
            provider_api_key=provider_api_key,
        )
        if provider_api_key and secret_entry_enabled():
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
        workspace_path = str(workspace_input.value or "").strip()
        runtime.root_path = (
            expanded_path_text(workspace_path) if workspace_path else None
        )
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
        if runtime.inspector_open:
            controller.preferences.workbench_open = True
        remember_recent_runtime_values(runtime)
        if (
            controller.current_user is None
            and requested_db_path
            and expand_app_path(requested_db_path) != controller.store.db_path
        ):
            controller.switch_database(expand_app_path(requested_db_path))
        else:
            controller.save_active_session()
        controller.save_preferences()
        refresh_all()
        return True

    def apply_settings_and_close() -> None:
        if apply_settings():
            dialogs["settings"].close()

    def apply_information_security_settings() -> None:
        protection = active_runtime().protection
        raw_corpus_directory = str(protection_corpus_input.value or "").strip()
        corpus_directory = (
            expanded_path_text(raw_corpus_directory) if raw_corpus_directory else ""
        )
        raw_corrections_path = str(protection_corrections_input.value or "").strip()
        corrections_path = (
            expanded_path_text(raw_corrections_path) if raw_corrections_path else ""
        )
        protection.enabled = bool(protection_enabled_switch.value)
        allowed_categories = _parse_information_security_categories(
            str(protection_categories_input.value or "")
        )
        category_catalog = _parse_information_security_category_catalog(
            str(protection_category_catalog_input.value or "")
        )
        protection.allowed_sensitivity_labels = allowed_categories
        protection.sensitivity_categories = (
            _ensure_information_security_category_catalog(
                categories=category_catalog,
                allowed_labels=allowed_categories,
            )
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
        provider_protocol = ProviderProtocol(
            str(provider_select.value or active_runtime().provider_protocol)
        )
        base_url = _normalized_provider_base_url(
            str(
                base_url_input.value
                or active_runtime().provider_connection.api_base_url
                or ""
            )
        )
        provider_api_key = str(provider_api_key_input.value or "").strip()
        auth_scheme = ProviderAuthScheme(str(auth_scheme_select.value))
        if (
            not provider_api_key
            and auth_scheme.requires_secret()
            and controller.session_secret_state(
                PROVIDER_API_KEY_FIELD, session_id=controller.active_session_id
            )
            == "expired"
        ):
            open_credential_reentry_dialog(
                PROVIDER_API_KEY_FIELD,
                resume=lambda: refresh_model_options(notify=notify),
            )
            return
        current_model = str(model_input.value or active_runtime().selected_model or "")
        models = _discover_model_names(
            provider_protocol=provider_protocol,
            base_url=base_url,
            auth_scheme=auth_scheme,
            api_key=provider_api_key
            or controller.provider_api_key(session_id=controller.active_session_id),
        )
        if not models:
            set_model_options(model_options_state["values"], selected=current_model)
            if notify:
                ui.notify("No models were discovered from the configured endpoint.")
            return
        set_model_options(models, selected=current_model)
        if notify:
            ui.notify(f"Loaded {len(models)} models")

    def validate_selected_model_for_apply(
        *,
        provider_protocol: ProviderProtocol,
        base_url: str | None,
        auth_scheme: ProviderAuthScheme,
        selected_model: str | None,
        api_key: str | None,
    ) -> bool:
        if not selected_model or not base_url:
            return True
        models = _discover_model_names(
            provider_protocol=provider_protocol,
            base_url=base_url,
            auth_scheme=auth_scheme,
            api_key=api_key,
        )
        message = _selected_model_unavailable_message(
            selected_model=selected_model,
            discovered_models=models,
        )
        if message is None:
            if not models:
                ui.notify(
                    "Could not verify the selected model from the configured endpoint."
                )
            return True
        ui.notify(message, type="negative")
        return False

    def update_provider_secret_for_connection(
        *,
        old_identity: tuple[str, str | None, str],
        new_identity: tuple[str, str | None, str],
        provider_api_key: str,
    ) -> None:
        if provider_api_key and secret_entry_enabled():
            controller.set_session_secret(PROVIDER_API_KEY_FIELD, provider_api_key)
            return
        if old_identity != new_identity:
            controller.clear_session_secret(PROVIDER_API_KEY_FIELD)

    def refresh_quick_model_options(*, notify: bool = True) -> None:
        runtime = active_runtime()
        if (
            runtime.provider_connection.auth_scheme.requires_secret()
            and controller.session_secret_state(
                PROVIDER_API_KEY_FIELD, session_id=controller.active_session_id
            )
            == "expired"
        ):
            open_credential_reentry_dialog(
                PROVIDER_API_KEY_FIELD,
                resume=lambda: refresh_quick_model_options(notify=notify),
            )
            return
        current_model = str(model_quick_select.value or runtime.selected_model or "")
        models = _discover_model_names(
            provider_protocol=runtime.provider_protocol,
            base_url=runtime.provider_connection.api_base_url,
            auth_scheme=runtime.provider_connection.auth_scheme,
            api_key=controller.provider_api_key(
                session_id=controller.active_session_id
            ),
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
        old_provider_identity = _provider_connection_identity(
            provider_protocol=runtime.provider_protocol,
            base_url=runtime.provider_connection.api_base_url,
            auth_scheme=runtime.provider_connection.auth_scheme,
        )
        provider_protocol = ProviderProtocol(str(provider_quick_select.value))
        base_url = _normalized_provider_base_url(str(base_url_quick_input.value or ""))
        auth_scheme = ProviderAuthScheme(str(auth_scheme_quick_select.value))
        new_provider_identity = _provider_connection_identity(
            provider_protocol=provider_protocol,
            base_url=base_url,
            auth_scheme=auth_scheme,
        )
        provider_api_key = str(provider_api_key_quick_input.value or "").strip()
        existing_provider_api_key = (
            controller.provider_api_key(session_id=controller.active_session_id)
            if old_provider_identity == new_provider_identity
            else None
        )
        if not validate_selected_model_for_apply(
            provider_protocol=provider_protocol,
            base_url=base_url,
            auth_scheme=auth_scheme,
            selected_model=runtime.selected_model,
            api_key=provider_api_key or existing_provider_api_key,
        ):
            return
        runtime.provider_protocol = provider_protocol
        runtime.provider_connection.api_base_url = base_url
        runtime.provider_connection.auth_scheme = auth_scheme
        update_provider_secret_for_connection(
            old_identity=old_provider_identity,
            new_identity=new_provider_identity,
            provider_api_key=provider_api_key,
        )
        if provider_api_key and secret_entry_enabled():
            provider_api_key_quick_input.value = ""
        remember_recent_runtime_values(runtime)
        controller.save_active_session()
        dialogs["provider_settings"].close()
        refresh_all()

    def apply_model_settings_and_close() -> None:
        runtime = active_runtime()
        selected_model = str(model_quick_select.value or "").strip() or None
        if not validate_selected_model_for_apply(
            provider_protocol=runtime.provider_protocol,
            base_url=runtime.provider_connection.api_base_url,
            auth_scheme=runtime.provider_connection.auth_scheme,
            selected_model=selected_model,
            api_key=controller.provider_api_key(
                session_id=controller.active_session_id
            ),
        ):
            return
        runtime.selected_model = selected_model
        remember_recent_runtime_values(runtime)
        controller.save_active_session()
        dialogs["model_settings"].close()
        refresh_all()

    def apply_mode_settings_and_close() -> None:
        runtime = active_runtime()
        runtime.response_mode_strategy = ResponseModeStrategy(
            str(mode_quick_select.value)
        )
        controller.save_active_session()
        dialogs["mode_settings"].close()
        refresh_all()

    def apply_workspace_settings_and_close() -> None:
        runtime = active_runtime()
        workspace_path = str(workspace_quick_input.value or "").strip()
        runtime.root_path = (
            expanded_path_text(workspace_path) if workspace_path else None
        )
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
            candidate = expand_app_path(raw).resolve()
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
            if os.name == "nt":
                drive_roots = _available_windows_drive_roots()
                if drive_roots:
                    ui.label("Drives").classes("text-xs llmt-muted q-px-sm")
                    with ui.row().classes("w-full items-center gap-1 q-px-sm"):
                        for drive_root in drive_roots:
                            ui.button(
                                str(drive_root),
                                icon="storage",
                                on_click=lambda _event, target=drive_root: (
                                    render_workspace_browser(target)
                                ),
                            ).props("flat no-caps")
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
                    with ui.row().classes("llmt-brand items-center gap-1 no-wrap"):
                        _render_brand_icon(
                            controller.admin_settings.branding.icon_name,
                            "llmt-brand-icon",
                        )
                        ui.label(controller.admin_settings.branding.short_name).classes(
                            "text-xs llmt-muted"
                        )
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
                    information_security_button.set_visibility(
                        controller.information_protection_enabled()
                    )
                with ui.row().classes("llmt-header-actions items-center gap-2"):
                    status_chip = ui.badge("ready").props("outline")
                    with (
                        ui.button(icon="account_circle").props("flat round"),
                        ui.menu().classes("llmt-account-menu"),
                    ):
                        account_name, account_role = _account_menu_identity_labels(
                            controller.current_user
                        )
                        ui.label(account_name).classes("text-sm q-px-sm q-pt-sm")
                        ui.label(account_role).classes(
                            "text-xs llmt-muted q-px-sm q-pb-sm"
                        )
                        for label in _account_menu_action_labels(
                            controller.current_user
                        ):
                            if label == "Settings":
                                ui.menu_item(label, on_click=open_settings_dialog)
                            elif label == "Admin":
                                ui.menu_item(label, on_click=open_admin_dialog)
                            elif label == "Log out":
                                ui.menu_item(label, on_click=logout_hosted_user)
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
                    "llmt-context-meter-row w-full"
                ) as context_meter_row:
                    pass
                with ui.row().classes(
                    "llmt-selected-tools w-full items-center gap-1"
                ) as selected_tools_row:
                    pass
                with ui.row().classes(
                    "llmt-composer-meta w-full items-center gap-1"
                ) as composer_meta_row:
                    pass
                composer_disclaimer_label = ui.label(
                    "AI can make mistakes. Check important info."
                ).classes("llmt-composer-disclaimer llmt-muted")
                composer_disclaimer_label.set_visibility(
                    controller.active_record.runtime.show_footer_help
                )

        with ui.column() as workbench_column:
            pass

    with ui.dialog() as admin_dialog, ui.card().classes("w-[560px] max-h-[88vh]"):
        dialogs["admin"] = admin_dialog
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Admin").classes("text-lg")
            ui.button(icon="close", on_click=admin_dialog.close).props("flat round")
        ui.label("Manage local assistant users.").classes("text-sm llmt-muted")
        with ui.expansion("Feature Flags (Beta)", icon="science", value=False).classes(
            "llmt-settings-section w-full"
        ):
            ui.label(
                "These capabilities are experimental, incomplete, or not fully validated."
            ).classes("text-xs llmt-muted")
            admin_deep_task_switch = ui.switch(
                "Deep Task mode",
                value=controller.deep_task_mode_enabled(),
                on_change=set_admin_deep_task_enabled,
            ).classes("w-full")
            with admin_deep_task_switch:
                ui.tooltip(
                    "Shows Deep Task as a selectable chat mode before the first message."
                ).props("delay=700")
            admin_information_protection_switch = ui.switch(
                "Information Protection",
                value=controller.admin_settings.information_protection_enabled,
                on_change=lambda event: set_admin_beta_feature_flag(
                    "information_protection_enabled", bool(event.value)
                ),
            ).classes("w-full")
            with admin_information_protection_switch:
                ui.tooltip(
                    "Shows Information Security settings and enables protection checks."
                ).props("delay=700")
            admin_skills_switch = ui.switch(
                "Local skills",
                value=controller.admin_settings.skills_enabled,
                on_change=lambda event: set_admin_beta_feature_flag(
                    "skills_enabled", bool(event.value)
                ),
            ).classes("w-full")
            with admin_skills_switch:
                ui.tooltip(
                    "Shows discovered local skills and permits explicit skill invocation."
                ).props("delay=700")
            admin_ollama_native_provider_switch = ui.switch(
                "Native Ollama provider",
                value=controller.admin_settings.ollama_native_provider_enabled,
                on_change=lambda event: set_admin_beta_feature_flag(
                    "ollama_native_provider_enabled", bool(event.value)
                ),
            ).classes("w-full")
            with admin_ollama_native_provider_switch:
                ui.tooltip("Shows the Ollama native provider protocol.").props(
                    "delay=700"
                )
            admin_ask_sage_native_provider_switch = ui.switch(
                "Native Ask Sage provider",
                value=controller.admin_settings.ask_sage_native_provider_enabled,
                on_change=lambda event: set_admin_beta_feature_flag(
                    "ask_sage_native_provider_enabled", bool(event.value)
                ),
            ).classes("w-full")
            with admin_ask_sage_native_provider_switch:
                ui.tooltip("Shows the Ask Sage native provider protocol.").props(
                    "delay=700"
                )
            admin_write_file_switch = ui.switch(
                "File write tool",
                value=controller.admin_settings.write_file_tool_enabled,
                on_change=lambda event: set_admin_beta_feature_flag(
                    "write_file_tool_enabled", bool(event.value)
                ),
            ).classes("w-full")
            with admin_write_file_switch:
                ui.tooltip(
                    "Shows write_file and permits it to be selected subject to normal approvals."
                ).props("delay=700")
            admin_atlassian_switch = ui.switch(
                "Atlassian tools",
                value=controller.admin_settings.atlassian_tools_enabled,
                on_change=lambda event: set_admin_beta_feature_flag(
                    "atlassian_tools_enabled", bool(event.value)
                ),
            ).classes("w-full")
            with admin_atlassian_switch:
                ui.tooltip("Shows Jira, Confluence, and Bitbucket tools.").props(
                    "delay=700"
                )
            admin_gitlab_switch = ui.switch(
                "GitLab tools",
                value=controller.admin_settings.gitlab_tools_enabled,
                on_change=lambda event: set_admin_beta_feature_flag(
                    "gitlab_tools_enabled", bool(event.value)
                ),
            ).classes("w-full")
            with admin_gitlab_switch:
                ui.tooltip("Shows GitLab tools.").props("delay=700")
        with ui.column().classes("llmt-settings-section w-full gap-2"):
            ui.label("Branding").classes("text-base")
            admin_branding_app_name_input = ui.input("App name").classes("w-full")
            admin_branding_short_name_input = ui.input("Short name").classes("w-full")
            admin_branding_icon_name_input = ui.input("Icon or symbol").classes(
                "w-full"
            )
            with admin_branding_icon_name_input:
                ui.tooltip(
                    "Material icon name or literal symbol used in the app chrome."
                ).props("delay=700")
            admin_branding_favicon_input = (
                ui.textarea("Favicon SVG").props("autogrow outlined").classes("w-full")
            )
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Reset", on_click=reset_admin_branding).props("flat")
                ui.button("Save branding", on_click=save_admin_branding).props("flat")
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

    def settings_expansion(title: str) -> Any:
        return (
            ui.expansion(title, value=_settings_section_default_open(title))
            .props("expand-separator")
            .classes("llmt-settings-expansion w-full")
        )

    with ui.dialog() as settings_dialog, ui.card().classes("llmt-settings-card"):
        dialogs["settings"] = settings_dialog
        ui.label("Settings").classes("text-lg q-pb-sm")
        with (
            ui.scroll_area().classes("llmt-settings-body"),
            ui.column().classes("w-full gap-2 q-pr-sm q-pb-sm"),
        ):
            with (
                settings_expansion("Connection"),
                ui.column().classes("w-full gap-2 q-pt-sm"),
            ):
                provider_select = ui.select(
                    provider_protocol_options(),
                    label="Provider",
                ).classes("w-full")
                with ui.row().classes("w-full items-end gap-2 no-wrap"):
                    model_input = (
                        ui.select(
                            (
                                [controller.active_record.runtime.selected_model]
                                if controller.active_record.runtime.selected_model
                                else []
                            ),
                            label="Model",
                            value=controller.active_record.runtime.selected_model,
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
                    [strategy.value for strategy in ResponseModeStrategy],
                    label="Response mode",
                ).classes("w-full")
                with ui.row().classes("w-full items-end gap-2"):
                    temperature_input = (
                        ui.input("Temperature")
                        .props("type=number step=0.05 min=0 max=1")
                        .classes("grow")
                    )
                    timeout_seconds_input = (
                        ui.input("Provider timeout seconds")
                        .props("type=number step=1 min=1")
                        .classes("grow")
                    )
                with ui.row().classes("w-full items-end gap-2 no-wrap"):
                    base_url_input = (
                        ui.select(
                            base_url_options(controller.active_record.runtime),
                            label="Base URL",
                            with_input=True,
                            new_value_mode="add-unique",
                        )
                        .props("clearable")
                        .classes("grow")
                    )
                    with base_url_input:
                        ui.tooltip(_provider_base_url_help_text()).props("delay=700")
                    render_provider_endpoint_help_button()
                auth_scheme_select = ui.select(
                    NICEGUI_PROVIDER_AUTH_OPTIONS,
                    label="Auth scheme",
                    value=controller.active_record.runtime.provider_connection.auth_scheme.value,
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
            with (
                settings_expansion("Workspace"),
                ui.row().classes("w-full items-end gap-2 no-wrap q-pt-sm"),
            ):
                workspace_input = ui.input("Workspace root").classes("grow")
                ui.button(
                    icon="folder_open",
                    on_click=lambda: open_workspace_browser(target="settings"),
                ).props("flat round color=primary")
            with (
                settings_expansion("Display"),
                ui.column().classes("w-full gap-1 q-pt-sm"),
            ):
                dark_mode_switch = ui.switch(
                    "Dark mode",
                    value=controller.preferences.theme_mode == "dark",
                ).classes("w-full")
                show_token_usage_switch = ui.switch("Show token usage").classes(
                    "w-full"
                )
                show_footer_help_switch = ui.switch("Show footer disclaimer").classes(
                    "w-full"
                )
                inspector_open_switch = ui.switch("Auto-open workbench").classes(
                    "w-full"
                )
            with (
                settings_expansion("Persistence"),
                ui.row().classes("w-full items-end gap-2 no-wrap q-pt-sm"),
            ):
                database_path_input = ui.input("SQLite database").classes("grow")
                if controller.current_user is not None:
                    database_path_input.disable()
                ui.button(
                    icon="folder_open",
                    on_click=lambda: open_workspace_browser(target="database"),
                ).props("flat round color=primary")
            with (
                settings_expansion("Session permissions"),
                ui.column().classes("w-full gap-1 q-pt-sm"),
            ):
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
            with (
                settings_expansion("Chat limits"),
                ui.column().classes("w-full gap-2 q-pt-sm"),
            ):
                with ui.row().classes("w-full items-end gap-2"):
                    max_context_tokens_input = (
                        ui.input("Max context tokens")
                        .props("type=number step=1000 min=1")
                        .classes("grow")
                    )
                    max_tool_round_trips_input = (
                        ui.input("Max tool rounds")
                        .props("type=number step=1 min=1")
                        .classes("grow")
                    )
                with ui.row().classes("w-full items-end gap-2"):
                    max_tool_calls_per_round_input = (
                        ui.input("Max tool calls per round")
                        .props("type=number step=1 min=1")
                        .classes("grow")
                    )
                    max_total_tool_calls_per_turn_input = (
                        ui.input("Max total tool calls per turn")
                        .props("type=number step=1 min=1")
                        .classes("grow")
                    )
            with (
                settings_expansion("Tool limits"),
                ui.column().classes("w-full gap-2 q-pt-sm"),
            ):
                with ui.row().classes("w-full items-end gap-2"):
                    max_entries_per_call_input = (
                        ui.input("Max entries per call")
                        .props("type=number step=1 min=1")
                        .classes("grow")
                    )
                    max_recursive_depth_input = (
                        ui.input("Max recursive depth")
                        .props("type=number step=1 min=1")
                        .classes("grow")
                    )
                with ui.row().classes("w-full items-end gap-2"):
                    max_files_scanned_input = (
                        ui.input("Max files scanned")
                        .props("type=number step=1 min=1")
                        .classes("grow")
                    )
                    max_search_matches_input = (
                        ui.input("Max search matches")
                        .props("type=number step=1 min=1")
                        .classes("grow")
                    )
                with ui.row().classes("w-full items-end gap-2"):
                    max_read_lines_input = (
                        ui.input("Max read lines")
                        .props("type=number step=1 min=1")
                        .classes("grow")
                    )
                    max_read_input_bytes_input = (
                        ui.input("Max read input bytes")
                        .props("type=number step=1024 min=1")
                        .classes("grow")
                    )
                with ui.row().classes("w-full items-end gap-2"):
                    max_file_size_characters_input = (
                        ui.input("Max file size characters")
                        .props("type=number step=1024 min=1")
                        .classes("grow")
                    )
                    max_read_file_chars_input = (
                        ui.input("Max read file chars")
                        .props("type=number step=1024 min=1 clearable")
                        .classes("grow")
                    )
                max_tool_result_chars_input = (
                    ui.input("Max tool result chars")
                    .props("type=number step=1024 min=1")
                    .classes("w-full")
                )
            with (
                settings_expansion("Deep Task limits"),
                ui.column().classes("w-full gap-2 q-pt-sm"),
            ):
                deep_task_max_turns_input = (
                    ui.input("Deep Task max turns")
                    .props("type=number step=1 min=1")
                    .classes("w-full")
                )
                with ui.row().classes("w-full items-end gap-2"):
                    deep_task_max_tool_invocations_input = (
                        ui.input("Deep Task max tool invocations")
                        .props("type=number step=1 min=1 clearable")
                        .classes("grow")
                    )
                    deep_task_max_elapsed_seconds_input = (
                        ui.input("Deep Task max elapsed seconds")
                        .props("type=number step=1 min=1 clearable")
                        .classes("grow")
                    )
                deep_task_include_replay_switch = ui.switch(
                    "Include replay in Deep Task artifacts"
                ).classes("w-full")
            with (
                settings_expansion("Tool credentials"),
                ui.column().classes("w-full gap-0") as tool_credentials_column,
            ):
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
            protection_category_catalog_input = (
                ui.textarea(
                    "Category catalog",
                    placeholder=(
                        "TRIVIAL: public, routine | Freely discussable information\n"
                        "MINOR: low, limited | Low-sensitivity source wording"
                    ),
                    value=_format_information_security_category_catalog(
                        controller.active_record.runtime.protection.sensitivity_categories
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
            ui.label(
                "Corpus documents may be Markdown, text, PDF, Word, PowerPoint, or spreadsheets. "
                "Optional categories can come from front matter, category-named folders, or a local source metadata sidecar."
            ).classes("text-xs llmt-muted")
            with ui.column().classes("w-full gap-1") as protection_issues_column:
                pass
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=information_security_dialog.close).props(
                "flat"
            )
            ui.button("Apply", on_click=apply_information_security_settings_and_close)

    with ui.dialog() as credential_reentry_dialog, ui.card().classes("w-[420px]"):
        dialogs["credential_reentry"] = credential_reentry_dialog
        credential_reentry_dialog.props("persistent")
        credential_reentry_title = ui.label("Re-enter credential").classes("text-lg")
        credential_reentry_label = ui.label("").classes("text-sm llmt-muted")
        credential_reentry_input = (
            ui.input("Credential")
            .props("type=password autocomplete=off")
            .classes("w-full")
        )
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=cancel_credential_reentry).props("flat")
            ui.button("Continue", on_click=submit_credential_reentry)

    with ui.dialog() as provider_settings_dialog, ui.card().classes("w-[460px]"):
        dialogs["provider_settings"] = provider_settings_dialog
        ui.label("Provider").classes("text-lg")
        provider_quick_select = ui.select(
            provider_protocol_options(),
            label="Provider",
        ).classes("w-full")
        with ui.row().classes("w-full items-end gap-2 no-wrap"):
            base_url_quick_input = (
                ui.select(
                    base_url_options(controller.active_record.runtime),
                    label="Base URL",
                    with_input=True,
                    new_value_mode="add-unique",
                )
                .props("clearable")
                .classes("grow")
            )
            with base_url_quick_input:
                ui.tooltip(_provider_base_url_help_text()).props("delay=700")
            render_provider_endpoint_help_button()
        auth_scheme_quick_select = ui.select(
            NICEGUI_PROVIDER_AUTH_OPTIONS,
            label="Auth scheme",
            value=controller.active_record.runtime.provider_connection.auth_scheme.value,
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
                    (
                        [controller.active_record.runtime.selected_model]
                        if controller.active_record.runtime.selected_model
                        else []
                    ),
                    label="Model",
                    value=controller.active_record.runtime.selected_model,
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
        ui.label("Response mode").classes("text-lg")
        mode_quick_select = ui.select(
            [strategy.value for strategy in ResponseModeStrategy],
            label="Response mode",
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

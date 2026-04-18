"""Shared session-control helpers for interactive repository chat apps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, TypeVar

from llm_tools.apps.chat_config import TextualChatConfig
from llm_tools.tool_api import SideEffectClass, ToolSpec

ProviderT = TypeVar("ProviderT")


@dataclass(slots=True)
class ChatControlNotice:
    """One transcript-ready message emitted by a command handler."""

    role: Literal["system", "error"]
    text: str


@dataclass(slots=True)
class ChatControlState:
    """Mutable session-scoped control state shared by chat frontends."""

    active_model_name: str
    default_enabled_tools: set[str]
    enabled_tools: set[str]
    require_approval_for: set[SideEffectClass]
    inspector_open: bool = False


@dataclass(slots=True)
class ModelCatalogOutcome:
    """Result of listing available models for a chat frontend."""

    model_ids: list[str] | None = None
    notice: ChatControlNotice | None = None
    unavailable: bool = False


@dataclass(slots=True)
class ModelSwitchOutcome(Generic[ProviderT]):
    """Result of switching the active model for a chat frontend."""

    provider: ProviderT | None = None
    notice: ChatControlNotice | None = None
    unavailable: bool = False


@dataclass(slots=True)
class ChatCommandOutcome(Generic[ProviderT]):
    """Outcome of one parsed chat command."""

    handled: bool
    notices: list[ChatControlNotice] = field(default_factory=list)
    provider: ProviderT | None = None
    request_copy: bool = False
    request_exit: bool = False


def resolve_default_enabled_tools(
    config: TextualChatConfig,
    *,
    available_tool_names: set[str],
) -> set[str]:
    """Return the session-default tool set for the given frontend config."""
    configured_tools = config.policy.enabled_tools
    if configured_tools is None:
        return set(available_tool_names)
    return {
        tool_name for tool_name in configured_tools if tool_name in available_tool_names
    }


def build_chat_control_state(
    config: TextualChatConfig,
    *,
    available_tool_names: set[str],
) -> ChatControlState:
    """Build the default mutable session controls for a chat frontend."""
    default_enabled_tools = resolve_default_enabled_tools(
        config,
        available_tool_names=available_tool_names,
    )
    return ChatControlState(
        active_model_name=config.llm.model_name,
        default_enabled_tools=set(default_enabled_tools),
        enabled_tools=set(default_enabled_tools),
        require_approval_for=set(config.policy.require_approval_for),
        inspector_open=config.ui.inspector_open_by_default,
    )


def build_startup_message(
    *,
    root_path: Path,
    model_name: str,
    exit_hint: str,
) -> str:
    """Return the shared startup transcript copy for chat frontends."""
    return (
        f"Root: {root_path}\nModel: {model_name}\nUse /help for guidance. {exit_hint}"
    )


def build_help_text(*, exit_hint: str) -> str:
    """Return the shared inline help text for chat frontends."""
    return (
        "Ask grounded questions about the selected root. Use /model to "
        "inspect or switch models, /tools to manage tools, /approvals "
        "to toggle approvals, /inspect to toggle the inspector, and "
        f"/copy to open transcript export. {exit_hint}"
    )


def build_tool_state_payload(
    state: ChatControlState,
    *,
    available_tool_specs: Mapping[str, ToolSpec],
) -> dict[str, object]:
    """Return the shared inspector payload describing current tool controls."""
    return {
        "enabled_tools": sorted(state.enabled_tools),
        "disabled_tools": sorted(
            set(available_tool_specs).difference(state.enabled_tools)
        ),
        "require_approval_for": sorted(
            side_effect.value for side_effect in state.require_approval_for
        ),
    }


def build_tools_command_text(
    state: ChatControlState,
    *,
    available_tool_specs: Mapping[str, ToolSpec],
) -> str:
    """Return the shared transcript text for the `/tools` command."""
    approval_suffix = (
        " (approval required)"
        if SideEffectClass.LOCAL_READ in state.require_approval_for
        else ""
    )
    enabled_lines = [
        (
            f"- {tool_name}: "
            f"{available_tool_specs[tool_name].side_effects.value}"
            f"{approval_suffix}"
        )
        for tool_name in sorted(state.enabled_tools)
    ]
    disabled_lines = [
        f"- {tool_name}"
        for tool_name in sorted(
            set(available_tool_specs).difference(state.enabled_tools)
        )
    ]
    return "\n\n".join(
        [
            "Enabled tools:\n"
            + ("\n".join(enabled_lines) if enabled_lines else "- none"),
            "Disabled tools:\n"
            + ("\n".join(disabled_lines) if disabled_lines else "- none"),
        ]
    )


def handle_tools_command(
    user_message: str,
    *,
    state: ChatControlState,
    available_tool_specs: Mapping[str, ToolSpec],
) -> ChatCommandOutcome[ProviderT]:
    """Mutate tool controls for one `/tools` command."""
    parts = user_message.strip().split()
    if len(parts) == 1:
        return ChatCommandOutcome(
            handled=True,
            notices=[
                ChatControlNotice(
                    role="system",
                    text=build_tools_command_text(
                        state,
                        available_tool_specs=available_tool_specs,
                    ),
                )
            ],
        )
    if len(parts) == 2 and parts[1].lower() == "reset":
        state.enabled_tools = set(state.default_enabled_tools)
        return ChatCommandOutcome(
            handled=True,
            notices=[
                ChatControlNotice(
                    role="system",
                    text="Restored the default session tool set.",
                )
            ],
        )
    if len(parts) == 3 and parts[1].lower() in {"enable", "disable"}:
        tool_name = parts[2].strip()
        if tool_name not in available_tool_specs:
            return ChatCommandOutcome(
                handled=True,
                notices=[
                    ChatControlNotice(role="error", text=f"Unknown tool: {tool_name}")
                ],
            )
        if parts[1].lower() == "enable":
            state.enabled_tools.add(tool_name)
            message = f"Enabled tool: {tool_name}"
        else:
            state.enabled_tools.discard(tool_name)
            message = f"Disabled tool: {tool_name}"
        return ChatCommandOutcome(
            handled=True,
            notices=[ChatControlNotice(role="system", text=message)],
        )
    return ChatCommandOutcome(
        handled=True,
        notices=[
            ChatControlNotice(
                role="system",
                text=(
                    "Usage: /tools | /tools enable <tool_name> | "
                    "/tools disable <tool_name> | /tools reset"
                ),
            )
        ],
    )


def handle_approvals_command(
    user_message: str,
    *,
    state: ChatControlState,
) -> ChatCommandOutcome[ProviderT]:
    """Mutate approval controls for one `/approvals` command."""
    parts = user_message.strip().split()
    if len(parts) == 1:
        enabled = SideEffectClass.LOCAL_READ in state.require_approval_for
        return ChatCommandOutcome(
            handled=True,
            notices=[
                ChatControlNotice(
                    role="system",
                    text=(
                        "Approvals are ON for local_read tools."
                        if enabled
                        else "Approvals are OFF for local_read tools."
                    ),
                )
            ],
        )
    if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
        if parts[1].lower() == "on":
            state.require_approval_for.add(SideEffectClass.LOCAL_READ)
            message = "Enabled approvals for local_read tools."
        else:
            state.require_approval_for.discard(SideEffectClass.LOCAL_READ)
            message = "Disabled approvals for local_read tools."
        return ChatCommandOutcome(
            handled=True,
            notices=[ChatControlNotice(role="system", text=message)],
        )
    return ChatCommandOutcome(
        handled=True,
        notices=[
            ChatControlNotice(
                role="system",
                text="Usage: /approvals | /approvals on | /approvals off",
            )
        ],
    )


def handle_model_command(
    user_message: str,
    *,
    state: ChatControlState,
    busy: bool,
    list_models: Callable[[], ModelCatalogOutcome],
    switch_model: Callable[[str], ModelSwitchOutcome[ProviderT]],
) -> ChatCommandOutcome[ProviderT]:
    """Handle `/model` listing and switching semantics."""
    if busy:
        return ChatCommandOutcome(
            handled=True,
            notices=[
                ChatControlNotice(
                    role="system",
                    text="Stop the active turn before changing models.",
                )
            ],
        )
    parts = user_message.strip().split(maxsplit=1)
    cleaned_model_name = parts[1].strip() if len(parts) > 1 else ""
    if not cleaned_model_name:
        catalog = list_models()
        if catalog.unavailable:
            return ChatCommandOutcome(handled=True)
        if catalog.notice is not None:
            return ChatCommandOutcome(handled=True, notices=[catalog.notice])
        if not catalog.model_ids:
            return ChatCommandOutcome(
                handled=True,
                notices=[
                    ChatControlNotice(
                        role="system",
                        text=(
                            f"Current model: {state.active_model_name}\n"
                            "No models were returned by models.list."
                        ),
                    )
                ],
            )
        available = "\n".join(f"- {model_id}" for model_id in catalog.model_ids)
        return ChatCommandOutcome(
            handled=True,
            notices=[
                ChatControlNotice(
                    role="system",
                    text=(
                        f"Current model: {state.active_model_name}\n"
                        f"Available models:\n{available}"
                    ),
                )
            ],
        )
    if cleaned_model_name == state.active_model_name:
        return ChatCommandOutcome(
            handled=True,
            notices=[
                ChatControlNotice(
                    role="system",
                    text=f"Current model: {cleaned_model_name}",
                )
            ],
        )
    switch_outcome = switch_model(cleaned_model_name)
    if switch_outcome.unavailable:
        return ChatCommandOutcome(handled=True)
    if switch_outcome.notice is not None and switch_outcome.provider is None:
        return ChatCommandOutcome(
            handled=True,
            notices=[switch_outcome.notice],
        )
    previous_model = state.active_model_name
    state.active_model_name = cleaned_model_name
    notices = (
        [switch_outcome.notice]
        if switch_outcome.notice is not None
        else [
            ChatControlNotice(
                role="system",
                text=f"Switched model from {previous_model} to {cleaned_model_name}.",
            )
        ]
    )
    return ChatCommandOutcome(
        handled=True,
        notices=notices,
        provider=switch_outcome.provider,
    )


def handle_chat_command(
    user_message: str,
    *,
    state: ChatControlState,
    available_tool_specs: Mapping[str, ToolSpec],
    busy: bool,
    list_models: Callable[[], ModelCatalogOutcome],
    switch_model: Callable[[str], ModelSwitchOutcome[ProviderT]],
    exit_mode: Literal["request_exit", "notice"],
    exit_notice: str,
) -> ChatCommandOutcome[ProviderT]:
    """Handle a frontend chat command against shared session controls."""
    normalized = user_message.strip().lower()
    if normalized == "/help":
        return ChatCommandOutcome(
            handled=True,
            notices=[
                ChatControlNotice(
                    role="system", text=build_help_text(exit_hint=exit_notice)
                )
            ],
        )
    if normalized.startswith("/tools"):
        return handle_tools_command(
            user_message,
            state=state,
            available_tool_specs=available_tool_specs,
        )
    if normalized.startswith("/approvals"):
        return handle_approvals_command(user_message, state=state)
    if normalized == "/inspect":
        state.inspector_open = not state.inspector_open
        return ChatCommandOutcome(handled=True)
    if normalized.startswith("/model"):
        return handle_model_command(
            user_message,
            state=state,
            busy=busy,
            list_models=list_models,
            switch_model=switch_model,
        )
    if normalized == "/copy":
        return ChatCommandOutcome(handled=True, request_copy=True)
    if normalized in {"quit", "exit"}:
        if exit_mode == "request_exit":
            return ChatCommandOutcome(handled=True, request_exit=True)
        return ChatCommandOutcome(
            handled=True,
            notices=[ChatControlNotice(role="system", text=exit_notice)],
        )
    return ChatCommandOutcome(handled=False)

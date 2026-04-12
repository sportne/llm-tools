"""Controller helpers for the interactive repository chat screen lifecycle."""

from __future__ import annotations

from os import getenv
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4

from textual.containers import VerticalScroll
from textual.timer import Timer
from textual.widgets import Button, Static

from llm_tools.apps.textual_chat.models import ChatLLMConfig, ProviderPreset
from llm_tools.apps.textual_chat.presentation import (
    AssistantMarkdownEntry,
    TranscriptEntry,
    format_final_response,
    format_final_response_metadata,
    pretty_json,
)
from llm_tools.apps.textual_chat.prompts import build_chat_system_prompt
from llm_tools.apps.textual_chat.screens import (
    ComposerTextArea,
    CredentialModal,
    InterruptConfirmModal,
    TranscriptCopyModal,
)
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderModeStrategy
from llm_tools.tool_api import (
    SideEffectClass,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    ToolSpec,
)
from llm_tools.tools import register_filesystem_tools, register_text_tools
from llm_tools.workflow_api import (
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
    WorkflowExecutor,
)

if TYPE_CHECKING:
    from llm_tools.apps.textual_chat.app import ChatScreen

STATUS_MIN_DISPLAY_SECONDS = 0.4
STATUS_ANIMATION_INTERVAL_SECONDS = 0.25


class ChatScreenController:
    """Encapsulate turn lifecycle and UI state transitions for ChatScreen."""

    def __init__(self, screen: ChatScreen) -> None:
        self._screen = screen
        self._status_base_text = ""
        self._status_pending_text: str | None = None
        self._status_visible_since = 0.0
        self._status_animation_step = 0
        self._status_hold_timer: Timer | None = None
        self._status_animation_timer: Timer | None = None

    def initialize_provider(self) -> bool:
        if self._screen._provider is not None:
            return True
        try:
            self._screen._provider = self._screen._create_provider(
                self._screen._config.llm,
                api_key=self._screen._credential_secret,
                model_name=self._screen._active_model_name,
            )
        except Exception as exc:
            self.append_transcript("error", str(exc))
            self.set_status("")
            return False
        return True

    def ensure_provider_ready(self) -> bool:
        if self._screen._provider is not None:
            return True
        metadata = self._screen._config.llm.credential_prompt_metadata()
        if not metadata.expects_api_key:
            return self.initialize_provider()
        env_value = getenv(self._screen._config.llm.api_key_env_var or "")
        if env_value:
            self._screen._credential_secret = env_value
            return self.initialize_provider()
        if not self._screen._credential_prompt_completed:
            self._screen.app.push_screen(
                CredentialModal(metadata),
                callback=self.handle_credential_submit,
            )
            return False
        return self.initialize_provider()

    def append_transcript(
        self,
        role: str,
        text: str,
        *,
        assistant_completion_state: str = "complete",
    ) -> TranscriptEntry:
        transcript = self._screen.query_one("#transcript", VerticalScroll)
        entry = TranscriptEntry(
            role=role,
            text=text,
            assistant_completion_state=assistant_completion_state,
        )
        transcript.mount(entry)
        transcript.scroll_end(animate=False)
        return entry

    def append_assistant_markdown(
        self,
        markdown_text: str,
        *,
        metadata_text: str,
        fallback_text: str,
    ) -> AssistantMarkdownEntry | TranscriptEntry:
        transcript = self._screen.query_one("#transcript", VerticalScroll)
        try:
            entry = AssistantMarkdownEntry(
                markdown_text=markdown_text,
                metadata_text=metadata_text,
            )
            transcript.mount(entry)
            transcript.scroll_end(animate=False)
            return entry
        except Exception:
            return self.append_transcript("assistant", fallback_text)

    def set_status(self, text: str) -> None:
        requested_text = text.strip()
        current_text = self._status_base_text
        now = monotonic()
        if requested_text == current_text:
            self._status_pending_text = None
            self._stop_status_hold_timer()
            return
        if current_text:
            elapsed = now - self._status_visible_since
            if elapsed < STATUS_MIN_DISPLAY_SECONDS:
                self._status_pending_text = requested_text
                self._schedule_status_transition(STATUS_MIN_DISPLAY_SECONDS - elapsed)
                return
        self._apply_status_text(requested_text)

    def _apply_status_text(self, text: str) -> None:
        self._stop_status_hold_timer()
        self._status_pending_text = None
        self._status_base_text = text
        self._status_animation_step = 0
        self._status_visible_since = monotonic() if text else 0.0
        self._render_status_text()
        if not text:
            if self._status_animation_timer is not None:
                self._status_animation_timer.pause()
            return
        if self._status_animation_timer is None:
            self._status_animation_timer = self._screen.set_interval(
                STATUS_ANIMATION_INTERVAL_SECONDS,
                self._advance_status_animation,
                pause=True,
            )
        self._status_animation_timer.reset()
        self._status_animation_timer.resume()

    def _schedule_status_transition(self, delay_seconds: float) -> None:
        self._stop_status_hold_timer()
        self._status_hold_timer = self._screen.set_timer(
            delay_seconds,
            self._flush_pending_status,
        )

    def _stop_status_hold_timer(self) -> None:
        if self._status_hold_timer is not None:
            self._status_hold_timer.stop()
            self._status_hold_timer = None

    def _flush_pending_status(self) -> None:
        pending_text = self._status_pending_text
        self._status_hold_timer = None
        if pending_text is None:
            return
        self._apply_status_text(pending_text)

    def _advance_status_animation(self) -> None:
        if not self._status_base_text:
            if self._status_animation_timer is not None:
                self._status_animation_timer.pause()
            return
        self._status_animation_step = (
            1 if self._status_animation_step >= 3 else self._status_animation_step + 1
        )
        self._render_status_text()

    def _render_status_text(self) -> None:
        rendered = self._status_base_text
        if rendered and self._status_animation_step:
            rendered = f"{rendered}{'.' * self._status_animation_step}"
        self._screen.query_one("#status-text", Static).update(rendered)
        self._refresh_approval_actions()

    def _refresh_approval_actions(self) -> None:
        has_pending_approval = self._screen._pending_approval is not None
        decision_in_flight = self._screen._approval_decision_in_flight
        approve_button = self._screen.query_one("#approve-button", Button)
        deny_button = self._screen.query_one("#deny-button", Button)
        approve_button.display = has_pending_approval
        deny_button.display = has_pending_approval
        approve_button.disabled = not has_pending_approval or decision_in_flight
        deny_button.disabled = not has_pending_approval or decision_in_flight

    def refresh_footer(self) -> None:
        session_tokens = (
            self._screen._footer_session_tokens
            if self._screen._footer_session_tokens is not None
            else "-"
        )
        active_context_tokens = (
            self._screen._footer_active_context_tokens
            if self._screen._footer_active_context_tokens is not None
            else "-"
        )
        footer = f"session tokens: {session_tokens} | active context tokens: {active_context_tokens}"
        if self._screen._footer_confidence is not None:
            footer += f" | confidence: {self._screen._footer_confidence:.2f}"
        if self._screen._busy:
            footer += " | Enter send | Shift+Enter newline | Stop active turn | F6 copy transcript | F7 inspector"
        else:
            footer += (
                " | Enter send | Shift+Enter newline | F6 copy transcript | "
                "F7 inspector | /help | quit"
            )
        self._screen.query_one("#footer-bar", Static).update(footer)
        self._screen.query_one("#stop-button", Button).disabled = not self._screen._busy
        self._refresh_approval_actions()

    def update_footer_metrics(
        self,
        *,
        session_tokens: int | None,
        active_context_tokens: int | None,
        confidence: float | None,
    ) -> None:
        self._screen._footer_session_tokens = session_tokens
        self._screen._footer_active_context_tokens = active_context_tokens
        self._screen._footer_confidence = confidence
        self.refresh_footer()

    def refresh_inspector(self) -> None:
        inspector_pane = self._screen.query_one("#inspector-pane", VerticalScroll)
        inspector_pane.display = self._screen._inspector_open
        self._screen.query_one("#tool-state-box", Static).update(
            self._render_tool_state()
        )
        self._screen.query_one("#pending-approval-box", Static).update(
            self._render_pending_approval()
        )
        self._screen.query_one("#provider-messages-box", Static).update(
            self._render_inspector_entries(self._screen._inspector_provider_messages)
        )
        self._screen.query_one("#parsed-response-box", Static).update(
            self._render_inspector_entries(self._screen._inspector_parsed_responses)
        )
        self._screen.query_one("#tool-execution-box", Static).update(
            self._render_inspector_entries(self._screen._inspector_tool_executions)
        )

    def _render_tool_state(self) -> str:
        enabled_tools = sorted(self._screen._enabled_tools)
        disabled_tools = sorted(
            set(self._screen._available_tool_specs).difference(
                self._screen._enabled_tools
            )
        )
        approvals_enabled = sorted(
            side_effect.value for side_effect in self._screen._require_approval_for
        )
        payload = {
            "enabled_tools": enabled_tools,
            "disabled_tools": disabled_tools,
            "require_approval_for": approvals_enabled,
        }
        return pretty_json(payload)

    def _render_pending_approval(self) -> str:
        if self._screen._pending_approval is None:
            return "No pending approval."
        return pretty_json(self._screen._pending_approval.model_dump(mode="json"))

    def _render_inspector_entries(self, entries: list[dict[str, object]]) -> str:
        if not entries:
            return ""
        return "\n\n".join(
            f"{entry['label']}\n{pretty_json(entry['payload'])}" for entry in entries
        )

    def append_inspector_entry(
        self,
        entries: list[dict[str, object]],
        *,
        label: str,
        payload: object,
    ) -> None:
        entries.append({"label": label, "payload": payload})
        self.refresh_inspector()

    def clear_composer(self) -> None:
        self._screen.query_one("#composer", ComposerTextArea).load_text("")

    def _describe_active_model(self) -> str:
        return f"Current model: {self._screen._active_model_name}"

    def _show_available_models(self) -> None:
        provider = self._screen._provider
        if provider is None:
            self.append_transcript(
                "system",
                f"{self._describe_active_model()}\nProvider is not configured yet.",
            )
            return
        try:
            model_ids = provider.list_available_models()
        except Exception as exc:
            self.append_transcript(
                "system",
                f"{self._describe_active_model()}\nUnable to list available models: {exc}",
            )
            return
        if not model_ids:
            self.append_transcript(
                "system",
                f"{self._describe_active_model()}\nNo models were returned by models.list.",
            )
            return
        available = "\n".join(f"- {model_id}" for model_id in model_ids)
        self.append_transcript(
            "system",
            f"{self._describe_active_model()}\nAvailable models:\n{available}",
        )

    def _switch_active_model(self, new_model_name: str) -> None:
        if self._screen._busy:
            self.append_transcript(
                "system",
                "Stop the active turn before changing models.",
            )
            return
        cleaned_model_name = new_model_name.strip()
        if not cleaned_model_name:
            self._show_available_models()
            return
        if cleaned_model_name == self._screen._active_model_name:
            self.append_transcript("system", f"Current model: {cleaned_model_name}")
            return
        try:
            provider = self._screen._create_provider(
                self._screen._config.llm,
                api_key=self._screen._credential_secret,
                model_name=cleaned_model_name,
            )
        except Exception as exc:
            self.append_transcript(
                "error",
                f"Unable to switch model to {cleaned_model_name}: {exc}",
            )
            return
        previous = self._screen._active_model_name
        self._screen._provider = provider
        self._screen._active_model_name = cleaned_model_name
        self.append_transcript(
            "system",
            f"Switched model from {previous} to {cleaned_model_name}.",
        )

    def handle_credential_submit(self, secret_value: str | None) -> None:
        self._screen._credential_secret = secret_value
        self._screen._credential_prompt_completed = True
        self.initialize_provider()
        self._screen.query_one("#composer", ComposerTextArea).focus()

    def handle_interrupt_confirmation(self, confirmed: bool | None) -> None:
        if confirmed:
            pending_draft = self._screen.query_one("#composer", ComposerTextArea).text
            self._screen._pending_interrupt_draft = (
                pending_draft if pending_draft.strip() else None
            )
            self.cancel_active_turn(status_text="stopping")
        self._screen.query_one("#composer", ComposerTextArea).focus()

    def handle_turn_error(self, error_message: str) -> None:
        self.append_transcript("error", error_message)
        self.set_status("")
        self._screen._busy = False
        self._screen._active_runner = None
        self._screen._active_assistant_entry = None
        self._screen._pending_approval = None
        self._screen._approval_decision_in_flight = False
        self._screen._pending_interrupt_draft = None
        self.refresh_footer()
        self.refresh_inspector()
        self._screen.query_one("#composer", ComposerTextArea).focus()

    def handle_turn_status(self, event: object) -> None:
        typed_event = ChatWorkflowStatusEvent.model_validate(event)
        self.set_status(typed_event.status)

    def handle_turn_approval_requested(self, event: object) -> None:
        typed_event = ChatWorkflowApprovalEvent.model_validate(event)
        self._screen._pending_approval = typed_event.approval
        self._screen._approval_decision_in_flight = False
        self.set_status(f"approval required for {typed_event.approval.tool_name}")
        self.append_transcript(
            "system",
            (
                f"Approval requested for {typed_event.approval.tool_name}: "
                f"{typed_event.approval.policy_reason}"
            ),
        )
        self._refresh_approval_actions()
        self.refresh_inspector()

    def handle_turn_approval_resolved(self, event: object) -> None:
        typed_event = ChatWorkflowApprovalResolvedEvent.model_validate(event)
        self._screen._pending_approval = None
        self._screen._approval_decision_in_flight = False
        resolution_text = {
            "approved": "Approved pending approval request.",
            "denied": "Denied pending approval request.",
            "timed_out": "Pending approval request timed out.",
            "cancelled": "Pending approval request was cancelled.",
        }[typed_event.resolution]
        self.append_transcript("system", resolution_text)
        if typed_event.resolution == "approved":
            self.set_status("resuming turn")
        elif typed_event.resolution == "denied":
            self.set_status("continuing without approval")
        elif typed_event.resolution == "timed_out":
            self.set_status("approval timed out")
        else:
            self.set_status("")
        self._refresh_approval_actions()
        self.refresh_inspector()

    def handle_turn_inspector_event(self, event: object) -> None:
        typed_event = ChatWorkflowInspectorEvent.model_validate(event)
        label = (
            f"Turn {self._screen._active_turn_number} Round {typed_event.round_index} "
            f"{typed_event.kind.replace('_', ' ')}"
        )
        target = {
            "provider_messages": self._screen._inspector_provider_messages,
            "parsed_response": self._screen._inspector_parsed_responses,
            "tool_execution": self._screen._inspector_tool_executions,
        }[typed_event.kind]
        self.append_inspector_entry(target, label=label, payload=typed_event.payload)

    def show_final_response(self, result: ChatWorkflowTurnResult) -> None:
        final_response = result.final_response
        if final_response is None:
            return
        self._screen._active_assistant_entry = None
        self.append_assistant_markdown(
            final_response.answer.rstrip(),
            metadata_text=format_final_response_metadata(final_response),
            fallback_text=format_final_response(final_response),
        )

    def handle_turn_result(self, event: object) -> None:
        typed_event = ChatWorkflowResultEvent.model_validate(event)
        result = ChatWorkflowTurnResult.model_validate(typed_event.result)
        self._screen._session_state = (
            result.session_state or self._screen._session_state
        )
        self._screen._pending_approval = None
        self._screen._approval_decision_in_flight = False
        if result.context_warning:
            self.append_transcript("system", result.context_warning)
        if result.status == "needs_continuation" and result.continuation_reason:
            self.append_transcript("system", result.continuation_reason)
        if result.final_response is not None:
            self.show_final_response(result)
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
                if self._screen._active_assistant_entry is None:
                    self._screen._active_assistant_entry = self.append_transcript(
                        "assistant",
                        interrupted_message.content,
                        assistant_completion_state="interrupted",
                    )
                else:
                    self._screen._active_assistant_entry.update_text(
                        interrupted_message.content,
                        assistant_completion_state="interrupted",
                    )
        self.update_footer_metrics(
            session_tokens=result.token_usage.session_tokens
            if result.token_usage
            else None,
            active_context_tokens=(
                result.token_usage.active_context_tokens if result.token_usage else None
            ),
            confidence=(
                result.final_response.confidence if result.final_response else None
            ),
        )
        self.set_status("")
        self._screen._busy = False
        self._screen._active_runner = None
        pending_draft = self._screen._pending_interrupt_draft
        self._screen._pending_interrupt_draft = None
        self.refresh_footer()
        self.refresh_inspector()
        self._screen.query_one("#composer", ComposerTextArea).focus()
        if pending_draft is not None:
            self.submit_draft(pending_draft)

    def toggle_inspector(self) -> None:
        self._screen._inspector_open = not self._screen._inspector_open
        self.refresh_inspector()

    def resolve_active_approval(self, approved: bool) -> None:
        runner = self._screen._active_runner
        if runner is None or self._screen._pending_approval is None:
            return
        if not runner.resolve_pending_approval(approved):
            return
        self._screen._approval_decision_in_flight = True
        self.set_status("approving" if approved else "denying")
        self.refresh_inspector()

    def _tools_command_text(self) -> str:
        enabled_tools = sorted(self._screen._enabled_tools)
        disabled_tools = sorted(
            set(self._screen._available_tool_specs).difference(
                self._screen._enabled_tools
            )
        )
        approval_suffix = (
            " (approval required)"
            if SideEffectClass.LOCAL_READ in self._screen._require_approval_for
            else ""
        )
        enabled_lines = [
            (
                f"- {tool_name}: "
                f"{self._screen._available_tool_specs[tool_name].side_effects.value}"
                f"{approval_suffix}"
            )
            for tool_name in enabled_tools
        ]
        disabled_lines = [f"- {tool_name}" for tool_name in disabled_tools]
        parts = [
            "Enabled tools:\n"
            + ("\n".join(enabled_lines) if enabled_lines else "- none"),
            "Disabled tools:\n"
            + ("\n".join(disabled_lines) if disabled_lines else "- none"),
        ]
        return "\n\n".join(parts)

    def _handle_tools_command(self, user_message: str) -> bool:
        parts = user_message.strip().split()
        if len(parts) == 1:
            self.append_transcript("system", self._tools_command_text())
            return True
        if len(parts) == 2 and parts[1].lower() == "reset":
            self._screen._enabled_tools = set(self._screen._default_enabled_tools)
            self.append_transcript("system", "Restored the default session tool set.")
            self.refresh_inspector()
            return True
        if len(parts) == 3 and parts[1].lower() in {"enable", "disable"}:
            tool_name = parts[2].strip()
            if tool_name not in self._screen._available_tool_specs:
                self.append_transcript("error", f"Unknown tool: {tool_name}")
                return True
            if parts[1].lower() == "enable":
                self._screen._enabled_tools.add(tool_name)
                self.append_transcript("system", f"Enabled tool: {tool_name}")
            else:
                self._screen._enabled_tools.discard(tool_name)
                self.append_transcript("system", f"Disabled tool: {tool_name}")
            self.refresh_inspector()
            return True
        self.append_transcript(
            "system",
            (
                "Usage: /tools | /tools enable <tool_name> | "
                "/tools disable <tool_name> | /tools reset"
            ),
        )
        return True

    def _handle_approvals_command(self, user_message: str) -> bool:
        parts = user_message.strip().split()
        if len(parts) == 1:
            enabled = SideEffectClass.LOCAL_READ in self._screen._require_approval_for
            self.append_transcript(
                "system",
                (
                    "Approvals are ON for local_read tools."
                    if enabled
                    else "Approvals are OFF for local_read tools."
                ),
            )
            return True
        if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
            if parts[1].lower() == "on":
                self._screen._require_approval_for.add(SideEffectClass.LOCAL_READ)
                self.append_transcript(
                    "system", "Enabled approvals for local_read tools."
                )
            else:
                self._screen._require_approval_for.discard(SideEffectClass.LOCAL_READ)
                self.append_transcript(
                    "system", "Disabled approvals for local_read tools."
                )
            self.refresh_inspector()
            return True
        self.append_transcript(
            "system", "Usage: /approvals | /approvals on | /approvals off"
        )
        return True

    def handle_inline_command(self, user_message: str) -> bool:
        normalized = user_message.strip().lower()
        if normalized == "/help":
            self.append_transcript(
                "system",
                (
                    "Ask grounded questions about the selected root. Use /model to "
                    "inspect or switch models, /tools to manage tools, /approvals "
                    "to toggle approvals, /inspect to toggle the inspector, and "
                    "/copy to open a selectable transcript view. Use quit or exit "
                    "to leave."
                ),
            )
            return True
        if normalized.startswith("/tools"):
            return self._handle_tools_command(user_message)
        if normalized.startswith("/approvals"):
            return self._handle_approvals_command(user_message)
        if normalized == "/inspect":
            self.toggle_inspector()
            return True
        if normalized.startswith("/model"):
            parts = user_message.strip().split(maxsplit=1)
            self._switch_active_model(parts[1] if len(parts) > 1 else "")
            return True
        if normalized == "/copy":
            self.open_transcript_copy()
            return True
        if normalized in {"quit", "exit"}:
            self._screen.app.exit()
            return True
        return False

    def transcript_copy_text(self) -> str:
        transcript = self._screen.query_one("#transcript", VerticalScroll)
        parts: list[str] = []
        for child in transcript.children:
            text = getattr(child, "transcript_text", "").rstrip()
            if text:
                parts.append(text)
        return "\n\n".join(parts).rstrip()

    def open_transcript_copy(self) -> None:
        self._screen.app.push_screen(TranscriptCopyModal(self.transcript_copy_text()))

    def cancel_active_turn(self, *, status_text: str) -> None:
        if self._screen._active_runner is None:
            return
        self.set_status(status_text)
        self._screen._active_runner.cancel()

    def submit_draft(self, raw_draft: str) -> None:
        if not raw_draft.strip():
            return
        if self.handle_inline_command(raw_draft):
            self.clear_composer()
            return
        if self._screen._busy:
            self._screen.app.push_screen(
                InterruptConfirmModal(),
                callback=self.handle_interrupt_confirmation,
            )
            return
        if not self.ensure_provider_ready():
            return
        self.clear_composer()
        self.append_transcript("user", raw_draft)
        self._screen._active_assistant_entry = None
        self._screen._active_turn_number += 1
        self._screen._busy = True
        self.refresh_footer()
        self._screen._run_turn_worker(raw_draft)


def create_provider(
    config: ChatLLMConfig,
    *,
    api_key: str | None,
    model_name: str,
) -> OpenAICompatibleProvider:
    """Create a provider client for the configured OpenAI-compatible backend."""
    request_params = {"timeout": config.timeout_seconds}
    if config.provider is ProviderPreset.OPENAI:
        return OpenAICompatibleProvider.for_openai(
            model=model_name,
            api_key=api_key or getenv(config.api_key_env_var or "OPENAI_API_KEY"),
            mode_strategy=ProviderModeStrategy.AUTO,
            default_request_params=request_params,
        )
    if config.provider is ProviderPreset.OLLAMA:
        base_url = config.api_base_url or "http://127.0.0.1:11434/v1"
        return OpenAICompatibleProvider.for_ollama(
            model=model_name,
            base_url=base_url,
            api_key=api_key or "ollama",
            mode_strategy=ProviderModeStrategy.AUTO,
            default_request_params=request_params,
        )
    if not config.api_base_url:
        raise ValueError("Custom OpenAI-compatible providers require api_base_url.")
    return OpenAICompatibleProvider(
        model=model_name,
        base_url=config.api_base_url,
        api_key=api_key or getenv(config.api_key_env_var or "OPENAI_API_KEY"),
        mode_strategy=ProviderModeStrategy.AUTO,
        default_request_params=request_params,
    )


def build_chat_registry() -> ToolRegistry:
    """Return the full generic tool registry available to the chat app."""
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    return registry


def build_chat_policy(screen: ChatScreen) -> ToolPolicy:
    """Build the session-scoped chat policy from current UI state."""
    return ToolPolicy(
        allowed_tools=set(screen._enabled_tools),
        allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
        require_approval_for=set(screen._require_approval_for),
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
        redaction=screen._config.policy.redaction.model_copy(deep=True),
    )


def build_chat_executor(
    screen: ChatScreen | None = None,
) -> tuple[ToolRegistry, WorkflowExecutor]:
    """Return the chat registry and policy-aware executor for one turn."""
    registry = build_chat_registry()
    if screen is None:
        policy = ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
            allow_network=False,
            allow_filesystem=True,
            allow_subprocess=False,
        )
    else:
        policy = build_chat_policy(screen)
    return registry, WorkflowExecutor(registry=registry, policy=policy)


def build_available_tool_specs() -> dict[str, ToolSpec]:
    """Return all chat-visible tool specs keyed by tool name."""
    registry = build_chat_registry()
    return {tool.spec.name: tool.spec for tool in registry.list_registered_tools()}


def build_chat_context(screen: ChatScreen) -> ToolContext:
    """Build the tool context passed into chat workflow execution."""
    effective_read_limit = (
        screen._config.tool_limits.max_read_file_chars
        if screen._config.tool_limits.max_read_file_chars is not None
        else max(1, screen._config.session.max_context_tokens * 4)
    )
    effective_tool_limits = screen._config.tool_limits.model_copy(
        update={"max_read_file_chars": effective_read_limit}
    )
    return ToolContext(
        invocation_id=f"textual-chat-{uuid4()}",
        workspace=str(screen._root_path),
        metadata={
            "source_filters": screen._config.source_filters.model_dump(mode="json"),
            "tool_limits": effective_tool_limits.model_dump(mode="json"),
        },
    )


def build_chat_system_prompt_for_screen(
    screen: ChatScreen, registry: ToolRegistry
) -> str:
    """Build the repository-chat system prompt from the active registry."""
    return build_chat_system_prompt(
        tool_registry=registry,
        tool_limits=screen._config.tool_limits,
        enabled_tool_names=screen._enabled_tools,
    )

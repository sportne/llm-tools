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
)
from llm_tools.apps.textual_chat.prompts import build_chat_system_prompt
from llm_tools.apps.textual_chat.screens import (
    ComposerTextArea,
    CredentialModal,
    InterruptConfirmModal,
    TranscriptCopyModal,
)
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderModeStrategy
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools import register_chat_tools
from llm_tools.workflow_api import (
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
        self._screen.query_one("#status-bar", Static).update(rendered)

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
            footer += " | Enter send | Shift+Enter newline | Stop active turn | F6 copy transcript"
        else:
            footer += " | Enter send | Shift+Enter newline | F6 copy transcript | /help | quit"
        self._screen.query_one("#footer-bar", Static).update(footer)
        self._screen.query_one("#stop-button", Button).disabled = not self._screen._busy

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
        self._screen._pending_interrupt_draft = None
        self.refresh_footer()
        self._screen.query_one("#composer", ComposerTextArea).focus()

    def handle_turn_status(self, event: object) -> None:
        typed_event = ChatWorkflowStatusEvent.model_validate(event)
        self.set_status(typed_event.status)

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
        self._screen.query_one("#composer", ComposerTextArea).focus()
        if pending_draft is not None:
            self.submit_draft(pending_draft)

    def handle_inline_command(self, user_message: str) -> bool:
        normalized = user_message.strip().lower()
        if normalized == "/help":
            self.append_transcript(
                "system",
                "Ask grounded questions about the selected root. Use /model to inspect or switch models. Use /copy to open a selectable transcript view. Use quit or exit to leave.",
            )
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
        if self._screen._busy:
            if raw_draft.strip().lower().startswith("/model"):
                self.append_transcript(
                    "system",
                    "Stop the active turn before changing models.",
                )
                return
            self._screen.app.push_screen(
                InterruptConfirmModal(),
                callback=self.handle_interrupt_confirmation,
            )
            return
        if self.handle_inline_command(raw_draft):
            self.clear_composer()
            return
        if not self.ensure_provider_ready():
            return
        self.clear_composer()
        self.append_transcript("user", raw_draft)
        self._screen._active_assistant_entry = None
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


def build_chat_executor() -> tuple[ToolRegistry, WorkflowExecutor]:
    """Return the read-only chat registry and executor."""
    registry = ToolRegistry()
    register_chat_tools(registry)
    policy = ToolPolicy(
        allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
    )
    return registry, WorkflowExecutor(registry=registry, policy=policy)


def build_chat_context(screen: ChatScreen) -> ToolContext:
    """Build the tool context passed into chat workflow execution."""
    return ToolContext(
        invocation_id=f"textual-chat-{uuid4()}",
        workspace=str(screen._root_path),
        metadata={
            "source_filters": screen._config.source_filters.model_dump(mode="json"),
            "session_config": screen._config.session.model_dump(mode="json"),
            "tool_limits": screen._config.tool_limits.model_dump(mode="json"),
        },
    )


def build_chat_system_prompt_for_screen(
    screen: ChatScreen, registry: ToolRegistry
) -> str:
    """Build the repository-chat system prompt from the active registry."""
    return build_chat_system_prompt(
        tool_registry=registry,
        tool_limits=screen._config.tool_limits,
    )

"""Additional branch-coverage tests for the Textual chat app."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

from tests.apps._imports import import_textual_chat_modules
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static

from llm_tools.apps.textual_chat.config import load_textual_chat_config
from llm_tools.apps.textual_chat.models import (
    ChatCredentialPromptMetadata,
    ProviderPreset,
    TextualChatConfig,
)
from llm_tools.apps.textual_chat.screens import (
    ComposerTextArea,
    CredentialModal,
    InterruptConfirmModal,
    TranscriptCopyModal,
)
from llm_tools.tool_api import SideEffectClass
from llm_tools.workflow_api import (
    ChatSessionState,
    ChatTokenUsage,
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowApprovalState,
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowTurnResult,
)
from llm_tools.workflow_api.models import ApprovalRequest

_CHAT_MODULES = import_textual_chat_modules()
chat_main_module = _CHAT_MODULES.main
ChatApp = _CHAT_MODULES.app.ChatApp
run_chat_app = _CHAT_MODULES.app.run_chat_app
build_available_tool_specs = _CHAT_MODULES.controller.build_available_tool_specs
build_chat_context = _CHAT_MODULES.controller.build_chat_context
build_chat_executor = _CHAT_MODULES.controller.build_chat_executor
build_chat_policy = _CHAT_MODULES.controller.build_chat_policy
build_chat_system_prompt_for_screen = (
    _CHAT_MODULES.controller.build_chat_system_prompt_for_screen
)
create_provider = _CHAT_MODULES.controller.create_provider


class _ProviderOk:
    def __init__(self, models: list[str] | None = None) -> None:
        self._models = models or ["demo-model"]

    def list_available_models(self) -> list[str]:
        return self._models


class _ProviderBoom:
    def list_available_models(self) -> list[str]:
        raise RuntimeError("listing failed")


class _Cancelable:
    def __init__(self) -> None:
        self.cancelled = False
        self.resolutions: list[bool] = []

    def cancel(self) -> None:
        self.cancelled = True

    def resolve_pending_approval(self, approved: bool) -> bool:
        self.resolutions.append(approved)
        return True


def test_textual_chat_config_errors_and_model_metadata(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ValueError):
        load_textual_chat_config(missing)

    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(ValueError):
        load_textual_chat_config(directory)

    invalid_yaml = tmp_path / "bad.yaml"
    invalid_yaml.write_text(":\n- [", encoding="utf-8")
    with pytest.raises(ValueError):
        load_textual_chat_config(invalid_yaml)

    bad_root = tmp_path / "list.yaml"
    bad_root.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_textual_chat_config(bad_root)

    bad_section = tmp_path / "section.yaml"
    bad_section.write_text("llm: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_textual_chat_config(bad_section)

    config = TextualChatConfig()
    assert config.llm.credential_prompt_metadata().expects_api_key is False
    openai_config = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"provider": ProviderPreset.OPENAI})
        }
    )
    metadata = openai_config.llm.credential_prompt_metadata()
    assert metadata.expects_api_key is True
    assert metadata.api_key_env_var == "OPENAI_API_KEY"


def test_textual_chat_provider_factory_and_main_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_calls: list[dict[str, object]] = []
    ollama_calls: list[dict[str, object]] = []
    custom_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "llm_tools.apps.textual_chat.controller.OpenAICompatibleProvider.for_openai",
        lambda **kwargs: openai_calls.append(kwargs) or SimpleNamespace(kind="openai"),
    )
    monkeypatch.setattr(
        "llm_tools.apps.textual_chat.controller.OpenAICompatibleProvider.for_ollama",
        lambda **kwargs: ollama_calls.append(kwargs) or SimpleNamespace(kind="ollama"),
    )
    create_provider(
        TextualChatConfig().llm.model_copy(update={"provider": ProviderPreset.OPENAI}),
        api_key="secret",
        model_name="gpt-demo",
    )
    create_provider(
        TextualChatConfig().llm.model_copy(update={"provider": ProviderPreset.OLLAMA}),
        api_key=None,
        model_name="llama-demo",
    )

    class _CustomProvider:
        @classmethod
        def for_openai(cls, **kwargs: object) -> object:
            return SimpleNamespace(kind="openai-from-custom", kwargs=kwargs)

        @classmethod
        def for_ollama(cls, **kwargs: object) -> object:
            return SimpleNamespace(kind="ollama-from-custom", kwargs=kwargs)

        def __init__(self, **kwargs: object) -> None:
            custom_calls.append(kwargs)

    monkeypatch.setattr(
        "llm_tools.apps.textual_chat.controller.OpenAICompatibleProvider",
        _CustomProvider,
    )
    create_provider(
        TextualChatConfig().llm.model_copy(
            update={
                "provider": ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
                "api_base_url": "http://custom/v1",
            }
        ),
        api_key="secret",
        model_name="custom-demo",
    )
    with pytest.raises(ValueError):
        create_provider(
            TextualChatConfig().llm.model_copy(
                update={
                    "provider": ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
                    "api_base_url": None,
                }
            ),
            api_key=None,
            model_name="custom-demo",
        )
    assert openai_calls and ollama_calls and custom_calls

    called: list[int] = []
    monkeypatch.setattr(
        "llm_tools.apps.textual_chat.__main__._main",
        lambda: called.append(1) or 7,
    )
    assert chat_main_module.main() == 7
    assert called == [1]

    run_called: list[object] = []
    monkeypatch.setattr(
        "llm_tools.apps.textual_chat.app.ChatApp.run",
        lambda self: run_called.append(self),
    )
    assert run_chat_app(root_path=Path("."), config=TextualChatConfig()) == 0
    assert len(run_called) == 1


def test_textual_chat_screen_widgets_and_controller_branches(tmp_path: Path) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=TextualChatConfig(),
            provider=_ProviderOk(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            screen._run_turn_worker = lambda raw: None  # type: ignore[method-assign]

            screen.query_one("#composer", ComposerTextArea).load_text("draft")
            screen.handle_composer_submit()
            screen.handle_send_button()
            screen.handle_stop_button()
            screen._busy = False

            screen._controller.set_status("thinking")
            screen._controller.set_status("thinking")
            screen._controller._flush_pending_status()
            screen._controller._advance_status_animation()
            screen._controller.set_status("")
            screen._controller.update_footer_metrics(
                session_tokens=2,
                active_context_tokens=3,
                confidence=0.4,
            )
            assert "confidence: 0.40" in str(
                screen.query_one("#footer-bar", Static).renderable
            )

            screen._controller._show_available_models()
            assert "Available models:" in screen._controller.transcript_copy_text()

            screen._provider = _ProviderBoom()
            screen._controller._show_available_models()
            screen._provider = _ProviderOk(models=[])
            screen._controller._show_available_models()
            screen._create_provider = lambda config, api_key, model_name: _ProviderOk(
                models=["lazy-a", "lazy-b"]
            )
            screen._provider = None
            screen._controller._show_available_models()
            assert "lazy-a" in screen._controller.transcript_copy_text()

            switched: list[str] = []
            screen._create_provider = lambda config, api_key, model_name: (
                switched.append(model_name) or _ProviderOk()
            )
            screen._provider = _ProviderOk()
            screen._controller._switch_active_model(screen._active_model_name)
            screen._controller._switch_active_model("")
            screen._controller._switch_active_model("next-model")
            assert "next-model" in switched

            screen._busy = True
            screen._controller._switch_active_model("blocked-model")
            screen._busy = False
            screen._create_provider = lambda config, api_key, model_name: (
                _ for _ in ()
            ).throw(RuntimeError("bad switch"))
            screen._controller._switch_active_model("bad-model")

            screen._controller.handle_inline_command("/help")
            screen._controller.handle_inline_command("/inspect")
            assert screen.query_one("#inspector-pane", VerticalScroll).display is True
            screen._controller.handle_inline_command("/tools")
            screen._controller.handle_inline_command("/tools disable read_file")
            assert "read_file" not in screen._enabled_tools
            screen._controller.handle_inline_command("/tools enable read_file")
            assert "read_file" in screen._enabled_tools
            screen._controller.handle_inline_command("/tools reset")
            screen._controller.handle_inline_command("/tools maybe")
            screen._controller.handle_inline_command("/tools enable missing_tool")
            assert (
                "Unknown tool: missing_tool"
                in screen._controller.transcript_copy_text()
            )
            screen._controller.handle_inline_command("/approvals")
            screen._controller.handle_inline_command("/approvals on")
            assert screen._require_approval_for
            screen._controller.handle_inline_command("/approvals off")
            assert not screen._require_approval_for
            screen._controller.handle_inline_command("/approvals maybe")
            screen._controller.handle_inline_command("/copy")
            assert app.screen_stack[-1].id == "transcript-copy-modal"
            app.pop_screen()
            assert screen._controller.handle_inline_command("unknown") is False
            assert (
                screen._controller._render_pending_approval() == "No pending approval."
            )
            assert screen._controller._render_inspector_entries([]) == ""
            screen._controller.append_inspector_entry(
                screen._inspector_provider_messages,
                label="Manual",
                payload={"ok": True},
            )
            screen.query_one("#transcript", VerticalScroll).mount(Static(""))
            assert "Manual" in str(
                screen.query_one("#provider-messages-box", Static).renderable
            )
            screen._controller.toggle_inspector()
            assert screen.query_one("#inspector-pane", VerticalScroll).display is False
            screen._controller.toggle_inspector()
            assert screen.query_one("#inspector-pane", VerticalScroll).display is True

            exited = {"value": False}
            original_exit = app.exit
            app.exit = lambda *args, **kwargs: exited.__setitem__("value", True)
            assert screen._controller.handle_inline_command("quit") is True
            assert exited["value"] is True
            app.exit = original_exit

            screen._active_runner = _Cancelable()
            screen._controller.cancel_active_turn(status_text="stopping")
            assert screen._active_runner.cancelled is True
            screen._active_runner = None
            screen._controller.cancel_active_turn(status_text="stopping")

            screen._busy = True
            screen._controller.submit_draft("/model x")
            screen._controller.submit_draft("hello")
            assert isinstance(app.screen_stack[-1], InterruptConfirmModal)
            app.pop_screen()
            screen._busy = False
            screen._controller.handle_interrupt_confirmation(True)
            screen._controller.handle_interrupt_confirmation(False)

            screen._provider = _ProviderOk()
            captured: list[str] = []
            screen._run_turn_worker = lambda raw: captured.append(raw)  # type: ignore[method-assign]
            screen._controller.submit_draft("hello")
            assert captured == ["hello"]
            assert screen._busy is True

            screen._controller.handle_turn_error("boom")
            assert screen._busy is False

            screen._create_provider = lambda config, api_key, model_name: _ProviderOk()
            screen._provider = None
            screen._credential_prompt_completed = True
            assert screen._controller.ensure_provider_ready() is True
            screen._controller.initialize_provider()

            screen._pending_interrupt_draft = "carry"
            called_submit: list[str] = []
            original_submit = screen._controller.submit_draft
            screen._controller.submit_draft = lambda raw: called_submit.append(raw)  # type: ignore[method-assign]
            result = ChatWorkflowTurnResult(
                status="needs_continuation",
                new_messages=[{"role": "assistant", "content": "partial"}],
                continuation_reason="need more",
                token_usage=ChatTokenUsage(
                    total_tokens=1, session_tokens=2, active_context_tokens=3
                ),
                session_state=ChatSessionState(),
                context_warning="trimmed",
            )
            screen._controller.handle_turn_result(
                ChatWorkflowResultEvent(result=result).model_dump(mode="json")
            )
            assert called_submit == ["carry"]
            screen._controller.submit_draft = original_submit  # type: ignore[method-assign]

            interrupted = ChatWorkflowTurnResult(
                status="interrupted",
                new_messages=[
                    {
                        "role": "assistant",
                        "content": "partial",
                        "completion_state": "interrupted",
                    }
                ],
                interruption_reason="stop",
                token_usage=ChatTokenUsage(total_tokens=1),
                session_state=ChatSessionState(),
            )
            screen._controller.handle_turn_result(
                ChatWorkflowResultEvent(result=interrupted).model_dump(mode="json")
            )
            screen._controller.handle_turn_status(
                {"event_type": "status", "status": "thinking"}
            )
            approval = ChatWorkflowApprovalState(
                approval_request=ApprovalRequest(
                    approval_id="approval-1",
                    invocation_index=1,
                    request={"tool_name": "read_file", "arguments": {"path": "a.txt"}},
                    tool_name="read_file",
                    tool_version="0.1.0",
                    policy_reason="approval required",
                    policy_metadata={},
                    requested_at="2026-01-01T00:00:00Z",
                    expires_at="2026-01-01T00:05:00Z",
                ),
                tool_name="read_file",
                redacted_arguments={"path": "a.txt"},
                policy_reason="approval required",
                policy_metadata={},
            )
            screen._controller.handle_turn_approval_requested(
                ChatWorkflowApprovalEvent(approval=approval).model_dump(mode="json")
            )
            assert screen.query_one("#approve-button", Button).display is True
            screen._active_runner = None
            screen._controller.resolve_active_approval(approved=True)

            class _Unresolvable:
                def resolve_pending_approval(self, approved: bool) -> bool:
                    del approved
                    return False

            screen._active_runner = _Unresolvable()
            screen._controller.resolve_active_approval(approved=False)
            screen._active_runner = _Cancelable()
            screen._controller.resolve_active_approval(approved=True)
            screen._controller.handle_turn_approval_resolved(
                ChatWorkflowApprovalResolvedEvent(
                    approval=approval,
                    resolution="approved",
                ).model_dump(mode="json")
            )
            screen._controller.handle_turn_approval_resolved(
                ChatWorkflowApprovalResolvedEvent(
                    approval=approval,
                    resolution="denied",
                ).model_dump(mode="json")
            )
            screen._controller.handle_turn_approval_resolved(
                ChatWorkflowApprovalResolvedEvent(
                    approval=approval,
                    resolution="timed_out",
                ).model_dump(mode="json")
            )
            screen._controller.handle_turn_approval_resolved(
                ChatWorkflowApprovalResolvedEvent(
                    approval=approval,
                    resolution="cancelled",
                ).model_dump(mode="json")
            )
            screen._controller.handle_turn_inspector_event(
                ChatWorkflowInspectorEvent(
                    round_index=1,
                    kind="parsed_response",
                    payload={"invocations": []},
                ).model_dump(mode="json")
            )
            assert (
                "parsed response"
                in str(
                    screen.query_one("#parsed-response-box", Static).renderable
                ).lower()
            )

            registry, executor = build_chat_executor()
            assert registry.list_registered_tools()
            assert executor is not None
            registry_for_screen, executor_for_screen = build_chat_executor(screen)
            assert registry_for_screen.list_registered_tools()
            assert executor_for_screen._policy.allowed_tools == set(
                screen._enabled_tools
            )
            policy = build_chat_policy(screen)
            assert policy.allowed_tools == set(screen._enabled_tools)
            assert policy.allow_filesystem is True
            assert build_available_tool_specs()
            context = build_chat_context(screen)
            assert context.workspace == str(tmp_path)
            assert "tool_limits" in context.metadata
            prompt = build_chat_system_prompt_for_screen(screen, registry)
            assert "Available tools:" in prompt
            screen._require_approval_for.add(SideEffectClass.LOCAL_READ)
            assert "approval required" in screen._controller._tools_command_text()
            assert screen._controller.transcript_copy_text()

            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_textual_chat_modals_and_credential_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        config = TextualChatConfig()
        config = config.model_copy(
            update={
                "llm": config.llm.model_copy(update={"provider": ProviderPreset.OPENAI})
            }
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        app = ChatApp(root_path=tmp_path, config=config, provider=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.screen_stack[-1].id == "credential-modal"
            modal = app.screen_stack[-1]
            modal.query_one("#credential-input", Input).value = "secret"
            modal.handle_submit()
            await pilot.pause()

            copy_modal = TranscriptCopyModal("hello")
            app.push_screen(copy_modal)
            await pilot.pause()
            copy_modal.handle_copy_selection()
            copy_modal.handle_copy_all()
            copy_modal.handle_close()

            confirm_modal = InterruptConfirmModal()
            app.push_screen(confirm_modal)
            await pilot.pause()
            confirm_modal.handle_confirm()
            confirm_modal.handle_cancel()

            cred_modal = CredentialModal(ChatCredentialPromptMetadata())
            app.push_screen(cred_modal)
            await pilot.pause()
            cred_modal.handle_cancel()

            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_textual_chat_prompt_builder_and_strip_titles() -> None:
    from llm_tools.apps.textual_chat.prompts import (
        _strip_schema_titles,
        build_chat_system_prompt,
    )
    from llm_tools.tool_api import ToolRegistry
    from llm_tools.tools import register_filesystem_tools, register_text_tools
    from llm_tools.tools.filesystem import ToolLimits

    payload = {
        "title": "Root",
        "properties": {
            "child": {"title": "Child", "type": "string"},
            "items": [{"title": "Nested", "type": "number"}],
        },
    }
    stripped = _strip_schema_titles(payload)
    assert "title" not in stripped
    assert "title" not in stripped["properties"]["child"]
    assert "title" not in stripped["properties"]["items"][0]

    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    prompt = build_chat_system_prompt(
        tool_registry=registry,
        tool_limits=ToolLimits(),
    )
    assert "Available tools:" in prompt
    assert "Final response fields:" in prompt
    assert "missing_information" in prompt

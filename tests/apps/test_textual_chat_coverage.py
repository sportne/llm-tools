"""Focused per-file coverage tests for the Textual chat app."""

from __future__ import annotations

import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

import llm_tools.apps.textual_chat as chat_module
import llm_tools.apps.textual_chat.app as app_module
import llm_tools.apps.textual_chat.controller as controller_module
from llm_tools.apps.textual_chat import __main__ as chat_main_module
from llm_tools.apps.textual_chat.app import ChatApp, ChatScreen
from llm_tools.apps.textual_chat.models import (
    ChatLLMConfig,
    ProviderPreset,
    TextualChatConfig,
)
from llm_tools.apps.textual_chat.presentation import (
    AssistantMarkdownEntry,
    TranscriptEntry,
    format_citation,
    format_final_response_metadata,
)
from llm_tools.apps.textual_chat.screens import ComposerTextArea, CredentialModal
from llm_tools.workflow_api import (
    ChatCitation,
    ChatFinalResponse,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
)


class _Provider:
    def list_available_models(self) -> list[str]:
        return ["demo-model"]


def test_textual_chat_cli_and_module_entrypoints_cover_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "chat.yaml"
    config_path.write_text(
        """
llm:
  provider: ollama
  model_name: base-model
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_run_chat_app(*, root_path: Path, config: TextualChatConfig) -> int:
        captured["root_path"] = root_path
        captured["config"] = config
        return 23

    monkeypatch.setattr(chat_module, "run_chat_app", _fake_run_chat_app)
    result = chat_module.main(
        [
            str(tmp_path),
            "--config",
            str(config_path),
            "--provider",
            "custom_openai_compatible",
            "--model",
            "gpt-demo",
            "--temperature",
            "0.7",
            "--api-base-url",
            "http://example.test/v1",
            "--max-context-tokens",
            "321",
            "--max-tool-round-trips",
            "3",
            "--max-tool-calls-per-round",
            "2",
            "--max-total-tool-calls-per-turn",
            "4",
            "--max-entries-per-call",
            "7",
            "--max-recursive-depth",
            "8",
            "--max-search-matches",
            "9",
            "--max-read-lines",
            "10",
            "--max-file-size-characters",
            "11",
            "--max-tool-result-chars",
            "12",
        ]
    )

    assert result == 23
    config = captured["config"]
    assert isinstance(config, TextualChatConfig)
    assert config.llm.provider is ProviderPreset.CUSTOM_OPENAI_COMPATIBLE
    assert config.llm.model_name == "gpt-demo"
    assert config.llm.temperature == 0.7
    assert config.llm.api_base_url == "http://example.test/v1"
    assert config.session.max_context_tokens == 321
    assert config.session.max_tool_round_trips == 3
    assert config.session.max_tool_calls_per_round == 2
    assert config.session.max_total_tool_calls_per_turn == 4
    assert config.tool_limits.max_entries_per_call == 7
    assert config.tool_limits.max_recursive_depth == 8
    assert config.tool_limits.max_search_matches == 9
    assert config.tool_limits.max_read_lines == 10
    assert config.tool_limits.max_file_size_characters == 11
    assert config.tool_limits.max_tool_result_chars == 12
    assert captured["root_path"] == tmp_path.resolve()

    monkeypatch.setattr(chat_module, "main", lambda: 19)
    assert chat_main_module._main() == 19
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("llm_tools.apps.textual_chat.__main__", run_name="__main__")
    assert exc.value.code == 19


def test_textual_chat_models_and_presentation_helpers_cover_edge_cases() -> None:
    with pytest.raises(ValueError):
        ChatLLMConfig(model_name=" ")
    with pytest.raises(ValueError):
        ChatLLMConfig(api_base_url=" ")
    with pytest.raises(ValueError):
        ChatLLMConfig(temperature=1.5)
    with pytest.raises(ValueError):
        ChatLLMConfig(timeout_seconds=0)

    metadata = ChatLLMConfig(
        provider=ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
        api_key_env_var=None,
    ).credential_prompt_metadata()
    assert metadata.expects_api_key is True
    assert metadata.api_key_env_var == "OPENAI_API_KEY"

    assert format_citation(ChatCitation(source_path="src/app.py")) == "src/app.py"
    assert (
        format_citation(ChatCitation(source_path="src/app.py", line_start=4))
        == "src/app.py:4"
    )
    response = ChatFinalResponse(
        answer="Done",
        citations=[ChatCitation(source_path="src/app.py", line_start=1)],
        uncertainty=["unclear"],
        missing_information=["missing"],
        follow_up_suggestions=["next"],
    )
    metadata_text = format_final_response_metadata(response)
    assert "Missing Information:" in metadata_text
    assert "Follow-up Suggestions:" in metadata_text

    markdown_entry = AssistantMarkdownEntry(markdown_text="", metadata_text="Extra")
    assert markdown_entry.transcript_text.endswith("Extra")

    user_entry = TranscriptEntry(role="user", text="question")
    system_entry = TranscriptEntry(role="system", text="notice")
    error_entry = TranscriptEntry(role="error", text="boom")
    user_entry.update_text("updated")
    assert user_entry.transcript_text == "You:\nupdated"
    assert system_entry.transcript_text == "System:\nnotice"
    assert error_entry.transcript_text == "Error: boom"


def test_textual_chat_composer_key_handling() -> None:
    composer = ComposerTextArea()
    inserted: list[str] = []
    posted: list[object] = []
    stopped: list[str] = []
    prevented: list[str] = []

    composer.insert = lambda text: inserted.append(text)  # type: ignore[method-assign]
    composer.post_message = lambda message: posted.append(message)  # type: ignore[method-assign]

    shift_event = SimpleNamespace(
        key="shift+enter",
        stop=lambda: stopped.append("shift"),
        prevent_default=lambda: prevented.append("shift"),
    )
    enter_event = SimpleNamespace(
        key="enter",
        stop=lambda: stopped.append("enter"),
        prevent_default=lambda: prevented.append("enter"),
    )

    composer.on_key(shift_event)
    composer.on_key(enter_event)

    assert inserted == ["\n"]
    assert len(posted) == 1
    assert isinstance(posted[0], ComposerTextArea.SubmitRequested)
    assert posted[0].control is composer
    assert stopped == ["shift", "enter"]
    assert prevented == ["shift", "enter"]


def test_textual_chat_controller_and_worker_cover_remaining_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        config = TextualChatConfig().model_copy(
            update={
                "llm": TextualChatConfig().llm.model_copy(
                    update={
                        "provider": ProviderPreset.OPENAI,
                        "api_key_env_var": "OPENAI_API_KEY",
                    }
                )
            }
        )
        monkeypatch.setattr(controller_module, "getenv", lambda name: None)
        app = ChatApp(root_path=tmp_path, config=config, provider=None)
        async with app.run_test() as pilot:
            await pilot.pause()

            assert isinstance(app.screen_stack[-1], CredentialModal)
            app.pop_screen()
            await pilot.pause()
            screen = app.screen

            captured_provider_calls: list[tuple[str | None, str]] = []
            screen._create_provider = lambda config, api_key, model_name: (
                captured_provider_calls.append((api_key, model_name)) or _Provider()
            )
            monkeypatch.setattr(controller_module, "getenv", lambda name: "env-secret")
            assert screen._controller.ensure_provider_ready() is True
            assert captured_provider_calls[-1] == (
                "env-secret",
                screen._active_model_name,
            )

            monkeypatch.setattr(controller_module, "getenv", lambda name: None)
            screen._provider = None
            screen._credential_prompt_completed = False
            assert screen._controller.ensure_provider_ready() is False
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], CredentialModal)
            app.pop_screen()
            await pilot.pause()

            screen._create_provider = lambda config, api_key, model_name: (
                _ for _ in ()
            ).throw(RuntimeError("init failed"))
            screen._provider = None
            assert screen._controller.initialize_provider() is False
            assert "init failed" in screen._controller.transcript_copy_text()

            class _BrokenAssistantEntry:
                def __init__(self, **kwargs: object) -> None:
                    raise RuntimeError("render failed")

            monkeypatch.setattr(
                controller_module,
                "AssistantMarkdownEntry",
                _BrokenAssistantEntry,
            )
            fallback_entry = screen._controller.append_assistant_markdown(
                "markdown",
                metadata_text="meta",
                fallback_text="fallback",
            )
            assert isinstance(fallback_entry, TranscriptEntry)

            screen._controller.show_final_response(SimpleNamespace(final_response=None))

            composer = screen.query_one("#composer", ComposerTextArea)
            composer.load_text("/help")
            screen._controller.submit_draft("/help")
            assert composer.text == ""

            composer.load_text("question")
            screen._provider = None
            screen._credential_prompt_completed = False
            screen._controller.submit_draft("question")
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], CredentialModal)
            app.pop_screen()
            await pilot.pause()

            existing = screen._controller.append_transcript(
                "assistant",
                "old",
                assistant_completion_state="interrupted",
            )
            screen._active_assistant_entry = existing
            interrupted = ChatWorkflowTurnResult(
                status="interrupted",
                new_messages=[
                    {
                        "role": "assistant",
                        "content": "replacement",
                        "completion_state": "interrupted",
                    }
                ],
                interruption_reason="stopped",
            )
            screen._controller.handle_turn_result(
                ChatWorkflowResultEvent(result=interrupted).model_dump(mode="json")
            )
            assert "replacement" in existing.transcript_text

            screen.action_open_transcript_copy()
            await pilot.pause()
            assert app.screen_stack[-1].id == "transcript-copy-modal"
            app.pop_screen()

            worker_events: list[str] = []
            error_messages: list[str] = []
            app.call_from_thread = lambda fn, *args: fn(*args)  # type: ignore[method-assign]
            screen._controller.handle_turn_status = lambda event: worker_events.append(
                "status"
            )  # type: ignore[method-assign]
            screen._controller.handle_turn_result = lambda event: worker_events.append(
                "result"
            )  # type: ignore[method-assign]
            screen._controller.handle_turn_error = lambda message: (
                error_messages.append(message)
            )  # type: ignore[method-assign]
            monkeypatch.setattr(
                app_module,
                "build_chat_executor",
                lambda: (
                    SimpleNamespace(name="registry"),
                    SimpleNamespace(name="executor"),
                ),
            )
            monkeypatch.setattr(
                app_module,
                "build_chat_context",
                lambda screen_arg: {"workspace": str(tmp_path)},
            )
            monkeypatch.setattr(
                app_module,
                "build_chat_system_prompt_for_screen",
                lambda screen_arg, registry: "prompt",
            )

            class _Runner:
                def __iter__(self) -> object:
                    return iter(
                        [
                            ChatWorkflowStatusEvent(status="thinking"),
                            ChatWorkflowResultEvent(
                                result=ChatWorkflowTurnResult(
                                    status="completed",
                                    new_messages=[],
                                    final_response=ChatFinalResponse(answer="done"),
                                )
                            ),
                        ]
                    )

                def cancel(self) -> None:
                    return None

            monkeypatch.setattr(
                app_module,
                "run_interactive_chat_session_turn",
                lambda **kwargs: _Runner(),
            )

            screen._provider = _Provider()
            ChatScreen._run_turn_worker.__wrapped__(screen, "hello")
            assert worker_events == ["status", "result"]
            assert screen._active_runner is None

            screen._provider = None
            ChatScreen._run_turn_worker.__wrapped__(screen, "hello")
            assert error_messages == ["Chat provider is not configured."]

            app.exit()
            await pilot.pause()

    asyncio.run(_run())

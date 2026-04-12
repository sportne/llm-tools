"""Tests for the Textual repository chat app shell."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.containers import VerticalScroll
from textual.widgets import Static

from llm_tools.apps.textual_chat import _resolve_chat_config, build_parser
from llm_tools.apps.textual_chat.app import ChatApp, ChatScreen
from llm_tools.apps.textual_chat.config import load_textual_chat_config
from llm_tools.apps.textual_chat.models import TextualChatConfig
from llm_tools.apps.textual_chat.presentation import (
    AssistantMarkdownEntry,
    TranscriptEntry,
    format_citation,
    format_final_response,
)
from llm_tools.workflow_api import (
    ChatCitation,
    ChatFinalResponse,
    ChatSessionState,
    ChatTokenUsage,
    ChatWorkflowResultEvent,
    ChatWorkflowTurnResult,
)


class _FakeProvider:
    def list_available_models(self) -> list[str]:
        return ["demo-model"]


def test_textual_chat_config_loading_and_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "chat.yaml"
    config_path.write_text(
        """
llm:
  provider: ollama
  model_name: base-model
session:
  max_context_tokens: 99
""".strip(),
        encoding="utf-8",
    )
    loaded = load_textual_chat_config(config_path)
    assert loaded.llm.model_name == "base-model"
    parser = build_parser()
    args = parser.parse_args(
        [
            str(tmp_path),
            "--config",
            str(config_path),
            "--model",
            "override-model",
            "--max-context-tokens",
            "123",
        ]
    )
    resolved = _resolve_chat_config(args)
    assert resolved.llm.model_name == "override-model"
    assert resolved.session.max_context_tokens == 123


def test_textual_chat_presentation_helpers_render_expected_text() -> None:
    assert format_citation(ChatCitation(source_path="src/app.py", line_start=4, line_end=6)) == "src/app.py:4-6"
    formatted = format_final_response(
        ChatFinalResponse(
            answer="Done",
            citations=[ChatCitation(source_path="src/app.py", line_start=2)],
            uncertainty=["Unclear"],
            missing_information=["TBD"],
            follow_up_suggestions=["Inspect src"],
        )
    )
    assert "Citations:" in formatted
    entry = TranscriptEntry(role="assistant", text="draft")
    entry.update_text("partial", assistant_completion_state="interrupted")
    assert "Assistant (interrupted):" in str(entry.render())
    markdown_entry = AssistantMarkdownEntry(markdown_text="1. **Bold** and `code`")
    assert markdown_entry.transcript_text.startswith("Assistant:\n1. **Bold**")


def test_textual_chat_app_launches_with_shell_layout_and_startup_message(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=TextualChatConfig(),
            provider=_FakeProvider(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)
            transcript = app.screen.query_one("#transcript", VerticalScroll)
            texts = [str(getattr(child, "renderable", child.render())) for child in transcript.children]
            assert any("Root:" in text for text in texts)
            assert "quit or exit" in texts[0]
            assert "F6 copy transcript" in str(app.screen.query_one("#footer-bar", Static).renderable)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_textual_chat_controller_renders_completed_turn(tmp_path: Path) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=TextualChatConfig(),
            provider=_FakeProvider(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            result = ChatWorkflowTurnResult(
                status="completed",
                new_messages=[{"role": "user", "content": "question"}],
                final_response=ChatFinalResponse(
                    answer="Answer",
                    citations=[{"source_path": "src/app.py", "line_start": 1}],
                    confidence=0.5,
                ),
                token_usage=ChatTokenUsage(total_tokens=9, session_tokens=9, active_context_tokens=20),
                session_state=ChatSessionState(),
            )
            screen._controller.handle_turn_result(
                ChatWorkflowResultEvent(result=result).model_dump(mode="json")
            )
            await pilot.pause()
            transcript = screen.query_one("#transcript", VerticalScroll)
            assert any(isinstance(child, AssistantMarkdownEntry) for child in transcript.children)
            footer = str(screen.query_one("#footer-bar", Static).renderable)
            assert "confidence: 0.50" in footer
            app.exit()
            await pilot.pause()

    asyncio.run(_run())

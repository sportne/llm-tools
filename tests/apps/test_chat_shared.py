"""Focused unit tests for shared repository-chat helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_tools.apps.chat_config import (
    ChatLLMConfig,
    ChatPolicyConfig,
    ProviderPreset,
    TextualChatConfig,
    load_textual_chat_config,
)
from llm_tools.apps.chat_controls import (
    ChatControlNotice,
    ChatControlState,
    ModelCatalogOutcome,
    ModelSwitchOutcome,
    build_chat_control_state,
    build_help_text,
    build_startup_message,
    build_tool_state_payload,
    build_tools_command_text,
    handle_approvals_command,
    handle_chat_command,
    handle_model_command,
    handle_tools_command,
    resolve_default_enabled_tools,
)
from llm_tools.apps.chat_presentation import (
    format_citation,
    format_final_response,
    format_final_response_metadata,
    format_transcript_text,
    pretty_json,
)
from llm_tools.tool_api import SideEffectClass, ToolSpec
from llm_tools.workflow_api import ChatCitation, ChatFinalResponse


@pytest.fixture
def tool_specs() -> dict[str, ToolSpec]:
    return {
        "read_file": ToolSpec(
            name="read_file",
            description="Read a file",
            side_effects=SideEffectClass.LOCAL_READ,
        ),
        "search_text": ToolSpec(
            name="search_text",
            description="Search text",
            side_effects=SideEffectClass.NONE,
        ),
    }


@pytest.fixture
def control_state() -> ChatControlState:
    return ChatControlState(
        active_model_name="model-a",
        default_enabled_tools={"read_file"},
        enabled_tools={"read_file"},
        require_approval_for=set(),
    )


def test_chat_config_validation_and_loading_paths(tmp_path: Path) -> None:
    metadata = ChatLLMConfig(
        provider=ProviderPreset.OPENAI
    ).credential_prompt_metadata()
    assert metadata.api_key_env_var == "OPENAI_API_KEY"
    assert metadata.expects_api_key is True

    custom_metadata = ChatLLMConfig(
        provider=ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
        prompt_for_api_key_if_missing=False,
    ).credential_prompt_metadata()
    assert custom_metadata.api_key_env_var == "OPENAI_API_KEY"
    assert custom_metadata.prompt_for_api_key_if_missing is False

    ollama_metadata = ChatLLMConfig(api_key_env_var=None).credential_prompt_metadata()
    assert ollama_metadata.api_key_env_var == "API key"
    assert ollama_metadata.expects_api_key is False

    with pytest.raises(ValidationError):
        ChatLLMConfig(model_name="   ")
    with pytest.raises(ValidationError):
        ChatLLMConfig(api_base_url="   ")
    with pytest.raises(ValidationError):
        ChatLLMConfig(temperature=1.5)
    with pytest.raises(ValidationError):
        ChatLLMConfig(timeout_seconds=0)
    with pytest.raises(ValidationError):
        ChatPolicyConfig(enabled_tools=["read_file", " "])

    assert isinstance(TextualChatConfig(), TextualChatConfig)

    missing = tmp_path / "missing.yaml"
    with pytest.raises(ValueError, match="Configuration file not found"):
        load_textual_chat_config(missing)

    with pytest.raises(ValueError, match="Configuration path is not a file"):
        load_textual_chat_config(tmp_path)

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("llm: [", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_textual_chat_config(invalid_yaml)

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    loaded = load_textual_chat_config(empty)
    assert loaded.llm.model_name == "gemma4:26b"

    bad_root = tmp_path / "bad-root.yaml"
    bad_root.write_text("- nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected mapping at root"):
        load_textual_chat_config(bad_root)

    bad_section = tmp_path / "bad-section.yaml"
    bad_section.write_text("llm: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chat config 'llm' must be a mapping"):
        load_textual_chat_config(bad_section)

    invalid_model = tmp_path / "invalid-model.yaml"
    invalid_model.write_text("llm:\n  temperature: 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid chat config"):
        load_textual_chat_config(invalid_model)


def test_chat_control_builders_and_tool_commands(
    tool_specs: dict[str, ToolSpec], control_state: ChatControlState
) -> None:
    default_config = TextualChatConfig()
    assert resolve_default_enabled_tools(
        default_config, available_tool_names={"read_file", "search_text"}
    ) == {"read_file", "search_text"}

    filtered_config = TextualChatConfig(
        policy=ChatPolicyConfig(enabled_tools=["read_file", "missing"]),
    )
    built_state = build_chat_control_state(
        filtered_config,
        available_tool_names={"read_file", "search_text"},
    )
    assert built_state.enabled_tools == {"read_file"}
    assert built_state.default_enabled_tools == {"read_file"}

    startup = build_startup_message(
        root_path=Path("/workspace"),
        model_name="model-a",
        exit_hint="Use exit to close.",
    )
    assert (
        startup
        == "Root: /workspace\nModel: model-a\nUse /help for guidance. Use exit to close."
    )
    assert "Use /model" in build_help_text(exit_hint="Use exit to close.")

    payload = build_tool_state_payload(control_state, available_tool_specs=tool_specs)
    assert payload == {
        "enabled_tools": ["read_file"],
        "disabled_tools": ["search_text"],
        "require_approval_for": [],
    }

    listed = handle_tools_command(
        "/tools",
        state=control_state,
        available_tool_specs=tool_specs,
    )
    assert listed.handled is True
    assert "Enabled tools" in listed.notices[0].text

    unknown = handle_tools_command(
        "/tools enable nope",
        state=control_state,
        available_tool_specs=tool_specs,
    )
    assert unknown.notices == [
        ChatControlNotice(role="error", text="Unknown tool: nope")
    ]

    enabled = handle_tools_command(
        "/tools enable search_text",
        state=control_state,
        available_tool_specs=tool_specs,
    )
    assert enabled.notices[0].text == "Enabled tool: search_text"
    assert control_state.enabled_tools == {"read_file", "search_text"}

    control_state.require_approval_for.add(SideEffectClass.LOCAL_READ)
    tools_text = build_tools_command_text(
        control_state, available_tool_specs=tool_specs
    )
    assert "(approval required)" in tools_text

    disabled = handle_tools_command(
        "/tools disable search_text",
        state=control_state,
        available_tool_specs=tool_specs,
    )
    assert disabled.notices[0].text == "Disabled tool: search_text"
    assert control_state.enabled_tools == {"read_file"}

    reset = handle_tools_command(
        "/tools reset",
        state=control_state,
        available_tool_specs=tool_specs,
    )
    assert reset.notices[0].text == "Restored the default session tool set."

    usage = handle_tools_command(
        "/tools nope now",
        state=control_state,
        available_tool_specs=tool_specs,
    )
    assert usage.notices[0].text.startswith("Usage: /tools")


def test_chat_control_approvals_model_and_dispatch_paths(
    tool_specs: dict[str, ToolSpec], control_state: ChatControlState
) -> None:
    approvals_status = handle_approvals_command("/approvals", state=control_state)
    assert approvals_status.notices[0].text == "Approvals are OFF for local_read tools."

    approvals_on = handle_approvals_command("/approvals on", state=control_state)
    assert approvals_on.notices[0].text == "Enabled approvals for local_read tools."
    assert SideEffectClass.LOCAL_READ in control_state.require_approval_for

    approvals_off = handle_approvals_command("/approvals off", state=control_state)
    assert approvals_off.notices[0].text == "Disabled approvals for local_read tools."
    assert SideEffectClass.LOCAL_READ not in control_state.require_approval_for

    approvals_usage = handle_approvals_command("/approvals maybe", state=control_state)
    assert approvals_usage.notices[0].text.startswith("Usage: /approvals")

    busy = handle_model_command(
        "/model other",
        state=control_state,
        busy=True,
        list_models=lambda: ModelCatalogOutcome(model_ids=["ignored"]),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
    )
    assert busy.notices[0].text == "Stop the active turn before changing models."

    unavailable = handle_model_command(
        "/model",
        state=control_state,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(unavailable=True),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
    )
    assert unavailable.handled is True and unavailable.notices == []

    notice_only = handle_model_command(
        "/model",
        state=control_state,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(
            notice=ChatControlNotice(role="system", text="models unavailable")
        ),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
    )
    assert notice_only.notices[0].text == "models unavailable"

    no_models = handle_model_command(
        "/model",
        state=control_state,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=[]),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
    )
    assert "No models were returned" in no_models.notices[0].text

    listed = handle_model_command(
        "/model",
        state=control_state,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=["model-a", "model-b"]),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
    )
    assert "Available models" in listed.notices[0].text

    same_model = handle_model_command(
        "/model model-a",
        state=control_state,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=["model-a"]),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
    )
    assert same_model.notices[0].text == "Current model: model-a"

    switch_unavailable = handle_model_command(
        "/model model-b",
        state=control_state,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=["model-a", "model-b"]),
        switch_model=lambda name: ModelSwitchOutcome(unavailable=True),
    )
    assert switch_unavailable.notices == []

    switch_notice_only = handle_model_command(
        "/model model-b",
        state=control_state,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=["model-a", "model-b"]),
        switch_model=lambda name: ModelSwitchOutcome(
            notice=ChatControlNotice(role="error", text="switch failed")
        ),
    )
    assert switch_notice_only.notices[0].text == "switch failed"
    assert control_state.active_model_name == "model-a"

    switched = handle_model_command(
        "/model model-b",
        state=control_state,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=["model-a", "model-b"]),
        switch_model=lambda name: ModelSwitchOutcome(provider={"model": name}),
    )
    assert switched.provider == {"model": "model-b"}
    assert switched.notices[0].text == "Switched model from model-a to model-b."
    assert control_state.active_model_name == "model-b"

    switched_with_notice = handle_model_command(
        "/model model-c",
        state=control_state,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=["model-b", "model-c"]),
        switch_model=lambda name: ModelSwitchOutcome(
            provider={"model": name},
            notice=ChatControlNotice(role="system", text="switched with caution"),
        ),
    )
    assert switched_with_notice.notices[0].text == "switched with caution"
    assert control_state.active_model_name == "model-c"

    help_outcome = handle_chat_command(
        "/help",
        state=control_state,
        available_tool_specs=tool_specs,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=[]),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
        exit_mode="notice",
        exit_notice="Close the browser tab.",
    )
    assert "Ask grounded questions" in help_outcome.notices[0].text

    inspect_before = control_state.inspector_open
    inspect_outcome = handle_chat_command(
        "/inspect",
        state=control_state,
        available_tool_specs=tool_specs,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=[]),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
        exit_mode="notice",
        exit_notice="Close the browser tab.",
    )
    assert inspect_outcome.handled is True
    assert control_state.inspector_open is (not inspect_before)

    copy_outcome = handle_chat_command(
        "/copy",
        state=control_state,
        available_tool_specs=tool_specs,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=[]),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
        exit_mode="notice",
        exit_notice="Close the browser tab.",
    )
    assert copy_outcome.request_copy is True

    exit_request = handle_chat_command(
        "exit",
        state=control_state,
        available_tool_specs=tool_specs,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=[]),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
        exit_mode="request_exit",
        exit_notice="ignored",
    )
    assert exit_request.request_exit is True

    exit_notice = handle_chat_command(
        "quit",
        state=control_state,
        available_tool_specs=tool_specs,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=[]),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
        exit_mode="notice",
        exit_notice="Close the browser tab.",
    )
    assert exit_notice.notices[0].text == "Close the browser tab."

    unhandled = handle_chat_command(
        "hello",
        state=control_state,
        available_tool_specs=tool_specs,
        busy=False,
        list_models=lambda: ModelCatalogOutcome(model_ids=[]),
        switch_model=lambda name: ModelSwitchOutcome(provider=name),
        exit_mode="notice",
        exit_notice="Close the browser tab.",
    )
    assert unhandled.handled is False


def test_chat_presentation_helpers_cover_all_sections() -> None:
    response = ChatFinalResponse(
        answer="Answer body",
        citations=[
            ChatCitation(source_path="README.md"),
            ChatCitation(source_path="src/app.py", line_start=12),
            ChatCitation(source_path="src/lib.py", line_start=3, line_end=8),
        ],
        uncertainty=["Need to verify line endings"],
        missing_information=["No production config"],
        follow_up_suggestions=["Run smoke tests"],
    )

    assert pretty_json(None) == ""
    pretty = pretty_json({"when": datetime(2026, 4, 19, tzinfo=UTC)})
    assert '"when"' in pretty

    assert format_citation(response.citations[0]) == "README.md"
    assert format_citation(response.citations[1]) == "src/app.py:12"
    assert format_citation(response.citations[2]) == "src/lib.py:3-8"

    formatted = format_final_response(response)
    assert "Answer body" in formatted
    assert "Citations:" in formatted
    assert "Uncertainty:" in formatted
    assert "Missing Information:" in formatted
    assert "Follow-up Suggestions:" in formatted

    metadata = format_final_response_metadata(response)
    assert "Citations:" in metadata
    assert "Follow-up Suggestions:" in metadata
    assert format_final_response_metadata(ChatFinalResponse(answer="Only answer")) == ""

    assert format_transcript_text("assistant", "done\n") == "Assistant:\ndone"
    assert (
        format_transcript_text(
            "assistant",
            "partial\n",
            assistant_completion_state="interrupted",
        )
        == "Assistant (interrupted):\npartial"
    )
    assert format_transcript_text("user", "hello\n") == "You:\nhello"
    assert format_transcript_text("error", "boom\n") == "Error: boom"
    assert format_transcript_text("system", "note\n") == "System:\nnote"

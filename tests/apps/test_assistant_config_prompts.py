"""Coverage for assistant config, context, and prompt helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_tools.apps.assistant_config import (
    AssistantConfig,
    AssistantResearchConfig,
    AssistantWorkspaceConfig,
    load_assistant_config,
)
from llm_tools.apps.assistant_execution import build_assistant_context
from llm_tools.apps.assistant_prompts import (
    build_assistant_system_prompt,
    build_research_system_prompt,
)
from llm_tools.apps.chat_config import ProviderAuthScheme, ProviderConnectionConfig
from llm_tools.tool_api import ToolRegistry
from llm_tools.tools import register_filesystem_tools
from llm_tools.tools.filesystem import ToolLimits


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    return registry


def test_assistant_config_validators_trim_optional_paths() -> None:
    assert AssistantWorkspaceConfig(default_root="  ").default_root is None
    assert AssistantWorkspaceConfig(default_root="  /repo  ").default_root == "/repo"
    assert AssistantResearchConfig(store_dir="  ").store_dir is None
    assert AssistantResearchConfig(store_dir="  /state  ").store_dir == "/state"


def test_load_assistant_config_rejects_bad_paths_and_shapes(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ValueError, match="not found"):
        load_assistant_config(missing)

    with pytest.raises(ValueError, match="not a file"):
        load_assistant_config(tmp_path)

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected mapping"):
        load_assistant_config(scalar)

    bad_section = tmp_path / "bad-section.yaml"
    bad_section.write_text("llm: nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'llm' must be a mapping"):
        load_assistant_config(bad_section)

    invalid_model = tmp_path / "invalid-model.yaml"
    invalid_model.write_text("research:\n  default_max_turns: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid assistant config"):
        load_assistant_config(invalid_model)

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert isinstance(load_assistant_config(empty), AssistantConfig)


def test_load_assistant_config_deeply_overlays_project_defaults(
    tmp_path: Path,
) -> None:
    base_config = AssistantConfig(
        llm={
            "selected_model": "base-model",
            "provider_connection": ProviderConnectionConfig(
                api_base_url="http://127.0.0.1:11434/v1",
                auth_scheme=ProviderAuthScheme.NONE,
            ),
        },
        policy={"enabled_tools": ["read_file", "search_text"]},
    )
    config_file = tmp_path / "assistant.yaml"
    config_file.write_text(
        "llm:\n"
        "  selected_model: overlay-model\n"
        "policy:\n"
        "  enabled_tools:\n"
        "    - list_directory\n",
        encoding="utf-8",
    )

    config = load_assistant_config(config_file, base_config=base_config)

    assert config.llm.selected_model == "overlay-model"
    assert config.llm.provider_connection.api_base_url == "http://127.0.0.1:11434/v1"
    assert config.llm.provider_connection.auth_scheme is ProviderAuthScheme.NONE
    assert config.policy.enabled_tools == ["list_directory"]


def test_build_assistant_context_uses_session_only_environment(tmp_path: Path) -> None:
    config = AssistantConfig()

    context = build_assistant_context(
        root_path=tmp_path,
        config=config,
        app_name="test-app",
        env_overrides={"TOKEN": "  secret  ", "BLANK": "  "},
        include_process_env=False,
    )

    assert context.workspace == str(tmp_path)
    assert context.env == {"TOKEN": "secret"}
    assert context.metadata["assistant_mode"] == "assistant_app"
    assert context.metadata["tool_limits"]["max_read_file_chars"] == (
        config.session.max_context_tokens * 4
    )


def test_assistant_prompt_protocol_variants_are_protocol_specific() -> None:
    registry = _registry()
    limits = ToolLimits(max_file_size_characters=1234, max_tool_result_chars=5678)

    prompt_tools = build_assistant_system_prompt(
        tool_registry=registry,
        tool_limits=limits,
        enabled_tool_names={"read_file"},
        workspace_enabled=False,
        interaction_protocol="prompt_tools",
    )
    native_tools = build_assistant_system_prompt(
        tool_registry=registry,
        tool_limits=limits,
        enabled_tool_names={"read_file"},
        interaction_protocol="native_tools",
    )
    staged_json = build_assistant_system_prompt(
        tool_registry=registry,
        tool_limits=limits,
        enabled_tool_names={"read_file"},
        interaction_protocol="staged_json",
    )
    action_envelope = build_assistant_system_prompt(
        tool_registry=registry,
        tool_limits=limits,
        enabled_tool_names={"read_file"},
        interaction_protocol="action_envelope",
    )

    assert "Prompt-tool protocol" in prompt_tools
    assert "No workspace root is configured" in prompt_tools
    assert "read_file" in prompt_tools
    assert "list_directory" not in prompt_tools
    assert "Native tool protocol" in native_tools
    assert "Final answer content should include" in native_tools
    assert "Structured interaction protocol" in staged_json
    assert "Required action format" in action_envelope
    assert "max_tool_result_chars" not in prompt_tools
    assert "5678" in action_envelope


def test_research_prompt_variants_include_harness_guidance() -> None:
    registry = _registry()
    limits = ToolLimits(max_tool_result_chars=222)

    staged = build_research_system_prompt(
        tool_registry=registry,
        tool_limits=limits,
        enabled_tool_names={"search_text"},
        workspace_enabled=False,
        staged_schema_protocol=True,
    )
    envelope = build_research_system_prompt(
        tool_registry=registry,
        tool_limits=limits,
        enabled_tool_names={"search_text"},
        workspace_enabled=True,
        staged_schema_protocol=False,
    )

    assert "durable research assistant" in staged
    assert "No workspace root is configured" in staged
    assert "Structured interaction protocol" in staged
    assert "search_text" in staged
    assert "read_file" not in staged
    assert "Required action format" in envelope
    assert "workspace root is configured" in envelope
    assert "222" in envelope

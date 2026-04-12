"""Pure-Python tests for the Textual workbench controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_tools.apps.textual_workbench.controller import WorkbenchController
from llm_tools.apps.textual_workbench.models import ProviderPreset, WorkbenchMode
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import SideEffectClass
from llm_tools.workflow_api import WorkflowTurnResult


class _FakeProvider:
    last_mode: str | None = None
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @classmethod
    def for_openai(cls, **kwargs: Any) -> _FakeProvider:
        return cls(**kwargs)

    @classmethod
    def for_ollama(cls, **kwargs: Any) -> _FakeProvider:
        return cls(**kwargs)

    def run_native_tool_calling(self, **kwargs: Any) -> ParsedModelResponse:
        type(self).last_mode = "native_tool_calling"
        type(self).last_kwargs = kwargs
        return ParsedModelResponse(final_response="native ok")

    def run_structured_output(self, **kwargs: Any) -> ParsedModelResponse:
        type(self).last_mode = "structured_output"
        type(self).last_kwargs = kwargs
        return ParsedModelResponse(final_response="structured ok")

    def run_prompt_schema(self, **kwargs: Any) -> ParsedModelResponse:
        type(self).last_mode = "prompt_schema"
        type(self).last_kwargs = kwargs
        return ParsedModelResponse(final_response="prompt ok")


def test_default_registry_contains_safe_builtins_only() -> None:
    controller = WorkbenchController()
    config = controller.default_config()

    assert controller.list_tool_names(config) == [
        "read_file",
        "write_file",
        "list_directory",
        "run_git_status",
        "run_git_diff",
        "run_git_log",
        "file_text_search",
        "directory_text_search",
    ]


def test_atlassian_is_available_when_enabled() -> None:
    controller = WorkbenchController()
    config = controller.default_config().model_copy(
        update={"enable_atlassian_tools": True}
    )

    assert "search_jira" in controller.list_tool_names(config)
    assert "read_jira_issue" in controller.list_tool_names(config)


def test_provider_preset_defaults_are_applied() -> None:
    controller = WorkbenchController()
    config = controller.default_config()

    openai_config = controller.apply_provider_preset(config, ProviderPreset.OPENAI)
    custom_config = controller.apply_provider_preset(
        config, ProviderPreset.CUSTOM_OPENAI_COMPATIBLE
    )

    assert openai_config.model == "gpt-4.1-mini"
    assert openai_config.base_url == ""
    ollama_config = controller.apply_provider_preset(config, ProviderPreset.OLLAMA)
    assert ollama_config.model == "gemma4:26b"
    assert ollama_config.base_url == "http://localhost:11434/v1"
    assert custom_config.model == ""
    assert custom_config.base_url == ""


def test_cycle_helpers_walk_presets_and_modes() -> None:
    controller = WorkbenchController()
    config = controller.default_config()

    cycled_provider = controller.cycle_provider_preset(
        config.model_copy(update={"provider_preset": ProviderPreset.OPENAI})
    )
    cycled_mode = controller.cycle_mode(
        config.model_copy(update={"mode": WorkbenchMode.NATIVE_TOOL_CALLING})
    )

    assert cycled_provider.provider_preset is ProviderPreset.OLLAMA
    assert cycled_mode.mode is WorkbenchMode.STRUCTURED_OUTPUT


def test_export_tools_matches_mode_shape() -> None:
    controller = WorkbenchController()
    config = controller.default_config()

    native_export = controller.export_tools(config)
    structured_export = controller.export_tools(
        config.model_copy(update={"mode": WorkbenchMode.STRUCTURED_OUTPUT})
    )
    prompt_export = controller.export_tools(
        config.model_copy(update={"mode": WorkbenchMode.PROMPT_SCHEMA})
    )

    assert isinstance(native_export, list)
    assert isinstance(structured_export, dict)
    assert isinstance(prompt_export, str)
    assert [tool["function"]["name"] for tool in native_export] == [
        "read_file",
        "list_directory",
        "run_git_status",
        "run_git_diff",
        "run_git_log",
        "file_text_search",
        "directory_text_search",
    ]
    assert "write_file" not in [tool["function"]["name"] for tool in native_export]
    assert (
        "write_file"
        not in structured_export["properties"]["actions"]["items"]["properties"][
            "tool_name"
        ]["enum"]
    )
    assert "Tool: write_file" not in prompt_export


def test_execute_direct_tool_produces_tool_result(tmp_path: Path) -> None:
    controller = WorkbenchController()
    config = controller.default_config().model_copy(
        update={
            "workspace": str(tmp_path),
            "allow_local_write": True,
        }
    )

    write_result = controller.execute_direct_tool(
        config,
        tool_name="write_file",
        arguments_text='{"path":"notes.txt","content":"hello"}',
    )
    read_result = controller.execute_direct_tool(
        config,
        tool_name="read_file",
        arguments_text='{"path":"notes.txt"}',
    )

    assert write_result.tool_result.ok is True
    assert read_result.tool_result.output is not None
    assert read_result.tool_result.output["content"] == "hello"


def test_execute_direct_tool_rejects_invalid_json() -> None:
    controller = WorkbenchController()

    with pytest.raises(ValueError):
        controller.execute_direct_tool(
            controller.default_config(),
            tool_name="read_file",
            arguments_text="not json",
        )


@pytest.mark.parametrize(
    ("mode", "expected_mode", "expected_response"),
    [
        (WorkbenchMode.NATIVE_TOOL_CALLING, "native_tool_calling", "native ok"),
        (WorkbenchMode.STRUCTURED_OUTPUT, "structured_output", "structured ok"),
        (WorkbenchMode.PROMPT_SCHEMA, "prompt_schema", "prompt ok"),
    ],
)
def test_run_model_turn_routes_to_the_selected_provider_mode(
    mode: WorkbenchMode,
    expected_mode: str,
    expected_response: str,
) -> None:
    controller = WorkbenchController(provider_factory=_FakeProvider)
    config = controller.default_config().model_copy(update={"mode": mode})

    result = controller.run_model_turn(config, prompt="hello")

    assert _FakeProvider.last_mode == expected_mode
    assert _FakeProvider.last_kwargs is not None
    assert "tool_descriptions" in _FakeProvider.last_kwargs
    assert "registry" not in _FakeProvider.last_kwargs
    assert result.parsed_response.final_response == expected_response
    assert isinstance(result.workflow_result, WorkflowTurnResult)


def test_run_model_turn_can_stop_after_parse_without_execution() -> None:
    controller = WorkbenchController(provider_factory=_FakeProvider)
    config = controller.default_config().model_copy(
        update={"execute_after_parse": False}
    )

    result = controller.run_model_turn(config, prompt="hello")

    assert result.parsed_response.final_response == "native ok"
    assert result.workflow_result is None


def test_run_model_turn_validation_errors_are_raised_cleanly() -> None:
    controller = WorkbenchController(provider_factory=_FakeProvider)
    config = controller.default_config().model_copy(
        update={
            "provider_preset": ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
            "base_url": "",
        }
    )

    with pytest.raises(ValueError):
        controller.run_model_turn(config, prompt="hello")

    with pytest.raises(ValueError):
        controller.run_model_turn(controller.default_config(), prompt="  ")


def test_get_tool_details_returns_schema_for_selected_tool() -> None:
    controller = WorkbenchController()
    spec_payload, input_schema = controller.get_tool_details(
        controller.default_config(),
        "read_file",
    )

    assert spec_payload is not None
    assert input_schema is not None
    assert spec_payload["name"] == "read_file"
    assert input_schema["title"] == "ReadFileInput"


def test_get_tool_details_returns_none_for_blank_or_missing_tools() -> None:
    controller = WorkbenchController()

    assert controller.get_tool_details(controller.default_config(), "") == (None, None)
    assert controller.get_tool_details(controller.default_config(), "missing_tool") == (
        None,
        None,
    )


def test_build_policy_and_direct_execution_validation_cover_extra_branches() -> None:
    controller = WorkbenchController()
    config = controller.default_config().model_copy(
        update={
            "allow_local_write": True,
            "allow_external_read": True,
            "allow_external_write": True,
            "allow_network": False,
            "allow_filesystem": False,
            "allow_subprocess": False,
            "enable_filesystem_tools": False,
            "enable_git_tools": False,
            "enable_text_tools": False,
        }
    )
    policy = controller.build_policy(config)

    assert SideEffectClass.LOCAL_WRITE in policy.allowed_side_effects
    assert SideEffectClass.EXTERNAL_READ in policy.allowed_side_effects
    assert SideEffectClass.EXTERNAL_WRITE in policy.allowed_side_effects
    assert controller.list_tool_names(config) == []

    with pytest.raises(ValueError):
        controller.execute_direct_tool(
            controller.default_config(),
            tool_name="",
            arguments_text="{}",
        )

    with pytest.raises(ValueError):
        controller.execute_direct_tool(
            controller.default_config(),
            tool_name="read_file",
            arguments_text='["not", "an", "object"]',
        )

"""Pure-Python tests for the Textual workbench controller."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from llm_tools.apps.textual_workbench.controller import WorkbenchController
from llm_tools.apps.textual_workbench.models import ProviderModeStrategy, ProviderPreset
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.llm_providers import ProviderModeStrategy as ProviderRunMode
from llm_tools.tool_api import SideEffectClass
from llm_tools.workflow_api import WorkflowTurnResult
from llm_tools.workflow_api.models import WorkflowInvocationStatus


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

    def run(self, **kwargs: Any) -> ParsedModelResponse:
        type(self).last_mode = "run"
        type(self).last_kwargs = kwargs
        return ParsedModelResponse(final_response="run ok")

    async def run_async(self, **kwargs: Any) -> ParsedModelResponse:
        type(self).last_mode = "run_async"
        type(self).last_kwargs = kwargs
        return ParsedModelResponse(final_response="run async ok")


def test_default_registry_contains_safe_builtins_only() -> None:
    controller = WorkbenchController()
    config = controller.default_config()

    assert controller.list_tool_names(config) == [
        "read_file",
        "write_file",
        "list_directory",
        "find_files",
        "get_file_info",
        "run_git_status",
        "run_git_diff",
        "run_git_log",
        "search_text",
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


def test_cycle_helpers_walk_presets_and_provider_mode_strategy() -> None:
    controller = WorkbenchController()
    config = controller.default_config()

    cycled_provider = controller.cycle_provider_preset(
        config.model_copy(update={"provider_preset": ProviderPreset.OPENAI})
    )
    cycled_mode = controller.cycle_provider_mode_strategy(
        config.model_copy(update={"provider_mode_strategy": ProviderModeStrategy.AUTO})
    )

    assert cycled_provider.provider_preset is ProviderPreset.OLLAMA
    assert cycled_mode.provider_mode_strategy is ProviderModeStrategy.TOOLS


def test_export_tools_returns_schema_and_respects_visibility() -> None:
    controller = WorkbenchController()
    config = controller.default_config()

    exported = controller.export_tools(config).exported_tools

    assert isinstance(exported, dict)
    assert "ActionEnvelope" in exported["title"]
    exported_str = str(exported)
    assert "read_file" in exported_str
    assert "list_directory" in exported_str
    assert "write_file" not in exported_str


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
    assert all(
        outcome.status is WorkflowInvocationStatus.EXECUTED
        for outcome in write_result.workflow_result.outcomes
    )


def test_execute_direct_tool_rejects_invalid_json() -> None:
    controller = WorkbenchController()

    with pytest.raises(ValueError):
        controller.execute_direct_tool(
            controller.default_config(),
            tool_name="read_file",
            arguments_text="not json",
        )


@pytest.mark.parametrize(
    "strategy",
    [
        ProviderModeStrategy.AUTO,
        ProviderModeStrategy.TOOLS,
        ProviderModeStrategy.JSON,
        ProviderModeStrategy.MD_JSON,
    ],
)
def test_run_model_turn_uses_single_provider_run_path(
    strategy: ProviderModeStrategy,
) -> None:
    controller = WorkbenchController(provider_factory=_FakeProvider)
    config = controller.default_config().model_copy(
        update={"provider_mode_strategy": strategy}
    )

    result = controller.run_model_turn(config, prompt="hello")

    assert _FakeProvider.last_mode == "run"
    assert _FakeProvider.last_kwargs is not None
    assert "response_model" in _FakeProvider.last_kwargs
    assert result.parsed_response.final_response == "run ok"
    assert isinstance(result.workflow_result, WorkflowTurnResult)
    assert result.workflow_result.outcomes == []
    provider = controller._build_provider(config)
    assert provider.kwargs["mode_strategy"] is ProviderRunMode(strategy.value)


def test_run_model_turn_can_stop_after_parse_without_execution() -> None:
    controller = WorkbenchController(provider_factory=_FakeProvider)
    config = controller.default_config().model_copy(
        update={"execute_after_parse": False}
    )

    result = controller.run_model_turn(config, prompt="hello")

    assert result.parsed_response.final_response == "run ok"
    assert result.workflow_result is None


def test_direct_execution_can_enqueue_and_resolve_approval(tmp_path: Path) -> None:
    controller = WorkbenchController()
    config = controller.default_config().model_copy(
        update={
            "workspace": str(tmp_path),
            "require_approval_for_local_read": True,
        }
    )

    pending = controller.execute_direct_tool(
        config,
        tool_name="list_directory",
        arguments_text='{"path":"."}',
    )
    queue = controller.list_pending_approvals(config)

    assert pending.tool_result is None
    assert (
        pending.workflow_result.outcomes[0].status
        is WorkflowInvocationStatus.APPROVAL_REQUESTED
    )
    assert len(queue) == 1

    denied = controller.resolve_pending_approval(
        config,
        approval_id=queue[0].approval_id,
        approved=False,
    )
    assert (
        denied.workflow_result.outcomes[0].status
        is WorkflowInvocationStatus.APPROVAL_DENIED
    )


def test_session_rebuild_clears_pending_approvals_on_config_change(
    tmp_path: Path,
) -> None:
    controller = WorkbenchController()
    config = controller.default_config().model_copy(
        update={
            "workspace": str(tmp_path),
            "require_approval_for_local_read": True,
        }
    )

    controller.execute_direct_tool(
        config,
        tool_name="list_directory",
        arguments_text='{"path":"."}',
    )
    assert controller.list_pending_approvals(config) != []

    changed = config.model_copy(update={"allow_external_read": True})
    export_result = controller.export_tools(changed)
    assert export_result.session_rebuilt is True
    assert controller.list_pending_approvals(changed) == []


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


@pytest.mark.parametrize(
    "strategy",
    [
        ProviderModeStrategy.AUTO,
        ProviderModeStrategy.TOOLS,
        ProviderModeStrategy.JSON,
        ProviderModeStrategy.MD_JSON,
    ],
)
def test_async_model_turn_routes_to_single_provider_path(
    strategy: ProviderModeStrategy,
) -> None:
    async def run() -> None:
        controller = WorkbenchController(provider_factory=_FakeProvider)
        config = controller.default_config().model_copy(
            update={"provider_mode_strategy": strategy}
        )
        result = await controller.run_model_turn_async(config, prompt="hello")

        assert _FakeProvider.last_mode == "run_async"
        assert _FakeProvider.last_kwargs is not None
        assert "response_model" in _FakeProvider.last_kwargs
        assert result.parsed_response.final_response == "run async ok"
        assert isinstance(result.workflow_result, WorkflowTurnResult)
        assert result.workflow_result.outcomes == []

    asyncio.run(run())


def test_async_direct_execution_and_approval_paths(tmp_path: Path) -> None:
    async def run() -> None:
        controller = WorkbenchController()
        config = controller.default_config().model_copy(
            update={
                "workspace": str(tmp_path),
                "allow_local_write": True,
                "require_approval_for_local_read": True,
            }
        )

        write_result = await controller.execute_direct_tool_async(
            config,
            tool_name="write_file",
            arguments_text='{"path":"notes.txt","content":"hello"}',
        )
        pending = await controller.execute_direct_tool_async(
            config,
            tool_name="list_directory",
            arguments_text='{"path":"."}',
        )

        assert write_result.tool_result is not None
        assert write_result.tool_result.ok is True
        assert pending.tool_result is None
        assert (
            pending.workflow_result.outcomes[0].status
            is WorkflowInvocationStatus.APPROVAL_REQUESTED
        )

        queue = controller.list_pending_approvals(config)
        denied = await controller.resolve_pending_approval_async(
            config,
            approval_id=queue[0].approval_id,
            approved=False,
        )
        finalized = await controller.finalize_expired_approvals_async(config)

        assert (
            denied.workflow_result.outcomes[0].status
            is WorkflowInvocationStatus.APPROVAL_DENIED
        )
        assert finalized.workflow_results == []

    asyncio.run(run())

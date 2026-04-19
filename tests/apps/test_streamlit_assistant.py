"""Tests for the Streamlit assistant app layer."""

from __future__ import annotations

import importlib
import runpy
import sys
import tomllib
from pathlib import Path

import pytest
from tests.apps._imports import import_streamlit_assistant_modules

from llm_tools.apps.assistant_config import (
    AssistantResearchConfig,
    AssistantWorkspaceConfig,
    StreamlitAssistantConfig,
    load_streamlit_assistant_config,
)
from llm_tools.apps.assistant_prompts import (
    build_assistant_system_prompt,
    build_research_system_prompt,
)
from llm_tools.apps.assistant_runtime import (
    AssistantHarnessTurnProvider,
    build_assistant_available_tool_specs,
    build_assistant_context,
    build_assistant_executor,
    build_assistant_policy,
    build_assistant_registry,
    build_tool_capabilities,
    resolve_assistant_default_enabled_tools,
)
from llm_tools.harness_api import (
    BudgetPolicy,
    HarnessSessionService,
    InMemoryHarnessStateStore,
    ScriptedParsedResponseProvider,
)
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import SideEffectClass, ToolContext
from llm_tools.workflow_api import ChatSessionState

_MODULES = import_streamlit_assistant_modules()
build_parser = _MODULES.app.build_parser
process_streamlit_assistant_turn = _MODULES.app.process_streamlit_assistant_turn
_resolve_assistant_config = _MODULES.app._resolve_assistant_config
AssistantResearchSessionController = _MODULES.app.AssistantResearchSessionController


class _FakeProvider:
    def __init__(self, responses: list[ParsedModelResponse]) -> None:
        self._responses = list(responses)

    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        return self._responses.pop(0)


def test_streamlit_assistant_package_imports_without_loading_streamlit() -> None:
    module = importlib.import_module("llm_tools.apps.streamlit_assistant")
    main_module = importlib.import_module("llm_tools.apps.streamlit_assistant.__main__")

    assert hasattr(module, "main")
    assert hasattr(module, "run_streamlit_assistant_app")
    assert hasattr(main_module, "main")


def test_console_script_target_is_declared_in_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["llm-tools-streamlit-assistant"]
        == "llm_tools.apps.streamlit_assistant:main"
    )


def test_module_main_helpers_dispatch_to_package_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_assistant")
    main_module = importlib.import_module("llm_tools.apps.streamlit_assistant.__main__")

    monkeypatch.setattr(package, "main", lambda: 9)

    assert main_module._main() == 9
    assert main_module.main() == 9


def test_module_entrypoint_raises_system_exit_with_main_return_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_assistant")
    called: list[str] = []
    monkeypatch.setattr(package, "main", lambda: called.append("main") or 0)

    sys.modules.pop("llm_tools.apps.streamlit_assistant.__main__", None)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(
            "llm_tools.apps.streamlit_assistant.__main__", run_name="__main__"
        )

    assert exc.value.code == 0
    assert called == ["main"]


def test_resolve_assistant_config_uses_new_model(tmp_path: Path) -> None:
    config_path = tmp_path / "assistant.yml"
    config_path.write_text(
        """
llm:
  provider: ollama
  model_name: demo-model
workspace:
  default_root: .
research:
  default_max_turns: 9
""".strip(),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["--config", str(config_path), "--model", "override-model"]
    )
    config = _resolve_assistant_config(args)

    assert isinstance(config, StreamlitAssistantConfig)
    assert config.llm.model_name == "override-model"
    assert config.workspace.default_root == "."
    assert config.research.default_max_turns == 9


def test_assistant_runtime_exposes_full_registry_and_defaults_to_no_enabled_tools() -> (
    None
):
    tool_specs = build_assistant_available_tool_specs()
    config = StreamlitAssistantConfig()

    assert "write_file" in tool_specs
    assert "search_gitlab_code" in tool_specs
    assert "search_jira" in tool_specs
    assert resolve_assistant_default_enabled_tools(config) == set()


def test_build_tool_capabilities_reports_blocked_states() -> None:
    tool_specs = build_assistant_available_tool_specs()
    capabilities = build_tool_capabilities(
        tool_specs=tool_specs,
        enabled_tools={"read_file", "search_gitlab_code", "write_file"},
        root_path=None,
        env={},
        allow_network=False,
        allow_filesystem=False,
        allow_subprocess=False,
        require_approval_for={SideEffectClass.LOCAL_WRITE},
    )

    all_items = {
        item.tool_name: item for items in capabilities.values() for item in items
    }
    assert all_items["read_file"].status == "missing_workspace"
    assert all_items["search_gitlab_code"].status == "missing_credentials"
    assert all_items["write_file"].approval_required is True


def test_process_streamlit_assistant_turn_supports_direct_answers() -> None:
    outcome = process_streamlit_assistant_turn(
        root_path=None,
        config=StreamlitAssistantConfig(),
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    final_response={
                        "answer": "Plain answer",
                        "citations": [],
                        "confidence": 0.8,
                        "uncertainty": [],
                        "missing_information": [],
                        "follow_up_suggestions": [],
                    }
                )
            ]
        ),
        session_state=ChatSessionState(),
        user_message="hello",
    )

    assert outcome.session_state.turns
    assert outcome.transcript_entries[-1].final_response is not None
    assert outcome.transcript_entries[-1].final_response.answer == "Plain answer"


def test_research_controller_can_launch_list_and_stop_sessions() -> None:
    store = InMemoryHarnessStateStore()

    def service_factory() -> HarnessSessionService:
        _, workflow_executor = build_assistant_executor()
        return HarnessSessionService(
            store=store,
            workflow_executor=workflow_executor,
            provider=ScriptedParsedResponseProvider(
                [ParsedModelResponse(final_response="research complete")]
            ),
            workspace=".",
        )

    controller = AssistantResearchSessionController(
        service_factory=service_factory,
        budget_policy=BudgetPolicy(max_turns=2),
        include_replay_by_default=False,
        list_limit=5,
    )

    inspection = controller.launch(prompt="Investigate the workspace")
    listed = controller.list_recent()
    stopped = controller.stop(inspection.snapshot.session_id)

    assert listed.sessions
    assert listed.sessions[0].snapshot.session_id == inspection.snapshot.session_id
    assert "Research session:" in controller.summary_text(inspection)
    assert stopped.snapshot.session_id == inspection.snapshot.session_id


def test_package_main_and_runner_dispatch_to_app_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_assistant")
    called: list[str] = []

    monkeypatch.setattr(
        "llm_tools.apps.streamlit_assistant.app.run_streamlit_assistant_app",
        lambda **kwargs: called.append(f"run:{kwargs['root_path']}"),
    )
    monkeypatch.setattr(
        "llm_tools.apps.streamlit_assistant.app.main",
        lambda argv=None: called.append(f"main:{argv}") or 0,
    )

    package.run_streamlit_assistant_app(
        root_path=None, config=StreamlitAssistantConfig()
    )
    assert package.main() == 0
    assert called == ["run:None", "main:None"]


def test_assistant_prompts_cover_normal_and_research_modes() -> None:
    registry = build_assistant_registry()
    enabled = {"read_file", "search_jira"}

    assistant_prompt = build_assistant_system_prompt(
        tool_registry=registry,
        tool_limits=StreamlitAssistantConfig().tool_limits,
        enabled_tool_names=enabled,
        workspace_enabled=False,
    )
    research_prompt = build_research_system_prompt(
        tool_registry=registry,
        tool_limits=StreamlitAssistantConfig().tool_limits,
        enabled_tool_names=enabled,
        workspace_enabled=True,
    )

    assert "general-purpose assistant" in assistant_prompt
    assert "No workspace root is configured" in assistant_prompt
    assert "search_jira" in assistant_prompt
    assert "durable research assistant" in research_prompt
    assert "A workspace root is configured" in research_prompt


def test_assistant_config_validators_normalize_blank_values() -> None:
    assert AssistantWorkspaceConfig(default_root=None).default_root is None
    assert AssistantWorkspaceConfig(default_root="   ").default_root is None
    assert (
        AssistantWorkspaceConfig(default_root=" ./workspace ").default_root
        == "./workspace"
    )
    assert AssistantResearchConfig(store_dir=None).store_dir is None
    assert AssistantResearchConfig(store_dir="   ").store_dir is None
    assert AssistantResearchConfig(store_dir=" .assistant ").store_dir == ".assistant"


def test_load_streamlit_assistant_config_handles_missing_nonfile_and_empty_yaml(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.yml"
    empty_path = tmp_path / "empty.yml"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration file not found"):
        load_streamlit_assistant_config(missing_path)

    with pytest.raises(ValueError, match="Configuration path is not a file"):
        load_streamlit_assistant_config(tmp_path)

    config = load_streamlit_assistant_config(empty_path)

    assert isinstance(config, StreamlitAssistantConfig)


def test_load_streamlit_assistant_config_rejects_non_mapping_and_invalid_yaml(
    tmp_path: Path,
) -> None:
    scalar_path = tmp_path / "scalar.yml"
    scalar_path.write_text("- item\n", encoding="utf-8")
    invalid_yaml_path = tmp_path / "invalid.yml"
    invalid_yaml_path.write_text("llm: [oops\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected mapping at root of YAML file"):
        load_streamlit_assistant_config(scalar_path)

    with pytest.raises(ValueError, match="Invalid YAML at"):
        load_streamlit_assistant_config(invalid_yaml_path)


def test_load_streamlit_assistant_config_wraps_nested_validation_errors(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid-research.yml"
    config_path.write_text(
        "research:\n  default_max_turns: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid assistant config at"):
        load_streamlit_assistant_config(config_path)


def test_assistant_config_validation_rejects_invalid_shapes(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yml"
    config_path.write_text("session: nope", encoding="utf-8")

    with pytest.raises(ValueError):
        _resolve_assistant_config(
            build_parser().parse_args(["--config", str(config_path)])
        )


def test_assistant_runtime_builds_policy_and_context() -> None:
    tool_specs = build_assistant_available_tool_specs()
    policy = build_assistant_policy(
        enabled_tools={"read_file", "write_file"},
        tool_specs=tool_specs,
        require_approval_for={SideEffectClass.LOCAL_WRITE},
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
        redaction_config=StreamlitAssistantConfig().policy.redaction,
    )
    context = build_assistant_context(
        root_path=Path(".").resolve(),
        config=StreamlitAssistantConfig(),
        app_name="assistant-test",
    )

    assert policy.allowed_tools == {"read_file", "write_file"}
    assert policy.allow_filesystem is True
    assert context.workspace
    assert context.metadata["assistant_mode"] == "streamlit_assistant"
    assert "tool_limits" in context.metadata


class _RecordingProvider:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None

    def run(self, **kwargs: object) -> ParsedModelResponse:
        self.messages = kwargs["messages"]  # type: ignore[index]
        return ParsedModelResponse(final_response="done")

    async def run_async(self, **kwargs: object) -> ParsedModelResponse:
        self.messages = kwargs["messages"]  # type: ignore[index]
        return ParsedModelResponse(final_response="done")


def test_assistant_harness_turn_provider_builds_research_messages() -> None:
    provider = _RecordingProvider()
    harness_provider = AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )
    response = harness_provider.run(
        state=object(),
        selected_task_ids=["task-1"],
        context=ToolContext(
            invocation_id="turn-1",
            metadata={"harness_turn_context": {"turn_index": 1}},
        ),
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=build_assistant_executor()[1].prepare_model_interaction(
            ActionEnvelopeAdapter(),
            context=ToolContext(invocation_id="demo"),
            final_response_model=str,
        ),
    )

    assert response.final_response == "done"
    assert provider.messages is not None
    assert provider.messages[0]["content"] == "research-system"
    assert '"task-1"' in provider.messages[1]["content"]

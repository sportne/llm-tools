"""Smoke tests for curated Streamlit assistant config examples."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tests.apps._imports import import_streamlit_assistant_modules

from llm_tools.apps.assistant_config import (
    AssistantResearchConfig,
    AssistantWorkspaceConfig,
    StreamlitAssistantConfig,
    load_streamlit_assistant_config,
)
from llm_tools.apps.chat_config import ChatLLMConfig, ChatPolicyConfig, ChatUIConfig
from llm_tools.harness_api import HarnessStopReason, ScriptedParsedResponseProvider
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.llm_providers import ProviderModeStrategy
from llm_tools.tool_api import SideEffectClass, ToolInvocationRequest
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import ChatSessionConfig, ChatSessionState
from llm_tools.workflow_api.protection import ProtectionConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples" / "assistant_configs"
EXPECTED_TOP_LEVEL_KEYS = list(StreamlitAssistantConfig.model_fields)
SECTION_MODELS = {
    "llm": ChatLLMConfig,
    "session": ChatSessionConfig,
    "tool_limits": ToolLimits,
    "policy": ChatPolicyConfig,
    "protection": ProtectionConfig,
    "ui": ChatUIConfig,
    "workspace": AssistantWorkspaceConfig,
    "research": AssistantResearchConfig,
}

LOCAL_ONLY_TOOLS = {
    "list_directory",
    "find_files",
    "get_file_info",
    "read_file",
    "search_text",
}
ENTERPRISE_DATA_TOOLS = {
    "search_jira",
    "read_jira_issue",
    "search_confluence",
    "read_confluence_attachment",
    "read_confluence_page",
    "search_bitbucket_code",
    "read_bitbucket_file",
    "read_bitbucket_pull_request",
    "search_gitlab_code",
    "read_gitlab_file",
    "read_gitlab_merge_request",
}
HARNESS_RESEARCH_TOOLS = {
    "list_directory",
    "find_files",
    "get_file_info",
    "read_file",
    "search_text",
    "run_git_status",
    "run_git_diff",
    "run_git_log",
}
_MODULES = import_streamlit_assistant_modules()


class _SequenceProvider:
    def __init__(self, responses: list[ParsedModelResponse]) -> None:
        self._responses = list(responses)

    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        return self._responses.pop(0)


def _load_raw_example(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _assert_supported_section_keys(raw: dict[str, object]) -> None:
    assert list(raw.keys()) == EXPECTED_TOP_LEVEL_KEYS
    for section_name, model in SECTION_MODELS.items():
        section = raw[section_name]
        assert isinstance(section, dict)
        assert set(section).issubset(model.model_fields)


def _final_response(answer: str) -> dict[str, object]:
    return {
        "answer": answer,
        "citations": [],
        "confidence": 0.7,
        "uncertainty": [],
        "missing_information": [],
        "follow_up_suggestions": [],
    }


def test_local_only_chat_example_loads_cleanly() -> None:
    path = EXAMPLES_DIR / "local-only-chat.yaml"
    raw = _load_raw_example(path)

    _assert_supported_section_keys(raw)

    config = load_streamlit_assistant_config(path)

    assert config.llm.provider.value == "ollama"
    assert config.llm.provider_mode_strategy is ProviderModeStrategy.AUTO
    assert config.llm.provider_mode_strategy is ProviderModeStrategy.AUTO
    assert config.workspace.default_root == "."
    assert config.research.enabled is False
    assert set(config.policy.enabled_tools or []) == LOCAL_ONLY_TOOLS


def test_enterprise_data_chat_example_loads_cleanly() -> None:
    path = EXAMPLES_DIR / "enterprise-data-chat.yaml"
    raw = _load_raw_example(path)

    _assert_supported_section_keys(raw)

    config = load_streamlit_assistant_config(path)

    assert config.llm.provider.value == "custom_openai_compatible"
    assert config.llm.provider_mode_strategy is ProviderModeStrategy.JSON
    assert config.workspace.default_root is None
    assert config.research.enabled is False
    assert set(config.policy.enabled_tools or []) == ENTERPRISE_DATA_TOOLS


def test_harness_research_chat_example_loads_cleanly() -> None:
    path = EXAMPLES_DIR / "harness-research-chat.yaml"
    raw = _load_raw_example(path)

    _assert_supported_section_keys(raw)

    config = load_streamlit_assistant_config(path)

    assert config.llm.provider.value == "ollama"
    assert config.workspace.default_root == "."
    assert config.research.enabled is True
    assert config.research.store_dir is None
    assert config.research.max_recent_sessions == 8
    assert config.research.default_max_turns == 12
    assert config.research.default_max_tool_invocations == 48
    assert config.research.default_max_elapsed_seconds == 900
    assert config.research.include_replay_by_default is True
    assert set(config.policy.enabled_tools or []) == HARNESS_RESEARCH_TOOLS


def test_local_only_chat_example_supports_direct_turn_without_workspace() -> None:
    config = load_streamlit_assistant_config(EXAMPLES_DIR / "local-only-chat.yaml")

    outcome = _MODULES.app.process_streamlit_assistant_turn(
        root_path=None,
        config=config,
        provider=_SequenceProvider(
            [ParsedModelResponse(final_response=_final_response("Direct answer"))]
        ),
        session_state=ChatSessionState(),
        user_message="What can this assistant do without tools?",
    )

    assert outcome.session_state.turns
    assert outcome.transcript_entries[-1].final_response is not None
    assert outcome.transcript_entries[-1].final_response.answer == "Direct answer"


def test_local_only_chat_example_supports_workspace_read_turn(
    tmp_path: Path,
) -> None:
    config = load_streamlit_assistant_config(EXAMPLES_DIR / "local-only-chat.yaml")
    (tmp_path / "notes.txt").write_text("Workspace facts for the assistant.", "utf-8")
    runtime = _MODULES.models.StreamlitRuntimeConfig(
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        api_base_url=config.llm.api_base_url,
        root_path=str(tmp_path),
        enabled_tools=list(config.policy.enabled_tools or []),
        allow_filesystem=True,
    )

    outcome = _MODULES.app.process_streamlit_assistant_turn(
        root_path=tmp_path,
        config=config,
        runtime_config=runtime,
        provider=_SequenceProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        ToolInvocationRequest(
                            tool_name="read_file",
                            arguments={"path": "notes.txt"},
                        )
                    ]
                ),
                ParsedModelResponse(
                    final_response=_final_response("Read the workspace note.")
                ),
            ]
        ),
        session_state=ChatSessionState(),
        user_message="Read notes.txt and summarize it.",
    )

    assert outcome.session_state.turns
    assert outcome.transcript_entries[-1].final_response is not None
    assert outcome.transcript_entries[-1].final_response.answer == (
        "Read the workspace note."
    )


def test_enterprise_data_chat_example_respects_network_and_credentials_gate() -> None:
    config = load_streamlit_assistant_config(EXAMPLES_DIR / "enterprise-data-chat.yaml")
    runtime = _MODULES.models.StreamlitRuntimeConfig(
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        api_base_url=config.llm.api_base_url,
        enabled_tools=list(config.policy.enabled_tools or []),
    )
    tool_specs = _MODULES.app._all_tool_specs()
    env = {
        "JIRA_BASE_URL": "https://jira.example.internal",
        "JIRA_USERNAME": "user",
        "JIRA_API_TOKEN": "token",
        "CONFLUENCE_BASE_URL": "https://confluence.example.internal",
        "CONFLUENCE_USERNAME": "user",
        "CONFLUENCE_API_TOKEN": "token",
        "BITBUCKET_BASE_URL": "https://bitbucket.example.internal",
        "BITBUCKET_USERNAME": "user",
        "BITBUCKET_API_TOKEN": "token",
        "GITLAB_BASE_URL": "https://gitlab.example.internal",
        "GITLAB_API_TOKEN": "token",
    }

    blocked = _MODULES.app._exposed_tool_names_for_runtime(
        tool_specs=tool_specs,
        runtime=runtime,
        root=None,
        env=env,
    )
    runtime.allow_network = True
    exposed = _MODULES.app._exposed_tool_names_for_runtime(
        tool_specs=tool_specs,
        runtime=runtime,
        root=None,
        env=env,
    )

    assert "search_jira" not in blocked
    assert {
        "search_jira",
        "read_confluence_page",
        "search_bitbucket_code",
        "search_gitlab_code",
    }.issubset(exposed)

    outcome = _MODULES.app.process_streamlit_assistant_turn(
        root_path=None,
        config=config,
        runtime_config=runtime,
        provider=_SequenceProvider(
            [ParsedModelResponse(final_response=_final_response("Enterprise answer"))]
        ),
        session_state=ChatSessionState(),
        user_message="Search enterprise systems for release notes.",
    )

    assert outcome.transcript_entries[-1].final_response is not None
    assert outcome.transcript_entries[-1].final_response.answer == "Enterprise answer"


@pytest.mark.filterwarnings(
    "ignore:coroutine 'BaseSubprocessTransport.__del__':RuntimeWarning"
)
def test_harness_research_chat_example_can_launch_scripted_research_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_streamlit_assistant_config(
        EXAMPLES_DIR / "harness-research-chat.yaml"
    )
    config = config.model_copy(
        update={
            "research": config.research.model_copy(
                update={"store_dir": str(tmp_path / "research")}
            )
        }
    )
    runtime = _MODULES.models.StreamlitRuntimeConfig(
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        api_base_url=config.llm.api_base_url,
        root_path=str(tmp_path),
        enabled_tools=list(config.policy.enabled_tools or []),
        allow_filesystem=True,
        allow_subprocess=True,
        require_approval_for={SideEffectClass.LOCAL_WRITE},
    )
    provider = ScriptedParsedResponseProvider(
        [ParsedModelResponse(final_response="Research complete")]
    )

    monkeypatch.setattr(_MODULES.app, "_current_api_key", lambda llm_config: None)
    monkeypatch.setattr(
        _MODULES.app,
        "build_live_harness_provider",
        lambda **kwargs: provider,
    )

    controller = _MODULES.app._build_research_controller(
        config=config,
        runtime=runtime,
    )
    inspection = controller.launch(prompt="Summarize the repository state.")

    assert inspection.summary.stop_reason is HarnessStopReason.COMPLETED
    assert inspection.summary.total_turns >= 1
    assert controller.inspect(inspection.snapshot.session_id).summary.stop_reason is (
        HarnessStopReason.COMPLETED
    )

"""Smoke tests for curated LLM Tools Assistant config examples."""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from llm_tools.apps.assistant_app.controller import (
    NiceGUIChatController,
    _exposed_tool_names_for_runtime,
)
from llm_tools.apps.assistant_app.models import NiceGUIRuntimeConfig
from llm_tools.apps.assistant_app.store import SQLiteNiceGUIChatStore
from llm_tools.apps.assistant_config import (
    AssistantConfig,
    AssistantResearchConfig,
    AssistantWorkspaceConfig,
    load_assistant_config,
)
from llm_tools.apps.assistant_tool_registry import (
    build_assistant_available_tool_specs,
)
from llm_tools.apps.chat_config import (
    ChatLLMConfig,
    ChatPolicyConfig,
    ChatUIConfig,
    ProviderProtocol,
)
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.llm_providers import ResponseModeStrategy
from llm_tools.tool_api import ToolInvocationRequest
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import ChatSessionConfig, ProtectionConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples" / "assistant_configs"
EXPECTED_TOP_LEVEL_KEYS = list(AssistantConfig.model_fields)
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


class _SequenceProvider:
    def __init__(self, responses: list[ParsedModelResponse]) -> None:
        self._responses = list(responses)

    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        return self._responses.pop(0)

    def uses_staged_schema_protocol(self) -> bool:
        return False


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


def _controller(
    tmp_path: Path,
    config: AssistantConfig,
    runtime: NiceGUIRuntimeConfig,
    provider: _SequenceProvider,
) -> NiceGUIChatController:
    store = SQLiteNiceGUIChatStore(
        tmp_path / "chat.sqlite3",
        db_key_file=tmp_path / "db.key",
        user_key_file=tmp_path / "user-kek.key",
    )
    store.initialize()
    controller = NiceGUIChatController(
        store=store,
        config=config,
        root_path=Path(runtime.root_path) if runtime.root_path else None,
        provider_factory=lambda _runtime: provider,
    )
    controller.active_record.runtime = runtime
    controller.save_active_session()
    return controller


def _drain(controller: NiceGUIChatController) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        controller.drain_events()
        if not controller.active_turns:
            controller.drain_events()
            return
        time.sleep(0.01)
    raise AssertionError("NiceGUI turn did not finish")


def test_local_only_chat_example_loads_cleanly() -> None:
    path = EXAMPLES_DIR / "local-only-chat.yaml"
    raw = _load_raw_example(path)

    _assert_supported_section_keys(raw)

    config = load_assistant_config(path)

    assert config.llm.provider_protocol is ProviderProtocol.OPENAI_API
    assert config.llm.provider_connection.api_base_url == "http://127.0.0.1:11434/v1"
    assert config.llm.provider_connection.requires_bearer_token is False
    assert config.llm.selected_model == "gemma4:26b"
    assert config.llm.response_mode_strategy is ResponseModeStrategy.AUTO
    assert config.workspace.default_root == "."
    assert config.research.enabled is False
    assert set(config.policy.enabled_tools or []) == LOCAL_ONLY_TOOLS


def test_enterprise_data_chat_example_loads_cleanly() -> None:
    path = EXAMPLES_DIR / "enterprise-data-chat.yaml"
    raw = _load_raw_example(path)

    _assert_supported_section_keys(raw)

    config = load_assistant_config(path)

    assert config.llm.provider_protocol is ProviderProtocol.OPENAI_API
    assert (
        config.llm.provider_connection.api_base_url == "https://llm.example.internal/v1"
    )
    assert config.llm.provider_connection.requires_bearer_token is True
    assert config.llm.selected_model == "assistant-model"
    assert config.llm.response_mode_strategy is ResponseModeStrategy.JSON
    assert config.workspace.default_root is None
    assert config.research.enabled is False
    assert set(config.policy.enabled_tools or []) == ENTERPRISE_DATA_TOOLS


def test_harness_research_chat_example_loads_cleanly() -> None:
    path = EXAMPLES_DIR / "harness-research-chat.yaml"
    raw = _load_raw_example(path)

    _assert_supported_section_keys(raw)

    config = load_assistant_config(path)

    assert config.llm.provider_protocol is ProviderProtocol.OPENAI_API
    assert config.llm.provider_connection.api_base_url == "http://127.0.0.1:11434/v1"
    assert config.llm.provider_connection.requires_bearer_token is False
    assert config.llm.selected_model == "gemma4:26b"
    assert config.workspace.default_root == "."
    assert config.research.enabled is True
    assert config.research.store_dir is None
    assert config.research.max_recent_sessions == 8
    assert config.research.default_max_turns == 12
    assert config.research.default_max_tool_invocations == 48
    assert config.research.default_max_elapsed_seconds == 900
    assert config.research.include_replay_by_default is True
    assert set(config.policy.enabled_tools or []) == HARNESS_RESEARCH_TOOLS


def test_local_only_chat_example_supports_direct_turn_without_workspace(
    tmp_path: Path,
) -> None:
    config = load_assistant_config(EXAMPLES_DIR / "local-only-chat.yaml")
    runtime = NiceGUIRuntimeConfig(
        provider_protocol=config.llm.provider_protocol,
        provider_connection=config.llm.provider_connection.model_copy(deep=True),
        selected_model=config.llm.selected_model,
    )
    controller = _controller(
        tmp_path,
        config,
        runtime,
        _SequenceProvider(
            [ParsedModelResponse(final_response=_final_response("Direct answer"))]
        ),
    )

    assert controller.submit_prompt("What can this assistant do without tools?") is None
    _drain(controller)

    assert controller.active_record.workflow_session_state.turns
    assert controller.active_record.transcript[-1].final_response is not None
    assert controller.active_record.transcript[-1].final_response.answer == (
        "Direct answer"
    )


def test_local_only_chat_example_supports_workspace_read_turn(
    tmp_path: Path,
) -> None:
    config = load_assistant_config(EXAMPLES_DIR / "local-only-chat.yaml")
    (tmp_path / "notes.txt").write_text("Workspace facts for the assistant.", "utf-8")
    runtime = NiceGUIRuntimeConfig(
        provider_protocol=config.llm.provider_protocol,
        provider_connection=config.llm.provider_connection.model_copy(deep=True),
        selected_model=config.llm.selected_model,
        root_path=str(tmp_path),
        enabled_tools=list(config.policy.enabled_tools or []),
        allow_filesystem=True,
    )
    controller = _controller(
        tmp_path,
        config,
        runtime,
        _SequenceProvider(
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
    )

    assert controller.submit_prompt("Read notes.txt and summarize it.") is None
    _drain(controller)

    assert controller.active_record.workflow_session_state.turns
    assert controller.active_record.transcript[-1].final_response is not None
    assert controller.active_record.transcript[-1].final_response.answer == (
        "Read the workspace note."
    )


def test_enterprise_data_chat_example_respects_network_and_credentials_gate() -> None:
    config = load_assistant_config(EXAMPLES_DIR / "enterprise-data-chat.yaml")
    runtime = NiceGUIRuntimeConfig(
        provider_protocol=config.llm.provider_protocol,
        provider_connection=config.llm.provider_connection.model_copy(deep=True),
        selected_model=config.llm.selected_model,
        enabled_tools=list(config.policy.enabled_tools or []),
        allow_network=False,
    )
    tool_specs = build_assistant_available_tool_specs()
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

    blocked = _exposed_tool_names_for_runtime(
        tool_specs=tool_specs,
        runtime=runtime,
        root=None,
        env=env,
    )
    runtime.allow_network = True
    exposed = _exposed_tool_names_for_runtime(
        tool_specs=tool_specs,
        runtime=runtime,
        root=None,
        env=env,
    )

    assert "search_jira" not in blocked
    assert {"search_jira", "read_gitlab_file", "read_confluence_page"}.issubset(exposed)

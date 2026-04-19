"""Smoke tests for curated Streamlit assistant config examples."""

from __future__ import annotations

from pathlib import Path

import yaml

from llm_tools.apps.assistant_config import (
    AssistantResearchConfig,
    AssistantWorkspaceConfig,
    StreamlitAssistantConfig,
    load_streamlit_assistant_config,
)
from llm_tools.apps.chat_config import ChatLLMConfig, ChatPolicyConfig, ChatUIConfig
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import ChatSessionConfig
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
    "read_confluence_content",
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


def test_local_only_chat_example_loads_cleanly() -> None:
    path = EXAMPLES_DIR / "local-only-chat.yaml"
    raw = _load_raw_example(path)

    _assert_supported_section_keys(raw)

    config = load_streamlit_assistant_config(path)

    assert config.llm.provider.value == "ollama"
    assert config.workspace.default_root == "."
    assert config.research.enabled is False
    assert set(config.policy.enabled_tools or []) == LOCAL_ONLY_TOOLS


def test_enterprise_data_chat_example_loads_cleanly() -> None:
    path = EXAMPLES_DIR / "enterprise-data-chat.yaml"
    raw = _load_raw_example(path)

    _assert_supported_section_keys(raw)

    config = load_streamlit_assistant_config(path)

    assert config.llm.provider.value == "custom_openai_compatible"
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
    assert config.research.store_dir == ".llm-tools/research"
    assert config.research.max_recent_sessions == 8
    assert config.research.default_max_turns == 12
    assert config.research.default_max_tool_invocations == 48
    assert config.research.default_max_elapsed_seconds == 900
    assert config.research.include_replay_by_default is True
    assert set(config.policy.enabled_tools or []) == HARNESS_RESEARCH_TOOLS

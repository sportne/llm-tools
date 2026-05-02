"""Focused tests for assistant runtime capability reporting."""

from __future__ import annotations

from types import SimpleNamespace

from llm_tools.apps.assistant_config import AssistantConfig
from llm_tools.apps.assistant_runtime import (
    AssistantRuntimeBundle,
    AssistantToolCapability,
    AssistantToolCapabilityReasonCode,
    NiceGUIHarnessContextBuilder,
    build_assistant_available_tool_specs,
    build_assistant_runtime_bundle,
    build_tool_capabilities,
    build_tool_group_capability_summaries,
)
from llm_tools.harness_api import BudgetPolicy, create_root_task
from llm_tools.tool_api import SideEffectClass, ToolSpec
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import ChatSessionConfig, ChatSessionState, ProtectionConfig


class _RuntimeProvider:
    def uses_staged_schema_protocol(self) -> bool:
        return False

    def run(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("provider should not run during runtime assembly")

    def run_structured(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("provider should not run during runtime assembly")

    def run_text(self, **kwargs: object) -> str:
        del kwargs
        raise AssertionError("provider should not run during runtime assembly")


class _PromptToolRuntimeProvider(_RuntimeProvider):
    def uses_prompt_tool_protocol(self) -> bool:
        return True


def _flatten_capabilities(capability_groups):
    return {
        item.tool_name: item for items in capability_groups.values() for item in items
    }


def _runtime(**overrides: object) -> SimpleNamespace:
    config = AssistantConfig()
    values: dict[str, object] = {
        "provider": config.llm.provider,
        "provider_mode_strategy": config.llm.provider_mode_strategy,
        "model_name": config.llm.model_name,
        "api_base_url": config.llm.api_base_url,
        "api_key_env_var": config.llm.api_key_env_var,
        "temperature": config.llm.temperature,
        "timeout_seconds": config.llm.timeout_seconds,
        "root_path": None,
        "default_workspace_root": config.workspace.default_root,
        "enabled_tools": [],
        "tool_urls": {},
        "require_approval_for": set(config.policy.require_approval_for),
        "allow_network": True,
        "allow_filesystem": True,
        "allow_subprocess": True,
        "inspector_open": config.ui.inspector_open_by_default,
        "show_token_usage": config.ui.show_token_usage,
        "show_footer_help": config.ui.show_footer_help,
        "session_config": ChatSessionConfig(),
        "tool_limits": ToolLimits(),
        "research": config.research.model_copy(deep=True),
        "protection": ProtectionConfig(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _admin(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "write_file_tool_enabled": False,
        "atlassian_tools_enabled": False,
        "gitlab_tools_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _bundle(
    *,
    runtime: SimpleNamespace,
    admin_settings: SimpleNamespace | None = None,
    provider: _RuntimeProvider | None = None,
    env_overrides: dict[str, str] | None = None,
    information_protection_enabled: bool = False,
    chat_has_pending_protection_prompt: bool = False,
) -> AssistantRuntimeBundle:
    effective_provider = provider or _RuntimeProvider()
    return build_assistant_runtime_bundle(
        config=AssistantConfig(),
        runtime=runtime,
        admin_settings=admin_settings or _admin(),
        session_id="session-1",
        provider_factory=lambda _runtime: effective_provider,  # type: ignore[return-value]
        env_overrides=env_overrides,
        information_protection_enabled=information_protection_enabled,
        chat_has_pending_protection_prompt=chat_has_pending_protection_prompt,
    )


def test_assistant_runtime_bundle_reflects_runtime_advanced_settings() -> None:
    runtime = _runtime(
        temperature=0.55,
        timeout_seconds=33.5,
        show_token_usage=False,
        show_footer_help=False,
        inspector_open=True,
        session_config=ChatSessionConfig(
            max_context_tokens=1000,
            max_tool_round_trips=2,
            max_tool_calls_per_round=1,
            max_total_tool_calls_per_turn=3,
        ),
    )

    bundle = _bundle(runtime=runtime)

    assert bundle.effective_config.llm.temperature == 0.55
    assert bundle.effective_config.llm.timeout_seconds == 33.5
    assert bundle.effective_config.ui.show_token_usage is False
    assert bundle.effective_config.ui.show_footer_help is False
    assert bundle.effective_config.ui.inspector_open_by_default is True
    assert bundle.effective_config.session.max_context_tokens == 1000
    assert bundle.effective_config.session.max_tool_round_trips == 2
    assert bundle.effective_config.session.max_tool_calls_per_round == 1
    assert bundle.effective_config.session.max_total_tool_calls_per_turn == 3


def test_assistant_runtime_bundle_applies_admin_feature_flags() -> None:
    runtime = _runtime(enabled_tools=["read_file", "write_file"])

    hidden = _bundle(runtime=runtime, admin_settings=_admin())
    visible = _bundle(
        runtime=runtime,
        admin_settings=_admin(write_file_tool_enabled=True),
    )

    assert "write_file" not in hidden.tool_specs
    assert "write_file" not in hidden.enabled_tool_names
    assert "read_file" in hidden.tool_specs
    assert "write_file" in visible.tool_specs
    assert "write_file" in visible.enabled_tool_names


def test_assistant_runtime_bundle_calculates_exposed_tools_from_capabilities() -> None:
    tool_specs = build_assistant_available_tool_specs()
    gitlab_secrets = {
        name: "present" for name in tool_specs["read_gitlab_file"].required_secrets
    }
    runtime = _runtime(
        enabled_tools=["read_gitlab_file"],
        allow_network=False,
    )
    admin_settings = _admin(gitlab_tools_enabled=True)

    blocked = _bundle(
        runtime=runtime,
        admin_settings=admin_settings,
        env_overrides=gitlab_secrets,
    )
    runtime.allow_network = True
    exposed = _bundle(
        runtime=runtime,
        admin_settings=admin_settings,
        env_overrides=gitlab_secrets,
    )

    assert "read_gitlab_file" in blocked.enabled_tool_names
    assert "read_gitlab_file" not in blocked.exposed_tool_names
    assert "read_gitlab_file" in exposed.exposed_tool_names


def test_assistant_runtime_bundle_propagates_session_env_overrides(
    tmp_path,
) -> None:
    runtime = _runtime(root_path=str(tmp_path), enabled_tools=["read_file"])
    bundle = _bundle(
        runtime=runtime,
        env_overrides={"JIRA_BASE_URL": "https://jira.example.test"},
    )

    runner = bundle.build_chat_runner(
        session_state=ChatSessionState(),
        user_message="Inspect the workspace.",
    )
    state = create_root_task(
        schema_version="3",
        session_id="harness-1",
        root_task_id="task-1",
        title="Test",
        intent="Test",
        budget_policy=BudgetPolicy(max_turns=1),
        started_at="2026-01-01T00:00:00Z",
    )
    harness_context = NiceGUIHarnessContextBuilder(
        env_overrides=bundle.env_overrides
    ).build(state=state, selected_task_ids=["task-1"], turn_index=1)

    assert runner._base_context.env == {"JIRA_BASE_URL": "https://jira.example.test"}
    assert harness_context.tool_context.env == {
        "JIRA_BASE_URL": "https://jira.example.test"
    }


def test_assistant_runtime_bundle_builds_mode_prompts(tmp_path) -> None:
    runtime = _runtime(root_path=str(tmp_path), enabled_tools=["read_file"])

    bundle = _bundle(
        runtime=runtime,
        provider=_PromptToolRuntimeProvider(),
    )

    assert "Prompt-tool protocol" in bundle.chat_system_prompt
    assert "durable research assistant" in bundle.deep_task_system_prompt
    assert "read_file" in bundle.chat_system_prompt
    assert "read_file" in bundle.deep_task_system_prompt


def test_assistant_runtime_bundle_wires_protection_when_ready(
    monkeypatch,
    tmp_path,
) -> None:
    document = tmp_path / "policy.md"
    document.write_text("TRIVIAL information may be discussed.", encoding="utf-8")
    controllers: list[dict[str, object]] = []

    def _fake_controller(**kwargs: object) -> dict[str, object]:
        controllers.append(dict(kwargs))
        return {"controller": len(controllers)}

    monkeypatch.setattr(
        "llm_tools.apps.assistant_runtime.build_protection_controller",
        _fake_controller,
    )
    runtime = _runtime(
        protection=ProtectionConfig(
            enabled=True,
            document_paths=[str(tmp_path)],
            allowed_sensitivity_labels=["TRIVIAL"],
        )
    )

    bundle = _bundle(
        runtime=runtime,
        information_protection_enabled=True,
    )

    assert bundle.chat_protection_controller == {"controller": 1}
    assert bundle.deep_task_protection_controller == {"controller": 2}
    assert controllers[0]["config"] is runtime.protection
    assert controllers[1]["config"] is runtime.protection


def test_assistant_runtime_bundle_pending_chat_protection_forces_chat_only(
    monkeypatch,
) -> None:
    controllers: list[dict[str, object]] = []

    def _fake_controller(**kwargs: object) -> dict[str, object]:
        controllers.append(dict(kwargs))
        return {"controller": len(controllers)}

    monkeypatch.setattr(
        "llm_tools.apps.assistant_runtime.build_protection_controller",
        _fake_controller,
    )

    bundle = _bundle(
        runtime=_runtime(protection=ProtectionConfig(enabled=False)),
        information_protection_enabled=True,
        chat_has_pending_protection_prompt=True,
    )

    assert bundle.chat_protection_controller == {"controller": 1}
    assert bundle.deep_task_protection_controller is None
    assert controllers[0]["config"].enabled is True


def test_build_tool_capabilities_disabled_tools_preserve_legacy_defaults() -> None:
    tool_specs = build_assistant_available_tool_specs()

    capabilities = build_tool_capabilities(
        tool_specs=tool_specs,
        enabled_tools=set(),
        root_path=None,
        env={},
        allow_network=False,
        allow_filesystem=False,
        allow_subprocess=False,
        require_approval_for={SideEffectClass.LOCAL_WRITE},
    )

    read_file = _flatten_capabilities(capabilities)["read_file"]
    assert read_file.status == "disabled"
    assert read_file.detail is None
    assert read_file.primary_reason is None
    assert read_file.reasons == []
    assert read_file.approval_required is False
    assert read_file.approval_gate.required is False
    assert read_file.exposed_to_model is False


def test_assistant_tool_capability_backfills_approval_gate_for_legacy_payload() -> None:
    capability = AssistantToolCapability.model_validate(
        {
            "tool_name": "write_file",
            "group": "Local Files",
            "status": "available",
            "approval_required": True,
            "side_effects": SideEffectClass.LOCAL_WRITE,
        }
    )

    assert capability.approval_required is True
    assert capability.approval_gate.required is True
    assert capability.approval_gate.side_effects is SideEffectClass.LOCAL_WRITE
    assert (
        capability.approval_gate.reason_code
        is AssistantToolCapabilityReasonCode.APPROVAL_REQUIRED
    )


def test_build_tool_capabilities_preserve_blocker_order_and_structured_reasons() -> (
    None
):
    tool_specs = build_assistant_available_tool_specs()
    gitlab_spec = tool_specs["search_gitlab_code"]

    capabilities = build_tool_capabilities(
        tool_specs=tool_specs,
        enabled_tools={"search_gitlab_code"},
        root_path=".",
        env={},
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=True,
        require_approval_for=set(),
    )

    gitlab = _flatten_capabilities(capabilities)["search_gitlab_code"]
    missing = sorted(gitlab_spec.required_secrets)

    assert gitlab.status == "missing_credentials"
    assert gitlab.primary_reason is not None
    assert (
        gitlab.primary_reason.code
        is AssistantToolCapabilityReasonCode.MISSING_CREDENTIALS
    )
    assert [reason.code for reason in gitlab.reasons] == [
        AssistantToolCapabilityReasonCode.MISSING_CREDENTIALS,
        AssistantToolCapabilityReasonCode.NETWORK_PERMISSION_BLOCKED,
    ]
    assert gitlab.reasons[0].missing_secrets == missing
    assert gitlab.reasons[1].blocked_capability == "network"
    assert gitlab.detail == (
        f"Missing credentials: {', '.join(missing)} "
        "Current session permissions do not allow this tool."
    )


def test_build_tool_capabilities_permission_blocked_can_be_primary_status() -> None:
    tool_specs = build_assistant_available_tool_specs()
    gitlab_spec = tool_specs["search_gitlab_code"]
    env = {secret: "present" for secret in gitlab_spec.required_secrets}

    capabilities = build_tool_capabilities(
        tool_specs=tool_specs,
        enabled_tools={"search_gitlab_code"},
        root_path=".",
        env=env,
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=True,
        require_approval_for=set(),
    )

    gitlab = _flatten_capabilities(capabilities)["search_gitlab_code"]
    assert gitlab.status == "permission_blocked"
    assert gitlab.primary_reason is not None
    assert (
        gitlab.primary_reason.code
        is AssistantToolCapabilityReasonCode.NETWORK_PERMISSION_BLOCKED
    )
    assert gitlab.detail == "Current session permissions do not allow this tool."
    assert gitlab.exposed_to_model is False


def test_build_tool_capabilities_approval_gate_is_additive_to_availability() -> None:
    tool_specs = build_assistant_available_tool_specs()

    capabilities = build_tool_capabilities(
        tool_specs=tool_specs,
        enabled_tools={"write_file"},
        root_path=".",
        env={},
        allow_network=True,
        allow_filesystem=True,
        allow_subprocess=True,
        require_approval_for={SideEffectClass.LOCAL_WRITE},
    )

    write_file = _flatten_capabilities(capabilities)["write_file"]
    assert write_file.status == "available"
    assert write_file.exposed_to_model is True
    assert write_file.reasons == []
    assert write_file.primary_reason is None
    assert write_file.approval_required is True
    assert write_file.approval_gate.required is True
    assert write_file.approval_gate.side_effects is SideEffectClass.LOCAL_WRITE
    assert (
        write_file.approval_gate.reason_code
        is AssistantToolCapabilityReasonCode.APPROVAL_REQUIRED
    )


def test_build_tool_group_capability_summaries_count_statuses_and_approval() -> None:
    tool_specs = {
        "read_local": ToolSpec(
            name="read_local",
            description="Read from disk",
            tags=["filesystem"],
            side_effects=SideEffectClass.LOCAL_READ,
            requires_filesystem=True,
        ),
        "write_local": ToolSpec(
            name="write_local",
            description="Write to disk",
            tags=["filesystem"],
            side_effects=SideEffectClass.LOCAL_WRITE,
        ),
        "plain_text": ToolSpec(
            name="plain_text",
            description="Text helper",
            tags=["filesystem", "text"],
            side_effects=SideEffectClass.NONE,
        ),
    }

    capabilities = build_tool_capabilities(
        tool_specs=tool_specs,
        enabled_tools={"read_local", "write_local"},
        root_path=None,
        env={},
        allow_network=True,
        allow_filesystem=True,
        allow_subprocess=True,
        require_approval_for={SideEffectClass.LOCAL_WRITE},
    )
    summaries = build_tool_group_capability_summaries(capabilities)

    local_files = summaries["Local Files"]
    assert local_files.total_tools == 3
    assert local_files.enabled_tools == 2
    assert local_files.exposed_tools == 1
    assert local_files.available_tools == 1
    assert local_files.disabled_tools == 1
    assert local_files.missing_workspace_tools == 1
    assert local_files.missing_credentials_tools == 0
    assert local_files.permission_blocked_tools == 0
    assert local_files.approval_gated_tools == 1

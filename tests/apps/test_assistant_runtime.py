"""Focused tests for assistant runtime capability reporting."""

from __future__ import annotations

from llm_tools.apps.assistant_runtime import (
    AssistantToolCapability,
    AssistantToolCapabilityReasonCode,
    build_assistant_available_tool_specs,
    build_tool_capabilities,
    build_tool_group_capability_summaries,
)
from llm_tools.tool_api import SideEffectClass, ToolSpec


def _flatten_capabilities(capability_groups):
    return {
        item.tool_name: item for items in capability_groups.values() for item in items
    }


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
            tags=["text"],
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
    assert local_files.total_tools == 2
    assert local_files.enabled_tools == 2
    assert local_files.exposed_tools == 1
    assert local_files.available_tools == 1
    assert local_files.disabled_tools == 0
    assert local_files.missing_workspace_tools == 1
    assert local_files.missing_credentials_tools == 0
    assert local_files.permission_blocked_tools == 0
    assert local_files.approval_gated_tools == 1

    text = summaries["Text"]
    assert text.total_tools == 1
    assert text.enabled_tools == 0
    assert text.disabled_tools == 1
    assert text.approval_gated_tools == 0

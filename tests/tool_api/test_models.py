"""Tests for canonical tool API models and enums."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from llm_tools.tool_api import (
    ErrorCode,
    ExecutionRecord,
    PolicyDecision,
    PolicyVerdict,
    RiskLevel,
    SideEffectClass,
    ToolContext,
    ToolError,
    ToolInvocationRequest,
    ToolResult,
    ToolSpec,
)


def test_tool_api_re_exports_enums() -> None:
    assert SideEffectClass.NONE == "none"
    assert RiskLevel.LOW == "low"
    assert ErrorCode.TOOL_NOT_FOUND == "tool_not_found"
    assert PolicyVerdict.ALLOW == "allow"


def test_side_effect_class_values_are_stable() -> None:
    assert [member.value for member in SideEffectClass] == [
        "none",
        "local_read",
        "local_write",
        "external_read",
        "external_write",
    ]


def test_risk_level_values_are_stable() -> None:
    assert [member.value for member in RiskLevel] == [
        "low",
        "medium",
        "high",
    ]


def test_error_code_values_are_stable() -> None:
    assert [member.value for member in ErrorCode] == [
        "tool_not_found",
        "input_validation_error",
        "output_validation_error",
        "policy_denied",
        "timeout",
        "dependency_missing",
        "execution_failed",
        "runtime_error",
    ]


def test_policy_verdict_values_are_stable() -> None:
    assert [member.value for member in PolicyVerdict] == [
        "allow",
        "deny",
        "require_approval",
    ]


def test_enum_members_behave_like_strings() -> None:
    assert SideEffectClass.LOCAL_READ.startswith("local_")
    assert ErrorCode.RUNTIME_ERROR.endswith("_error")
    assert RiskLevel.HIGH.upper() == "HIGH"


def test_enum_values_serialize_cleanly() -> None:
    payload = {
        "side_effect": SideEffectClass.EXTERNAL_WRITE.value,
        "risk": RiskLevel.MEDIUM.value,
        "error": ErrorCode.EXECUTION_FAILED.value,
        "verdict": PolicyVerdict.REQUIRE_APPROVAL.value,
    }

    assert json.loads(json.dumps(payload)) == {
        "side_effect": "external_write",
        "risk": "medium",
        "error": "execution_failed",
        "verdict": "require_approval",
    }


def test_tool_spec_defaults_match_the_canonical_contract() -> None:
    spec = ToolSpec(name="read_file", description="Read a file from disk.")

    assert spec.version == "0.1.0"
    assert spec.tags == []
    assert spec.side_effects is SideEffectClass.NONE
    assert spec.idempotent is True
    assert spec.deterministic is True
    assert spec.timeout_seconds is None
    assert spec.risk_level is RiskLevel.LOW
    assert spec.requires_network is False
    assert spec.requires_filesystem is False
    assert spec.requires_subprocess is False
    assert spec.required_secrets == []
    assert spec.cost_hint is None
    assert spec.retain_output_in_execution_record is True


def test_tool_context_defaults_include_observability_sinks() -> None:
    context = ToolContext(invocation_id="inv-123")

    assert context.logs == []
    assert context.artifacts == []
    assert context.metadata == {}


@pytest.mark.parametrize(
    ("model_type", "payload", "field_name"),
    [
        (ToolSpec, {"name": "", "description": "desc"}, "name"),
        (ToolContext, {"invocation_id": ""}, "invocation_id"),
        (ToolInvocationRequest, {"tool_name": ""}, "tool_name"),
        (
            ToolError,
            {"code": ErrorCode.RUNTIME_ERROR.value, "message": "boom", "details": []},
            "details",
        ),
    ],
)
def test_models_reject_invalid_payloads(
    model_type: type[Any], payload: dict[str, Any], field_name: str
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        model_type.model_validate(payload)

    assert field_name in str(exc_info.value)


def test_tool_error_and_tool_result_round_trip_nested_data() -> None:
    error = ToolError(
        code=ErrorCode.EXECUTION_FAILED,
        message="Subprocess failed.",
        retryable=True,
        details={"exit_code": 1},
    )
    result = ToolResult(
        ok=False,
        tool_name="run_process",
        tool_version="0.1.0",
        error=error,
        logs=["starting"],
        artifacts=["stderr.txt"],
        metadata={"attempt": 1},
    )

    validated = ToolResult.model_validate_json(result.model_dump_json())

    assert validated == result
    assert validated.error is not None
    assert validated.error.code is ErrorCode.EXECUTION_FAILED


def test_tool_spec_serializes_enums_as_json_values() -> None:
    spec = ToolSpec(
        name="fetch_url",
        description="Fetch a remote URL.",
        side_effects=SideEffectClass.EXTERNAL_READ,
        risk_level=RiskLevel.MEDIUM,
        requires_network=True,
        tags=["http"],
    )

    assert spec.model_dump(mode="json") == {
        "name": "fetch_url",
        "version": "0.1.0",
        "description": "Fetch a remote URL.",
        "tags": ["http"],
        "side_effects": "external_read",
        "idempotent": True,
        "deterministic": True,
        "timeout_seconds": None,
        "risk_level": "medium",
        "requires_network": True,
        "requires_filesystem": False,
        "requires_subprocess": False,
        "writes_internal_workspace_cache": False,
        "required_secrets": [],
        "cost_hint": None,
        "retain_output_in_execution_record": True,
    }


def test_execution_record_serializes_nested_models_and_error_codes() -> None:
    record = ExecutionRecord(
        invocation_id="inv-123",
        tool_name="fetch_url",
        tool_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
        duration_ms=1000,
        request=ToolInvocationRequest(
            tool_name="fetch_url",
            arguments={"url": "https://example.com"},
        ),
        validated_input={"url": "https://example.com"},
        ok=False,
        error_code=ErrorCode.TIMEOUT,
        policy_decision=PolicyDecision(
            allowed=True,
            reason="network allowed",
        ),
        logs=["starting"],
        artifacts=["trace.json"],
        metadata={"provider": "test"},
    )

    assert record.model_dump(mode="json") == {
        "invocation_id": "inv-123",
        "tool_name": "fetch_url",
        "tool_version": "0.1.0",
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:00:01Z",
        "duration_ms": 1000,
        "request": {
            "tool_name": "fetch_url",
            "arguments": {"url": "https://example.com"},
        },
        "validated_input": {"url": "https://example.com"},
        "redacted_input": None,
        "validated_output": None,
        "redacted_output": None,
        "ok": False,
        "error_code": "timeout",
        "policy_decision": {
            "allowed": True,
            "reason": "network allowed",
            "requires_approval": False,
            "metadata": {},
        },
        "logs": ["starting"],
        "artifacts": ["trace.json"],
        "metadata": {"provider": "test"},
        "source_provenance": [],
    }


def test_tool_spec_schema_snapshot_is_stable() -> None:
    schema = ToolSpec.model_json_schema()

    assert schema["required"] == ["name", "description"]
    assert schema["properties"]["side_effects"] == {
        "$ref": "#/$defs/SideEffectClass",
        "default": "none",
    }
    assert schema["properties"]["risk_level"] == {
        "$ref": "#/$defs/RiskLevel",
        "default": "low",
    }
    assert schema["$defs"]["SideEffectClass"]["enum"] == [
        "none",
        "local_read",
        "local_write",
        "external_read",
        "external_write",
    ]
    assert schema["$defs"]["RiskLevel"]["enum"] == [
        "low",
        "medium",
        "high",
    ]


def test_execution_record_schema_snapshot_captures_nested_contracts() -> None:
    schema = ExecutionRecord.model_json_schema()

    assert schema["required"] == [
        "invocation_id",
        "tool_name",
        "tool_version",
        "started_at",
        "request",
    ]
    assert schema["properties"]["request"] == {"$ref": "#/$defs/ToolInvocationRequest"}
    assert schema["properties"]["error_code"]["anyOf"] == [
        {"$ref": "#/$defs/ErrorCode"},
        {"type": "null"},
    ]
    assert schema["properties"]["policy_decision"]["anyOf"] == [
        {"$ref": "#/$defs/PolicyDecision"},
        {"type": "null"},
    ]
    assert schema["$defs"]["ErrorCode"]["enum"] == [
        "tool_not_found",
        "input_validation_error",
        "output_validation_error",
        "policy_denied",
        "timeout",
        "dependency_missing",
        "execution_failed",
        "runtime_error",
    ]
    assert schema["$defs"]["ToolInvocationRequest"]["required"] == ["tool_name"]

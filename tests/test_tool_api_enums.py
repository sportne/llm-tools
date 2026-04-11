"""Tests for canonical tool API enums."""

from __future__ import annotations

import json

from llm_tools.tool_api import ErrorCode, PolicyVerdict, RiskLevel, SideEffectClass


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

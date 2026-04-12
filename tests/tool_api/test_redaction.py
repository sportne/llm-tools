"""Tests for the redaction framework."""

from __future__ import annotations

from llm_tools.tool_api.redaction import (
    RedactionConfig,
    RedactionRule,
    RedactionTarget,
    Redactor,
)


def test_field_name_rules_redact_case_and_dash_variants() -> None:
    redactor = Redactor(
        RedactionConfig(
            rules=[
                RedactionRule(
                    field_names={"api_key"},
                    targets={RedactionTarget.INPUT},
                )
            ]
        ),
        tool_name="demo",
    )

    redacted = redactor.redact_structured(
        {
            "API-KEY": "abc",
            "nested": {"api_key": "def"},
        },
        target=RedactionTarget.INPUT,
    )

    assert redacted == {
        "API-KEY": "[REDACTED]",
        "nested": {"api_key": "[REDACTED]"},
    }


def test_path_rules_support_wildcard_list_segments() -> None:
    redactor = Redactor(
        RedactionConfig(
            rules=[
                RedactionRule(
                    paths={"items[*].secret"},
                    targets={RedactionTarget.INPUT},
                )
            ]
        ),
        tool_name="demo",
    )

    redacted = redactor.redact_structured(
        {
            "secret": "root",
            "items": [
                {"secret": "a", "keep": "x"},
                {"secret": "b", "keep": "y"},
            ],
        },
        target=RedactionTarget.INPUT,
    )

    assert redacted == {
        "secret": "root",
        "items": [
            {"secret": "[REDACTED]", "keep": "x"},
            {"secret": "[REDACTED]", "keep": "y"},
        ],
    }


def test_tool_specific_path_rules_precede_global_field_rules() -> None:
    redactor = Redactor(
        RedactionConfig(
            rules=[
                RedactionRule(
                    field_names={"secret"},
                    replacement="[GLOBAL]",
                )
            ],
            tool_rules={
                "demo_tool": [
                    RedactionRule(
                        paths={"payload.secret"},
                        targets={RedactionTarget.INPUT},
                        replacement="[TOOL_PATH]",
                    )
                ]
            },
        ),
        tool_name="demo_tool",
    )

    redacted = redactor.redact_structured(
        {
            "payload": {"secret": "x"},
            "secret": "y",
        },
        target=RedactionTarget.INPUT,
    )

    assert redacted == {
        "payload": {"secret": "[TOOL_PATH]"},
        "secret": "[GLOBAL]",
    }


def test_first_match_wins_for_replacement_selection() -> None:
    redactor = Redactor(
        RedactionConfig(
            rules=[
                RedactionRule(
                    field_names={"token"},
                    replacement="[FIRST]",
                ),
                RedactionRule(
                    field_names={"token"},
                    replacement="[SECOND]",
                ),
            ]
        ),
        tool_name="demo",
    )

    redacted = redactor.redact_structured(
        {"token": "abc"},
        target=RedactionTarget.OUTPUT,
    )
    assert redacted == {"token": "[FIRST]"}


def test_surface_level_redaction_for_logs_and_artifacts() -> None:
    redactor = Redactor(
        RedactionConfig(
            rules=[
                RedactionRule(
                    targets={RedactionTarget.LOGS},
                    replacement="[LOG]",
                ),
                RedactionRule(
                    targets={RedactionTarget.ARTIFACTS},
                    replacement="[ART]",
                ),
            ],
            redact_logs=True,
            redact_artifacts=True,
        ),
        tool_name="demo",
    )

    redacted_logs = redactor.redact_string_entries(
        ["line-1", "line-2"],
        target=RedactionTarget.LOGS,
    )
    redacted_artifacts = redactor.redact_string_entries(
        ["a.txt"],
        target=RedactionTarget.ARTIFACTS,
    )

    assert redacted_logs == ["[LOG]", "[LOG]"]
    assert redacted_artifacts == ["[ART]"]

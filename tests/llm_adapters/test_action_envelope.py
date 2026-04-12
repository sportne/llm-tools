"""Tests for the canonical action-envelope adapter."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field, model_validator

from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.tool_api import ToolInvocationRequest, ToolSpec


class EchoInput(BaseModel):
    value: str


class ReadInput(BaseModel):
    path: str
    encoding: str = "utf-8"


def _specs() -> list[ToolSpec]:
    return [
        ToolSpec(name="echo", description="Echo"),
        ToolSpec(name="read_file", description="Read"),
    ]


def _input_models() -> dict[str, type[BaseModel]]:
    return {
        "echo": EchoInput,
        "read_file": ReadInput,
    }


def test_build_response_model_constrains_tool_names() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(_specs(), _input_models())
    schema = adapter.export_schema(response_model)

    action_tool_name = schema["properties"]["actions"]["items"]["discriminator"][
        "propertyName"
    ]
    assert action_tool_name == "tool_name"


def test_parse_model_output_with_typed_actions() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(_specs(), _input_models())

    parsed = adapter.parse_model_output(
        {
            "actions": [
                {"tool_name": "echo", "arguments": {"value": "hi"}},
                {"tool_name": "read_file", "arguments": {"path": "README.md"}},
            ],
            "final_response": None,
        },
        response_model=response_model,
    )

    assert parsed.final_response is None
    assert parsed.invocations == [
        ToolInvocationRequest(tool_name="echo", arguments={"value": "hi"}),
        ToolInvocationRequest(
            tool_name="read_file",
            arguments={"path": "README.md", "encoding": "utf-8"},
        ),
    ]


def test_parse_model_output_with_typed_final_response() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(_specs(), _input_models())

    parsed = adapter.parse_model_output(
        {"actions": [], "final_response": "done"},
        response_model=response_model,
    )

    assert parsed.invocations == []
    assert parsed.final_response == "done"


def test_parse_model_output_accepts_json_strings() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(_specs(), _input_models())

    parsed = adapter.parse_model_output(
        json.dumps(
            {
                "actions": [{"tool_name": "echo", "arguments": {"value": "hello"}}],
                "final_response": None,
            }
        ),
        response_model=response_model,
    )

    assert parsed.invocations[0].tool_name == "echo"
    assert parsed.invocations[0].arguments == {"value": "hello"}


def test_parse_model_output_rejects_invalid_envelope_mode() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(_specs(), _input_models())

    with pytest.raises(ValueError):
        adapter.parse_model_output(
            {
                "actions": [{"tool_name": "echo", "arguments": {"value": "x"}}],
                "final_response": "bad",
            },
            response_model=response_model,
        )
    with pytest.raises(ValueError):
        adapter.parse_model_output(
            {"actions": [], "final_response": None},
            response_model=response_model,
        )


def test_parse_model_output_rejects_invalid_tool_arguments() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(_specs(), _input_models())

    with pytest.raises(ValueError):
        adapter.parse_model_output(
            {
                "actions": [
                    {
                        "tool_name": "echo",
                        "arguments": {"value": 123},
                    }
                ],
                "final_response": None,
            },
            response_model=response_model,
        )


def test_no_tools_response_model_rejects_actions() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model([], {})

    with pytest.raises(ValueError):
        adapter.parse_model_output(
            {
                "actions": [{"tool_name": "echo", "arguments": {}}],
                "final_response": None,
            },
            response_model=response_model,
        )

    parsed = adapter.parse_model_output(
        {"actions": [], "final_response": "no tools"},
        response_model=response_model,
    )
    assert parsed.final_response == "no tools"


def test_parse_model_output_uses_loose_fallback_without_response_model() -> None:
    adapter = ActionEnvelopeAdapter()

    parsed = adapter.parse_model_output(
        {
            "actions": [{"tool_name": "anything", "arguments": {"x": 1}}],
            "final_response": None,
        }
    )

    assert parsed.invocations == [
        ToolInvocationRequest(tool_name="anything", arguments={"x": 1})
    ]


def test_export_tool_descriptions_builds_schema() -> None:
    adapter = ActionEnvelopeAdapter()

    schema = adapter.export_tool_descriptions(_specs(), _input_models())

    assert isinstance(schema, dict)
    assert "ActionEnvelope" in schema["title"]


def test_parse_model_output_rejects_empty_final_response() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(_specs(), _input_models())

    with pytest.raises(ValueError):
        adapter.parse_model_output(
            {"actions": [], "final_response": "   "},
            response_model=response_model,
        )


class _LooseAnythingEnvelope(BaseModel):
    actions: list[object] = Field(default_factory=list)
    final_response: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> _LooseAnythingEnvelope:
        has_actions = len(self.actions) > 0
        has_final_response = self.final_response is not None
        if has_actions == has_final_response:
            raise ValueError("Must contain actions or final response.")
        return self


class _LooseDictEnvelope(BaseModel):
    actions: list[dict[str, Any]] = Field(default_factory=list)
    final_response: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> _LooseDictEnvelope:
        has_actions = len(self.actions) > 0
        has_final_response = self.final_response is not None
        if has_actions == has_final_response:
            raise ValueError("Must contain actions or final response.")
        return self


def test_parse_model_output_rejects_non_object_action_payload() -> None:
    adapter = ActionEnvelopeAdapter()

    with pytest.raises(ValueError):
        adapter.parse_model_output(
            {"actions": [123], "final_response": None},
            response_model=_LooseAnythingEnvelope,
        )


def test_parse_model_output_rejects_missing_tool_name_in_action() -> None:
    adapter = ActionEnvelopeAdapter()

    with pytest.raises(ValueError):
        adapter.parse_model_output(
            {"actions": [{"arguments": {}}], "final_response": None},
            response_model=_LooseDictEnvelope,
        )


def test_parse_model_output_rejects_non_object_arguments() -> None:
    adapter = ActionEnvelopeAdapter()

    with pytest.raises(ValueError):
        adapter.parse_model_output(
            {
                "actions": [{"tool_name": "echo", "arguments": ["bad"]}],
                "final_response": None,
            },
            response_model=_LooseDictEnvelope,
        )


def test_parse_model_output_normalizes_basemodel_arguments() -> None:
    adapter = ActionEnvelopeAdapter()

    parsed = adapter.parse_model_output(
        {
            "actions": [
                {"tool_name": "echo", "arguments": EchoInput(value="hello")},
            ],
            "final_response": None,
        },
        response_model=_LooseAnythingEnvelope,
    )

    assert parsed.invocations == [
        ToolInvocationRequest(tool_name="echo", arguments={"value": "hello"})
    ]


class _PayloadWithModelDump:
    def model_dump(self, *, mode: str, exclude_none: bool) -> dict[str, Any]:
        del mode, exclude_none
        return {
            "actions": [{"tool_name": "anything", "arguments": {"x": 1}}],
            "final_response": None,
        }


def test_parse_model_output_accepts_payload_objects_with_model_dump() -> None:
    adapter = ActionEnvelopeAdapter()

    parsed = adapter.parse_model_output(_PayloadWithModelDump())

    assert parsed.invocations == [
        ToolInvocationRequest(tool_name="anything", arguments={"x": 1})
    ]


def test_build_response_model_handles_sanitized_class_name_edge_cases() -> None:
    adapter = ActionEnvelopeAdapter()
    specs = [
        ToolSpec(name="---", description="non alphanumeric"),
        ToolSpec(name="123-tool", description="starts with digit"),
    ]
    input_models = {
        "---": EchoInput,
        "123-tool": ReadInput,
    }
    response_model = adapter.build_response_model(specs, input_models)
    parsed = adapter.parse_model_output(
        {
            "actions": [
                {"tool_name": "---", "arguments": {"value": "x"}},
                {"tool_name": "123-tool", "arguments": {"path": "README.md"}},
            ],
            "final_response": None,
        },
        response_model=response_model,
    )

    assert [invocation.tool_name for invocation in parsed.invocations] == [
        "---",
        "123-tool",
    ]

"""Tests for the canonical action-envelope adapter."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field, model_validator

from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.tool_api import ToolInvocationRequest, ToolSpec
from llm_tools.workflow_api.chat_models import ChatFinalResponse


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


def test_parse_model_output_uses_validated_typed_arguments_not_raw_payload() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(_specs(), _input_models())

    parsed = adapter.parse_model_output(
        {
            "actions": [
                {
                    "tool_name": "read_file",
                    "arguments": {
                        "path": "README.md",
                        "encoding": "utf-8",
                        "extra_flag": True,
                    },
                }
            ],
            "final_response": None,
        },
        response_model=response_model,
    )

    assert parsed.invocations == [
        ToolInvocationRequest(
            tool_name="read_file",
            arguments={"path": "README.md", "encoding": "utf-8"},
        )
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


def test_parse_model_output_preserves_json_like_final_response_string() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(_specs(), _input_models())
    final_response = '{"status": "done"}'

    parsed = adapter.parse_model_output(
        {"actions": [], "final_response": final_response},
        response_model=response_model,
    )

    assert parsed.final_response == final_response


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


def test_build_response_model_can_simplify_json_schema_contract() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(
        _specs(),
        _input_models(),
        final_response_model=ChatFinalResponse,
        simplify_json_schema=True,
    )
    schema = adapter.export_schema(response_model)

    assert schema["title"] == "ActionEnvelopeSimplified"
    assert schema["properties"]["mode"]["enum"] == ["actions", "final"]
    assert schema["properties"]["actions"]["items"]["$ref"] == "#/$defs/_LooseAction"
    assert schema["$defs"]["_LooseAction"]["title"] == "_LooseAction"
    assert (
        schema["properties"]["final_response"]["anyOf"][0]["additionalProperties"]
        is True
    )


def test_parse_model_output_accepts_simplified_json_action_arguments() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(
        _specs(),
        _input_models(),
        simplify_json_schema=True,
    )

    parsed = adapter.parse_model_output(
        {
            "mode": "actions",
            "actions": [
                {
                    "tool_name": "read_file",
                    "arguments": {
                        "path": "README.md",
                        "encoding": 123,
                        "extra_flag": True,
                    },
                }
            ],
            "final_response": None,
        },
        response_model=response_model,
    )

    assert parsed.final_response is None
    assert parsed.invocations == [
        ToolInvocationRequest(
            tool_name="read_file",
            arguments={
                "path": "README.md",
                "encoding": 123,
                "extra_flag": True,
            },
        )
    ]


def test_parse_model_output_normalizes_simplified_json_final_response_model() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(
        _specs(),
        _input_models(),
        final_response_model=ChatFinalResponse,
        simplify_json_schema=True,
    )

    parsed = adapter.parse_model_output(
        {
            "mode": "final",
            "actions": [],
            "final_response": {"answer": "done", "missing_information": ["none"]},
        },
        response_model=response_model,
    )

    assert parsed.invocations == []
    assert parsed.final_response == {
        "answer": "done",
        "missing_information": ["none"],
    }


def test_build_decision_step_model_constrains_tool_names() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_decision_step_model(_specs())
    schema = adapter.export_schema(response_model)

    assert schema["title"] == "DecisionStep"
    assert schema["properties"]["mode"]["enum"] == ["tool", "finalize"]
    assert sorted(schema["properties"]["tool_name"]["anyOf"][0]["enum"]) == [
        "echo",
        "read_file",
    ]


def test_decision_step_model_rejects_invalid_mode_combinations() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_decision_step_model(_specs())

    with pytest.raises(ValueError):
        response_model.model_validate({"mode": "tool"})
    with pytest.raises(ValueError):
        response_model.model_validate({"mode": "finalize", "tool_name": "echo"})
    with pytest.raises(ValueError):
        response_model.model_validate({"mode": "tool", "tool_name": "missing"})


def test_decision_step_model_without_tools_only_allows_finalize() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_decision_step_model([])
    schema = adapter.export_schema(response_model)

    assert schema["properties"]["mode"]["const"] == "finalize"
    assert response_model.model_validate({"mode": "finalize"}).mode == "finalize"
    with pytest.raises(ValueError):
        response_model.model_validate({"mode": "tool", "tool_name": "missing"})


def test_parse_staged_tool_invocation_and_final_response_steps() -> None:
    adapter = ActionEnvelopeAdapter()
    invocation_model = adapter.build_tool_invocation_step_model(
        tool_name="read_file",
        input_model=ReadInput,
    )
    final_model = adapter.build_final_response_step_model(
        final_response_model=ChatFinalResponse
    )

    parsed_invocation = adapter.parse_tool_invocation_step(
        {
            "mode": "tool",
            "tool_name": "read_file",
            "arguments": {"path": "README.md"},
        },
        response_model=invocation_model,
    )
    parsed_final = adapter.parse_final_response_step(
        {
            "mode": "finalize",
            "final_response": {"answer": "done", "citations": []},
        },
        response_model=final_model,
    )

    assert parsed_invocation.invocations == [
        ToolInvocationRequest(
            tool_name="read_file",
            arguments={"path": "README.md", "encoding": "utf-8"},
        )
    ]
    assert parsed_invocation.final_response is None
    assert parsed_final.invocations == []
    assert parsed_final.final_response is not None
    assert parsed_final.final_response["answer"] == "done"
    assert parsed_final.final_response["citations"] == []


def test_tool_invocation_step_model_rejects_wrong_tool_and_invalid_arguments() -> None:
    adapter = ActionEnvelopeAdapter()
    invocation_model = adapter.build_tool_invocation_step_model(
        tool_name="read_file",
        input_model=ReadInput,
    )

    with pytest.raises(ValueError):
        adapter.parse_tool_invocation_step(
            {
                "mode": "tool",
                "tool_name": "echo",
                "arguments": {"value": "hello"},
            },
            response_model=invocation_model,
        )
    with pytest.raises(ValueError):
        adapter.parse_tool_invocation_step(
            {
                "mode": "tool",
                "tool_name": "read_file",
                "arguments": {"encoding": "utf-8"},
            },
            response_model=invocation_model,
        )


def test_final_response_step_model_rejects_invalid_chat_final_response() -> None:
    adapter = ActionEnvelopeAdapter()
    final_model = adapter.build_final_response_step_model(
        final_response_model=ChatFinalResponse
    )

    with pytest.raises(ValueError):
        adapter.parse_final_response_step(
            {
                "mode": "finalize",
                "final_response": {
                    "answer": "done",
                    "citations": ["README.md"],
                },
            },
            response_model=final_model,
        )


def test_parse_single_action_step_tool_and_final_response() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_single_action_step_model(
        _specs(),
        final_response_model=ChatFinalResponse,
    )

    parsed_tool = adapter.parse_single_action_step(
        {
            "mode": "tool",
            "tool_name": "read_file",
            "arguments": {"path": "README.md"},
        },
        response_model=response_model,
        tool_specs=_specs(),
        input_models=_input_models(),
    )
    parsed_final = adapter.parse_single_action_step(
        {
            "mode": "finalize",
            "final_response": {"answer": "done", "citations": []},
        },
        response_model=response_model,
        tool_specs=_specs(),
        input_models=_input_models(),
    )

    assert parsed_tool.invocations == [
        ToolInvocationRequest(
            tool_name="read_file",
            arguments={"path": "README.md", "encoding": "utf-8"},
        )
    ]
    assert parsed_final.final_response is not None
    assert parsed_final.final_response["answer"] == "done"
    assert parsed_final.final_response["citations"] == []


def test_single_action_step_rejects_mixed_or_invalid_tool_payloads() -> None:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_single_action_step_model(
        _specs(),
        final_response_model=ChatFinalResponse,
    )

    with pytest.raises(ValueError):
        adapter.parse_single_action_step(
            {
                "mode": "tool",
                "tool_name": "read_file",
                "arguments": {"path": "README.md"},
                "final_response": {"answer": "mixed"},
            },
            response_model=response_model,
            tool_specs=_specs(),
            input_models=_input_models(),
        )
    with pytest.raises(ValueError):
        adapter.parse_single_action_step(
            {"mode": "tool", "tool_name": "missing", "arguments": {}},
            response_model=response_model,
            tool_specs=_specs(),
            input_models=_input_models(),
        )
    with pytest.raises(ValueError):
        adapter.parse_single_action_step(
            {"mode": "tool", "tool_name": "read_file", "arguments": {}},
            response_model=response_model,
            tool_specs=_specs(),
            input_models=_input_models(),
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


class _LooseStringActionEnvelope(BaseModel):
    actions: list[str] = Field(default_factory=list)
    final_response: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> _LooseStringActionEnvelope:
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


def test_parse_model_output_rejects_stringified_action_after_validation() -> None:
    adapter = ActionEnvelopeAdapter()

    with pytest.raises(ValueError):
        adapter.parse_model_output(
            {
                "actions": ['{"tool_name": "echo", "arguments": {"value": "hello"}}'],
                "final_response": None,
            },
            response_model=_LooseStringActionEnvelope,
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

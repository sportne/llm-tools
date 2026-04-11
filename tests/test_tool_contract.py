"""Tests for the base tool contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest
from pydantic import BaseModel

from llm_tools.tool_api import Tool, ToolContext, ToolSpec


class ExampleInput(BaseModel):
    value: str


class ExampleOutput(BaseModel):
    value: str


def test_valid_tool_subclass_exposes_expected_contract() -> None:
    class EchoTool(Tool[ExampleInput, ExampleOutput]):
        spec = ToolSpec(name="echo", description="Echo a string.")
        input_model = ExampleInput
        output_model = ExampleOutput

        def invoke(self, context: ToolContext, args: ExampleInput) -> ExampleOutput:
            return ExampleOutput(value=f"{context.invocation_id}:{args.value}")

    tool = EchoTool()

    assert EchoTool.spec.name == "echo"
    assert EchoTool.input_model is ExampleInput
    assert EchoTool.output_model is ExampleOutput
    result = tool.invoke(ToolContext(invocation_id="inv-1"), ExampleInput(value="hi"))
    assert isinstance(result, ExampleOutput)
    assert result.value == "inv-1:hi"


def test_abstract_intermediate_subclass_can_defer_required_class_attributes() -> None:
    class AbstractToolBase(Tool[ExampleInput, ExampleOutput], ABC):
        @abstractmethod
        def invoke(self, context: ToolContext, args: ExampleInput) -> ExampleOutput:
            raise NotImplementedError

    class ConcreteTool(AbstractToolBase):
        spec = ToolSpec(name="concrete", description="Concrete tool.")
        input_model = ExampleInput
        output_model = ExampleOutput

        def invoke(self, context: ToolContext, args: ExampleInput) -> ExampleOutput:
            return ExampleOutput(value=args.value)

    tool = ConcreteTool()
    assert (
        tool.invoke(ToolContext(invocation_id="inv-2"), ExampleInput(value="ok")).value
        == "ok"
    )


@pytest.mark.parametrize(
    ("tool_name", "namespace", "message"),
    [
        (
            "MissingSpecTool",
            {"input_model": ExampleInput, "output_model": ExampleOutput},
            "spec",
        ),
        (
            "MissingInputModelTool",
            {
                "spec": ToolSpec(name="missing-input", description="desc"),
                "output_model": ExampleOutput,
            },
            "input_model",
        ),
        (
            "MissingOutputModelTool",
            {
                "spec": ToolSpec(name="missing-output", description="desc"),
                "input_model": ExampleInput,
            },
            "output_model",
        ),
        (
            "WrongSpecTypeTool",
            {
                "spec": "not-a-spec",
                "input_model": ExampleInput,
                "output_model": ExampleOutput,
            },
            "ToolSpec",
        ),
        (
            "WrongInputModelTool",
            {
                "spec": ToolSpec(name="wrong-input", description="desc"),
                "input_model": str,
                "output_model": ExampleOutput,
            },
            "BaseModel",
        ),
        (
            "WrongOutputModelTool",
            {
                "spec": ToolSpec(name="wrong-output", description="desc"),
                "input_model": ExampleInput,
                "output_model": str,
            },
            "BaseModel",
        ),
    ],
)
def test_invalid_tool_subclasses_fail_at_definition_time(
    tool_name: str,
    namespace: dict[str, object],
    message: str,
) -> None:
    def invoke(self: object, context: ToolContext, args: ExampleInput) -> ExampleOutput:
        del self, context, args
        return ExampleOutput(value="unused")

    namespace_with_invoke = {
        "__module__": __name__,
        "__annotations__": {},
        **namespace,
        "invoke": invoke,
    }

    with pytest.raises(TypeError, match=message):
        type(tool_name, (Tool,), namespace_with_invoke)

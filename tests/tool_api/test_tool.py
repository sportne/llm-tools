"""Tests for the base tool contract."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import pytest
from pydantic import BaseModel

from llm_tools.tool_api import (
    Tool,
    ToolContext,
    ToolExecutionContext,
    ToolInvocationRequest,
    ToolRegistry,
    ToolRuntime,
    ToolSpec,
)


class ExampleInput(BaseModel):
    value: str


class ExampleOutput(BaseModel):
    value: str


def test_valid_tool_subclass_exposes_expected_contract() -> None:
    class EchoTool(Tool[ExampleInput, ExampleOutput]):
        spec = ToolSpec(name="echo", description="Echo a string.")
        input_model = ExampleInput
        output_model = ExampleOutput

        def _invoke_impl(
            self, context: ToolExecutionContext, args: ExampleInput
        ) -> ExampleOutput:
            return ExampleOutput(value=f"{context.invocation_id}:{args.value}")

    tool = EchoTool()
    registry = ToolRegistry()
    registry.register(tool)
    runtime = ToolRuntime(registry)

    assert EchoTool.spec.name == "echo"
    assert EchoTool.input_model is ExampleInput
    assert EchoTool.output_model is ExampleOutput

    result = runtime.execute(
        ToolInvocationRequest(tool_name="echo", arguments={"value": "hi"}),
        ToolContext(invocation_id="inv-1"),
    )
    assert result.ok is True
    assert result.output == {"value": "inv-1:hi"}


def test_public_execution_context_type_is_not_instantiable() -> None:
    with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
        ToolExecutionContext()


def test_direct_tool_execution_is_rejected_without_runtime_permit() -> None:
    class EchoTool(Tool[ExampleInput, ExampleOutput]):
        spec = ToolSpec(name="echo", description="Echo a string.")
        input_model = ExampleInput
        output_model = ExampleOutput

        def _invoke_impl(
            self, context: ToolExecutionContext, args: ExampleInput
        ) -> ExampleOutput:
            return ExampleOutput(value=f"{context.invocation_id}:{args.value}")

    tool = EchoTool()

    with pytest.raises(RuntimeError, match="Direct tool execution is disabled"):
        tool.invoke(ToolContext(invocation_id="inv-1"), ExampleInput(value="hi"))


def test_async_only_tool_subclass_is_valid() -> None:
    class AsyncEchoTool(Tool[ExampleInput, ExampleOutput]):
        spec = ToolSpec(name="async-echo", description="Async echo tool.")
        input_model = ExampleInput
        output_model = ExampleOutput

        async def _ainvoke_impl(
            self, context: ToolExecutionContext, args: ExampleInput
        ) -> ExampleOutput:
            return ExampleOutput(value=f"{context.invocation_id}:{args.value}")

    registry = ToolRegistry()
    registry.register(AsyncEchoTool())
    runtime = ToolRuntime(registry)
    result = asyncio.run(
        runtime.execute_async(
            ToolInvocationRequest(tool_name="async-echo", arguments={"value": "hi"}),
            ToolContext(invocation_id="inv-async"),
        )
    )
    assert result.ok is True
    assert result.output == {"value": "inv-async:hi"}


def test_abstract_intermediate_subclass_can_defer_required_class_attributes() -> None:
    class AbstractToolBase(Tool[ExampleInput, ExampleOutput], ABC):
        @abstractmethod
        def _invoke_impl(
            self, context: ToolExecutionContext, args: ExampleInput
        ) -> ExampleOutput:
            raise NotImplementedError

    class ConcreteTool(AbstractToolBase):
        spec = ToolSpec(name="concrete", description="Concrete tool.")
        input_model = ExampleInput
        output_model = ExampleOutput

        def _invoke_impl(
            self, context: ToolExecutionContext, args: ExampleInput
        ) -> ExampleOutput:
            del context
            return ExampleOutput(value=args.value)

    registry = ToolRegistry()
    registry.register(ConcreteTool())
    runtime = ToolRuntime(registry)
    result = runtime.execute(
        ToolInvocationRequest(tool_name="concrete", arguments={"value": "ok"}),
        ToolContext(invocation_id="inv-2"),
    )
    assert result.output == {"value": "ok"}


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
    def _invoke_impl(
        self: object,
        context: ToolExecutionContext,
        args: ExampleInput,
    ) -> ExampleOutput:
        del self, context, args
        return ExampleOutput(value="unused")

    namespace_with_impl = {
        "__module__": __name__,
        "__annotations__": {},
        **namespace,
        "_invoke_impl": _invoke_impl,
    }

    with pytest.raises(TypeError, match=message):
        type(tool_name, (Tool,), namespace_with_impl)


def test_tool_subclass_without_invoke_or_ainvoke_is_rejected() -> None:
    with pytest.raises(TypeError, match="at least one execution method"):
        type(
            "MissingExecutionMethodsTool",
            (Tool,),
            {
                "__module__": __name__,
                "__annotations__": {},
                "spec": ToolSpec(name="missing-exec", description="desc"),
                "input_model": ExampleInput,
                "output_model": ExampleOutput,
            },
        )

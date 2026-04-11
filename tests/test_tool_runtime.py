"""Tests for runtime execution and normalization."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from llm_tools.tool_api import (
    ErrorCode,
    SideEffectClass,
    Tool,
    ToolContext,
    ToolInvocationRequest,
    ToolPolicy,
    ToolRegistry,
    ToolRuntime,
    ToolSpec,
)


class RuntimeInput(BaseModel):
    value: str


class RuntimeOutput(BaseModel):
    value: str


class EchoTool(Tool[RuntimeInput, RuntimeOutput]):
    spec = ToolSpec(
        name="echo",
        description="Echo a string.",
        side_effects=SideEffectClass.NONE,
    )
    input_model = RuntimeInput
    output_model = RuntimeOutput

    def invoke(self, context: ToolContext, args: RuntimeInput) -> RuntimeOutput:
        return RuntimeOutput(value=f"{context.invocation_id}:{args.value}")


class DictOutputTool(Tool[RuntimeInput, RuntimeOutput]):
    spec = ToolSpec(
        name="dict_echo",
        description="Return a dict-like payload.",
        side_effects=SideEffectClass.NONE,
    )
    input_model = RuntimeInput
    output_model = RuntimeOutput

    def invoke(self, context: ToolContext, args: RuntimeInput) -> RuntimeOutput:
        del context
        return cast(RuntimeOutput, {"value": args.value.upper()})


class LocalWriteTool(Tool[RuntimeInput, RuntimeOutput]):
    spec = ToolSpec(
        name="write_file",
        description="Write a file.",
        side_effects=SideEffectClass.LOCAL_WRITE,
    )
    input_model = RuntimeInput
    output_model = RuntimeOutput

    def invoke(self, context: ToolContext, args: RuntimeInput) -> RuntimeOutput:
        del context
        return RuntimeOutput(value=args.value)


class FailingTool(Tool[RuntimeInput, RuntimeOutput]):
    spec = ToolSpec(
        name="failing_tool",
        description="Always raises.",
        side_effects=SideEffectClass.NONE,
    )
    input_model = RuntimeInput
    output_model = RuntimeOutput

    def invoke(self, context: ToolContext, args: RuntimeInput) -> RuntimeOutput:
        del context, args
        raise ValueError("boom")


class InvalidOutputTool(Tool[RuntimeInput, RuntimeOutput]):
    spec = ToolSpec(
        name="invalid_output",
        description="Returns invalid output.",
        side_effects=SideEffectClass.NONE,
    )
    input_model = RuntimeInput
    output_model = RuntimeOutput

    def invoke(self, context: ToolContext, args: RuntimeInput) -> RuntimeOutput:
        del context, args
        return cast(RuntimeOutput, "not-a-valid-output")


class BrokenPolicy(ToolPolicy):
    def evaluate(
        self,
        tool: Tool[Any, Any],
        context: ToolContext,
    ):  # type: ignore[override]
        del tool, context
        raise RuntimeError("policy exploded")


def _context() -> ToolContext:
    return ToolContext(invocation_id="inv-1")


def _registry(*tools: Tool[Any, Any]) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_runtime_executes_successfully_and_attaches_execution_record() -> None:
    runtime = ToolRuntime(_registry(EchoTool()))

    result = runtime.execute(
        ToolInvocationRequest(tool_name="echo", arguments={"value": "hi"}),
        _context(),
    )

    assert result.ok is True
    assert result.output == {"value": "inv-1:hi"}
    assert result.error is None

    record = result.metadata["execution_record"]
    assert record["invocation_id"] == "inv-1"
    assert record["tool_name"] == "echo"
    assert record["tool_version"] == "0.1.0"
    assert record["request"] == {"tool_name": "echo", "arguments": {"value": "hi"}}
    assert record["validated_input"] == {"value": "hi"}
    assert record["ok"] is True
    assert record["error_code"] is None


def test_runtime_accepts_output_that_validates_against_output_model() -> None:
    runtime = ToolRuntime(_registry(DictOutputTool()))

    result = runtime.execute(
        ToolInvocationRequest(tool_name="dict_echo", arguments={"value": "hello"}),
        _context(),
    )

    assert result.ok is True
    assert result.output == {"value": "HELLO"}
    assert result.error is None


def test_runtime_normalizes_invalid_input() -> None:
    runtime = ToolRuntime(_registry(EchoTool()))

    result = runtime.execute(
        ToolInvocationRequest(tool_name="echo", arguments={"value": 123}),
        _context(),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.INPUT_VALIDATION_ERROR
    assert result.error.details["tool_name"] == "echo"
    assert result.metadata["execution_record"]["error_code"] == "input_validation_error"


def test_runtime_normalizes_denied_policy_with_approval_metadata() -> None:
    runtime = ToolRuntime(
        _registry(LocalWriteTool()),
        policy=ToolPolicy(
            allowed_side_effects={
                SideEffectClass.NONE,
                SideEffectClass.LOCAL_READ,
                SideEffectClass.LOCAL_WRITE,
            },
            require_approval_for={SideEffectClass.LOCAL_WRITE},
        ),
    )

    result = runtime.execute(
        ToolInvocationRequest(tool_name="write_file", arguments={"value": "data"}),
        _context(),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.POLICY_DENIED
    assert result.error.details["policy_decision"]["requires_approval"] is True
    assert result.metadata["execution_record"]["policy_decision"]["reason"] == (
        "approval required"
    )


def test_runtime_normalizes_tool_execution_exceptions() -> None:
    runtime = ToolRuntime(_registry(FailingTool()))

    result = runtime.execute(
        ToolInvocationRequest(tool_name="failing_tool", arguments={"value": "hi"}),
        _context(),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    assert result.error.details["exception_type"] == "ValueError"
    assert result.error.details["exception_message"] == "boom"
    assert result.metadata["execution_record"]["error_code"] == "execution_failed"


def test_runtime_normalizes_invalid_output() -> None:
    runtime = ToolRuntime(_registry(InvalidOutputTool()))

    result = runtime.execute(
        ToolInvocationRequest(tool_name="invalid_output", arguments={"value": "hi"}),
        _context(),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.OUTPUT_VALIDATION_ERROR
    assert result.error.details["tool_name"] == "invalid_output"
    assert result.metadata["execution_record"]["error_code"] == (
        "output_validation_error"
    )


def test_runtime_normalizes_missing_tool_lookup() -> None:
    runtime = ToolRuntime(_registry())

    result = runtime.execute(
        ToolInvocationRequest(tool_name="missing_tool", arguments={}),
        _context(),
    )

    assert result.ok is False
    assert result.tool_name == "missing_tool"
    assert result.tool_version == "unknown"
    assert result.error is not None
    assert result.error.code is ErrorCode.TOOL_NOT_FOUND
    assert result.metadata["execution_record"]["tool_name"] == "missing_tool"
    assert result.metadata["execution_record"]["error_code"] == "tool_not_found"


def test_runtime_normalizes_unexpected_runtime_failures() -> None:
    runtime = ToolRuntime(_registry(EchoTool()), policy=BrokenPolicy())

    result = runtime.execute(
        ToolInvocationRequest(tool_name="echo", arguments={"value": "hi"}),
        _context(),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.RUNTIME_ERROR
    assert result.error.details["exception_type"] == "RuntimeError"
    assert result.error.details["exception_message"] == "policy exploded"
    assert result.metadata["execution_record"]["tool_name"] == "echo"
    assert result.metadata["execution_record"]["error_code"] == "runtime_error"

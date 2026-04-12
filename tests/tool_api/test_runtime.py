"""Tests for runtime execution and normalization."""

from __future__ import annotations

import asyncio
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
        context.logs.append("echo-start")
        context.artifacts.append("echo.txt")
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
        del args
        context.logs.append("about-to-fail")
        context.artifacts.append("failure.log")
        raise ValueError("boom")


class SecretInput(BaseModel):
    username: str
    password: str
    nested: dict[str, Any]


class SecretOutput(BaseModel):
    status: str


class SecretLoggingTool(Tool[SecretInput, SecretOutput]):
    spec = ToolSpec(
        name="secret_logging",
        description="Capture sensitive input and emit observability data.",
        side_effects=SideEffectClass.NONE,
    )
    input_model = SecretInput
    output_model = SecretOutput

    def invoke(self, context: ToolContext, args: SecretInput) -> SecretOutput:
        context.logs.append("processing-secret-input")
        context.artifacts.append("secret.json")
        return SecretOutput(status=args.username)


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


class AsyncEchoTool(Tool[RuntimeInput, RuntimeOutput]):
    spec = ToolSpec(
        name="async_echo",
        description="Async echo tool.",
        side_effects=SideEffectClass.NONE,
    )
    input_model = RuntimeInput
    output_model = RuntimeOutput

    async def ainvoke(self, context: ToolContext, args: RuntimeInput) -> RuntimeOutput:
        context.logs.append("async-echo-start")
        context.artifacts.append("async.txt")
        return RuntimeOutput(value=f"{context.invocation_id}:{args.value}")


class DualEchoTool(Tool[RuntimeInput, RuntimeOutput]):
    spec = ToolSpec(
        name="dual_echo",
        description="Tool with sync and async implementations.",
        side_effects=SideEffectClass.NONE,
    )
    input_model = RuntimeInput
    output_model = RuntimeOutput

    def invoke(self, context: ToolContext, args: RuntimeInput) -> RuntimeOutput:
        context.logs.append("sync-path")
        return RuntimeOutput(value=f"sync:{args.value}")

    async def ainvoke(self, context: ToolContext, args: RuntimeInput) -> RuntimeOutput:
        context.logs.append("async-path")
        return RuntimeOutput(value=f"async:{args.value}")


class BrokenPolicy(ToolPolicy):
    def evaluate(
        self,
        tool: Tool[Any, Any],
        context: ToolContext,
    ) -> Any:
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
    assert result.logs == ["echo-start"]
    assert result.artifacts == ["echo.txt"]

    record = result.metadata["execution_record"]
    assert record["invocation_id"] == "inv-1"
    assert record["tool_name"] == "echo"
    assert record["tool_version"] == "0.1.0"
    assert record["request"] == {"tool_name": "echo", "arguments": {"value": "hi"}}
    assert record["validated_input"] == {"value": "hi"}
    assert record["redacted_input"] == {"value": "hi"}
    assert record["ok"] is True
    assert record["error_code"] is None
    assert record["duration_ms"] is not None
    assert record["duration_ms"] >= 0
    assert record["started_at"].endswith("Z")
    assert record["ended_at"].endswith("Z")
    assert record["logs"] == ["echo-start"]
    assert record["artifacts"] == ["echo.txt"]


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
    assert result.logs == ["about-to-fail"]
    assert result.artifacts == ["failure.log"]
    assert result.metadata["execution_record"]["error_code"] == "execution_failed"
    assert result.metadata["execution_record"]["logs"] == ["about-to-fail"]
    assert result.metadata["execution_record"]["artifacts"] == ["failure.log"]


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


def test_runtime_records_redacted_input_and_tool_emitted_observability() -> None:
    runtime = ToolRuntime(_registry(SecretLoggingTool()))

    result = runtime.execute(
        ToolInvocationRequest(
            tool_name="secret_logging",
            arguments={
                "username": "alice",
                "password": "super-secret",
                "nested": {
                    "api_key": "abc123",
                    "items": [
                        {"authorization": "Bearer token"},
                        {"note": "keep"},
                    ],
                },
            },
        ),
        _context(),
    )

    assert result.ok is True
    assert result.logs == ["processing-secret-input"]
    assert result.artifacts == ["secret.json"]

    record = result.metadata["execution_record"]
    assert record["validated_input"] == {
        "username": "alice",
        "password": "super-secret",
        "nested": {
            "api_key": "abc123",
            "items": [
                {"authorization": "Bearer token"},
                {"note": "keep"},
            ],
        },
    }
    assert record["redacted_input"] == {
        "username": "alice",
        "password": "[REDACTED]",
        "nested": {
            "api_key": "[REDACTED]",
            "items": [
                {"authorization": "[REDACTED]"},
                {"note": "keep"},
            ],
        },
    }
    assert record["logs"] == ["processing-secret-input"]
    assert record["artifacts"] == ["secret.json"]


def test_runtime_redaction_normalizes_field_name_variants() -> None:
    runtime = ToolRuntime(
        _registry(SecretLoggingTool()),
        policy=ToolPolicy(redacted_field_names={"API_KEY", "api-key", "access_token"}),
    )

    result = runtime.execute(
        ToolInvocationRequest(
            tool_name="secret_logging",
            arguments={
                "username": "alice",
                "password": "visible",
                "nested": {
                    "API_KEY": "secret-1",
                    "api-key": "secret-2",
                    "access_token": "secret-3",
                },
            },
        ),
        _context(),
    )

    assert result.ok is True
    assert result.metadata["execution_record"]["redacted_input"] == {
        "username": "alice",
        "password": "visible",
        "nested": {
            "API_KEY": "[REDACTED]",
            "api-key": "[REDACTED]",
            "access_token": "[REDACTED]",
        },
    }


def test_runtime_execute_async_supports_sync_only_tools() -> None:
    runtime = ToolRuntime(_registry(EchoTool()))
    result = asyncio.run(
        runtime.execute_async(
            ToolInvocationRequest(tool_name="echo", arguments={"value": "hi"}),
            _context(),
        )
    )

    assert result.ok is True
    assert result.output == {"value": "inv-1:hi"}
    assert result.logs == ["echo-start"]


def test_runtime_execute_async_supports_async_only_tools() -> None:
    runtime = ToolRuntime(_registry(AsyncEchoTool()))
    result = asyncio.run(
        runtime.execute_async(
            ToolInvocationRequest(tool_name="async_echo", arguments={"value": "hi"}),
            _context(),
        )
    )

    assert result.ok is True
    assert result.output == {"value": "inv-1:hi"}
    assert result.logs == ["async-echo-start"]
    assert result.artifacts == ["async.txt"]


def test_runtime_execute_sync_can_bridge_async_only_tools_without_loop() -> None:
    runtime = ToolRuntime(_registry(AsyncEchoTool()))
    result = runtime.execute(
        ToolInvocationRequest(tool_name="async_echo", arguments={"value": "hi"}),
        _context(),
    )
    assert result.ok is True
    assert result.output == {"value": "inv-1:hi"}


def test_runtime_execute_sync_rejects_async_only_tools_inside_event_loop() -> None:
    runtime = ToolRuntime(_registry(AsyncEchoTool()))

    async def run() -> None:
        result = runtime.execute(
            ToolInvocationRequest(tool_name="async_echo", arguments={"value": "hi"}),
            _context(),
        )
        assert result.ok is False
        assert result.error is not None
        assert result.error.code is ErrorCode.EXECUTION_FAILED
        assert (
            "Use ToolRuntime.execute_async()"
            in result.error.details["exception_message"]
        )

    asyncio.run(run())


def test_runtime_prefers_sync_for_execute_and_async_for_execute_async() -> None:
    runtime = ToolRuntime(_registry(DualEchoTool()))

    sync_result = runtime.execute(
        ToolInvocationRequest(tool_name="dual_echo", arguments={"value": "hello"}),
        _context(),
    )
    async_result = asyncio.run(
        runtime.execute_async(
            ToolInvocationRequest(tool_name="dual_echo", arguments={"value": "hello"}),
            _context(),
        )
    )

    assert sync_result.ok is True
    assert sync_result.output == {"value": "sync:hello"}
    assert sync_result.logs == ["sync-path"]
    assert async_result.ok is True
    assert async_result.output == {"value": "async:hello"}
    assert async_result.logs == ["async-path"]


def test_runtime_execute_async_normalizes_missing_tool_lookup() -> None:
    runtime = ToolRuntime(_registry())

    result = asyncio.run(
        runtime.execute_async(
            ToolInvocationRequest(tool_name="missing_tool", arguments={}),
            _context(),
        )
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.TOOL_NOT_FOUND
    assert result.metadata["execution_record"]["error_code"] == "tool_not_found"


def test_runtime_execute_async_normalizes_policy_denied() -> None:
    runtime = ToolRuntime(
        _registry(LocalWriteTool()),
        policy=ToolPolicy(allowed_side_effects={SideEffectClass.NONE}),
    )

    result = asyncio.run(
        runtime.execute_async(
            ToolInvocationRequest(tool_name="write_file", arguments={"value": "data"}),
            _context(),
        )
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.POLICY_DENIED
    assert result.metadata["execution_record"]["error_code"] == "policy_denied"


def test_runtime_execute_async_normalizes_invalid_input_execution_and_output() -> None:
    invalid_input_runtime = ToolRuntime(_registry(EchoTool()))
    invalid_input = asyncio.run(
        invalid_input_runtime.execute_async(
            ToolInvocationRequest(tool_name="echo", arguments={"value": 123}),
            _context(),
        )
    )
    assert invalid_input.ok is False
    assert invalid_input.error is not None
    assert invalid_input.error.code is ErrorCode.INPUT_VALIDATION_ERROR

    execution_failure_runtime = ToolRuntime(_registry(FailingTool()))
    execution_failure = asyncio.run(
        execution_failure_runtime.execute_async(
            ToolInvocationRequest(tool_name="failing_tool", arguments={"value": "hi"}),
            _context(),
        )
    )
    assert execution_failure.ok is False
    assert execution_failure.error is not None
    assert execution_failure.error.code is ErrorCode.EXECUTION_FAILED

    invalid_output_runtime = ToolRuntime(_registry(InvalidOutputTool()))
    invalid_output = asyncio.run(
        invalid_output_runtime.execute_async(
            ToolInvocationRequest(
                tool_name="invalid_output", arguments={"value": "hi"}
            ),
            _context(),
        )
    )
    assert invalid_output.ok is False
    assert invalid_output.error is not None
    assert invalid_output.error.code is ErrorCode.OUTPUT_VALIDATION_ERROR

"""Strict runtime orchestration for tool execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from llm_tools.tool_api.errors import ToolNotRegisteredError
from llm_tools.tool_api.models import (
    ErrorCode,
    ExecutionRecord,
    PolicyDecision,
    ToolContext,
    ToolError,
    ToolInvocationRequest,
    ToolResult,
)
from llm_tools.tool_api.policy import ToolPolicy
from llm_tools.tool_api.registry import ToolRegistry
from llm_tools.tool_api.tool import Tool


@dataclass(slots=True)
class _ExecutionState:
    """Mutable execution state captured across runtime phases."""

    invocation_id: str
    request: ToolInvocationRequest
    started_at: datetime
    tool_name: str
    tool_version: str = "unknown"
    validated_input: dict[str, Any] | None = None
    redacted_input: dict[str, Any] | None = None
    policy_decision: PolicyDecision | None = None
    logs: list[str] | None = None
    artifacts: list[str] | None = None
    ok: bool | None = None
    error_code: ErrorCode | None = None


class ToolRuntime:
    """Execute registered tools with validation, policy, and normalization."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy or ToolPolicy()

    def execute(
        self,
        request: ToolInvocationRequest,
        context: ToolContext,
    ) -> ToolResult:
        """Execute one tool invocation and return a normalized result."""
        state = _ExecutionState(
            invocation_id=context.invocation_id,
            request=request,
            started_at=self._utc_now(),
            tool_name=request.tool_name,
        )

        try:
            tool = self._resolve_tool(request)
        except ToolNotRegisteredError as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_error(
                    code=ErrorCode.TOOL_NOT_FOUND,
                    message=str(exc),
                    details={"tool_name": request.tool_name},
                ),
            )
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_runtime_error(state, exc),
            )

        state.tool_name = tool.spec.name
        state.tool_version = tool.spec.version

        try:
            policy_decision = self._evaluate_policy(tool, context)
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_runtime_error(state, exc),
            )

        state.policy_decision = policy_decision
        if not policy_decision.allowed:
            return self._finalize_failure(
                state,
                context,
                self._make_error(
                    code=ErrorCode.POLICY_DENIED,
                    message=f"Policy denied execution for tool '{state.tool_name}'.",
                    details={
                        "tool_name": state.tool_name,
                        "policy_decision": policy_decision.model_dump(mode="json"),
                    },
                ),
            )

        try:
            validated_input = self._validate_input(tool, request.arguments)
        except ValidationError as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_error(
                    code=ErrorCode.INPUT_VALIDATION_ERROR,
                    message=f"Input validation failed for tool '{state.tool_name}'.",
                    details={
                        "tool_name": state.tool_name,
                        "validation_errors": exc.errors(),
                    },
                ),
            )
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_runtime_error(state, exc),
            )

        state.validated_input = validated_input.model_dump(mode="json")
        state.redacted_input = self._redact_input(state.validated_input)

        try:
            raw_output = self._invoke_tool(tool, context, validated_input)
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_error(
                    code=ErrorCode.EXECUTION_FAILED,
                    message=f"Tool '{state.tool_name}' execution failed.",
                    details={
                        "tool_name": state.tool_name,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                ),
            )

        try:
            validated_output = self._validate_output(tool, raw_output)
        except ValidationError as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_error(
                    code=ErrorCode.OUTPUT_VALIDATION_ERROR,
                    message=f"Output validation failed for tool '{state.tool_name}'.",
                    details={
                        "tool_name": state.tool_name,
                        "validation_errors": exc.errors(),
                    },
                ),
            )
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_runtime_error(state, exc),
            )

        return self._finalize_success(
            state,
            context,
            output=validated_output.model_dump(mode="json"),
        )

    async def execute_async(
        self,
        request: ToolInvocationRequest,
        context: ToolContext,
    ) -> ToolResult:
        """Asynchronously execute one tool invocation and normalize the result."""
        state = _ExecutionState(
            invocation_id=context.invocation_id,
            request=request,
            started_at=self._utc_now(),
            tool_name=request.tool_name,
        )

        try:
            tool = self._resolve_tool(request)
        except ToolNotRegisteredError as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_error(
                    code=ErrorCode.TOOL_NOT_FOUND,
                    message=str(exc),
                    details={"tool_name": request.tool_name},
                ),
            )
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_runtime_error(state, exc),
            )

        state.tool_name = tool.spec.name
        state.tool_version = tool.spec.version

        try:
            policy_decision = self._evaluate_policy(tool, context)
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_runtime_error(state, exc),
            )

        state.policy_decision = policy_decision
        if not policy_decision.allowed:
            return self._finalize_failure(
                state,
                context,
                self._make_error(
                    code=ErrorCode.POLICY_DENIED,
                    message=f"Policy denied execution for tool '{state.tool_name}'.",
                    details={
                        "tool_name": state.tool_name,
                        "policy_decision": policy_decision.model_dump(mode="json"),
                    },
                ),
            )

        try:
            validated_input = self._validate_input(tool, request.arguments)
        except ValidationError as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_error(
                    code=ErrorCode.INPUT_VALIDATION_ERROR,
                    message=f"Input validation failed for tool '{state.tool_name}'.",
                    details={
                        "tool_name": state.tool_name,
                        "validation_errors": exc.errors(),
                    },
                ),
            )
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_runtime_error(state, exc),
            )

        state.validated_input = validated_input.model_dump(mode="json")
        state.redacted_input = self._redact_input(state.validated_input)

        try:
            raw_output = await self._invoke_tool_async(tool, context, validated_input)
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_error(
                    code=ErrorCode.EXECUTION_FAILED,
                    message=f"Tool '{state.tool_name}' execution failed.",
                    details={
                        "tool_name": state.tool_name,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                ),
            )

        try:
            validated_output = self._validate_output(tool, raw_output)
        except ValidationError as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_error(
                    code=ErrorCode.OUTPUT_VALIDATION_ERROR,
                    message=f"Output validation failed for tool '{state.tool_name}'.",
                    details={
                        "tool_name": state.tool_name,
                        "validation_errors": exc.errors(),
                    },
                ),
            )
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                self._make_runtime_error(state, exc),
            )

        return self._finalize_success(
            state,
            context,
            output=validated_output.model_dump(mode="json"),
        )

    def _resolve_tool(self, request: ToolInvocationRequest) -> Tool[Any, Any]:
        return self._registry.get(request.tool_name)

    def _evaluate_policy(
        self,
        tool: Tool[Any, Any],
        context: ToolContext,
    ) -> PolicyDecision:
        return self._policy.evaluate(tool, context)

    def _validate_input(
        self,
        tool: Tool[Any, Any],
        arguments: dict[str, Any],
    ) -> BaseModel:
        return cast(BaseModel, tool.input_model.model_validate(arguments))

    def _invoke_tool(
        self,
        tool: Tool[Any, Any],
        context: ToolContext,
        validated_input: BaseModel,
    ) -> Any:
        if tool.__class__._has_sync_implementation():
            return tool.invoke(context, validated_input)
        if tool.__class__._has_async_implementation():
            return self._run_async_tool_in_sync(tool, context, validated_input)
        raise RuntimeError(
            f"Tool '{tool.spec.name}' has no synchronous or asynchronous "
            "implementation."
        )

    async def _invoke_tool_async(
        self,
        tool: Tool[Any, Any],
        context: ToolContext,
        validated_input: BaseModel,
    ) -> Any:
        if tool.__class__._has_async_implementation():
            return await tool.ainvoke(context, validated_input)
        if tool.__class__._has_sync_implementation():
            return await asyncio.to_thread(tool.invoke, context, validated_input)
        raise RuntimeError(
            f"Tool '{tool.spec.name}' has no synchronous or asynchronous "
            "implementation."
        )

    def _validate_output(self, tool: Tool[Any, Any], output: Any) -> BaseModel:
        return cast(BaseModel, tool.output_model.model_validate(output))

    def _finalize_success(
        self,
        state: _ExecutionState,
        context: ToolContext,
        *,
        output: dict[str, Any],
    ) -> ToolResult:
        state.ok = True
        state.logs = list(context.logs)
        state.artifacts = list(context.artifacts)
        result = self._normalize_result(state, output=output, error=None)
        record = self._record_execution(state)
        result.metadata["execution_record"] = record.model_dump(mode="json")
        return result

    def _finalize_failure(
        self,
        state: _ExecutionState,
        context: ToolContext,
        error: ToolError,
    ) -> ToolResult:
        state.ok = False
        state.error_code = error.code
        state.logs = list(context.logs)
        state.artifacts = list(context.artifacts)
        result = self._normalize_result(state, output=None, error=error)
        record = self._record_execution(state)
        result.metadata["execution_record"] = record.model_dump(mode="json")
        return result

    def _normalize_result(
        self,
        state: _ExecutionState,
        *,
        output: dict[str, Any] | None,
        error: ToolError | None,
    ) -> ToolResult:
        return ToolResult(
            ok=error is None,
            tool_name=state.tool_name,
            tool_version=state.tool_version,
            output=output,
            error=error,
            logs=list(state.logs or []),
            artifacts=list(state.artifacts or []),
        )

    def _record_execution(self, state: _ExecutionState) -> ExecutionRecord:
        ended_at = self._utc_now()
        logs, artifacts = self._snapshot_observability(state)
        return ExecutionRecord(
            invocation_id=state.invocation_id,
            tool_name=state.tool_name,
            tool_version=state.tool_version,
            started_at=self._to_timestamp(state.started_at),
            ended_at=self._to_timestamp(ended_at),
            duration_ms=self._duration_ms(state.started_at, ended_at),
            request=state.request,
            validated_input=state.validated_input,
            redacted_input=state.redacted_input,
            ok=state.ok,
            error_code=state.error_code,
            policy_decision=state.policy_decision,
            logs=logs,
            artifacts=artifacts,
        )

    def _snapshot_observability(
        self, state: _ExecutionState
    ) -> tuple[list[str], list[str]]:
        state.logs = list(state.logs or [])
        state.artifacts = list(state.artifacts or [])
        return state.logs, state.artifacts

    def _redact_input(self, value: dict[str, Any]) -> dict[str, Any]:
        redacted = self._redact_value(value)
        if not isinstance(redacted, dict):
            return {}
        return redacted

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    "[REDACTED]"
                    if self._is_redacted_field_name(key)
                    else self._redact_value(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        return value

    def _is_redacted_field_name(self, field_name: str) -> bool:
        normalized = self._normalize_field_name(field_name)
        return any(
            normalized == self._normalize_field_name(candidate)
            for candidate in self._policy.redacted_field_names
        )

    @staticmethod
    def _normalize_field_name(field_name: str) -> str:
        return field_name.casefold().replace("-", "_")

    @staticmethod
    def _make_error(
        *,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> ToolError:
        return ToolError(
            code=code,
            message=message,
            details=details or {},
        )

    def _make_runtime_error(
        self,
        state: _ExecutionState,
        exc: Exception,
    ) -> ToolError:
        return self._make_error(
            code=ErrorCode.RUNTIME_ERROR,
            message=f"Runtime failed while executing tool '{state.tool_name}'.",
            details={
                "tool_name": state.tool_name,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _to_timestamp(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _duration_ms(started_at: datetime, ended_at: datetime) -> int:
        duration = ended_at - started_at
        return max(int(duration.total_seconds() * 1000), 0)

    @staticmethod
    def _run_async_tool_in_sync(
        tool: Tool[Any, Any],
        context: ToolContext,
        validated_input: BaseModel,
    ) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(tool.ainvoke(context, validated_input))
        raise RuntimeError(
            "Cannot execute an async-only tool from ToolRuntime.execute() while an "
            "event loop is already running. Use ToolRuntime.execute_async() instead."
        )

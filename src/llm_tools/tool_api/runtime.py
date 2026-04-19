"""Strict runtime orchestration for tool execution."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from llm_tools.tool_api.errors import ToolNotRegisteredError
from llm_tools.tool_api.execution import (
    ExecutionServices,
    FilesystemBroker,
    HostToolContext,
    RuntimeInspection,
    SecretView,
    SubprocessBroker,
    ToolExecutionContext,
    _create_execution_context,
    _ExecutionPermit,
    _issue_execution_permit_for_context,
    build_bitbucket_gateway,
    build_confluence_gateway,
    build_gitlab_gateway,
    build_jira_gateway,
)
from llm_tools.tool_api.models import (
    ErrorCode,
    ExecutionRecord,
    PolicyDecision,
    SideEffectClass,
    SourceProvenanceRef,
    ToolError,
    ToolInvocationRequest,
    ToolResult,
    ToolSpec,
)
from llm_tools.tool_api.policy import ToolPolicy
from llm_tools.tool_api.redaction import RedactionSummary, RedactionTarget, Redactor
from llm_tools.tool_api.registry import ToolRegistry
from llm_tools.tool_api.tool import Tool


class _ToolTimeoutError(TimeoutError):
    """Internal timeout used to normalize runtime timeouts."""


class _ToolTimeoutInterrupt(BaseException):
    """Internal interrupt used to stop sync execution at the timeout boundary."""


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
    validated_output: dict[str, Any] | None = None
    redacted_output: dict[str, Any] | None = None
    retain_output_in_execution_record: bool = True
    policy_decision: PolicyDecision | None = None
    logs: list[str] | None = None
    artifacts: list[str] | None = None
    emitted_provenance: list[SourceProvenanceRef] | None = None
    redaction_summary: RedactionSummary | None = None
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

    def inspect_invocation(
        self,
        request: ToolInvocationRequest,
        context: HostToolContext,
        *,
        approval_override: bool = False,
    ) -> RuntimeInspection:
        """Return resolved tool metadata plus the current policy decision."""
        tool = self._resolve_tool(request)
        decision = self._policy_decision(
            tool.spec,
            context,
            approval_override=approval_override,
        )
        return RuntimeInspection(
            tool_name=tool.spec.name,
            tool_version=tool.spec.version,
            policy_decision=decision,
        )

    def execute(
        self,
        request: ToolInvocationRequest,
        context: HostToolContext,
        *,
        approval_override: bool = False,
    ) -> ToolResult:
        """Execute one tool invocation and return a normalized result."""
        state = _ExecutionState(
            invocation_id=context.invocation_id,
            request=request,
            started_at=self._utc_now(),
            tool_name=request.tool_name,
        )
        redactor = self._new_redactor(state.tool_name)
        state.redaction_summary = redactor.summary

        try:
            tool = self._resolve_tool(request)
        except ToolNotRegisteredError as exc:
            return self._finalize_failure(
                state,
                context,
                None,
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
                None,
                self._make_runtime_error(state, exc),
            )

        state.tool_name = tool.spec.name
        state.tool_version = tool.spec.version
        state.retain_output_in_execution_record = (
            tool.spec.retain_output_in_execution_record
        )
        redactor = self._new_redactor(state.tool_name)
        state.redaction_summary = redactor.summary

        try:
            policy_decision = self._policy_decision(
                tool.spec,
                context,
                approval_override=approval_override,
            )
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                None,
                self._make_runtime_error(state, exc),
            )

        state.policy_decision = policy_decision
        if not policy_decision.allowed:
            return self._finalize_failure(
                state,
                context,
                None,
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
                None,
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
                None,
                self._make_runtime_error(state, exc),
            )

        raw_validated_input = validated_input.model_dump(mode="json")
        state.redacted_input = redactor.redact_structured(
            raw_validated_input,
            target=RedactionTarget.INPUT,
        )
        if self._policy.redaction.retain_unredacted_inputs:
            state.validated_input = raw_validated_input

        try:
            execution_context = self._build_execution_context(tool.spec, context)
            raw_output = self._invoke_tool(tool, execution_context, validated_input)
        except _ToolTimeoutError as exc:
            return self._finalize_failure(
                state,
                context,
                None,
                self._make_error(
                    code=ErrorCode.TIMEOUT,
                    message=f"Tool '{state.tool_name}' execution timed out.",
                    details={
                        "tool_name": state.tool_name,
                        "timeout_seconds": tool.spec.timeout_seconds,
                        "exception_message": str(exc),
                    },
                ),
            )
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                execution_context if "execution_context" in locals() else None,
                self._make_execution_failed_error(state, tool, exc),
            )

        try:
            validated_output = self._validate_output(tool, raw_output)
        except ValidationError as exc:
            return self._finalize_failure(
                state,
                context,
                execution_context,
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
                execution_context,
                self._make_runtime_error(state, exc),
            )

        raw_validated_output = validated_output.model_dump(mode="json")
        state.redacted_output = redactor.redact_structured(
            raw_validated_output,
            target=RedactionTarget.OUTPUT,
        )
        if self._policy.redaction.retain_unredacted_outputs:
            state.validated_output = raw_validated_output

        return self._finalize_success(state, context, execution_context)

    async def execute_async(
        self,
        request: ToolInvocationRequest,
        context: HostToolContext,
        *,
        approval_override: bool = False,
    ) -> ToolResult:
        """Asynchronously execute one tool invocation and normalize the result."""
        state = _ExecutionState(
            invocation_id=context.invocation_id,
            request=request,
            started_at=self._utc_now(),
            tool_name=request.tool_name,
        )
        redactor = self._new_redactor(state.tool_name)
        state.redaction_summary = redactor.summary

        try:
            tool = self._resolve_tool(request)
        except ToolNotRegisteredError as exc:
            return self._finalize_failure(
                state,
                context,
                None,
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
                None,
                self._make_runtime_error(state, exc),
            )

        state.tool_name = tool.spec.name
        state.tool_version = tool.spec.version
        state.retain_output_in_execution_record = (
            tool.spec.retain_output_in_execution_record
        )
        redactor = self._new_redactor(state.tool_name)
        state.redaction_summary = redactor.summary

        try:
            policy_decision = self._policy_decision(
                tool.spec,
                context,
                approval_override=approval_override,
            )
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                None,
                self._make_runtime_error(state, exc),
            )

        state.policy_decision = policy_decision
        if not policy_decision.allowed:
            return self._finalize_failure(
                state,
                context,
                None,
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
                None,
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
                None,
                self._make_runtime_error(state, exc),
            )

        raw_validated_input = validated_input.model_dump(mode="json")
        state.redacted_input = redactor.redact_structured(
            raw_validated_input,
            target=RedactionTarget.INPUT,
        )
        if self._policy.redaction.retain_unredacted_inputs:
            state.validated_input = raw_validated_input

        try:
            execution_context = self._build_execution_context(tool.spec, context)
            raw_output = await self._invoke_tool_async(
                tool,
                execution_context,
                validated_input,
            )
        except _ToolTimeoutError as exc:
            return self._finalize_failure(
                state,
                context,
                None,
                self._make_error(
                    code=ErrorCode.TIMEOUT,
                    message=f"Tool '{state.tool_name}' execution timed out.",
                    details={
                        "tool_name": state.tool_name,
                        "timeout_seconds": tool.spec.timeout_seconds,
                        "exception_message": str(exc),
                    },
                ),
            )
        except Exception as exc:
            return self._finalize_failure(
                state,
                context,
                execution_context if "execution_context" in locals() else None,
                self._make_execution_failed_error(state, tool, exc),
            )

        try:
            validated_output = self._validate_output(tool, raw_output)
        except ValidationError as exc:
            return self._finalize_failure(
                state,
                context,
                execution_context,
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
                execution_context,
                self._make_runtime_error(state, exc),
            )

        raw_validated_output = validated_output.model_dump(mode="json")
        state.redacted_output = redactor.redact_structured(
            raw_validated_output,
            target=RedactionTarget.OUTPUT,
        )
        if self._policy.redaction.retain_unredacted_outputs:
            state.validated_output = raw_validated_output

        return self._finalize_success(state, context, execution_context)

    def _resolve_tool(self, request: ToolInvocationRequest) -> Tool[Any, Any]:
        return self._registry._resolve_tool(request.tool_name)

    def _policy_decision(
        self,
        spec: ToolSpec,
        context: HostToolContext,
        *,
        approval_override: bool,
    ) -> PolicyDecision:
        decision = self._policy.evaluate(spec, context)
        if approval_override and decision.requires_approval:
            return PolicyDecision(
                allowed=True,
                reason="approved",
                requires_approval=False,
                metadata={
                    **decision.metadata,
                    "approval_override": True,
                },
            )
        return decision

    def _build_execution_context(
        self,
        spec: ToolSpec,
        context: HostToolContext,
    ) -> ToolExecutionContext:
        scoped_secrets = SecretView(
            {
                name: context.env[name]
                for name in spec.required_secrets
                if name in context.env
            }
        )
        services = self._build_services(spec, context, scoped_secrets)
        return _create_execution_context(
            invocation_id=context.invocation_id,
            workspace=context.workspace,
            metadata=dict(context.metadata),
            secrets=scoped_secrets,
            services=services,
        )

    def _build_services(
        self,
        spec: ToolSpec,
        context: HostToolContext,
        secrets: SecretView,
    ) -> ExecutionServices:
        tags = set(spec.tags)
        required_secrets = set(spec.required_secrets)
        services = ExecutionServices()
        if spec.requires_filesystem and self._policy.allow_filesystem:
            services.filesystem = FilesystemBroker(context)
        if spec.requires_subprocess and self._policy.allow_subprocess:
            services.subprocess = SubprocessBroker(context)
        if spec.requires_network and self._policy.allow_network:
            if "gitlab" in tags or required_secrets.intersection(
                {"GITLAB_BASE_URL", "GITLAB_API_TOKEN"}
            ):
                services.gitlab = build_gitlab_gateway(secrets)
            if "jira" in tags or required_secrets.intersection(
                {"JIRA_BASE_URL", "JIRA_USERNAME", "JIRA_API_TOKEN"}
            ):
                services.jira = build_jira_gateway(secrets)
            if "bitbucket" in tags or required_secrets.intersection(
                {"BITBUCKET_BASE_URL", "BITBUCKET_USERNAME", "BITBUCKET_API_TOKEN"}
            ):
                services.bitbucket = build_bitbucket_gateway(secrets)
            if "confluence" in tags or required_secrets.intersection(
                {"CONFLUENCE_BASE_URL", "CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN"}
            ):
                services.confluence = build_confluence_gateway(secrets)
        return services

    def _validate_input(
        self,
        tool: Tool[Any, Any],
        arguments: dict[str, Any],
    ) -> BaseModel:
        return cast(BaseModel, tool.input_model.model_validate(arguments))

    def _invoke_tool(
        self,
        tool: Tool[Any, Any],
        context: ToolExecutionContext,
        validated_input: BaseModel,
    ) -> Any:
        permit = _issue_execution_permit_for_context(context)
        timeout_seconds = tool.spec.timeout_seconds
        if tool.__class__._has_sync_implementation():
            return self._invoke_sync_tool(
                tool,
                context,
                validated_input,
                permit,
                timeout_seconds=timeout_seconds,
            )
        if tool.__class__._has_async_implementation():
            return self._run_async_tool_in_sync(
                tool,
                context,
                validated_input,
                permit,
                timeout_seconds=timeout_seconds,
            )
        raise RuntimeError(
            f"Tool '{tool.spec.name}' has no synchronous or asynchronous implementation."
        )

    def _invoke_sync_tool(
        self,
        tool: Tool[Any, Any],
        context: ToolExecutionContext,
        validated_input: BaseModel,
        permit: _ExecutionPermit,
        *,
        timeout_seconds: int | None,
    ) -> Any:
        with self._sync_timeout_guard(timeout_seconds):
            return tool.invoke(context, validated_input, _permit=permit)

    async def _invoke_tool_async(
        self,
        tool: Tool[Any, Any],
        context: ToolExecutionContext,
        validated_input: BaseModel,
    ) -> Any:
        permit = _issue_execution_permit_for_context(context)
        timeout_seconds = tool.spec.timeout_seconds
        if tool.__class__._has_async_implementation():
            coroutine = tool.ainvoke(context, validated_input, _permit=permit)
            try:
                if timeout_seconds is None:
                    return await coroutine
                return await asyncio.wait_for(coroutine, timeout=timeout_seconds)
            except TimeoutError as exc:
                raise _ToolTimeoutError(
                    "Tool execution exceeded the configured timeout."
                ) from exc
        if tool.__class__._has_sync_implementation():
            if timeout_seconds is None:
                return await asyncio.to_thread(
                    tool.invoke,
                    context,
                    validated_input,
                    _permit=permit,
                )
            return self._invoke_sync_tool(
                tool,
                context,
                validated_input,
                permit,
                timeout_seconds=timeout_seconds,
            )
        raise RuntimeError(
            f"Tool '{tool.spec.name}' has no synchronous or asynchronous implementation."
        )

    def _validate_output(self, tool: Tool[Any, Any], output: Any) -> BaseModel:
        return cast(BaseModel, tool.output_model.model_validate(output))

    def _finalize_success(
        self,
        state: _ExecutionState,
        context: HostToolContext,
        execution_context: ToolExecutionContext,
    ) -> ToolResult:
        state.ok = True
        self._capture_execution_observability(state, execution_context)
        self._apply_observability_redaction(state)
        result = self._normalize_result(state, context=context, error=None)
        record = self._record_execution(state, context=context)
        result.metadata["execution_record"] = record.model_dump(mode="json")
        return result

    def _finalize_failure(
        self,
        state: _ExecutionState,
        context: HostToolContext,
        execution_context: ToolExecutionContext | None,
        error: ToolError,
    ) -> ToolResult:
        state.ok = False
        if execution_context is not None:
            self._capture_execution_observability(state, execution_context)
        redacted_error = self._redact_error_details(state, error)
        state.error_code = redacted_error.code
        self._apply_observability_redaction(state)
        result = self._normalize_result(state, context=context, error=redacted_error)
        record = self._record_execution(state, context=context)
        result.metadata["execution_record"] = record.model_dump(mode="json")
        return result

    def _normalize_result(
        self,
        state: _ExecutionState,
        *,
        context: HostToolContext,
        error: ToolError | None,
    ) -> ToolResult:
        return ToolResult(
            ok=error is None,
            tool_name=state.tool_name,
            tool_version=state.tool_version,
            output=state.redacted_output,
            error=error,
            logs=list(state.logs or []),
            artifacts=list(state.artifacts or []),
            source_provenance=self._combined_provenance(context, state),
        )

    def _record_execution(
        self,
        state: _ExecutionState,
        *,
        context: HostToolContext,
    ) -> ExecutionRecord:
        ended_at = self._utc_now()
        record_request = state.request.model_copy(
            update={"arguments": state.redacted_input or {}}
        )
        return ExecutionRecord(
            invocation_id=state.invocation_id,
            tool_name=state.tool_name,
            tool_version=state.tool_version,
            started_at=self._to_timestamp(state.started_at),
            ended_at=self._to_timestamp(ended_at),
            duration_ms=self._duration_ms(state.started_at, ended_at),
            request=record_request,
            validated_input=state.validated_input,
            redacted_input=state.redacted_input,
            validated_output=(
                state.validated_output
                if state.retain_output_in_execution_record
                else None
            ),
            redacted_output=(
                state.redacted_output
                if state.retain_output_in_execution_record
                else None
            ),
            ok=state.ok,
            error_code=state.error_code,
            policy_decision=state.policy_decision,
            logs=list(state.logs or []),
            artifacts=list(state.artifacts or []),
            source_provenance=self._combined_provenance(context, state),
            metadata={
                "redaction": self._redaction_metadata(state),
            },
        )

    def _capture_execution_observability(
        self,
        state: _ExecutionState,
        context: ToolExecutionContext,
    ) -> None:
        state.logs = context.snapshot_logs()
        state.artifacts = context.snapshot_artifacts()
        state.emitted_provenance = context.snapshot_source_provenance()

    def _apply_observability_redaction(self, state: _ExecutionState) -> None:
        redactor = self._new_redactor(state.tool_name)
        if state.redaction_summary is not None:
            redactor.summary.matched_targets.update(
                state.redaction_summary.matched_targets
            )
            redactor.summary.matched_paths.update(state.redaction_summary.matched_paths)
            redactor.summary.applied_rule_count += (
                state.redaction_summary.applied_rule_count
            )

        state.logs = redactor.redact_string_entries(
            list(state.logs or []),
            target=RedactionTarget.LOGS,
        )
        state.artifacts = redactor.redact_string_entries(
            list(state.artifacts or []),
            target=RedactionTarget.ARTIFACTS,
        )
        state.redaction_summary = redactor.summary

    def _redact_error_details(
        self,
        state: _ExecutionState,
        error: ToolError,
    ) -> ToolError:
        if not error.details:
            return error

        redactor = self._new_redactor(state.tool_name)
        if state.redaction_summary is not None:
            redactor.summary.matched_targets.update(
                state.redaction_summary.matched_targets
            )
            redactor.summary.matched_paths.update(state.redaction_summary.matched_paths)
            redactor.summary.applied_rule_count += (
                state.redaction_summary.applied_rule_count
            )
        redacted_details = redactor.redact_structured(
            error.details,
            target=RedactionTarget.ERROR_DETAILS,
        )
        state.redaction_summary = redactor.summary
        return error.model_copy(update={"details": redacted_details})

    def _combined_provenance(
        self,
        context: HostToolContext,
        state: _ExecutionState,
    ) -> list[SourceProvenanceRef]:
        base = [entry.model_copy(deep=True) for entry in context.source_provenance]
        emitted = [
            entry.model_copy(deep=True) for entry in state.emitted_provenance or []
        ]
        return [*base, *emitted]

    def _redaction_metadata(self, state: _ExecutionState) -> dict[str, object]:
        summary = state.redaction_summary or RedactionSummary()
        return summary.as_metadata(
            retain_unredacted_inputs=self._policy.redaction.retain_unredacted_inputs,
            retain_unredacted_outputs=self._policy.redaction.retain_unredacted_outputs,
        )

    @staticmethod
    @contextmanager
    def _sync_timeout_guard(timeout_seconds: int | None) -> Any:
        if timeout_seconds is None:
            yield
            return

        if (
            os.name != "posix"
            or threading.current_thread() is not threading.main_thread()
        ):
            raise RuntimeError(
                "Synchronous tool timeouts require execution on the main thread of a POSIX runtime."
            )

        previous_handler = signal.getsignal(signal.SIGALRM)

        def _handle_timeout(signum: int, frame: object | None) -> None:
            del signum, frame
            raise _ToolTimeoutInterrupt()

        signal.signal(signal.SIGALRM, _handle_timeout)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
        try:
            yield
        except _ToolTimeoutInterrupt as exc:
            raise _ToolTimeoutError(
                "Tool execution exceeded the configured timeout."
            ) from exc
        finally:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
            signal.signal(signal.SIGALRM, previous_handler)

    def _new_redactor(self, tool_name: str) -> Redactor:
        return Redactor(self._policy.redaction, tool_name=tool_name)

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

    def _make_execution_failed_error(
        self,
        state: _ExecutionState,
        tool: Tool[Any, Any],
        exc: Exception,
    ) -> ToolError:
        details: dict[str, Any] = {
            "tool_name": state.tool_name,
            "exception_type": type(exc).__name__,
        }
        if self._should_suppress_exception_message(tool):
            details["failure_reason"] = "filesystem_target_invalid_or_unavailable"
        else:
            details["exception_message"] = str(exc)
        return self._make_error(
            code=ErrorCode.EXECUTION_FAILED,
            message=f"Tool '{state.tool_name}' execution failed.",
            details=details,
        )

    @staticmethod
    def _should_suppress_exception_message(tool: Tool[Any, Any]) -> bool:
        spec = tool.spec
        return (
            spec.requires_filesystem
            and spec.side_effects
            in {
                SideEffectClass.LOCAL_READ,
                SideEffectClass.LOCAL_WRITE,
            }
            and bool({"filesystem", "text"}.intersection(spec.tags))
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
        context: ToolExecutionContext,
        validated_input: BaseModel,
        permit: _ExecutionPermit,
        timeout_seconds: int | None,
    ) -> Any:
        async def _execute() -> Any:
            coroutine = tool.ainvoke(context, validated_input, _permit=permit)
            if timeout_seconds is None:
                return await coroutine
            return await asyncio.wait_for(coroutine, timeout=timeout_seconds)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(_execute())
            except TimeoutError as exc:
                raise _ToolTimeoutError(
                    "Tool execution exceeded the configured timeout."
                ) from exc
        raise RuntimeError(
            "Cannot execute an async-only tool from ToolRuntime.execute() while an "
            "event loop is already running. Use ToolRuntime.execute_async() instead."
        )

"""Thin workflow bridge for one model turn of parsing and execution."""

from __future__ import annotations

from pydantic import BaseModel

from llm_tools.llm_adapters import (
    ModelOutputParsingAdapter,
    ParsedModelResponse,
    ToolExposureAdapter,
)
from llm_tools.tool_api import (
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolRuntime,
    ToolSpec,
)
from llm_tools.workflow_api.models import WorkflowTurnResult


class WorkflowExecutor:
    """Bridge adapters and runtime for one parsed model turn."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy | None = None,
    ) -> None:
        self._registry = registry
        self._runtime = ToolRuntime(registry, policy=policy)

    def export_tools(self, adapter: ToolExposureAdapter) -> object:
        """Export registered tools through the supplied adapter."""
        tools = self._registry.list_registered_tools()
        specs: list[ToolSpec] = [tool.spec for tool in tools]
        input_models: dict[str, type[BaseModel]] = {
            tool.spec.name: tool.input_model for tool in tools
        }
        return adapter.export_tool_descriptions(specs, input_models)

    def execute_parsed_response(
        self,
        parsed_response: ParsedModelResponse,
        context: ToolContext,
    ) -> WorkflowTurnResult:
        """Execute parsed invocations or return a final-response-only result."""
        if parsed_response.final_response is not None:
            return WorkflowTurnResult(parsed_response=parsed_response)

        tool_results: list[ToolResult] = []
        invocation_count = len(parsed_response.invocations)
        for index, request in enumerate(parsed_response.invocations, start=1):
            tool_context = self._make_tool_context(
                base_context=context,
                index=index,
                invocation_count=invocation_count,
            )
            tool_results.append(self._runtime.execute(request, tool_context))

        return WorkflowTurnResult(
            parsed_response=parsed_response,
            tool_results=tool_results,
        )

    def execute_model_output(
        self,
        adapter: ModelOutputParsingAdapter,
        payload: object,
        context: ToolContext,
    ) -> WorkflowTurnResult:
        """Parse one model output and execute any resulting tool invocations."""
        parsed_response = adapter.parse_model_output(payload)
        return self.execute_parsed_response(parsed_response, context)

    def _make_tool_context(
        self,
        *,
        base_context: ToolContext,
        index: int,
        invocation_count: int,
    ) -> ToolContext:
        invocation_id = base_context.invocation_id
        if invocation_count > 1:
            invocation_id = f"{invocation_id}:{index}"

        return ToolContext(
            invocation_id=invocation_id,
            workspace=base_context.workspace,
            env=dict(base_context.env),
            logs=[],
            artifacts=[],
            metadata=dict(base_context.metadata),
        )

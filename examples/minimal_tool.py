"""Minimal custom-tool example for llm-tools."""

from __future__ import annotations

from pydantic import BaseModel

from llm_tools.tool_api import Tool, ToolContext, ToolExecutionContext, ToolInvocationRequest, ToolRegistry, ToolRuntime, ToolSpec


class EchoInput(BaseModel):
    value: str


class EchoOutput(BaseModel):
    echoed: str


class EchoTool(Tool[EchoInput, EchoOutput]):
    spec = ToolSpec(
        name="echo",
        description="Return the provided value.",
        tags=["example"],
    )
    input_model = EchoInput
    output_model = EchoOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: EchoInput
    ) -> EchoOutput:
        context.log("Echo tool invoked.")
        return EchoOutput(echoed=f"{context.invocation_id}:{args.value}")


def main() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    runtime = ToolRuntime(registry)

    result = runtime.execute(
        ToolInvocationRequest(tool_name="echo", arguments={"value": "hello"}),
        ToolContext(invocation_id="example-1"),
    )

    print("Echo result:", result.model_dump(mode="json"))


if __name__ == "__main__":
    main()

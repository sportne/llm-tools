"""Structured-response adapter example."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from llm_tools.llm_adapters import StructuredResponseAdapter
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools import register_filesystem_tools, register_text_tools
from llm_tools.workflow_api import WorkflowExecutor


def main() -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    adapter = StructuredResponseAdapter()
    executor = WorkflowExecutor(
        registry,
        policy=ToolPolicy(
            allowed_side_effects={
                SideEffectClass.NONE,
                SideEffectClass.LOCAL_READ,
                SideEffectClass.LOCAL_WRITE,
            }
        ),
    )

    print("Structured schema:", executor.export_tools(adapter))

    with TemporaryDirectory() as workspace:
        setup = executor.execute_model_output(
            adapter,
            {
                "actions": [
                    {
                        "tool_name": "write_file",
                        "arguments": {
                            "path": "docs/demo.txt",
                            "content": "hello structured response",
                            "create_parents": True,
                        },
                    }
                ]
            },
            ToolContext(invocation_id="structured-setup", workspace=workspace),
        )
        action_turn = executor.execute_model_output(
            adapter,
            {
                "actions": [
                    {
                        "tool_name": "read_file",
                        "arguments": {"path": "docs/demo.txt"},
                    }
                ]
            },
            ToolContext(invocation_id="structured-turn", workspace=workspace),
        )
        final_turn = executor.execute_model_output(
            adapter,
            {"final_response": "Already handled without a tool."},
            ToolContext(invocation_id="structured-final", workspace=workspace),
        )

        print("Setup turn:", setup.model_dump(mode="json"))
        print("Action turn:", action_turn.model_dump(mode="json"))
        print("Final turn:", final_turn.model_dump(mode="json"))


if __name__ == "__main__":
    main()

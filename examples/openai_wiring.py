"""Offline OpenAI adapter wiring example."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from llm_tools.llm_adapters import OpenAIToolCallingAdapter
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolInvocationRequest, ToolPolicy, ToolRegistry
from llm_tools.tools import register_filesystem_tools
from llm_tools.workflow_api import WorkflowExecutor


def main() -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    adapter = OpenAIToolCallingAdapter()
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

    tools = executor.export_tools(adapter)
    print("Exported OpenAI tools:", tools)

    with TemporaryDirectory() as workspace:
        setup = executor.execute_parsed_response(
            adapter.parse_model_output(
                {
                    "tool_calls": [
                        {
                            "id": "call_write",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path":"README.txt","content":"hello from openai wiring"}',
                            },
                        }
                    ]
                }
            ),
            ToolContext(invocation_id="openai-setup", workspace=workspace),
        )
        read_turn = executor.execute_model_output(
            adapter,
            {
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.txt"}',
                        },
                    }
                ]
            },
            ToolContext(invocation_id="openai-turn", workspace=workspace),
        )
        final_turn = executor.execute_model_output(
            adapter,
            {"content": "No tool needed."},
            ToolContext(invocation_id="openai-final", workspace=workspace),
        )

        print("Setup turn:", setup.model_dump(mode="json"))
        print("Read turn:", read_turn.model_dump(mode="json"))
        print("Final turn:", final_turn.model_dump(mode="json"))


if __name__ == "__main__":
    main()

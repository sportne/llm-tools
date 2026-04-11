"""Prompt-schema adapter example."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from llm_tools.llm_adapters import PromptSchemaAdapter, StructuredResponseAdapter
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools import register_filesystem_tools
from llm_tools.workflow_api import WorkflowExecutor


def main() -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    prompt_adapter = PromptSchemaAdapter()
    setup_adapter = StructuredResponseAdapter()
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

    print("Prompt instructions:")
    print(executor.export_tools(prompt_adapter))

    with TemporaryDirectory() as workspace:
        setup = executor.execute_model_output(
            setup_adapter,
            {
                "actions": [
                    {
                        "tool_name": "write_file",
                        "arguments": {
                            "path": "prompt/demo.txt",
                            "content": "hello prompt schema",
                            "create_parents": True,
                        },
                    }
                ]
            },
            ToolContext(invocation_id="prompt-setup", workspace=workspace),
        )
        action_turn = executor.execute_model_output(
            prompt_adapter,
            """```json
{"actions":[{"tool_name":"read_file","arguments":{"path":"prompt/demo.txt"}}]}
```""",
            ToolContext(invocation_id="prompt-turn", workspace=workspace),
        )
        final_turn = executor.execute_model_output(
            prompt_adapter,
            '{"final_response":"No tool required."}',
            ToolContext(invocation_id="prompt-final", workspace=workspace),
        )

        print("Setup turn:", setup.model_dump(mode="json"))
        print("Action turn:", action_turn.model_dump(mode="json"))
        print("Final turn:", final_turn.model_dump(mode="json"))


if __name__ == "__main__":
    main()

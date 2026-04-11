"""Live OpenAI API example using the OpenAI Python SDK."""

from __future__ import annotations

import os
import sys
from tempfile import TemporaryDirectory

from openai import OpenAI

from llm_tools.llm_adapters import OpenAIToolCallingAdapter
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools import register_filesystem_tools
from llm_tools.workflow_api import WorkflowExecutor


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required to run examples/openai_live.py.", file=sys.stderr)
        return 1

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

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

    with TemporaryDirectory() as workspace:
        setup = executor.execute_model_output(
            adapter,
            {
                "tool_calls": [
                    {
                        "id": "setup_call",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path":"notes/demo.txt","content":"hello from llm-tools","create_parents":true}',
                        },
                    }
                ]
            },
            ToolContext(invocation_id="live-setup", workspace=workspace),
        )
        print("Setup turn:", setup.model_dump(mode="json"))

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Read notes/demo.txt and either call a tool or answer directly.",
                }
            ],
            tools=executor.export_tools(adapter),
        )

        turn_result = executor.execute_model_output(
            adapter,
            response.choices[0].message,
            ToolContext(invocation_id="live-turn", workspace=workspace),
        )
        print("Model turn:", turn_result.model_dump(mode="json"))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

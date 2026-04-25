"""Live Ollama example using the low-level OpenAI-compatible provider layer."""

from __future__ import annotations

import os
import sys
from tempfile import TemporaryDirectory

from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools import register_filesystem_tools
from llm_tools.workflow_api import WorkflowExecutor


def main() -> int:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("OLLAMA_MODEL", "gemma4:26b")

    provider = OpenAICompatibleProvider.for_ollama(
        model=model,
        base_url=base_url,
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
    )

    registry = ToolRegistry()
    register_filesystem_tools(registry)
    adapter = ActionEnvelopeAdapter()
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
        context = ToolContext(invocation_id="live-turn", workspace=workspace)
        prepared = executor.prepare_model_interaction(
            adapter,
            context=context,
            include_requires_approval=True,
        )
        setup = executor.execute_model_output(
            adapter,
            {
                "actions": [
                    {
                        "tool_name": "write_file",
                        "arguments": {
                            "path": "notes/demo.txt",
                            "content": "hello from llm-tools",
                            "create_parents": True,
                        },
                    }
                ]
            },
            ToolContext(invocation_id="live-setup", workspace=workspace),
        )
        print("Setup turn:", setup.model_dump(mode="json"))

        try:
            parsed = provider.run(
                adapter=adapter,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Read notes/demo.txt and either call a tool or answer "
                            "directly."
                        ),
                    }
                ],
                response_model=prepared.response_model,
            )
        except Exception as exc:
            print(
                (
                    "Failed to reach Ollama at "
                    f"{base_url!r} with model {model!r}: {exc}"
                ),
                file=sys.stderr,
            )
            print(
                "Make sure Ollama is running and the requested model is available.",
                file=sys.stderr,
            )
            return 1

        turn_result = executor.execute_parsed_response(
            parsed,
            context,
        )
        print("Model turn:", turn_result.model_dump(mode="json"))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

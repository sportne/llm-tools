"""Offline async model-turn example using provider + workflow async APIs."""

from __future__ import annotations

import asyncio
from tempfile import TemporaryDirectory

from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools import register_filesystem_tools
from llm_tools.workflow_api import WorkflowExecutor


class _FakeAsyncCompletions:
    def __init__(self, response: object) -> None:
        self._response = response

    async def create(self, **kwargs: object) -> object:
        del kwargs
        return self._response


class _FakeAsyncClient:
    def __init__(self, response: object) -> None:
        self.chat = type(
            "_FakeChat",
            (),
            {"completions": _FakeAsyncCompletions(response)},
        )()


async def _run() -> None:
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

    provider = OpenAICompatibleProvider(
        model="demo-model",
        async_client=_FakeAsyncClient(
            {
                "actions": [{"tool_name": "read_file", "arguments": {"path": "README.txt"}}],
                "final_response": None,
            }
        ),
    )

    with TemporaryDirectory() as workspace:
        context = ToolContext(invocation_id="async-turn", workspace=workspace)
        prepared = executor.prepare_model_interaction(
            adapter,
            context=context,
            include_requires_approval=True,
        )
        setup = await executor.execute_model_output_async(
            adapter,
            {
                "actions": [
                    {
                        "tool_name": "write_file",
                        "arguments": {
                            "path": "README.txt",
                            "content": "hello from async model turn",
                        },
                    }
                ]
            },
            ToolContext(invocation_id="async-setup", workspace=workspace),
        )
        parsed = await provider.run_async(
            adapter=adapter,
            messages=[{"role": "user", "content": "Read README.txt"}],
            response_model=prepared.response_model,
        )
        result = await executor.execute_parsed_response_async(
            parsed,
            context,
        )

    print("Setup turn:", setup.model_dump(mode="json"))
    print("Async turn:", result.model_dump(mode="json"))


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

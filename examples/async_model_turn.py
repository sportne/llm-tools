"""Offline async model-turn example using provider + workflow async APIs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from tempfile import TemporaryDirectory

from llm_tools.llm_adapters import NativeToolCallingAdapter, StructuredOutputAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools import register_filesystem_tools
from llm_tools.workflow_api import WorkflowExecutor


@dataclass
class _FakeChoice:
    message: object


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


class _FakeAsyncCompletions:
    def __init__(self, response: object) -> None:
        self._response = response

    async def create(self, **kwargs: object) -> _FakeResponse:
        del kwargs
        return _FakeResponse(choices=[_FakeChoice(message=self._response)])


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
    adapter = NativeToolCallingAdapter()
    setup_adapter = StructuredOutputAdapter()
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
            }
        ),
    )

    with TemporaryDirectory() as workspace:
        setup = await executor.execute_model_output_async(
            setup_adapter,
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
        parsed = await provider.run_native_tool_calling_async(
            adapter=adapter,
            messages=[{"role": "user", "content": "Read README.txt"}],
            registry=registry,
        )
        result = await executor.execute_parsed_response_async(
            parsed,
            ToolContext(invocation_id="async-turn", workspace=workspace),
        )

    print("Setup turn:", setup.model_dump(mode="json"))
    print("Async turn:", result.model_dump(mode="json"))


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

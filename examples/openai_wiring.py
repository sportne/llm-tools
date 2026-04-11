"""Offline native-tool-calling example using the provider layer."""

from __future__ import annotations

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


class _FakeCompletions:
    def __init__(self, response: object) -> None:
        self._response = response

    def create(self, **kwargs: object) -> _FakeResponse:
        del kwargs
        return _FakeResponse(choices=[_FakeChoice(message=self._response)])


class _FakeClient:
    def __init__(self, response: object) -> None:
        self.chat = type(
            "_FakeChat",
            (),
            {"completions": _FakeCompletions(response)},
        )()


def main() -> None:
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
        client=_FakeClient(
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
    final_provider = OpenAICompatibleProvider(
        model="demo-model",
        client=_FakeClient({"content": "No tool needed."}),
    )

    print("Exported native tool definitions:", executor.export_tools(adapter))

    with TemporaryDirectory() as workspace:
        setup = executor.execute_model_output(
            setup_adapter,
            {
                "actions": [
                    {
                        "tool_name": "write_file",
                        "arguments": {
                            "path": "README.txt",
                            "content": "hello from openai-compatible wiring",
                        },
                    }
                ]
            },
            ToolContext(invocation_id="openai-setup", workspace=workspace),
        )
        parsed = provider.run_native_tool_calling(
            adapter=adapter,
            messages=[{"role": "user", "content": "Read README.txt"}],
            registry=registry,
        )
        read_turn = executor.execute_parsed_response(
            parsed,
            ToolContext(invocation_id="openai-turn", workspace=workspace),
        )
        final_turn = executor.execute_parsed_response(
            final_provider.run_native_tool_calling(
                adapter=adapter,
                messages=[{"role": "user", "content": "Reply directly"}],
                registry=registry,
            ),
            ToolContext(invocation_id="openai-final", workspace=workspace),
        )

        print("Setup turn:", setup.model_dump(mode="json"))
        print("Read turn:", read_turn.model_dump(mode="json"))
        print("Final turn:", final_turn.model_dump(mode="json"))


if __name__ == "__main__":
    main()

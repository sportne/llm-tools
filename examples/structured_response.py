"""Structured-output example using the provider layer."""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory

from llm_tools.llm_adapters import StructuredOutputAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools import register_filesystem_tools, register_text_tools
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
    register_text_tools(registry)
    adapter = StructuredOutputAdapter()
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

    print("Structured output schema:", executor.export_tools(adapter))

    with TemporaryDirectory() as workspace:
        setup = executor.execute_model_output(
            adapter,
            {
                "actions": [
                    {
                        "tool_name": "write_file",
                        "arguments": {
                            "path": "docs/demo.txt",
                            "content": "hello structured output",
                            "create_parents": True,
                        },
                    }
                ]
            },
            ToolContext(invocation_id="structured-setup", workspace=workspace),
        )
        provider = OpenAICompatibleProvider(
            model="demo-model",
            client=_FakeClient(
                '{"actions":[{"tool_name":"read_file","arguments":{"path":"docs/demo.txt"}}]}'
            ),
        )
        action_turn = executor.execute_parsed_response(
            provider.run_structured_output(
                adapter=adapter,
                messages=[{"role": "user", "content": "Read docs/demo.txt"}],
                registry=registry,
            ),
            ToolContext(invocation_id="structured-turn", workspace=workspace),
        )
        final_provider = OpenAICompatibleProvider(
            model="demo-model",
            client=_FakeClient('{"final_response":"Already handled without a tool."}'),
        )
        final_turn = executor.execute_parsed_response(
            final_provider.run_structured_output(
                adapter=adapter,
                messages=[{"role": "user", "content": "Reply directly"}],
                registry=registry,
            ),
            ToolContext(invocation_id="structured-final", workspace=workspace),
        )

        print("Setup turn:", setup.model_dump(mode="json"))
        print("Action turn:", action_turn.model_dump(mode="json"))
        print("Final turn:", final_turn.model_dump(mode="json"))


if __name__ == "__main__":
    main()

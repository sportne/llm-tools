"""Prompt-schema example using the provider layer."""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory

from llm_tools.llm_adapters import PromptSchemaAdapter, StructuredOutputAdapter
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
    prompt_adapter = PromptSchemaAdapter()
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
        provider = OpenAICompatibleProvider(
            model="demo-model",
            client=_FakeClient(
                """```json
{"actions":[{"tool_name":"read_file","arguments":{"path":"prompt/demo.txt"}}]}
```"""
            ),
        )
        action_turn = executor.execute_parsed_response(
            provider.run_prompt_schema(
                adapter=prompt_adapter,
                messages=[{"role": "user", "content": "Read prompt/demo.txt"}],
                registry=registry,
            ),
            ToolContext(invocation_id="prompt-turn", workspace=workspace),
        )
        final_provider = OpenAICompatibleProvider(
            model="demo-model",
            client=_FakeClient('{"final_response":"No tool required."}'),
        )
        final_turn = executor.execute_parsed_response(
            final_provider.run_prompt_schema(
                adapter=prompt_adapter,
                messages=[{"role": "user", "content": "Reply directly"}],
                registry=registry,
            ),
            ToolContext(invocation_id="prompt-final", workspace=workspace),
        )

        print("Setup turn:", setup.model_dump(mode="json"))
        print("Action turn:", action_turn.model_dump(mode="json"))
        print("Final turn:", final_turn.model_dump(mode="json"))


if __name__ == "__main__":
    main()

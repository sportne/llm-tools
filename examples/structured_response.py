"""Action-envelope example using the provider layer."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools import register_filesystem_tools, register_text_tools
from llm_tools.workflow_api import WorkflowExecutor


class _FakeCompletions:
    def __init__(self, response: object) -> None:
        self._response = response

    def create(self, **kwargs: object) -> object:
        del kwargs
        return self._response


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
        context = ToolContext(invocation_id="structured-turn", workspace=workspace)
        prepared = executor.prepare_model_interaction(
            adapter,
            context=context,
            include_requires_approval=True,
        )
        print("Action-envelope schema:", prepared.schema)

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
                {
                    "actions": [
                        {"tool_name": "read_file", "arguments": {"path": "docs/demo.txt"}}
                    ],
                    "final_response": None,
                }
            ),
        )
        action_turn = executor.execute_parsed_response(
            provider.run(
                adapter=adapter,
                messages=[{"role": "user", "content": "Read docs/demo.txt"}],
                response_model=prepared.response_model,
            ),
            context,
        )
        final_provider = OpenAICompatibleProvider(
            model="demo-model",
            client=_FakeClient(
                {
                    "actions": [],
                    "final_response": "Already handled without a tool.",
                }
            ),
        )
        final_turn = executor.execute_parsed_response(
            final_provider.run(
                adapter=adapter,
                messages=[{"role": "user", "content": "Reply directly"}],
                response_model=prepared.response_model,
            ),
            ToolContext(invocation_id="structured-final", workspace=workspace),
        )

        print("Setup turn:", setup.model_dump(mode="json"))
        print("Action turn:", action_turn.model_dump(mode="json"))
        print("Final turn:", final_turn.model_dump(mode="json"))


if __name__ == "__main__":
    main()

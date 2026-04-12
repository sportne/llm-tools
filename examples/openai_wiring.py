"""Offline action-envelope wiring example using the provider layer."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools import register_filesystem_tools
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
        client=_FakeClient(
            {
                "actions": [{"tool_name": "read_file", "arguments": {"path": "README.txt"}}],
                "final_response": None,
            }
        ),
    )
    final_provider = OpenAICompatibleProvider(
        model="demo-model",
        client=_FakeClient({"actions": [], "final_response": "No tool needed."}),
    )

    with TemporaryDirectory() as workspace:
        context = ToolContext(invocation_id="openai-turn", workspace=workspace)
        prepared = executor.prepare_model_interaction(
            adapter,
            context=context,
            include_requires_approval=True,
        )
        print("Exported action-envelope schema:", prepared.schema)

        setup = executor.execute_model_output(
            adapter,
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
        parsed = provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "Read README.txt"}],
            response_model=prepared.response_model,
        )
        read_turn = executor.execute_parsed_response(
            parsed,
            ToolContext(invocation_id="openai-turn", workspace=workspace),
        )
        final_turn = executor.execute_parsed_response(
            final_provider.run(
                adapter=adapter,
                messages=[{"role": "user", "content": "Reply directly"}],
                response_model=prepared.response_model,
            ),
            ToolContext(invocation_id="openai-final", workspace=workspace),
        )

        print("Setup turn:", setup.model_dump(mode="json"))
        print("Read turn:", read_turn.model_dump(mode="json"))
        print("Final turn:", final_turn.model_dump(mode="json"))


if __name__ == "__main__":
    main()

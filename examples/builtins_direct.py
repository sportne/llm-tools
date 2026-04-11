"""Direct built-in tool usage example."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from llm_tools.tool_api import SideEffectClass, ToolContext, ToolInvocationRequest, ToolPolicy, ToolRegistry, ToolRuntime
from llm_tools.tools import register_filesystem_tools, register_text_tools


def main() -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)

    runtime = ToolRuntime(
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
        write_result = runtime.execute(
            ToolInvocationRequest(
                tool_name="write_file",
                arguments={
                    "path": "notes/todo.txt",
                    "content": "buy milk\ncall mom",
                    "create_parents": True,
                },
            ),
            ToolContext(invocation_id="builtins-write", workspace=workspace),
        )
        read_result = runtime.execute(
            ToolInvocationRequest(
                tool_name="read_file",
                arguments={"path": "notes/todo.txt"},
            ),
            ToolContext(invocation_id="builtins-read", workspace=workspace),
        )
        search_result = runtime.execute(
            ToolInvocationRequest(
                tool_name="directory_text_search",
                arguments={"path": ".", "query": "milk"},
            ),
            ToolContext(invocation_id="builtins-search", workspace=workspace),
        )

        print("Write result:", write_result.model_dump(mode="json"))
        print("Read result:", read_result.model_dump(mode="json"))
        print("Search result:", search_result.model_dump(mode="json"))


if __name__ == "__main__":
    main()

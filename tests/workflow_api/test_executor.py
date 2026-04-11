"""End-to-end integration tests for the workflow executor."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType

import pytest
from pydantic import ValidationError

import llm_tools.tools.git.tools as git_tools
from llm_tools.llm_adapters import (
    OpenAIToolCallingAdapter,
    ParsedModelResponse,
    PromptSchemaAdapter,
    StructuredResponseAdapter,
)
from llm_tools.tool_api import (
    ErrorCode,
    SideEffectClass,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
)
from llm_tools.tools.atlassian import register_atlassian_tools
from llm_tools.tools.filesystem import register_filesystem_tools
from llm_tools.tools.git import register_git_tools
from llm_tools.tools.text import register_text_tools
from llm_tools.workflow_api import WorkflowExecutor, WorkflowTurnResult


def _executor(registry: ToolRegistry, *, allow_write: bool = False) -> WorkflowExecutor:
    allowed_side_effects = {
        SideEffectClass.NONE,
        SideEffectClass.LOCAL_READ,
        SideEffectClass.EXTERNAL_READ,
    }
    if allow_write:
        allowed_side_effects.add(SideEffectClass.LOCAL_WRITE)

    return WorkflowExecutor(
        registry,
        policy=ToolPolicy(allowed_side_effects=allowed_side_effects),
    )


def test_workflow_turn_result_rejects_mismatched_final_response_and_tool_results() -> (
    None
):
    with pytest.raises(ValidationError):
        WorkflowTurnResult(
            parsed_response=ParsedModelResponse(final_response="done"),
            tool_results=[
                ToolResult(
                    ok=True,
                    tool_name="read_file",
                    tool_version="0.1.0",
                    output={},
                )
            ],
        )


def test_workflow_turn_result_rejects_mismatched_invocation_and_result_counts() -> None:
    with pytest.raises(ValidationError):
        WorkflowTurnResult(
            parsed_response=ParsedModelResponse(
                invocations=[
                    {
                        "tool_name": "read_file",
                        "arguments": {},
                    },
                    {
                        "tool_name": "list_directory",
                        "arguments": {},
                    },
                ]
            ),
            tool_results=[
                ToolResult(
                    ok=True,
                    tool_name="read_file",
                    tool_version="0.1.0",
                    output={},
                )
            ],
        )


@pytest.mark.parametrize(
    ("adapter", "expected_type"),
    [
        (OpenAIToolCallingAdapter(), list),
        (StructuredResponseAdapter(), dict),
        (PromptSchemaAdapter(), str),
    ],
)
def test_workflow_executor_exports_registered_tools(
    adapter: object, expected_type: type[object]
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    executor = _executor(registry)

    exported = executor.export_tools(adapter)  # type: ignore[arg-type]

    assert isinstance(exported, expected_type)
    if isinstance(exported, list):
        assert [tool["function"]["name"] for tool in exported] == [
            "read_file",
            "write_file",
            "list_directory",
            "file_text_search",
            "directory_text_search",
        ]
    elif isinstance(exported, dict):
        assert exported["properties"]["actions"]["items"]["properties"]["tool_name"][
            "enum"
        ] == [
            "read_file",
            "write_file",
            "list_directory",
            "file_text_search",
            "directory_text_search",
        ]
    else:
        assert "read_file" in exported
        assert "directory_text_search" in exported


def test_workflow_executor_returns_final_response_without_tool_execution(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    result = executor.execute_parsed_response(
        ParsedModelResponse(final_response="All set."),
        ToolContext(
            invocation_id="turn-1",
            workspace=str(tmp_path),
            logs=["turn-log"],
            artifacts=["turn-artifact"],
        ),
    )

    assert result.parsed_response.final_response == "All set."
    assert result.tool_results == []


def test_workflow_executor_executes_single_parsed_invocation(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry, allow_write=True)
    context = ToolContext(invocation_id="turn-2", workspace=str(tmp_path))

    write_result = executor.execute_model_output(
        StructuredResponseAdapter(),
        {
            "actions": [
                {
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "docs/note.txt",
                        "content": "hello",
                        "create_parents": True,
                    },
                }
            ]
        },
        context,
    )

    assert len(write_result.tool_results) == 1
    assert write_result.tool_results[0].ok is True
    record = write_result.tool_results[0].metadata["execution_record"]
    assert record["invocation_id"] == "turn-2"


def test_workflow_executor_executes_multiple_invocations_sequentially(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry, allow_write=True)

    setup_result = executor.execute_model_output(
        StructuredResponseAdapter(),
        {
            "actions": [
                {
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "docs/note.txt",
                        "content": "hello world",
                        "create_parents": True,
                    },
                }
            ]
        },
        ToolContext(invocation_id="setup", workspace=str(tmp_path)),
    )
    assert setup_result.tool_results[0].ok is True

    result = executor.execute_model_output(
        StructuredResponseAdapter(),
        {
            "actions": [
                {"tool_name": "read_file", "arguments": {"path": "docs/note.txt"}},
                {
                    "tool_name": "list_directory",
                    "arguments": {"path": ".", "recursive": True},
                },
            ]
        },
        ToolContext(
            invocation_id="turn-3",
            workspace=str(tmp_path),
            logs=["base-log"],
            artifacts=["base-artifact"],
        ),
    )

    assert [tool_result.tool_name for tool_result in result.tool_results] == [
        "read_file",
        "list_directory",
    ]
    first_record = result.tool_results[0].metadata["execution_record"]
    second_record = result.tool_results[1].metadata["execution_record"]
    assert first_record["invocation_id"] == "turn-3:1"
    assert second_record["invocation_id"] == "turn-3:2"
    assert result.tool_results[0].artifacts != []
    assert result.tool_results[1].artifacts == []
    assert "base-artifact" not in result.tool_results[0].artifacts
    assert "base-log" not in result.tool_results[0].logs
    assert "base-artifact" not in result.tool_results[1].artifacts
    assert "base-log" not in result.tool_results[1].logs


def test_workflow_executor_executes_openai_adapter_tool_path(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry, allow_write=True)

    setup = executor.execute_model_output(
        StructuredResponseAdapter(),
        {
            "actions": [
                {
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "docs/note.txt",
                        "content": "hello openai",
                        "create_parents": True,
                    },
                }
            ]
        },
        ToolContext(invocation_id="setup-openai", workspace=str(tmp_path)),
    )
    assert setup.tool_results[0].ok is True

    result = executor.execute_model_output(
        OpenAIToolCallingAdapter(),
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "docs/note.txt"}',
                    },
                }
            ]
        },
        ToolContext(invocation_id="turn-4", workspace=str(tmp_path)),
    )

    assert result.tool_results[0].ok is True
    assert result.tool_results[0].output["content"] == "hello openai"


def test_workflow_executor_executes_openai_adapter_final_response_path(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    result = executor.execute_model_output(
        OpenAIToolCallingAdapter(),
        {"content": "No tool needed."},
        ToolContext(invocation_id="turn-5", workspace=str(tmp_path)),
    )

    assert result.parsed_response.final_response == "No tool needed."
    assert result.tool_results == []


def test_workflow_executor_executes_prompt_schema_paths(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry, allow_write=True)

    setup = executor.execute_model_output(
        StructuredResponseAdapter(),
        {
            "actions": [
                {
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "docs/note.txt",
                        "content": "hello prompt",
                        "create_parents": True,
                    },
                }
            ]
        },
        ToolContext(invocation_id="setup-prompt", workspace=str(tmp_path)),
    )
    assert setup.tool_results[0].ok is True

    action_result = executor.execute_model_output(
        PromptSchemaAdapter(),
        """```json
{"actions":[{"tool_name":"read_file","arguments":{"path":"docs/note.txt"}}]}
```""",
        ToolContext(invocation_id="turn-6", workspace=str(tmp_path)),
    )
    final_result = executor.execute_model_output(
        PromptSchemaAdapter(),
        '{"final_response": "Already answered."}',
        ToolContext(invocation_id="turn-7", workspace=str(tmp_path)),
    )

    assert action_result.tool_results[0].output["content"] == "hello prompt"
    assert final_result.parsed_response.final_response == "Already answered."


def test_workflow_executor_normalizes_unknown_tool_failure(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    result = executor.execute_model_output(
        StructuredResponseAdapter(),
        {"actions": [{"tool_name": "missing_tool", "arguments": {}}]},
        ToolContext(invocation_id="turn-8", workspace=str(tmp_path)),
    )

    assert result.tool_results[0].error is not None
    assert result.tool_results[0].error.code is ErrorCode.TOOL_NOT_FOUND


def test_workflow_executor_normalizes_policy_denied_failure(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    result = executor.execute_model_output(
        StructuredResponseAdapter(),
        {
            "actions": [
                {
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "docs/note.txt",
                        "content": "denied",
                        "create_parents": True,
                    },
                }
            ]
        },
        ToolContext(invocation_id="turn-9", workspace=str(tmp_path)),
    )

    assert result.tool_results[0].error is not None
    assert result.tool_results[0].error.code is ErrorCode.POLICY_DENIED


def test_workflow_executor_normalizes_input_validation_failure(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    result = executor.execute_model_output(
        StructuredResponseAdapter(),
        {
            "actions": [
                {
                    "tool_name": "list_directory",
                    "arguments": {"recursive": {"value": True}},
                }
            ]
        },
        ToolContext(invocation_id="turn-10", workspace=str(tmp_path)),
    )

    assert result.tool_results[0].error is not None
    assert result.tool_results[0].error.code is ErrorCode.INPUT_VALIDATION_ERROR


def test_workflow_executor_propagates_adapter_parse_failure(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    with pytest.raises(ValueError):
        executor.execute_model_output(
            PromptSchemaAdapter(),
            "not json",
            ToolContext(invocation_id="turn-11", workspace=str(tmp_path)),
        )


def test_workflow_executor_executes_git_and_jira_paths(
    tmp_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def enhanced_jql(self, jql: str, *, limit: int) -> dict[str, object]:
            del jql, limit
            return {
                "issues": [
                    {
                        "key": "DEMO-1",
                        "fields": {
                            "summary": "Issue",
                            "status": {"name": "Open"},
                            "issuetype": {"name": "Task"},
                            "assignee": {"displayName": "Alice"},
                        },
                    }
                ]
            }

    def fake_run(
        args: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n")

    fake_module = ModuleType("atlassian")
    fake_module.Jira = FakeJira
    monkeypatch.setitem(sys.modules, "atlassian", fake_module)
    monkeypatch.setattr(git_tools.subprocess, "run", fake_run)

    registry = ToolRegistry()
    register_git_tools(registry)
    register_atlassian_tools(registry)
    executor = _executor(registry)

    git_result = executor.execute_model_output(
        StructuredResponseAdapter(),
        {"actions": [{"tool_name": "run_git_status", "arguments": {"path": "."}}]},
        ToolContext(invocation_id="turn-12", workspace=str(tmp_path)),
    )
    jira_result = executor.execute_model_output(
        StructuredResponseAdapter(),
        {
            "actions": [
                {"tool_name": "search_jira", "arguments": {"jql": "project = DEMO"}}
            ]
        },
        ToolContext(
            invocation_id="turn-13",
            env={
                "JIRA_BASE_URL": "https://example.atlassian.net",
                "JIRA_USERNAME": "user@example.com",
                "JIRA_API_TOKEN": "token",
            },
        ),
    )

    assert git_result.tool_results[0].output["status_text"] == "ok\n"
    assert jira_result.tool_results[0].output["issues"][0]["key"] == "DEMO-1"

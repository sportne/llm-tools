"""Tests for built-in tool registration helpers."""

from __future__ import annotations

import pytest

from llm_tools.tool_api import DuplicateToolError, ToolRegistry
from llm_tools.tools import (
    register_atlassian_tools,
    register_filesystem_tools,
    register_git_tools,
    register_text_tools,
)


@pytest.mark.parametrize(
    ("register_tools", "expected_names"),
    [
        (
            register_filesystem_tools,
            {"read_file", "write_file", "list_directory"},
        ),
        (
            register_git_tools,
            {"run_git_status", "run_git_diff", "run_git_log"},
        ),
        (
            register_atlassian_tools,
            {"search_jira", "read_jira_issue"},
        ),
        (
            register_text_tools,
            {"file_text_search", "directory_text_search"},
        ),
    ],
)
def test_register_helpers_add_expected_tools(
    register_tools: object,
    expected_names: set[str],
) -> None:
    registry = ToolRegistry()
    register_tools(registry)  # type: ignore[misc]

    assert {spec.name for spec in registry.list_tools()} == expected_names


@pytest.mark.parametrize(
    "register_tools",
    [
        register_filesystem_tools,
        register_git_tools,
        register_atlassian_tools,
        register_text_tools,
    ],
)
def test_register_helpers_reject_duplicate_registration(register_tools: object) -> None:
    registry = ToolRegistry()
    register_tools(registry)  # type: ignore[misc]

    with pytest.raises(DuplicateToolError):
        register_tools(registry)  # type: ignore[misc]

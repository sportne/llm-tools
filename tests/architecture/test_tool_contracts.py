"""Contract-level invariants for built-in tool definitions."""

from __future__ import annotations

from pydantic import BaseModel
from tests.architecture._helpers import builtin_tools

from llm_tools.tool_api import SideEffectClass, ToolSpec


def test_builtin_tool_names_are_unique() -> None:
    tools = builtin_tools()
    names = [tool.spec.name for tool in tools]

    assert len(names) == len(set(names)), (
        f"Built-in tool names must be unique, found duplicates in {sorted(names)!r}."
    )


def test_builtin_tools_expose_valid_contracts() -> None:
    tools = builtin_tools()
    assert tools, "Expected at least one built-in tool to be registered."

    for tool in tools:
        spec = tool.spec
        tool_name = spec.name

        assert isinstance(spec, ToolSpec), (
            f"Built-in tool '{tool_name}' must expose a ToolSpec instance."
        )
        ToolSpec.model_validate(spec.model_dump(mode="json"))

        assert isinstance(tool.input_model, type) and issubclass(
            tool.input_model, BaseModel
        ), f"Built-in tool '{tool_name}' must declare a concrete input_model."
        assert isinstance(tool.output_model, type) and issubclass(
            tool.output_model, BaseModel
        ), f"Built-in tool '{tool_name}' must declare a concrete output_model."
        assert tool.input_model is not BaseModel, (
            f"Built-in tool '{tool_name}' must not use plain BaseModel as input_model."
        )
        assert tool.output_model is not BaseModel, (
            f"Built-in tool '{tool_name}' must not use plain BaseModel as output_model."
        )

        assert isinstance(spec.side_effects, SideEffectClass), (
            f"Built-in tool '{tool_name}' must declare a valid side_effects value."
        )
        assert (
            tool.__class__._has_sync_implementation()
            or tool.__class__._has_async_implementation()
        ), f"Built-in tool '{tool_name}' must implement invoke() or ainvoke()."
        assert all(isinstance(tag, str) and tag.strip() for tag in spec.tags), (
            f"Built-in tool '{tool_name}' must only expose non-empty string tags."
        )

        if spec.side_effects in {
            SideEffectClass.LOCAL_READ,
            SideEffectClass.LOCAL_WRITE,
        }:
            assert spec.requires_filesystem is True, (
                f"Built-in tool '{tool_name}' uses local filesystem side effects and "
                "must declare requires_filesystem=True."
            )

        if spec.writes_internal_workspace_cache:
            assert spec.requires_filesystem is True, (
                f"Built-in tool '{tool_name}' writes internal workspace cache data "
                "and must declare requires_filesystem=True."
            )
            assert spec.side_effects in {
                SideEffectClass.LOCAL_READ,
                SideEffectClass.LOCAL_WRITE,
            }, (
                f"Built-in tool '{tool_name}' writes internal workspace cache data "
                "and must declare a local filesystem side effect class."
            )

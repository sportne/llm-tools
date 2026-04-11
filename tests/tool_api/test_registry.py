"""Tests for the tool registry."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_tools.tool_api import (
    DuplicateToolError,
    SideEffectClass,
    Tool,
    ToolContext,
    ToolNotRegisteredError,
    ToolRegistry,
    ToolSpec,
)


class RegistryInput(BaseModel):
    value: str


class RegistryOutput(BaseModel):
    value: str


class ReadTool(Tool[RegistryInput, RegistryOutput]):
    spec = ToolSpec(
        name="read",
        description="Read something.",
        tags=["filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
    )
    input_model = RegistryInput
    output_model = RegistryOutput

    def invoke(self, context: ToolContext, args: RegistryInput) -> RegistryOutput:
        return RegistryOutput(value=f"{context.invocation_id}:{args.value}")


class WriteTool(Tool[RegistryInput, RegistryOutput]):
    spec = ToolSpec(
        name="write",
        description="Write something.",
        tags=["filesystem", "write"],
        side_effects=SideEffectClass.LOCAL_WRITE,
    )
    input_model = RegistryInput
    output_model = RegistryOutput

    def invoke(self, context: ToolContext, args: RegistryInput) -> RegistryOutput:
        return RegistryOutput(value=args.value)


class FetchTool(Tool[RegistryInput, RegistryOutput]):
    spec = ToolSpec(
        name="fetch",
        description="Fetch something.",
        tags=["http", "read"],
        side_effects=SideEffectClass.EXTERNAL_READ,
    )
    input_model = RegistryInput
    output_model = RegistryOutput

    def invoke(self, context: ToolContext, args: RegistryInput) -> RegistryOutput:
        return RegistryOutput(value=args.value)


def test_register_and_get_return_the_same_tool_instance() -> None:
    registry = ToolRegistry()
    tool = ReadTool()

    registry.register(tool)

    assert registry.get("read") is tool


def test_list_tools_returns_specs_in_registration_order() -> None:
    registry = ToolRegistry()
    read_tool = ReadTool()
    write_tool = WriteTool()
    fetch_tool = FetchTool()

    registry.register(read_tool)
    registry.register(write_tool)
    registry.register(fetch_tool)

    assert registry.list_tools() == [
        read_tool.spec,
        write_tool.spec,
        fetch_tool.spec,
    ]


def test_register_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry()
    registry.register(ReadTool())

    with pytest.raises(DuplicateToolError, match="read"):
        registry.register(ReadTool())


def test_get_raises_custom_error_for_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotRegisteredError, match="missing-tool"):
        registry.get("missing-tool")


def test_filter_tools_by_single_tag() -> None:
    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(FetchTool())

    assert registry.filter_tools(tags=["write"]) == [WriteTool.spec]


def test_filter_tools_by_multiple_tags_uses_any_match_semantics() -> None:
    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(FetchTool())

    assert registry.filter_tools(tags=["write", "http"]) == [
        WriteTool.spec,
        FetchTool.spec,
    ]


def test_filter_tools_by_single_side_effect() -> None:
    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(FetchTool())

    assert registry.filter_tools(side_effects=[SideEffectClass.LOCAL_WRITE]) == [
        WriteTool.spec
    ]


def test_filter_tools_by_multiple_side_effects_uses_any_match_semantics() -> None:
    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(FetchTool())

    assert registry.filter_tools(
        side_effects=[
            SideEffectClass.LOCAL_READ,
            SideEffectClass.EXTERNAL_READ,
        ]
    ) == [ReadTool.spec, FetchTool.spec]


def test_filter_tools_combines_tag_and_side_effect_filters() -> None:
    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(FetchTool())

    assert registry.filter_tools(
        tags=["read"],
        side_effects=[SideEffectClass.EXTERNAL_READ],
    ) == [FetchTool.spec]


def test_filter_tools_returns_empty_list_when_nothing_matches() -> None:
    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())

    assert (
        registry.filter_tools(
            tags=["http"],
            side_effects=[SideEffectClass.EXTERNAL_READ],
        )
        == []
    )

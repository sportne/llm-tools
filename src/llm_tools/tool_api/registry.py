"""Registry for concrete tool instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from llm_tools.tool_api.errors import DuplicateToolError, ToolNotRegisteredError
from llm_tools.tool_api.models import SideEffectClass, ToolSpec
from llm_tools.tool_api.tool import Tool


@dataclass(frozen=True, slots=True)
class RegisteredToolBinding:
    """Public read-only contract for one registered tool."""

    spec: ToolSpec
    input_model: type[BaseModel]
    output_model: type[BaseModel]


class ToolRegistry:
    """Stores and exposes registered tool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}

    def register(self, tool: Tool[Any, Any]) -> None:
        """Register a concrete tool instance by canonical tool name."""
        tool_name = tool.spec.name
        if tool_name in self._tools:
            raise DuplicateToolError(f"Tool '{tool_name}' is already registered.")

        self._tools[tool_name] = tool

    def get_spec(self, name: str) -> ToolSpec:
        """Return the registered tool spec for the given name."""
        return self._resolve_tool(name).spec

    def list_tools(self) -> list[ToolSpec]:
        """Return registered tool specs in registration order."""
        return [tool.spec for tool in self._tools.values()]

    def list_bindings(self) -> list[RegisteredToolBinding]:
        """Return public registered tool contracts in registration order."""
        return [
            RegisteredToolBinding(
                spec=tool.spec,
                input_model=tool.input_model,
                output_model=tool.output_model,
            )
            for tool in self._tools.values()
        ]

    def filter_tools(
        self,
        *,
        tags: list[str] | None = None,
        side_effects: list[SideEffectClass] | None = None,
    ) -> list[ToolSpec]:
        """Return tool specs matching the requested metadata filters."""
        requested_tags = set(tags or [])
        requested_side_effects = set(side_effects or [])
        matching_specs: list[ToolSpec] = []

        for tool in self._tools.values():
            spec = tool.spec
            if requested_tags and not requested_tags.intersection(spec.tags):
                continue
            if (
                requested_side_effects
                and spec.side_effects not in requested_side_effects
            ):
                continue
            matching_specs.append(spec)

        return matching_specs

    def _resolve_tool(self, name: str) -> Tool[Any, Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotRegisteredError(f"Tool '{name}' is not registered.") from exc

    def _iter_registered_tools(self) -> list[Tool[Any, Any]]:
        return list(self._tools.values())

"""Registry for concrete tool instances."""

from __future__ import annotations

from typing import Any

from llm_tools.tool_api.errors import DuplicateToolError, ToolNotRegisteredError
from llm_tools.tool_api.models import SideEffectClass, ToolSpec
from llm_tools.tool_api.tool import Tool


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

    def get(self, name: str) -> Tool[Any, Any]:
        """Return the registered tool instance for the given name."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotRegisteredError(f"Tool '{name}' is not registered.") from exc

    def list_tools(self) -> list[ToolSpec]:
        """Return registered tool specs in registration order."""
        return [tool.spec for tool in self._tools.values()]

    def list_registered_tools(self) -> list[Tool[Any, Any]]:
        """Return registered tool instances in registration order."""
        return list(self._tools.values())

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

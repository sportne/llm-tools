"""Policy model and evaluation for tool execution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llm_tools.tool_api.models import PolicyDecision, SideEffectClass, ToolContext
from llm_tools.tool_api.tool import Tool


class ToolPolicy(BaseModel):
    """Declarative policy for evaluating whether a tool may execute."""

    allowed_tools: set[str] | None = None
    denied_tools: set[str] = Field(default_factory=set)

    allowed_tags: set[str] | None = None
    denied_tags: set[str] = Field(default_factory=set)

    allowed_side_effects: set[SideEffectClass] = Field(
        default_factory=lambda: {
            SideEffectClass.NONE,
            SideEffectClass.LOCAL_READ,
        }
    )

    require_approval_for: set[SideEffectClass] = Field(default_factory=set)
    allow_network: bool = True
    allow_filesystem: bool = True
    allow_subprocess: bool = True

    def evaluate(self, tool: Tool[Any, Any], context: ToolContext) -> PolicyDecision:
        """Evaluate whether a tool is allowed under the current policy."""
        spec = tool.spec
        tool_name = spec.name
        tool_tags = set(spec.tags)

        if tool_name in self.denied_tools:
            return self._deny(
                tool_name,
                reason="tool name denied",
                metadata={"denied_by": "tool_name"},
            )

        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return self._deny(
                tool_name,
                reason="tool name not allowed",
                metadata={"allowed_tools": sorted(self.allowed_tools)},
            )

        denied_tag_matches = sorted(tool_tags.intersection(self.denied_tags))
        if denied_tag_matches:
            return self._deny(
                tool_name,
                reason="tool tag denied",
                metadata={"matched_tags": denied_tag_matches},
            )

        if self.allowed_tags is not None:
            allowed_tag_matches = sorted(tool_tags.intersection(self.allowed_tags))
            if not allowed_tag_matches:
                return self._deny(
                    tool_name,
                    reason="tool tags not allowed",
                    metadata={"allowed_tags": sorted(self.allowed_tags)},
                )

        if spec.side_effects not in self.allowed_side_effects:
            return self._deny(
                tool_name,
                reason="side effect not allowed",
                metadata={"side_effects": spec.side_effects.value},
            )

        if spec.requires_network and not self.allow_network:
            return self._deny(
                tool_name,
                reason="network access denied",
                metadata={"blocked_capability": "network"},
            )

        if spec.requires_filesystem and not self.allow_filesystem:
            return self._deny(
                tool_name,
                reason="filesystem access denied",
                metadata={"blocked_capability": "filesystem"},
            )

        if spec.requires_subprocess and not self.allow_subprocess:
            return self._deny(
                tool_name,
                reason="subprocess access denied",
                metadata={"blocked_capability": "subprocess"},
            )

        missing_secrets = [
            secret for secret in spec.required_secrets if secret not in context.env
        ]
        if missing_secrets:
            return self._deny(
                tool_name,
                reason="required secrets missing",
                metadata={"missing_secrets": sorted(missing_secrets)},
            )

        if spec.side_effects in self.require_approval_for:
            return PolicyDecision(
                allowed=False,
                reason="approval required",
                requires_approval=True,
                metadata={
                    "tool_name": tool_name,
                    "side_effects": spec.side_effects.value,
                },
            )

        return PolicyDecision(
            allowed=True,
            reason="allowed",
            metadata={"tool_name": tool_name},
        )

    @staticmethod
    def _deny(
        tool_name: str,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            allowed=False,
            reason=reason,
            metadata={"tool_name": tool_name, **(metadata or {})},
        )

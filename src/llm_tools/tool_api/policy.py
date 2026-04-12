"""Policy model and evaluation for tool execution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llm_tools.tool_api.models import (
    PolicyDecision,
    PolicyVerdict,
    SideEffectClass,
    ToolContext,
    ToolSpec,
)
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
    redacted_field_names: set[str] = Field(
        default_factory=lambda: {
            "password",
            "secret",
            "token",
            "api_key",
            "access_token",
            "refresh_token",
            "authorization",
        }
    )

    def evaluate(self, tool: Tool[Any, Any], context: ToolContext) -> PolicyDecision:
        """Evaluate whether a tool is allowed under the current policy."""
        spec = tool.spec
        tool_name = spec.name
        tool_tags = set(spec.tags)

        for decision in (
            self._evaluate_name_rules(tool_name),
            self._evaluate_tag_rules(tool_name, tool_tags),
            self._evaluate_side_effect_rules(tool_name, tool.spec),
            self._evaluate_secret_rules(tool_name, tool.spec, context),
        ):
            if decision is not None:
                return decision

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

    def verdict(self, tool: Tool[Any, Any], context: ToolContext) -> PolicyVerdict:
        """Return a high-level policy verdict for exposure and execution flows."""
        decision = self.evaluate(tool, context)
        if decision.allowed:
            return PolicyVerdict.ALLOW
        if decision.requires_approval:
            return PolicyVerdict.REQUIRE_APPROVAL
        return PolicyVerdict.DENY

    def _evaluate_tag_rules(
        self,
        tool_name: str,
        tool_tags: set[str],
    ) -> PolicyDecision | None:
        denied_tag_matches = sorted(tool_tags.intersection(self.denied_tags))
        if denied_tag_matches:
            return self._deny(
                tool_name,
                reason="tool tag denied",
                metadata={"matched_tags": denied_tag_matches},
            )

        if self.allowed_tags is None:
            return None

        allowed_tag_matches = sorted(tool_tags.intersection(self.allowed_tags))
        if allowed_tag_matches:
            return None

        return self._deny(
            tool_name,
            reason="tool tags not allowed",
            metadata={"allowed_tags": sorted(self.allowed_tags)},
        )

    def _evaluate_side_effect_rules(
        self,
        tool_name: str,
        spec: ToolSpec,
    ) -> PolicyDecision | None:
        if spec.side_effects not in self.allowed_side_effects:
            return self._deny(
                tool_name,
                reason="side effect not allowed",
                metadata={"side_effects": spec.side_effects.value},
            )

        for capability_name, is_required, is_allowed in (
            ("network", spec.requires_network, self.allow_network),
            ("filesystem", spec.requires_filesystem, self.allow_filesystem),
            ("subprocess", spec.requires_subprocess, self.allow_subprocess),
        ):
            if is_required and not is_allowed:
                return self._deny(
                    tool_name,
                    reason=f"{capability_name} access denied",
                    metadata={"blocked_capability": capability_name},
                )

        return None

    def _evaluate_secret_rules(
        self,
        tool_name: str,
        spec: ToolSpec,
        context: ToolContext,
    ) -> PolicyDecision | None:
        missing_secrets = [
            secret for secret in spec.required_secrets if secret not in context.env
        ]
        if not missing_secrets:
            return None

        return self._deny(
            tool_name,
            reason="required secrets missing",
            metadata={"missing_secrets": sorted(missing_secrets)},
        )

    def _evaluate_name_rules(self, tool_name: str) -> PolicyDecision | None:
        if tool_name in self.denied_tools:
            return self._deny(
                tool_name,
                reason="tool name denied",
                metadata={"denied_by": "tool_name"},
            )

        if self.allowed_tools is None or tool_name in self.allowed_tools:
            return None

        return self._deny(
            tool_name,
            reason="tool name not allowed",
            metadata={"allowed_tools": sorted(self.allowed_tools)},
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

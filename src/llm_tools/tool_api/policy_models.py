"""Policy model for tool execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

from llm_tools.tool_api.models import (
    SideEffectClass,
)
from llm_tools.tool_api.redaction import (
    DEFAULT_SENSITIVE_FIELD_NAMES,
    RedactionConfig,
    RedactionRule,
    RedactionTarget,
)

if TYPE_CHECKING:
    from llm_tools.tool_api.execution import HostToolContext
    from llm_tools.tool_api.models import PolicyDecision, PolicyVerdict, ToolSpec
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
    approval_timeout_seconds: int = Field(default=300, ge=1)
    allow_network: bool = True
    allow_filesystem: bool = True
    allow_subprocess: bool = True
    allow_internal_workspace_cache_writes: bool = True
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)
    redacted_field_names: set[str] = Field(
        default_factory=lambda: set(DEFAULT_SENSITIVE_FIELD_NAMES)
    )

    @model_validator(mode="after")
    def _merge_legacy_redaction_fields(self) -> ToolPolicy:
        """Map legacy redacted field names into the rich redaction config."""
        if not self.redacted_field_names:
            return self

        existing_signature = {
            (
                frozenset(rule.field_names),
                frozenset(rule.paths),
                frozenset(rule.targets),
                rule.replacement,
            )
            for rule in self.redaction.rules
        }
        legacy_rule = RedactionRule(
            field_names=set(self.redacted_field_names),
            targets={RedactionTarget.ALL},
        )
        legacy_signature = (
            frozenset(legacy_rule.field_names),
            frozenset(legacy_rule.paths),
            frozenset(legacy_rule.targets),
            legacy_rule.replacement,
        )
        if legacy_signature not in existing_signature:
            self.redaction.rules.append(legacy_rule)
        return self

    if TYPE_CHECKING:

        def evaluate(
            self,
            tool: Tool[Any, Any] | ToolSpec,
            context: HostToolContext,
        ) -> PolicyDecision:
            """Evaluate whether a tool is allowed under the current policy."""

        def verdict(
            self,
            tool: Tool[Any, Any] | ToolSpec,
            context: HostToolContext,
        ) -> PolicyVerdict:
            """Return a high-level policy verdict for exposure and execution flows."""


__all__ = [
    "ToolPolicy",
]

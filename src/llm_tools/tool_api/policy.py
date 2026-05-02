"""Policy model and evaluation for tool execution."""

from __future__ import annotations

from typing import Any

from llm_tools.tool_api.execution import HostToolContext
from llm_tools.tool_api.models import (
    PolicyDecision,
    PolicyVerdict,
    ToolSpec,
)
from llm_tools.tool_api.policy_models import (
    ToolPolicy as ToolPolicy,
)
from llm_tools.tool_api.tool import Tool


def _tool_policy_evaluate(
    self: ToolPolicy,
    tool: Tool[Any, Any] | ToolSpec,
    context: HostToolContext,
) -> PolicyDecision:
    """Evaluate whether a tool is allowed under the current policy."""
    spec = tool.spec if isinstance(tool, Tool) else tool
    tool_name = spec.name
    tool_tags = set(spec.tags)

    for decision in (
        _evaluate_name_rules(self, tool_name),
        _evaluate_tag_rules(self, tool_name, tool_tags),
        _evaluate_side_effect_rules(self, tool_name, spec),
        _evaluate_internal_cache_write_rules(self, tool_name, spec),
        _evaluate_secret_rules(tool_name, spec, context),
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


def _tool_policy_verdict(
    self: ToolPolicy,
    tool: Tool[Any, Any] | ToolSpec,
    context: HostToolContext,
) -> PolicyVerdict:
    """Return a high-level policy verdict for exposure and execution flows."""
    decision = _tool_policy_evaluate(self, tool, context)
    if decision.allowed:
        return PolicyVerdict.ALLOW
    if decision.requires_approval:
        return PolicyVerdict.REQUIRE_APPROVAL
    return PolicyVerdict.DENY


def _evaluate_tag_rules(
    policy: ToolPolicy,
    tool_name: str,
    tool_tags: set[str],
) -> PolicyDecision | None:
    denied_tag_matches = sorted(tool_tags.intersection(policy.denied_tags))
    if denied_tag_matches:
        return _deny(
            tool_name,
            reason="tool tag denied",
            metadata={"matched_tags": denied_tag_matches},
        )

    if policy.allowed_tags is None:
        return None

    allowed_tag_matches = sorted(tool_tags.intersection(policy.allowed_tags))
    if allowed_tag_matches:
        return None

    return _deny(
        tool_name,
        reason="tool tags not allowed",
        metadata={"allowed_tags": sorted(policy.allowed_tags)},
    )


def _evaluate_side_effect_rules(
    policy: ToolPolicy,
    tool_name: str,
    spec: ToolSpec,
) -> PolicyDecision | None:
    if spec.side_effects not in policy.allowed_side_effects:
        return _deny(
            tool_name,
            reason="side effect not allowed",
            metadata={"side_effects": spec.side_effects.value},
        )

    for capability_name, is_required, is_allowed in (
        ("network", spec.requires_network, policy.allow_network),
        ("filesystem", spec.requires_filesystem, policy.allow_filesystem),
        ("subprocess", spec.requires_subprocess, policy.allow_subprocess),
    ):
        if is_required and not is_allowed:
            return _deny(
                tool_name,
                reason=f"{capability_name} access denied",
                metadata={"blocked_capability": capability_name},
            )

    return None


def _evaluate_internal_cache_write_rules(
    policy: ToolPolicy,
    tool_name: str,
    spec: ToolSpec,
) -> PolicyDecision | None:
    if (
        spec.writes_internal_workspace_cache
        and not policy.allow_internal_workspace_cache_writes
    ):
        return _deny(
            tool_name,
            reason="internal workspace cache writes denied",
            metadata={"writes_internal_workspace_cache": True},
        )
    return None


def _evaluate_secret_rules(
    tool_name: str,
    spec: ToolSpec,
    context: HostToolContext,
) -> PolicyDecision | None:
    missing_secrets = [
        secret for secret in spec.required_secrets if secret not in context.env
    ]
    if not missing_secrets:
        return None

    return _deny(
        tool_name,
        reason="required secrets missing",
        metadata={"missing_secrets": sorted(missing_secrets)},
    )


def _evaluate_name_rules(
    policy: ToolPolicy,
    tool_name: str,
) -> PolicyDecision | None:
    if tool_name in policy.denied_tools:
        return _deny(
            tool_name,
            reason="tool name denied",
            metadata={"denied_by": "tool_name"},
        )

    if policy.allowed_tools is None or tool_name in policy.allowed_tools:
        return None

    return _deny(
        tool_name,
        reason="tool name not allowed",
        metadata={"allowed_tools": sorted(policy.allowed_tools)},
    )


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


ToolPolicy.evaluate = _tool_policy_evaluate  # type: ignore[method-assign]
ToolPolicy.verdict = _tool_policy_verdict  # type: ignore[method-assign]

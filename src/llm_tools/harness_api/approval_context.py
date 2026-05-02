"""Helpers for durable pending-approval tool context."""

from __future__ import annotations

from collections.abc import Mapping

from llm_tools.tool_api import ToolContext


def sanitize_pending_approval_context(context: ToolContext) -> ToolContext:
    """Return the durable subset of tool context safe to persist."""
    return context.model_copy(
        update={
            "env": {},
            "logs": [],
            "artifacts": [],
            "source_provenance": [],
        },
        deep=True,
    )


def rehydrate_pending_approval_context(
    context: ToolContext,
    *,
    env: Mapping[str, str] | None = None,
) -> ToolContext:
    """Rebuild a persisted approval context with the current process env."""
    return context.model_copy(
        update={"env": dict(env or {})},
        deep=True,
    )


__all__ = [
    "rehydrate_pending_approval_context",
    "sanitize_pending_approval_context",
]

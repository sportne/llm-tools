"""Assistant execution policy and context helpers."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from llm_tools.apps.assistant_config import StreamlitAssistantConfig
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolSpec
from llm_tools.tool_api.redaction import RedactionConfig


def build_assistant_policy(
    *,
    enabled_tools: set[str],
    tool_specs: dict[str, ToolSpec],
    require_approval_for: set[SideEffectClass],
    allow_network: bool,
    allow_filesystem: bool,
    allow_subprocess: bool,
    redaction_config: RedactionConfig,
) -> ToolPolicy:
    """Build the assistant workflow policy for one session."""
    allowed_side_effects = {SideEffectClass.NONE}
    for tool_name in enabled_tools:
        spec = tool_specs.get(tool_name)
        if spec is not None:
            allowed_side_effects.add(spec.side_effects)
    return ToolPolicy(
        allowed_tools=set(enabled_tools),
        allowed_side_effects=allowed_side_effects,
        require_approval_for=set(require_approval_for),
        allow_network=allow_network,
        allow_filesystem=allow_filesystem,
        allow_subprocess=allow_subprocess,
        redaction=redaction_config.model_copy(deep=True),
    )


def build_assistant_context(
    *,
    root_path: Path | None,
    config: StreamlitAssistantConfig,
    app_name: str,
    env_overrides: dict[str, str] | None = None,
) -> ToolContext:
    """Build the tool context passed into assistant workflow execution."""
    effective_read_limit = (
        config.tool_limits.max_read_file_chars
        if config.tool_limits.max_read_file_chars is not None
        else max(1, config.session.max_context_tokens * 4)
    )
    effective_tool_limits = config.tool_limits.model_copy(
        update={"max_read_file_chars": effective_read_limit}
    )
    env = dict(os.environ)
    for key, value in (env_overrides or {}).items():
        cleaned = value.strip()
        if cleaned:
            env[key] = cleaned

    return ToolContext(
        invocation_id=f"{app_name}-{uuid4()}",
        workspace=str(root_path) if root_path is not None else None,
        env=env,
        metadata={
            "tool_limits": effective_tool_limits.model_dump(mode="json"),
            "assistant_mode": "streamlit_assistant",
        },
    )


__all__ = ["build_assistant_context", "build_assistant_policy"]

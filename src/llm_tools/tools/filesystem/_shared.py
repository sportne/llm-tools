"""Shared helpers for filesystem tool implementations."""

from __future__ import annotations

import hashlib
from pathlib import Path

from llm_tools.tool_api import SourceProvenanceRef, ToolExecutionContext
from llm_tools.tool_api.execution import get_workspace_root
from llm_tools.tools.filesystem.models import SourceFilters, ToolLimits


def append_local_source_provenance(
    context: ToolExecutionContext, *, relative_path: str
) -> None:
    """Record local-file provenance for one workspace-relative path."""
    source_id = f"workspace:{relative_path}"
    context.add_source_provenance(
        SourceProvenanceRef(
            source_kind="local_file",
            source_id=source_id,
            content_hash=hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
            whole_source_reproduction_allowed=True,
            metadata={"path": relative_path},
        )
    )


def require_repository_metadata(
    context: ToolExecutionContext,
) -> tuple[Path, SourceFilters, ToolLimits]:
    """Return workspace root plus per-call filesystem metadata."""
    workspace = get_workspace_root(context)
    metadata = context.metadata
    source_filters = SourceFilters.model_validate(metadata.get("source_filters", {}))
    tool_limits = ToolLimits.model_validate(metadata.get("tool_limits", {}))
    return workspace, source_filters, tool_limits


def source_filters_for_call(
    source_filters: SourceFilters, *, include_hidden: bool
) -> SourceFilters:
    """Return source filters adjusted for one tool call."""
    if not include_hidden:
        return source_filters
    return source_filters.model_copy(update={"include_hidden": True})

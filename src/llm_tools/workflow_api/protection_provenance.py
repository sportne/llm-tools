"""Source-provenance helpers for workflow protection."""

from __future__ import annotations

from llm_tools.tool_api import (
    ProtectionProvenanceSnapshot,
    SourceProvenanceRef,
    ToolResult,
)


def collect_provenance_from_tool_results(
    tool_results: list[ToolResult],
) -> ProtectionProvenanceSnapshot:
    """Collect unique source provenance refs from tool results."""
    deduped: dict[tuple[str, str, str], SourceProvenanceRef] = {}
    for tool_result in tool_results:
        for entry in tool_result.source_provenance:
            key = (entry.source_kind, entry.source_id, entry.content_hash)
            if key not in deduped:
                deduped[key] = entry.model_copy(deep=True)
    return ProtectionProvenanceSnapshot(sources=list(deduped.values()))


__all__ = ["collect_provenance_from_tool_results"]

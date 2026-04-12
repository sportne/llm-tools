"""Presentation helpers for the Textual workbench."""

from __future__ import annotations

import json
from typing import Any

from llm_tools.tool_api import ToolResult


def pretty_json(value: Any) -> str:
    """Return a stable, human-readable JSON representation."""
    if value is None:
        return ""
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def format_event(message: str) -> str:
    """Return one transcript-style event line."""
    return message.strip()


def append_event(existing: list[str], message: str) -> str:
    """Append one event and return the rendered transcript body."""
    existing.append(format_event(message))
    return "\n\n".join(existing)


def extract_execution_record(result: ToolResult | None) -> Any:
    """Return the attached execution record payload when present."""
    if result is None:
        return None
    return result.metadata.get("execution_record")


def sanitize_tool_result(result: ToolResult | None) -> Any:
    """Return a tool result payload without the duplicated execution record."""
    if result is None:
        return None

    payload = result.model_dump(mode="json")
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("execution_record", None)
    payload["metadata"] = metadata
    return payload


def build_tool_details_payload(
    *,
    spec_payload: dict[str, Any] | None,
    input_schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize tool details for the inspector pane."""
    if spec_payload is None and input_schema is None:
        return None
    return {
        "spec": spec_payload,
        "input_schema": input_schema,
    }

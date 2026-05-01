"""Assistant app feature flag helpers."""

from __future__ import annotations

from collections.abc import Iterable

from llm_tools.apps.assistant_app.models import NiceGUIAdminSettings
from llm_tools.tool_api import ToolSpec

WRITE_FILE_TOOL_NAME = "write_file"
ATLASSIAN_TOOL_TAGS = frozenset({"jira", "confluence", "bitbucket"})


def assistant_tool_is_feature_visible(
    spec: ToolSpec, admin_settings: NiceGUIAdminSettings
) -> bool:
    """Return whether one tool is visible under admin feature flags."""
    if spec.name == WRITE_FILE_TOOL_NAME:
        return admin_settings.write_file_tool_enabled
    tags = set(spec.tags)
    if "gitlab" in tags:
        return admin_settings.gitlab_tools_enabled
    if tags.intersection(ATLASSIAN_TOOL_TAGS):
        return admin_settings.atlassian_tools_enabled
    return True


def filter_assistant_tool_specs_for_features(
    tool_specs: dict[str, ToolSpec], admin_settings: NiceGUIAdminSettings
) -> dict[str, ToolSpec]:
    """Return Assistant tool specs that are visible under admin feature flags."""
    return {
        name: spec
        for name, spec in tool_specs.items()
        if assistant_tool_is_feature_visible(spec, admin_settings)
    }


def visible_enabled_tool_names(
    enabled_tools: Iterable[str],
    tool_specs: dict[str, ToolSpec],
    admin_settings: NiceGUIAdminSettings,
) -> set[str]:
    """Return enabled tool names that remain visible under admin feature flags."""
    visible_specs = filter_assistant_tool_specs_for_features(tool_specs, admin_settings)
    return set(enabled_tools).intersection(visible_specs)


__all__ = [
    "ATLASSIAN_TOOL_TAGS",
    "WRITE_FILE_TOOL_NAME",
    "assistant_tool_is_feature_visible",
    "filter_assistant_tool_specs_for_features",
    "visible_enabled_tool_names",
]

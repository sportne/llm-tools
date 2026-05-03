"""Config models and loading helpers for the LLM Tools Assistant app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError as PydanticValidationError

from llm_tools.apps.assistant_config_models import (
    AssistantConfig as AssistantConfig,
)
from llm_tools.apps.assistant_config_models import (
    AssistantResearchConfig as AssistantResearchConfig,
)
from llm_tools.apps.assistant_config_models import (
    AssistantWorkspaceConfig as AssistantWorkspaceConfig,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Configuration path is not a file: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML at {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at root of YAML file: {path}")
    return raw


def _deep_overlay(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_overlay(base_value, value)
        else:
            merged[key] = value
    return merged


def _validate_assistant_config_sections(raw: dict[str, Any]) -> None:
    for section_name in (
        "llm",
        "session",
        "tool_limits",
        "policy",
        "protection",
        "ui",
        "workspace",
        "research",
    ):
        section_value = raw.get(section_name)
        if section_value is not None and not isinstance(section_value, dict):
            raise ValueError(f"assistant config '{section_name}' must be a mapping")


def load_assistant_config(
    path: Path, *, base_config: AssistantConfig | None = None
) -> AssistantConfig:
    """Load and validate the LLM Tools Assistant configuration file."""
    raw = _load_yaml(path)
    _validate_assistant_config_sections(raw)
    if base_config is not None:
        raw = _deep_overlay(base_config.model_dump(mode="python"), raw)
    try:
        return AssistantConfig.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValueError(f"Invalid assistant config at {path}: {exc}") from exc


__all__ = [
    "AssistantResearchConfig",
    "AssistantWorkspaceConfig",
    "AssistantConfig",
    "load_assistant_config",
]

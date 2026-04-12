"""Configuration loading and validation for the Textual chat app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError as PydanticValidationError

from llm_tools.apps.textual_chat.models import TextualChatConfig


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


def load_textual_chat_config(path: Path) -> TextualChatConfig:
    """Load and validate the standalone chat configuration file."""
    raw = _load_yaml(path)
    for section_name in (
        "llm",
        "source_filters",
        "session",
        "tool_limits",
        "policy",
        "ui",
    ):
        section_value = raw.get(section_name)
        if section_value is not None and not isinstance(section_value, dict):
            raise ValueError(f"chat config '{section_name}' must be a mapping")
    try:
        return TextualChatConfig.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValueError(f"Invalid chat config at {path}: {exc}") from exc

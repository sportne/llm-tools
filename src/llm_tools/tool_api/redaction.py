"""Redaction models and helpers for runtime observability payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RedactionTarget(str, Enum):  # noqa: UP042
    """Payload surfaces where redaction rules may apply."""

    INPUT = "input"
    OUTPUT = "output"
    ERROR_DETAILS = "error_details"
    LOGS = "logs"
    ARTIFACTS = "artifacts"
    ALL = "all"


class RedactionRule(BaseModel):
    """A single redaction rule using field names and/or structured paths."""

    field_names: set[str] = Field(default_factory=set)
    paths: set[str] = Field(default_factory=set)
    targets: set[RedactionTarget] = Field(default_factory=lambda: {RedactionTarget.ALL})
    replacement: str = "[REDACTED]"


class RedactionConfig(BaseModel):
    """Configurable redaction policy for runtime outputs and observability."""

    rules: list[RedactionRule] = Field(default_factory=list)
    tool_rules: dict[str, list[RedactionRule]] = Field(default_factory=dict)
    redact_logs: bool = True
    redact_artifacts: bool = True
    retain_unredacted_inputs: bool = False
    retain_unredacted_outputs: bool = False


DEFAULT_SENSITIVE_FIELD_NAMES: set[str] = {
    "password",
    "secret",
    "token",
    "api_key",
    "access_token",
    "refresh_token",
    "authorization",
}


def _normalize_name(name: str) -> str:
    return name.casefold().replace("-", "_")


def _parse_path(path: str) -> tuple[str, ...]:
    components: list[str] = []
    for raw_segment in path.split("."):
        segment = raw_segment.strip()
        if segment == "":
            continue

        while "[*]" in segment:
            prefix, _, suffix = segment.partition("[*]")
            if prefix:
                components.append(_normalize_name(prefix))
            components.append("[*]")
            segment = suffix
        if segment:
            components.append(_normalize_name(segment))

    return tuple(components)


def _path_to_string(path_components: tuple[str, ...]) -> str:
    parts: list[str] = []
    for component in path_components:
        if component == "[*]":
            if not parts:
                parts.append("[*]")
            else:
                parts[-1] = f"{parts[-1]}[*]"
            continue
        parts.append(component)
    return ".".join(parts)


@dataclass(slots=True)
class RedactionSummary:
    """Internal summary for redaction work performed during one execution."""

    matched_targets: set[str] = field(default_factory=set)
    matched_paths: set[str] = field(default_factory=set)
    applied_rule_count: int = 0

    def as_metadata(
        self,
        *,
        retain_unredacted_inputs: bool,
        retain_unredacted_outputs: bool,
    ) -> dict[str, object]:
        return {
            "matched_targets": sorted(self.matched_targets),
            "matched_paths": sorted(self.matched_paths),
            "applied_rule_count": self.applied_rule_count,
            "retain_unredacted_inputs": retain_unredacted_inputs,
            "retain_unredacted_outputs": retain_unredacted_outputs,
        }


class Redactor:
    """Apply redaction rules for one tool execution context."""

    def __init__(self, config: RedactionConfig, *, tool_name: str) -> None:
        self._config = config
        self._tool_name = tool_name
        self._summary = RedactionSummary()

    @property
    def summary(self) -> RedactionSummary:
        """Return the cumulative redaction summary."""
        return self._summary

    def redact_structured(self, value: Any, *, target: RedactionTarget) -> Any:
        """Redact structured payloads (dict/list scalars) for a specific target."""
        return self._redact_value(value, target=target, path_components=())

    def redact_string_entries(
        self,
        values: list[str],
        *,
        target: RedactionTarget,
    ) -> list[str]:
        """Redact plain-string log/artifact entries at the surface level."""
        if target is RedactionTarget.LOGS and not self._config.redact_logs:
            return list(values)
        if target is RedactionTarget.ARTIFACTS and not self._config.redact_artifacts:
            return list(values)

        if not values:
            return []

        replacement = self._surface_replacement(target)
        self._summary.matched_targets.add(target.value)
        self._summary.applied_rule_count += len(values)
        return [replacement for _ in values]

    def _redact_value(
        self,
        value: Any,
        *,
        target: RedactionTarget,
        path_components: tuple[str, ...],
    ) -> Any:
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                normalized_key = _normalize_name(key)
                next_components = (*path_components, normalized_key)
                path_rule = self._first_matching_path_rule(target, next_components)
                if path_rule is not None:
                    redacted[key] = path_rule.replacement
                    self._record_match(target, next_components)
                    continue

                field_rule = self._first_matching_field_rule(target, normalized_key)
                if field_rule is not None:
                    redacted[key] = field_rule.replacement
                    self._record_match(target, next_components)
                    continue

                redacted[key] = self._redact_value(
                    item,
                    target=target,
                    path_components=next_components,
                )
            return redacted

        if isinstance(value, list):
            redacted_items: list[Any] = []
            for item in value:
                next_components = (*path_components, "[*]")
                path_rule = self._first_matching_path_rule(target, next_components)
                if path_rule is not None:
                    redacted_items.append(path_rule.replacement)
                    self._record_match(target, next_components)
                else:
                    redacted_items.append(
                        self._redact_value(
                            item,
                            target=target,
                            path_components=next_components,
                        )
                    )
            return redacted_items

        return value

    def _record_match(
        self, target: RedactionTarget, path_components: tuple[str, ...]
    ) -> None:
        self._summary.matched_targets.add(target.value)
        self._summary.matched_paths.add(_path_to_string(path_components))
        self._summary.applied_rule_count += 1

    def _surface_replacement(self, target: RedactionTarget) -> str:
        for rule in self._iter_rules(target, mode="surface"):
            return rule.replacement
        return "[REDACTED]"

    def _first_matching_path_rule(
        self,
        target: RedactionTarget,
        path_components: tuple[str, ...],
    ) -> RedactionRule | None:
        for rule in self._iter_rules(target, mode="path"):
            for rule_path in rule.paths:
                if _parse_path(rule_path) == path_components:
                    return rule
        return None

    def _first_matching_field_rule(
        self,
        target: RedactionTarget,
        field_name: str,
    ) -> RedactionRule | None:
        for rule in self._iter_rules(target, mode="field"):
            if field_name in {_normalize_name(name) for name in rule.field_names}:
                return rule
        return None

    def _iter_rules(self, target: RedactionTarget, *, mode: str) -> list[RedactionRule]:
        tool_specific = self._config.tool_rules.get(self._tool_name, [])
        global_rules = self._config.rules

        def matches_target(rule: RedactionRule) -> bool:
            return RedactionTarget.ALL in rule.targets or target in rule.targets

        def matches_mode(rule: RedactionRule) -> bool:
            if mode == "path":
                return bool(rule.paths)
            if mode == "field":
                return bool(rule.field_names)
            return True

        rules: list[RedactionRule] = []
        for collection in (tool_specific, global_rules):
            for rule in collection:
                if matches_target(rule) and matches_mode(rule):
                    rules.append(rule)
        return rules

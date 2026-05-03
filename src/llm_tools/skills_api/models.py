"""Pydantic models for local skill packages."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class SkillScope(StrEnum):
    """Authority level for a discovered skill."""

    ENTERPRISE = "enterprise"
    USER = "user"
    PROJECT = "project"
    BUNDLED = "bundled"


class SkillInvocationType(StrEnum):
    """How a skill was brought into a turn."""

    EXPLICIT = "explicit"
    SELECTED = "selected"


class SkillRoot(BaseModel):
    """One local directory to scan for skills."""

    path: Path
    scope: SkillScope


class SkillDiscoveryOptions(BaseModel):
    """Controls for local skill discovery."""

    max_depth: int = Field(default=6, ge=0)
    max_directories_per_root: int = Field(default=2000, ge=1)
    ignore_hidden_directories: bool = True
    follow_symlinks: bool = False


class SkillMetadata(BaseModel):
    """Lightweight discovery metadata for one valid skill."""

    name: str
    description: str
    path: Path
    scope: SkillScope


class SkillError(BaseModel):
    """Path-specific skill validation or discovery error."""

    path: Path
    message: str


class SkillDiscoveryResult(BaseModel):
    """Skills and errors found while scanning local roots."""

    skills: tuple[SkillMetadata, ...] = ()
    errors: tuple[SkillError, ...] = ()


class SkillEnablement(BaseModel):
    """Caller-supplied skill availability gates."""

    disabled_paths: tuple[Path, ...] = ()
    disabled_names: tuple[str, ...] = ()
    disabled_scopes: tuple[SkillScope, ...] = ()


class SkillInvocation(BaseModel):
    """A request to resolve one skill by name or explicit path."""

    name: str | None = None
    path: Path | None = None

    @model_validator(mode="after")
    def validate_target(self) -> SkillInvocation:
        """Require exactly one invocation target."""
        if (self.name is None) == (self.path is None):
            msg = "provide exactly one of name or path"
            raise ValueError(msg)
        return self


class SkillResolution(BaseModel):
    """One resolved skill candidate."""

    skill: SkillMetadata


class LoadedSkillContext(BaseModel):
    """Model-visible context for a loaded skill."""

    name: str
    path: Path
    contents: str


class AvailableSkillLine(BaseModel):
    """One rendered available-skill catalog entry."""

    name: str
    description: str
    path: str


class SkillRenderReport(BaseModel):
    """Budgeting report for available-skill context rendering."""

    total_count: int
    included_count: int
    omitted_count: int
    truncated_description_chars: int
    truncated_description_count: int


class AvailableSkillsContext(BaseModel):
    """Model-visible catalog of available skills."""

    skill_root_lines: tuple[str, ...] = ()
    skill_lines: tuple[str, ...] = ()
    report: SkillRenderReport
    warning_message: str | None = None
    rendered_text: str


class SkillUsageRecord(BaseModel):
    """Durable metadata that a skill was used for a turn."""

    name: str
    scope: SkillScope
    path: Path
    invocation_type: SkillInvocationType
    content_hash: str | None = None

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str | None) -> str | None:
        """Normalize empty content hashes to absent values."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SkillMetadataBudget(BaseModel):
    """Character budget for available-skill metadata rendering."""

    characters: int = Field(default=8000, ge=1)

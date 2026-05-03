"""Reusable support for local agent skill packages."""

from llm_tools.skills_api.discovery import bundled_skill_root, discover_skills
from llm_tools.skills_api.errors import (
    SkillAmbiguousError,
    SkillDisabledError,
    SkillLoadError,
    SkillNotFoundError,
    SkillResolutionError,
)
from llm_tools.skills_api.loading import (
    build_skill_usage_record,
    load_skill_context,
    render_loaded_skill_context,
)
from llm_tools.skills_api.models import (
    AvailableSkillLine,
    AvailableSkillsContext,
    LoadedSkillContext,
    SkillDiscoveryOptions,
    SkillDiscoveryResult,
    SkillEnablement,
    SkillError,
    SkillInvocation,
    SkillInvocationType,
    SkillMetadata,
    SkillMetadataBudget,
    SkillRenderReport,
    SkillResolution,
    SkillRoot,
    SkillScope,
    SkillUsageRecord,
)
from llm_tools.skills_api.rendering import render_available_skills_context
from llm_tools.skills_api.resolution import (
    enabled_skills,
    is_skill_enabled,
    resolve_skill,
)

__all__ = [
    "AvailableSkillLine",
    "AvailableSkillsContext",
    "build_skill_usage_record",
    "bundled_skill_root",
    "discover_skills",
    "enabled_skills",
    "is_skill_enabled",
    "LoadedSkillContext",
    "load_skill_context",
    "render_available_skills_context",
    "render_loaded_skill_context",
    "resolve_skill",
    "SkillAmbiguousError",
    "SkillDisabledError",
    "SkillDiscoveryOptions",
    "SkillDiscoveryResult",
    "SkillEnablement",
    "SkillError",
    "SkillInvocation",
    "SkillInvocationType",
    "SkillLoadError",
    "SkillMetadata",
    "SkillMetadataBudget",
    "SkillNotFoundError",
    "SkillRenderReport",
    "SkillResolution",
    "SkillResolutionError",
    "SkillRoot",
    "SkillScope",
    "SkillUsageRecord",
]

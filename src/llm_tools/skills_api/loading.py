"""Load selected skill instructions."""

from __future__ import annotations

import hashlib

from llm_tools.skills_api.errors import SkillLoadError
from llm_tools.skills_api.models import (
    LoadedSkillContext,
    SkillInvocationType,
    SkillMetadata,
    SkillUsageRecord,
)


def load_skill_context(skill: SkillMetadata) -> LoadedSkillContext:
    """Read a skill's full SKILL.md body as loaded-skill context."""
    try:
        contents = skill.path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillLoadError(f"failed to load skill {skill.name}: {exc}") from exc
    return LoadedSkillContext(name=skill.name, path=skill.path, contents=contents)


def render_loaded_skill_context(context: LoadedSkillContext) -> str:
    """Render loaded skill context using the Codex-compatible envelope."""
    return (
        "<skill>\n"
        f"<name>{context.name}</name>\n"
        f"<path>{context.path}</path>\n"
        f"{context.contents}\n"
        "</skill>"
    )


def build_skill_usage_record(
    skill: SkillMetadata,
    *,
    invocation_type: SkillInvocationType,
    contents: str | None = None,
) -> SkillUsageRecord:
    """Build durable metadata for one skill use."""
    content_hash = None
    if contents is not None:
        content_hash = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    return SkillUsageRecord(
        name=skill.name,
        scope=skill.scope,
        path=skill.path,
        invocation_type=invocation_type,
        content_hash=content_hash,
    )

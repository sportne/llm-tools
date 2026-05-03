"""Skill resolution and enablement helpers."""

from __future__ import annotations

from pathlib import Path

from llm_tools.skills_api.errors import (
    SkillAmbiguousError,
    SkillDisabledError,
    SkillNotFoundError,
)
from llm_tools.skills_api.models import (
    SkillDiscoveryResult,
    SkillEnablement,
    SkillInvocation,
    SkillMetadata,
    SkillResolution,
    SkillScope,
)

_SCOPE_RANK: dict[SkillScope, int] = {
    SkillScope.ENTERPRISE: 0,
    SkillScope.USER: 1,
    SkillScope.PROJECT: 2,
    SkillScope.BUNDLED: 3,
}


def resolve_skill(
    discovery: SkillDiscoveryResult,
    invocation: SkillInvocation,
    enablement: SkillEnablement | None = None,
    *,
    allow_disabled: bool = False,
    case_insensitive: bool = False,
) -> SkillResolution:
    """Resolve one skill by path or name."""
    resolved_enablement = enablement or SkillEnablement()
    if invocation.path is not None:
        return SkillResolution(
            skill=_resolve_by_path(
                discovery,
                invocation.path,
                resolved_enablement,
                allow_disabled=allow_disabled,
            )
        )
    if invocation.name is None:
        raise SkillNotFoundError("skill invocation has no name or path")
    return SkillResolution(
        skill=_resolve_by_name(
            discovery,
            invocation.name,
            resolved_enablement,
            allow_disabled=allow_disabled,
            case_insensitive=case_insensitive,
        )
    )


def is_skill_enabled(skill: SkillMetadata, enablement: SkillEnablement | None) -> bool:
    """Return whether a skill is enabled by caller-supplied state."""
    if enablement is None:
        return True
    canonical_disabled_paths = {
        _canonical_path(path) for path in enablement.disabled_paths
    }
    if _canonical_path(skill.path) in canonical_disabled_paths:
        return False
    if skill.name in set(enablement.disabled_names):
        return False
    return skill.scope not in set(enablement.disabled_scopes)


def enabled_skills(
    discovery: SkillDiscoveryResult,
    enablement: SkillEnablement | None = None,
) -> tuple[SkillMetadata, ...]:
    """Return skills allowed by caller-supplied enablement state."""
    return tuple(
        skill for skill in discovery.skills if is_skill_enabled(skill, enablement)
    )


def _resolve_by_path(
    discovery: SkillDiscoveryResult,
    path: Path,
    enablement: SkillEnablement,
    *,
    allow_disabled: bool,
) -> SkillMetadata:
    target = _canonical_path(path)
    for skill in discovery.skills:
        if _canonical_path(skill.path) == target:
            if not allow_disabled and not is_skill_enabled(skill, enablement):
                raise SkillDisabledError(f"skill at path is disabled: {target}")
            return skill
    raise SkillNotFoundError(f"skill not found at path: {target}")


def _resolve_by_name(
    discovery: SkillDiscoveryResult,
    name: str,
    enablement: SkillEnablement,
    *,
    allow_disabled: bool,
    case_insensitive: bool,
) -> SkillMetadata:
    cleaned = name.strip()
    if not cleaned:
        raise SkillNotFoundError("skill name is empty")
    candidates = [
        skill
        for skill in discovery.skills
        if _name_matches(skill.name, cleaned, case_insensitive=case_insensitive)
        and (allow_disabled or is_skill_enabled(skill, enablement))
    ]
    if not candidates:
        raise SkillNotFoundError(f"skill not found: {cleaned}")

    best_rank = min(_SCOPE_RANK[skill.scope] for skill in candidates)
    best = [skill for skill in candidates if _SCOPE_RANK[skill.scope] == best_rank]
    if len(best) > 1:
        paths = ", ".join(str(skill.path) for skill in best)
        raise SkillAmbiguousError(f"skill name is ambiguous: {cleaned} ({paths})")
    return best[0]


def _name_matches(candidate: str, name: str, *, case_insensitive: bool) -> bool:
    if candidate == name:
        return True
    return case_insensitive and candidate.casefold() == name.casefold()


def _canonical_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)

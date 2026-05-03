"""Render model-visible available-skill context."""

from __future__ import annotations

from collections.abc import Sequence

from llm_tools.skills_api.models import (
    AvailableSkillsContext,
    SkillDiscoveryResult,
    SkillEnablement,
    SkillMetadata,
    SkillMetadataBudget,
    SkillRenderReport,
)
from llm_tools.skills_api.resolution import enabled_skills

SKILLS_INTRO = (
    "A skill is a set of local instructions to follow that is stored in a "
    "`SKILL.md` file. Below is the list of skills that can be used. Each entry "
    "includes a name, description, and file path so you can open the source for "
    "full instructions when using a specific skill."
)
SKILLS_HOW_TO_USE = (
    "- Discovery: The list above is the skills available in this session "
    "(name + description + file path). Skill bodies live on disk at the listed "
    "paths.\n"
    "- Trigger rules: If the user names a skill (with `$SkillName` or plain "
    "text) OR the task clearly matches a skill's description shown above, use "
    "that skill for the turn.\n"
    "- How to use a skill: After deciding to use a skill, load its `SKILL.md`; "
    "resolve relative references from the skill directory; only load supporting "
    "files needed for the current task.\n"
    "- Context hygiene: Keep context small and avoid bulk-loading references "
    "unless needed."
)
OMITTED_WARNING_PREFIX = (
    "Exceeded skills context budget. All skill descriptions were removed and"
)
TRUNCATED_WARNING = (
    "Skill descriptions were shortened to fit the skills context budget. The "
    "model can still see every included skill, but some descriptions are shorter."
)


def render_available_skills_context(
    discovery: SkillDiscoveryResult,
    budget: SkillMetadataBudget | None = None,
    enablement: SkillEnablement | None = None,
) -> AvailableSkillsContext | None:
    """Render a Codex-style available-skills block within a character budget."""
    skills = enabled_skills(discovery, enablement)
    if not skills:
        return None
    resolved_budget = budget or SkillMetadataBudget()
    skill_lines, report = _budget_skill_lines(skills, resolved_budget.characters)
    warning_message = _warning_message(report)
    rendered_text = _render_body(skill_lines)
    return AvailableSkillsContext(
        skill_lines=tuple(skill_lines),
        report=report,
        warning_message=warning_message,
        rendered_text=rendered_text,
    )


def _budget_skill_lines(
    skills: Sequence[SkillMetadata],
    budget: int,
) -> tuple[list[str], SkillRenderReport]:
    full_lines = [_render_skill_line(skill, skill.description) for skill in skills]
    if _lines_cost(full_lines) <= budget:
        return full_lines, SkillRenderReport(
            total_count=len(skills),
            included_count=len(skills),
            omitted_count=0,
            truncated_description_chars=0,
            truncated_description_count=0,
        )

    minimum_lines = [_render_skill_line(skill, "") for skill in skills]
    if _lines_cost(minimum_lines) <= budget:
        extra_budget = budget - _lines_cost(minimum_lines)
        rendered: list[str] = []
        truncated_chars = 0
        truncated_count = 0
        share = max(extra_budget // max(len(skills), 1), 0)
        for skill in skills:
            description = _description_within_budget(skill.description, share)
            if len(description) < len(skill.description):
                truncated_chars += len(skill.description) - len(description)
                truncated_count += 1
            rendered.append(_render_skill_line(skill, description))
        return rendered, SkillRenderReport(
            total_count=len(skills),
            included_count=len(skills),
            omitted_count=0,
            truncated_description_chars=truncated_chars,
            truncated_description_count=truncated_count,
        )

    included: list[str] = []
    used = 0
    omitted = 0
    truncated_chars = 0
    truncated_count = 0
    for skill, line in zip(skills, minimum_lines, strict=True):
        cost = len(line) + 1
        if used + cost <= budget:
            included.append(line)
            used += cost
        else:
            omitted += 1
        if skill.description:
            truncated_chars += len(skill.description)
            truncated_count += 1
    return included, SkillRenderReport(
        total_count=len(skills),
        included_count=len(included),
        omitted_count=omitted,
        truncated_description_chars=truncated_chars,
        truncated_description_count=truncated_count,
    )


def _render_body(skill_lines: Sequence[str]) -> str:
    lines = [
        "## Skills",
        SKILLS_INTRO,
        "### Available skills",
        *skill_lines,
        "### How to use skills",
        SKILLS_HOW_TO_USE,
    ]
    return "\n" + "\n".join(lines) + "\n"


def _render_skill_line(skill: SkillMetadata, description: str) -> str:
    path = str(skill.path).replace("\\", "/")
    if description:
        return f"- {skill.name}: {description} (file: {path})"
    return f"- {skill.name}: (file: {path})"


def _lines_cost(lines: Sequence[str]) -> int:
    return sum(len(line) + 1 for line in lines)


def _description_within_budget(description: str, budget: int) -> str:
    if budget <= 0:
        return ""
    return description[:budget]


def _warning_message(report: SkillRenderReport) -> str | None:
    if report.omitted_count > 0:
        skill_word = "skill" if report.omitted_count == 1 else "skills"
        verb = "was" if report.omitted_count == 1 else "were"
        return (
            f"{OMITTED_WARNING_PREFIX} {report.omitted_count} additional "
            f"{skill_word} {verb} not included in the model-visible skills list."
        )
    if report.truncated_description_count > 0:
        return TRUNCATED_WARNING
    return None

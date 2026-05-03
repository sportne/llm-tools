"""Tests for local skill discovery, resolution, loading, and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_tools.skills_api import (
    SkillAmbiguousError,
    SkillDisabledError,
    SkillDiscoveryOptions,
    SkillEnablement,
    SkillInvocation,
    SkillInvocationType,
    SkillLoadError,
    SkillMetadataBudget,
    SkillNotFoundError,
    SkillRoot,
    SkillScope,
    build_skill_usage_record,
    bundled_skill_root,
    discover_skills,
    is_skill_enabled,
    load_skill_context,
    render_available_skills_context,
    render_loaded_skill_context,
    resolve_skill,
)


def test_discover_skills_returns_valid_metadata_and_path_errors(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / "valid" / "SKILL.md",
        name="valid-skill",
        description="Use this valid skill.",
        body="Follow the workflow.",
    )
    _write_skill(
        tmp_path / "invalid" / "SKILL.md",
        name="Invalid Skill!",
        description="This name is invalid.",
        body="Bad name.",
    )

    result = discover_skills([SkillRoot(path=tmp_path, scope=SkillScope.PROJECT)])

    assert [skill.name for skill in result.skills] == ["valid-skill"]
    assert result.skills[0].scope is SkillScope.PROJECT
    assert result.skills[0].description == "Use this valid skill."
    assert len(result.errors) == 1
    assert result.errors[0].path.name == "SKILL.md"
    assert "invalid name" in result.errors[0].message


def test_discovery_requires_non_empty_single_line_description(tmp_path: Path) -> None:
    _write_raw_skill(
        tmp_path / "empty" / "SKILL.md",
        "---\nname: empty-description\ndescription: ''\n---\nBody\n",
    )
    _write_raw_skill(
        tmp_path / "multiline" / "SKILL.md",
        "---\nname: multiline-description\ndescription: |\n  first\n  second\n---\nBody\n",
    )

    result = discover_skills([SkillRoot(path=tmp_path, scope=SkillScope.USER)])

    assert result.skills == ()
    assert len(result.errors) == 2
    assert {error.path.parent.name for error in result.errors} == {
        "empty",
        "multiline",
    }


def test_discovery_reports_frontmatter_edge_errors(tmp_path: Path) -> None:
    _write_raw_skill(tmp_path / "missing" / "SKILL.md", "# No frontmatter\n")
    _write_raw_skill(tmp_path / "scalar" / "SKILL.md", "---\nplain\n---\nBody\n")
    _write_raw_skill(
        tmp_path / "missing-name" / "SKILL.md",
        "---\ndescription: Missing name.\n---\nBody\n",
    )
    _write_raw_skill(
        tmp_path / "long-description" / "SKILL.md",
        f"---\nname: long-description\ndescription: {'x' * 1025}\n---\nBody\n",
    )
    _write_raw_skill(
        tmp_path / "boolish" / "SKILL.md",
        "---\nname: off\ndescription: Bool-like names stay strings.\n---\nBody\n",
    )

    result = discover_skills([SkillRoot(path=tmp_path, scope=SkillScope.USER)])

    assert [skill.name for skill in result.skills] == ["off"]
    messages = {error.path.parent.name: error.message for error in result.errors}
    assert "missing YAML frontmatter" in messages["missing"]
    assert "frontmatter must be a mapping" in messages["scalar"]
    assert "missing or invalid `name`" in messages["missing-name"]
    assert "at most 1024 characters" in messages["long-description"]


def test_discovery_ignores_missing_roots_and_dedupes_paths(tmp_path: Path) -> None:
    _write_skill(tmp_path / "skill" / "SKILL.md", name="once", description="Once.")

    result = discover_skills(
        [
            SkillRoot(path=tmp_path / "missing", scope=SkillScope.USER),
            SkillRoot(path=tmp_path, scope=SkillScope.USER),
            SkillRoot(path=tmp_path, scope=SkillScope.PROJECT),
        ]
    )

    assert [skill.name for skill in result.skills] == ["once"]
    assert result.skills[0].scope is SkillScope.USER


def test_discovery_scans_bounded_recursion_and_skips_hidden_dirs(
    tmp_path: Path,
) -> None:
    _write_skill(
        tmp_path / "one" / "two" / "SKILL.md",
        name="nested",
        description="Nested skill.",
    )
    _write_skill(
        tmp_path / ".hidden" / "SKILL.md",
        name="hidden",
        description="Hidden skill.",
    )

    shallow = discover_skills(
        [SkillRoot(path=tmp_path, scope=SkillScope.PROJECT)],
        SkillDiscoveryOptions(max_depth=1),
    )
    deep = discover_skills(
        [SkillRoot(path=tmp_path, scope=SkillScope.PROJECT)],
        SkillDiscoveryOptions(max_depth=2),
    )

    assert shallow.skills == ()
    assert [skill.name for skill in deep.skills] == ["nested"]


def test_discovery_respects_directory_count_limit(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / "one" / "SKILL.md",
        name="one",
        description="One.",
    )

    result = discover_skills(
        [SkillRoot(path=tmp_path, scope=SkillScope.PROJECT)],
        SkillDiscoveryOptions(max_directories_per_root=1),
    )

    assert result.skills == ()


def test_discovery_skips_symlinked_skill_files_by_default(tmp_path: Path) -> None:
    external_skill = tmp_path / "external" / "SKILL.md"
    _write_skill(external_skill, name="linked", description="Linked.")
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / "SKILL.md").symlink_to(external_skill)
    except OSError as exc:  # pragma: no cover - platform permission edge
        pytest.skip(f"symlinks unavailable: {exc}")

    skipped = discover_skills([SkillRoot(path=root, scope=SkillScope.PROJECT)])
    followed = discover_skills(
        [SkillRoot(path=root, scope=SkillScope.PROJECT)],
        SkillDiscoveryOptions(follow_symlinks=True),
    )

    assert skipped.skills == ()
    assert [skill.name for skill in followed.skills] == ["linked"]


def test_resolve_skill_applies_scope_precedence_and_same_scope_ambiguity(
    tmp_path: Path,
) -> None:
    _write_skill(
        tmp_path / "project" / "SKILL.md", name="shared", description="Project."
    )
    _write_skill(tmp_path / "user" / "SKILL.md", name="shared", description="User.")
    _write_skill(
        tmp_path / "other-user" / "SKILL.md", name="shared", description="Other."
    )

    precedence = discover_skills(
        [
            SkillRoot(path=tmp_path / "project", scope=SkillScope.PROJECT),
            SkillRoot(path=tmp_path / "user", scope=SkillScope.USER),
        ]
    )
    resolved = resolve_skill(precedence, SkillInvocation(name="shared"))

    assert resolved.skill.scope is SkillScope.USER
    assert resolved.skill.path.parent.name == "user"

    ambiguous = discover_skills(
        [
            SkillRoot(path=tmp_path / "user", scope=SkillScope.USER),
            SkillRoot(path=tmp_path / "other-user", scope=SkillScope.USER),
        ]
    )
    with pytest.raises(SkillAmbiguousError):
        resolve_skill(ambiguous, SkillInvocation(name="shared"))

    explicit = resolve_skill(
        ambiguous,
        SkillInvocation(path=tmp_path / "other-user" / "SKILL.md"),
    )
    assert explicit.skill.path.parent.name == "other-user"


def test_resolve_skill_respects_enablement(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / "disabled" / "SKILL.md", name="off", description="Disabled."
    )
    discovery = discover_skills([SkillRoot(path=tmp_path, scope=SkillScope.USER)])
    skill = discovery.skills[0]
    enablement = SkillEnablement(disabled_paths=(skill.path,))

    with pytest.raises(SkillNotFoundError):
        resolve_skill(discovery, SkillInvocation(name="off"), enablement)

    with pytest.raises(SkillDisabledError):
        resolve_skill(discovery, SkillInvocation(path=skill.path), enablement)

    resolved = resolve_skill(
        discovery,
        SkillInvocation(path=skill.path),
        enablement,
        allow_disabled=True,
    )
    assert resolved.skill.name == "off"


def test_resolve_skill_enablement_names_scopes_and_case_matching(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path / "alpha" / "SKILL.md", name="Alpha", description="Alpha.")
    _write_skill(tmp_path / "beta" / "SKILL.md", name="beta", description="Beta.")
    discovery = discover_skills([SkillRoot(path=tmp_path, scope=SkillScope.PROJECT)])
    alpha = next(skill for skill in discovery.skills if skill.name == "Alpha")
    beta = next(skill for skill in discovery.skills if skill.name == "beta")

    assert is_skill_enabled(alpha, None) is True
    assert is_skill_enabled(alpha, SkillEnablement(disabled_names=("Alpha",))) is False
    assert (
        is_skill_enabled(
            alpha,
            SkillEnablement(disabled_scopes=(SkillScope.PROJECT,)),
        )
        is False
    )
    assert (
        resolve_skill(
            discovery,
            SkillInvocation(name="alpha"),
            case_insensitive=True,
        ).skill
        == alpha
    )
    assert (
        resolve_skill(
            discovery,
            SkillInvocation(name="Alpha"),
            SkillEnablement(disabled_names=("Alpha",)),
            allow_disabled=True,
        ).skill
        == alpha
    )
    assert resolve_skill(discovery, SkillInvocation(path=beta.path)).skill == beta

    with pytest.raises(SkillNotFoundError):
        resolve_skill(discovery, SkillInvocation(name=" "))
    with pytest.raises(SkillNotFoundError):
        resolve_skill(
            discovery, SkillInvocation(path=tmp_path / "missing" / "SKILL.md")
        )
    with pytest.raises(SkillNotFoundError):
        resolve_skill(discovery, SkillInvocation.model_construct())


def test_load_and_render_loaded_skill_context_and_usage_record(tmp_path: Path) -> None:
    skill_path = tmp_path / "doc" / "SKILL.md"
    _write_skill(
        skill_path, name="doc", description="Document things.", body="Write docs."
    )
    discovery = discover_skills([SkillRoot(path=tmp_path, scope=SkillScope.USER)])

    context = load_skill_context(discovery.skills[0])
    rendered = render_loaded_skill_context(context)
    usage = build_skill_usage_record(
        discovery.skills[0],
        invocation_type=SkillInvocationType.EXPLICIT,
        contents=context.contents,
    )

    assert rendered.startswith("<skill>\n<name>doc</name>\n<path>")
    assert "---\nname: doc\n" in rendered
    assert "Write docs." in rendered
    assert rendered.endswith("</skill>")
    assert usage.name == "doc"
    assert usage.content_hash is not None


def test_load_skill_context_reports_missing_file_and_usage_hash_is_optional(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path / "gone" / "SKILL.md", name="gone", description="Gone.")
    discovery = discover_skills([SkillRoot(path=tmp_path, scope=SkillScope.USER)])
    skill = discovery.skills[0]
    skill.path.unlink()

    with pytest.raises(SkillLoadError):
        load_skill_context(skill)

    usage = build_skill_usage_record(
        skill,
        invocation_type=SkillInvocationType.SELECTED,
    )
    assert usage.content_hash is None


def test_render_available_skills_context_filters_disabled_and_reports_budget(
    tmp_path: Path,
) -> None:
    _write_skill(
        tmp_path / "short" / "SKILL.md",
        name="short",
        description="Short description.",
    )
    _write_skill(
        tmp_path / "long" / "SKILL.md",
        name="long",
        description="Long description. " * 20,
    )
    discovery = discover_skills([SkillRoot(path=tmp_path, scope=SkillScope.USER)])
    disabled = next(skill for skill in discovery.skills if skill.name == "short")

    context = render_available_skills_context(
        discovery,
        SkillMetadataBudget(characters=320),
        SkillEnablement(disabled_paths=(disabled.path,)),
    )

    assert context is not None
    assert context.report.total_count == 1
    assert context.report.included_count == 1
    assert context.warning_message is not None
    assert "## Skills" in context.rendered_text
    assert "- long:" in context.rendered_text
    assert "- short:" not in context.rendered_text
    assert "### How to use skills" in context.rendered_text


def test_render_available_skills_context_budget_branches(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / "one" / "SKILL.md",
        name="one",
        description="First description.",
    )
    _write_skill(
        tmp_path / "two" / "SKILL.md",
        name="two",
        description="Second description.",
    )
    discovery = discover_skills([SkillRoot(path=tmp_path, scope=SkillScope.USER)])

    full = render_available_skills_context(discovery)
    assert full is not None
    assert full.warning_message is None
    assert full.report.truncated_description_count == 0

    all_disabled = render_available_skills_context(
        discovery,
        enablement=SkillEnablement(disabled_scopes=(SkillScope.USER,)),
    )
    assert all_disabled is None

    min_line_cost = sum(
        len(f"- {skill.name}: (file: {_rendered_path(skill.path)})") + 1
        for skill in discovery.skills
    )
    truncated = render_available_skills_context(
        discovery,
        SkillMetadataBudget(characters=min_line_cost),
    )
    assert truncated is not None
    assert truncated.report.included_count == 2
    assert truncated.report.truncated_description_count == 2
    assert truncated.warning_message is not None

    omitted = render_available_skills_context(
        discovery,
        SkillMetadataBudget(characters=1),
    )
    assert omitted is not None
    assert omitted.report.included_count == 0
    assert omitted.report.omitted_count == 2
    assert "2 additional skills were not included" in (omitted.warning_message or "")


def test_bundled_skill_root_is_opt_in_and_contains_examples() -> None:
    result = discover_skills([bundled_skill_root()])

    assert {skill.name for skill in result.skills} >= {
        "create-typed-tool",
        "debug-model-turn",
    }
    assert {skill.scope for skill in result.skills} == {SkillScope.BUNDLED}


def test_skill_invocation_requires_one_target() -> None:
    with pytest.raises(ValidationError):
        SkillInvocation()
    with pytest.raises(ValidationError):
        SkillInvocation(name="x", path=Path("x/SKILL.md"))


def _write_skill(
    path: Path,
    *,
    name: str,
    description: str,
    body: str = "Instructions.",
) -> None:
    _write_raw_skill(
        path,
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
    )


def _write_raw_skill(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _rendered_path(path: Path) -> str:
    return str(path).replace("\\", "/")

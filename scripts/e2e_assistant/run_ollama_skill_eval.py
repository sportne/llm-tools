"""Run a live Ollama probe that demonstrates local skill use."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import common

from llm_tools.llm_providers import OpenAICompatibleProvider  # noqa: E402
from llm_tools.skills_api import (  # noqa: E402
    SkillInvocation,
    SkillInvocationType,
    SkillRoot,
    SkillScope,
    build_skill_usage_record,
    discover_skills,
    load_skill_context,
    render_available_skills_context,
    render_loaded_skill_context,
    resolve_skill,
)

DEFAULT_MARKER = "SKILL_E2E_OK"
SKILL_NAME = "ollama-skill-e2e"


def build_parser() -> argparse.ArgumentParser:
    """Build the Ollama skill probe CLI."""
    parser = argparse.ArgumentParser(
        description="Run a live Ollama probe proving that SKILL.md context is used."
    )
    parser.add_argument("--model", default=common.DEFAULT_MODEL)
    parser.add_argument("--ollama-base-url", default=common.DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--output-dir")
    parser.add_argument("--marker", default=DEFAULT_MARKER)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the skill probe and write artifacts."""
    args = build_parser().parse_args(argv)
    output_dir = common.resolve_output_dir(args.output_dir, kind="ollama-skill")
    result = run_skill_probe(
        output_dir=output_dir,
        model=args.model,
        ollama_base_url=args.ollama_base_url,
        marker=args.marker,
        timeout_seconds=args.timeout_seconds,
    )
    common.write_json(output_dir / "skill_eval.json", result)
    common.write_text(output_dir / "skill_eval.md", _render_markdown(result))
    print(f"Ollama skill eval artifacts written to {output_dir}")
    return common.final_exit_code([result])


def run_skill_probe(
    *,
    output_dir: Path,
    model: str,
    ollama_base_url: str,
    marker: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run one local skill through an Ollama plain-text model call."""
    skill_root = _write_probe_skill(output_dir / "skill-root", marker=marker)
    discovery = discover_skills([SkillRoot(path=skill_root, scope=SkillScope.USER)])
    available_context = render_available_skills_context(discovery)
    result: dict[str, Any] = {
        "name": "ollama_skill_e2e",
        "kind": "skill",
        "model": model,
        "ollama_base_url": ollama_base_url,
        "skill_root": str(skill_root),
        "marker": marker,
        "discovery": common.dump_model(discovery),
        "available_skills_context": common.dump_model(available_context),
    }
    try:
        resolved = resolve_skill(discovery, SkillInvocation(name=SKILL_NAME))
        loaded_context = load_skill_context(resolved.skill)
        rendered_skill = render_loaded_skill_context(loaded_context)
        usage_record = build_skill_usage_record(
            resolved.skill,
            invocation_type=SkillInvocationType.EXPLICIT,
            contents=loaded_context.contents,
        )
        provider = OpenAICompatibleProvider.for_ollama(
            model=model,
            base_url=ollama_base_url,
            default_request_params={
                "temperature": 0,
                "timeout": timeout_seconds,
            },
        )
        response_text = provider.run_text(
            messages=_messages_for_skill_probe(
                available_context=available_context.rendered_text
                if available_context is not None
                else "",
                loaded_skill_context=rendered_skill,
                marker=marker,
            ),
            request_params={"temperature": 0},
        )
        checks = _evaluate_skill_response(response_text, marker)
        result.update(
            {
                "loaded_skill_context": common.dump_model(loaded_context),
                "usage_record": common.dump_model(usage_record),
                "response_text": response_text,
                "checks": checks,
                "status": (
                    common.SCENARIO_STATUS_PASSED
                    if all(checks.values())
                    else common.SCENARIO_STATUS_FAILED
                ),
                "summary": (
                    "Ollama followed the loaded skill instructions."
                    if all(checks.values())
                    else "Ollama did not return the skill marker as instructed."
                ),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": common.SCENARIO_STATUS_INFRA,
                "summary": "Ollama skill probe could not complete.",
                "failure": common.failure_payload(exc),
            }
        )
    return result


def _write_probe_skill(root: Path, *, marker: str) -> Path:
    skill_dir = root / SKILL_NAME
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {SKILL_NAME}",
                "description: Answer the Ollama skill E2E verification prompt with the requested marker.",
                "---",
                "",
                "# Ollama Skill E2E",
                "",
                "When this skill is loaded, answer verification prompts with the exact",
                f"marker `{marker}` and no explanatory prose.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _messages_for_skill_probe(
    *,
    available_context: str,
    loaded_skill_context: str,
    marker: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are running an llm-tools skill integration check. "
                "Follow the loaded SKILL.md instructions exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{available_context}\n"
                f"{loaded_skill_context}\n\n"
                "Task: Use the loaded skill to answer this verification prompt. "
                f"Return exactly `{marker}` and no other text."
            ),
        },
    ]


def _evaluate_skill_response(response_text: str, marker: str) -> dict[str, bool]:
    normalized = _normalize_response_text(response_text)
    return {
        "marker_present": marker in response_text,
        "normalized_exact_marker": normalized == marker,
    }


def _normalize_response_text(response_text: str) -> str:
    cleaned = response_text.strip()
    if cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned.strip("`").strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned.strip('"').strip()
    return cleaned


def _render_markdown(result: dict[str, Any]) -> str:
    checks = result.get("checks") or {}
    lines = [
        "# Ollama Skill E2E",
        "",
        f"- Status: {result.get('status')}",
        f"- Model: {result.get('model')}",
        f"- Skill: {SKILL_NAME}",
        f"- Marker: {result.get('marker')}",
        f"- Summary: {result.get('summary')}",
        "",
        "## Checks",
    ]
    if checks:
        lines.extend(f"- {name}: {value}" for name, value in checks.items())
    else:
        lines.append("- No checks were recorded.")
    if result.get("response_text") is not None:
        lines.extend(["", "## Response", "", str(result["response_text"])])
    if result.get("failure") is not None:
        lines.extend(["", "## Failure", "", str(result["failure"])])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

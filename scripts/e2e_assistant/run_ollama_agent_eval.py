"""Run a compact, repeatable Ollama agent reliability evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any

import common

DEFAULT_MODEL_PROFILES = [
    {
        "model": "gemma4:e2b",
        "profile": "small/new/non-reasoning",
        "rationale": "Small recent Gemma-family model.",
    },
    {
        "model": "gemma3:12b",
        "profile": "mid/older/non-reasoning",
        "rationale": "Older mid-sized Gemma baseline.",
    },
    {
        "model": "deepseek-r1:7b-qwen-distill-q4_K_M",
        "profile": "small/reasoning/distilled",
        "rationale": "Reasoning-style local model without adding a 32B run.",
    },
    {
        "model": "gemma4:26b",
        "profile": "large/new/non-reasoning",
        "rationale": "Large recent Gemma-family model.",
    },
]
DEFAULT_PROVIDER_MODES = [
    "json",
    "prompt_tools",
]
DEFAULT_SCENARIOS = ["chat_repo_lookup"]
DEFAULT_ROUND_CAP = 8


def build_parser() -> argparse.ArgumentParser:
    """Build the Ollama agent eval CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a compact Ollama agent modality evaluation and summarize the artifacts."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model id to test. May be repeated. Defaults to a curated 4-model "
            "small/mid/large/reasoning set."
        ),
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model ids. Overrides the default model set when provided.",
    )
    parser.add_argument(
        "--provider-modes",
        default=",".join(DEFAULT_PROVIDER_MODES),
        help="Comma-separated provider modes to test.",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated backend matrix scenarios to test.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=common.DEFAULT_OLLAMA_BASE_URL,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--output-dir",
        help="Artifact output directory. Defaults to a timestamped temp directory.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only summarize an existing output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run or summarize the repeatable Ollama agent evaluation."""
    args = build_parser().parse_args(argv)
    output_dir = _output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = _selected_profiles(args.model, args.models)
    modes = _csv(args.provider_modes)
    scenarios = _csv(args.scenarios)

    run_records: list[dict[str, Any]] = []
    if not args.skip_run:
        for profile in profiles:
            model = str(profile["model"])
            model_dir = output_dir / _safe_name(model)
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_backend_matrix.py")),
                "--model",
                model,
                "--provider-modes",
                ",".join(modes),
                "--scenarios",
                ",".join(scenarios),
                "--ollama-base-url",
                args.ollama_base_url,
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--output-dir",
                str(model_dir),
            ]
            completed = subprocess.run(command, check=False)
            run_records.append(
                {
                    "model": model,
                    "profile": profile.get("profile"),
                    "returncode": completed.returncode,
                    "output_dir": str(model_dir),
                }
            )

    summary = _summarize(
        output_dir=output_dir,
        profiles=profiles,
        modes=modes,
        scenarios=scenarios,
        run_records=run_records,
    )
    (output_dir / "agent_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "agent_eval_summary.md").write_text(
        _render_markdown(summary),
        encoding="utf-8",
    )
    print(f"Ollama agent eval artifacts written to {output_dir}")
    return 0


def _output_dir(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(gettempdir()) / f"llm-tools-ollama-agent-eval-{stamp}"


def _selected_profiles(
    model_args: list[str],
    models_csv: str | None,
) -> list[dict[str, str]]:
    selected = [value for raw in model_args for value in _csv(raw)]
    if models_csv:
        selected.extend(_csv(models_csv))
    if not selected:
        return [dict(profile) for profile in DEFAULT_MODEL_PROFILES]
    return [
        {
            "model": model,
            "profile": "custom",
            "rationale": "Selected from CLI.",
        }
        for model in selected
    ]


def _csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-")


def _summarize(
    *,
    output_dir: Path,
    profiles: list[dict[str, str]],
    modes: list[str],
    scenarios: list[str],
    run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        model = profile["model"]
        model_dir = output_dir / _safe_name(model)
        for mode in modes:
            for scenario in scenarios:
                artifact = model_dir / mode / f"{scenario}.json"
                rows.append(
                    _summarize_artifact(
                        artifact=artifact,
                        model=model,
                        profile=profile.get("profile", ""),
                        mode=mode,
                        scenario=scenario,
                    )
                )
    return {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "models": profiles,
        "provider_modes": modes,
        "scenarios": scenarios,
        "run_records": run_records,
        "rows": rows,
        "rollup": _rollup(rows),
    }


def _summarize_artifact(
    *,
    artifact: Path,
    model: str,
    profile: str,
    mode: str,
    scenario: str,
) -> dict[str, Any]:
    base = {
        "model": model,
        "profile": profile,
        "provider_mode": mode,
        "scenario": scenario,
        "artifact": str(artifact),
    }
    if not artifact.exists():
        return {
            **base,
            "status": "missing",
            "summary": "No artifact was written.",
            "resolved_mode": None,
            "checks": None,
            "tool_sequence": [],
            "tool_call_count": 0,
            "exact_duplicate_tool_calls": 0,
            "tool_name_counts": {},
            "repair_count": 0,
            "hit_default_round_cap": False,
            "final_response_present": False,
            "final_answer_chars": 0,
            "answer_quality": {
                "expected_keyword_hits": 0,
                "expected_keywords": [],
                "refusal_like": False,
                "quality_passed": False,
            },
            "failure": "No artifact was written.",
            "failure_type": "missing_artifact",
        }
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    tool_calls = _tool_calls(payload)
    exact_counts = Counter(_tool_call_key(call) for call in tool_calls)
    duplicate_exact_calls = sum(
        count - 1 for count in exact_counts.values() if count > 1
    )
    status_events = payload.get("status_events") or []
    final_response = payload.get("final_response")
    answer = (final_response or {}).get("answer", "")
    quality = _answer_quality(scenario=scenario, answer=answer)
    return {
        **base,
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "continuation_refusal_attempted": bool(
            payload.get("continuation_refusal_attempted")
        ),
        "resolved_mode": (payload.get("provider_health") or {}).get("resolved_mode"),
        "checks": payload.get("checks"),
        "tool_sequence": payload.get("tool_sequence") or [],
        "tool_call_count": len(tool_calls),
        "exact_duplicate_tool_calls": duplicate_exact_calls,
        "tool_name_counts": dict(Counter(call["tool_name"] for call in tool_calls)),
        "repair_count": sum(
            1
            for event in status_events
            if isinstance(event, str) and event.startswith("repairing ")
        ),
        "hit_default_round_cap": len(tool_calls) >= DEFAULT_ROUND_CAP,
        "final_response_present": final_response is not None,
        "final_answer_chars": len(answer),
        "answer_quality": quality,
        "failure": (payload.get("failure") or {}).get("message"),
        "failure_type": (payload.get("failure") or {}).get("type"),
    }


def _tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = (payload.get("inspector") or {}).get("parsed_responses") or []
    calls: list[dict[str, Any]] = []
    for response in parsed:
        if not isinstance(response, dict):
            continue
        for invocation in response.get("invocations") or []:
            if not isinstance(invocation, dict):
                continue
            tool_name = invocation.get("tool_name")
            arguments = invocation.get("arguments")
            if isinstance(tool_name, str):
                calls.append(
                    {
                        "tool_name": tool_name,
                        "arguments": arguments if isinstance(arguments, dict) else {},
                        "tool_call_id": invocation.get("tool_call_id"),
                    }
                )
    if calls:
        return calls
    return [
        {"tool_name": tool_name, "arguments": {}}
        for tool_name in payload.get("tool_sequence") or []
        if isinstance(tool_name, str)
    ]


def _tool_call_key(call: dict[str, Any]) -> str:
    return json.dumps(call, sort_keys=True, separators=(",", ":"), default=str)


def _rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") != "missing"]
    passed = [
        row for row in completed if row.get("status") == common.SCENARIO_STATUS_PASSED
    ]
    quality_passed = [
        row
        for row in completed
        if (row.get("answer_quality") or {}).get("quality_passed")
    ]
    return {
        "total_runs": len(rows),
        "completed_artifacts": len(completed),
        "passed": len(passed),
        "pass_rate": (len(passed) / len(completed)) if completed else 0.0,
        "quality_passed": len(quality_passed),
        "quality_pass_rate": (
            len(quality_passed) / len(completed) if completed else 0.0
        ),
        "round_cap_hits": sum(
            1 for row in completed if row.get("hit_default_round_cap")
        ),
        "total_exact_duplicate_tool_calls": sum(
            int(row.get("exact_duplicate_tool_calls") or 0) for row in completed
        ),
    }


def _answer_quality(*, scenario: str, answer: object) -> dict[str, Any]:
    text = answer if isinstance(answer, str) else ""
    lowered = text.lower()
    refusal_markers = [
        "unable to",
        "cannot access",
        "encountered errors",
        "not found",
        "workspace search",
        "prevents me",
    ]
    refusal_like = any(marker in lowered for marker in refusal_markers)
    if scenario != "chat_repo_lookup":
        return {
            "expected_keyword_hits": 0,
            "refusal_like": refusal_like,
            "quality_passed": bool(text.strip()) and not refusal_like,
        }
    expected_terms = [
        "harness_api",
        "harnesssessionservice",
        "assistantresearchsessioncontroller",
        "assistant_research_provider",
        "nicegui_chat",
        "durable orchestration",
        "resume",
        "inspect",
        "session",
    ]
    hits = sorted(term for term in expected_terms if term in lowered)
    return {
        "expected_keyword_hits": len(hits),
        "expected_keywords": hits,
        "refusal_like": refusal_like,
        "quality_passed": len(hits) >= 3 and not refusal_like and len(text) >= 200,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Ollama Agent Evaluation",
        "",
        f"Generated: {summary['generated_at']}",
        f"Artifacts: `{summary['output_dir']}`",
        "",
        "## Rollup",
        "",
    ]
    rollup = summary["rollup"]
    lines.extend(
        [
            f"- Runs with artifacts: {rollup['completed_artifacts']} / {rollup['total_runs']}",
            f"- Passed: {rollup['passed']}",
            f"- Pass rate: {rollup['pass_rate']:.0%}",
            f"- Quality passed: {rollup['quality_passed']}",
            f"- Quality pass rate: {rollup['quality_pass_rate']:.0%}",
            f"- Default round-cap hits: {rollup['round_cap_hits']}",
            (
                "- Exact duplicate tool calls: "
                f"{rollup['total_exact_duplicate_tool_calls']}"
            ),
            "",
            "## Results",
            "",
            "| Model | Profile | Mode | Scenario | Status | Quality | Tools | Exact Dupes | Repairs | Round Cap | Final Chars |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        quality = row.get("answer_quality") or {}
        lines.append(
            "| {model} | {profile} | {provider_mode} | {scenario} | {status} | "
            "{quality_passed} | {tool_call_count} | {exact_duplicate_tool_calls} | "
            "{repair_count} | {hit_default_round_cap} | {final_answer_chars} |".format(
                quality_passed=quality.get("quality_passed"),
                **row,
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

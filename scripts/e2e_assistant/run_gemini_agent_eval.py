"""Run a compact, repeatable Gemini agent reliability evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any

from run_ollama_agent_eval import (
    DEFAULT_SCENARIOS,
    _csv,
    _rollup,
    _safe_name,
    _summarize_artifact,
)

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_API_KEY_FILE = "google-ai-studio-api-key.txt"
GEMINI_API_KEY_ENV_VAR = "GOOGLE_AI_STUDIO_API_KEY"
NATIVE_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_PROVIDER_MODES = [
    "tools",
    "json",
    "prompt_tools",
]
DEFAULT_MODEL_PROFILES = [
    {
        "model": "gemini-3.1-flash-lite-preview",
        "profile": "small/gemini-3/reasoning",
        "rationale": "Fastest available Gemini 3-family text candidate.",
    },
    {
        "model": "gemini-3-flash-preview",
        "profile": "balanced/gemini-3/reasoning",
        "rationale": "Balanced Gemini 3 Flash text candidate.",
    },
    {
        "model": "gemini-3.1-pro-preview",
        "profile": "large/gemini-3/reasoning",
        "rationale": "Strongest available Gemini 3.1 Pro text candidate.",
    },
]
EXCLUDED_MODEL_FRAGMENTS = (
    "customtools",
    "image",
    "live",
    "nano-banana",
    "tts",
    "lyria",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the Gemini agent eval CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a compact Gemini agent modality evaluation and summarize the artifacts."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model id to test. May be repeated. Defaults to a discovered Gemini 3 "
            "small/balanced/large set."
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
        "--gemini-base-url",
        default=DEFAULT_GEMINI_BASE_URL,
    )
    parser.add_argument(
        "--api-key-file",
        default=DEFAULT_API_KEY_FILE,
        help="Path to the temporary Google AI Studio API key file.",
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
    """Run or summarize the repeatable Gemini agent evaluation."""
    args = build_parser().parse_args(argv)
    output_dir = _output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovery: dict[str, Any] | None = None
    api_key: str | None = None
    if not args.skip_run or Path(args.api_key_file).expanduser().exists():
        api_key = _read_api_key(Path(args.api_key_file))
        discovery = discover_gemini_models(
            api_key=api_key,
            openai_base_url=args.gemini_base_url,
        )

    profiles = _selected_profiles(args.model, args.models, discovery=discovery)
    modes = _csv(args.provider_modes)
    scenarios = _csv(args.scenarios)

    run_records: list[dict[str, Any]] = []
    if not args.skip_run:
        if api_key is None:
            api_key = _read_api_key(Path(args.api_key_file))
        for profile in profiles:
            model = str(profile["model"])
            model_dir = output_dir / _safe_name(model)
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_backend_matrix.py")),
                "--provider",
                "custom_openai_compatible",
                "--api-base-url",
                args.gemini_base_url,
                "--api-key-env-var",
                GEMINI_API_KEY_ENV_VAR,
                "--model",
                model,
                "--provider-modes",
                ",".join(modes),
                "--scenarios",
                ",".join(scenarios),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--output-dir",
                str(model_dir),
            ]
            env = dict(os.environ)
            env[GEMINI_API_KEY_ENV_VAR] = api_key
            completed = subprocess.run(command, check=False, env=env)
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
        discovery=discovery,
        gemini_base_url=args.gemini_base_url,
    )
    (output_dir / "agent_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "agent_eval_summary.md").write_text(
        _render_markdown(summary),
        encoding="utf-8",
    )
    print(f"Gemini agent eval artifacts written to {output_dir}")
    return 0


def _output_dir(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(gettempdir()) / f"llm-tools-gemini-agent-eval-{stamp}"


def _read_api_key(path: Path) -> str:
    key = path.expanduser().read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def discover_gemini_models(
    *,
    api_key: str,
    openai_base_url: str,
) -> dict[str, Any]:
    """Return Gemini 3 text models visible through native and OpenAI APIs."""
    native_payload = _fetch_json(
        NATIVE_MODELS_URL,
        headers={"x-goog-api-key": api_key},
    )
    openai_payload = _fetch_json(
        _openai_models_url(openai_base_url),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    native_ids = _native_text_model_ids(native_payload)
    openai_ids = _openai_text_model_ids(openai_payload)
    available_ids = sorted(set(native_ids).intersection(openai_ids))
    selected_ids = _default_selected_ids(available_ids)
    return {
        "native_gemini3_text_model_ids": native_ids,
        "openai_gemini3_text_model_ids": openai_ids,
        "available_gemini3_text_model_ids": available_ids,
        "selected_default_model_ids": selected_ids,
    }


def _fetch_json(url: str, *, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _openai_models_url(openai_base_url: str) -> str:
    return openai_base_url.rstrip("/") + "/models"


def _native_text_model_ids(payload: dict[str, Any]) -> list[str]:
    model_ids: list[str] = []
    for model in payload.get("models") or []:
        if not isinstance(model, dict):
            continue
        model_id = _normalize_model_id(str(model.get("name") or ""))
        methods = model.get("supportedGenerationMethods") or []
        if _is_gemini3_text_candidate(model_id) and "generateContent" in methods:
            model_ids.append(model_id)
    return sorted(set(model_ids))


def _openai_text_model_ids(payload: dict[str, Any]) -> list[str]:
    model_ids: list[str] = []
    for model in payload.get("data") or []:
        if not isinstance(model, dict):
            continue
        model_id = _normalize_model_id(str(model.get("id") or ""))
        if _is_gemini3_text_candidate(model_id):
            model_ids.append(model_id)
    return sorted(set(model_ids))


def _normalize_model_id(raw: str) -> str:
    return raw.removeprefix("models/")


def _is_gemini3_text_candidate(model_id: str) -> bool:
    lowered = model_id.lower()
    return lowered.startswith("gemini-3") and not any(
        fragment in lowered for fragment in EXCLUDED_MODEL_FRAGMENTS
    )


def _default_selected_ids(available_ids: list[str]) -> list[str]:
    preferred = [profile["model"] for profile in DEFAULT_MODEL_PROFILES]
    selected = [model for model in preferred if model in available_ids]
    if selected:
        return selected
    return available_ids[:3]


def _selected_profiles(
    model_args: list[str],
    models_csv: str | None,
    *,
    discovery: dict[str, Any] | None,
) -> list[dict[str, str]]:
    selected = [value for raw in model_args for value in _csv(raw)]
    if models_csv:
        selected.extend(_csv(models_csv))
    if selected:
        return [
            {
                "model": model,
                "profile": "custom",
                "rationale": "Selected from CLI.",
            }
            for model in selected
        ]

    selected_ids = []
    if discovery is not None:
        selected_ids = list(discovery.get("selected_default_model_ids") or [])
    if not selected_ids:
        selected_ids = [profile["model"] for profile in DEFAULT_MODEL_PROFILES]

    profile_by_model = {profile["model"]: profile for profile in DEFAULT_MODEL_PROFILES}
    return [
        dict(
            profile_by_model.get(
                model,
                {
                    "model": model,
                    "profile": "discovered/gemini-3",
                    "rationale": "Selected from discovered Gemini 3 text candidates.",
                },
            )
        )
        for model in selected_ids
    ]


def _summarize(
    *,
    output_dir: Path,
    profiles: list[dict[str, str]],
    modes: list[str],
    scenarios: list[str],
    run_records: list[dict[str, Any]],
    discovery: dict[str, Any] | None,
    gemini_base_url: str,
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
        "gemini_base_url": gemini_base_url,
        "models": profiles,
        "model_discovery": discovery,
        "provider_modes": modes,
        "scenarios": scenarios,
        "run_records": run_records,
        "rows": rows,
        "rollup": _rollup(rows),
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Gemini Agent Evaluation",
        "",
        f"Generated: {summary['generated_at']}",
        f"Artifacts: `{summary['output_dir']}`",
        f"Base URL: `{summary['gemini_base_url']}`",
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

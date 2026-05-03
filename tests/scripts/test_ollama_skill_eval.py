from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "e2e_assistant"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_ollama_skill_eval as skill_eval  # noqa: E402


def test_ollama_skill_eval_parser_defaults_to_common_model() -> None:
    args = skill_eval.build_parser().parse_args([])

    assert args.model
    assert args.marker == skill_eval.DEFAULT_MARKER


def test_ollama_skill_eval_writes_discoverable_probe_skill(tmp_path: Path) -> None:
    root = skill_eval._write_probe_skill(tmp_path / "skills", marker="MARKER")

    assert (root / skill_eval.SKILL_NAME / "SKILL.md").exists()


def test_ollama_skill_eval_response_normalization() -> None:
    assert skill_eval._normalize_response_text(" `SKILL_E2E_OK` \n") == "SKILL_E2E_OK"
    assert skill_eval._evaluate_skill_response("SKILL_E2E_OK", "SKILL_E2E_OK") == {
        "marker_present": True,
        "normalized_exact_marker": True,
    }


def test_ollama_skill_eval_reports_local_skill_errors_as_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("local skill setup broke")

    monkeypatch.setattr(skill_eval, "resolve_skill", boom)

    result = skill_eval.run_skill_probe(
        output_dir=tmp_path,
        model="test-model",
        ollama_base_url="http://127.0.0.1:11434/v1",
        marker=skill_eval.DEFAULT_MARKER,
        timeout_seconds=1.0,
    )

    assert result["status"] == skill_eval.common.SCENARIO_STATUS_FAILED
    assert result["failure"]["message"] == "local skill setup broke"


def test_ollama_skill_eval_reports_provider_errors_as_infra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingProvider:
        def run_text(self, **kwargs: object) -> str:
            del kwargs
            raise RuntimeError("ollama unavailable")

    class ProviderFactory:
        @staticmethod
        def for_ollama(**kwargs: object) -> FailingProvider:
            del kwargs
            return FailingProvider()

    monkeypatch.setattr(skill_eval, "OpenAICompatibleProvider", ProviderFactory)

    result = skill_eval.run_skill_probe(
        output_dir=tmp_path,
        model="test-model",
        ollama_base_url="http://127.0.0.1:11434/v1",
        marker=skill_eval.DEFAULT_MARKER,
        timeout_seconds=1.0,
    )

    assert result["status"] == skill_eval.common.SCENARIO_STATUS_INFRA
    assert result["failure"]["message"] == "ollama unavailable"

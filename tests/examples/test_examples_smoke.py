"""Smoke tests for runnable examples."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _run_example(name: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / name)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.mark.parametrize(
    "example_name",
    [
        "minimal_tool.py",
        "builtins_direct.py",
        "openai_wiring.py",
        "structured_response.py",
        "prompt_schema.py",
    ],
)
def test_offline_examples_run_successfully(example_name: str) -> None:
    result = _run_example(example_name)

    assert result.returncode == 0, result.stderr


def test_live_openai_example_fails_cleanly_without_api_key() -> None:
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "openai_live.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 1
    assert "OPENAI_API_KEY is required" in result.stderr

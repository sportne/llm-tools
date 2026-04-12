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


def test_live_ollama_example_fails_cleanly_when_server_is_unreachable() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["OLLAMA_BASE_URL"] = "http://127.0.0.1:9/v1"
    env["OLLAMA_MODEL"] = "gemma4:26b"

    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "openai_live.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 1
    assert "Failed to reach Ollama" in result.stderr

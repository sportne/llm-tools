"""Smoke tests for runnable examples."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.apps._imports import load_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


@dataclass(slots=True)
class _ExampleRunResult:
    exit_code: int
    stdout: str
    stderr: str


def _run_loaded_example(module: object) -> _ExampleRunResult:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    exit_code = 0

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        try:
            result = module.main()
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
        else:
            if isinstance(result, int):
                exit_code = result

    return _ExampleRunResult(
        exit_code=exit_code,
        stdout=stdout_buffer.getvalue(),
        stderr=stderr_buffer.getvalue(),
    )


def _run_example(
    name: str,
    *,
    fake_providers_enabled: bool = False,
) -> _ExampleRunResult:
    module = load_module_from_path(
        EXAMPLES_DIR / name,
        module_name=f"tests.examples.{Path(name).stem}",
        fake_providers_enabled=fake_providers_enabled,
    )
    return _run_loaded_example(module)


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

    assert result.exit_code == 0, result.stderr


def test_async_model_turn_example_runs_successfully_with_stubbed_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module_from_path(
        EXAMPLES_DIR / "async_model_turn.py",
        module_name="tests.examples.async_model_turn",
        fake_providers_enabled=True,
    )

    class _Dumpable:
        def __init__(self, label: str) -> None:
            self._label = label

        def model_dump(self, mode: str = "json") -> dict[str, str]:
            del mode
            return {"label": self._label}

    class _Provider:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def run_async(self, **kwargs: object) -> object:
            del kwargs
            return SimpleNamespace(final_response=None, invocations=[])

    class _Executor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def prepare_model_interaction(
            self, adapter: object, *, context: object, include_requires_approval: bool
        ) -> object:
            del adapter, context, include_requires_approval
            return SimpleNamespace(response_model=dict)

        async def execute_model_output_async(
            self, adapter: object, payload: object, context: object
        ) -> _Dumpable:
            del adapter, payload, context
            return _Dumpable("setup")

        async def execute_parsed_response_async(
            self, parsed: object, context: object
        ) -> _Dumpable:
            del parsed, context
            return _Dumpable("async")

    monkeypatch.setattr(module, "OpenAICompatibleProvider", _Provider)
    monkeypatch.setattr(module, "WorkflowExecutor", _Executor)

    result = _run_loaded_example(module)

    assert result.exit_code == 0, result.stderr
    assert "Setup turn:" in result.stdout
    assert "Async turn:" in result.stdout


def test_live_ollama_example_fails_cleanly_when_server_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module_from_path(
        EXAMPLES_DIR / "openai_live.py",
        module_name="tests.examples.openai_live",
    )

    class _BrokenProvider:
        def run(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError("connection refused")

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:26b")
    monkeypatch.setattr(
        module.OpenAICompatibleProvider,
        "for_ollama",
        classmethod(lambda cls, **kwargs: _BrokenProvider()),
    )

    result = _run_loaded_example(module)

    assert result.exit_code == 1
    assert "Failed to reach Ollama" in result.stderr

"""Launcher and packaging tests for the Textual workbench."""

from __future__ import annotations

import importlib
import runpy
import sys
import tomllib
from pathlib import Path

import pytest
from tests.apps._imports import import_textual_workbench_modules


def test_textual_workbench_package_imports_without_loading_textual() -> None:
    module = importlib.import_module("llm_tools.apps.textual_workbench")
    main_module = importlib.import_module("llm_tools.apps.textual_workbench.__main__")

    assert hasattr(module, "main")
    assert hasattr(module, "run_workbench_app")
    assert hasattr(main_module, "main")


def test_package_main_and_runner_dispatch_to_app_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.textual_workbench")
    called: list[str] = []

    monkeypatch.setattr(
        "llm_tools.apps.textual_workbench.app.run_workbench_app",
        lambda: called.append("run"),
    )
    monkeypatch.setattr(
        "llm_tools.apps.textual_workbench.app.main",
        lambda: called.append("main") or 0,
    )

    package.run_workbench_app()
    assert package.main() == 0
    assert called == ["run", "main"]


def test_console_script_target_is_declared_in_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["llm-tools-workbench"]
        == "llm_tools.apps.textual_workbench:main"
    )


def test_apps_optional_dependency_group_is_declared() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert "textual>=0.89.1,<1" in pyproject["project"]["optional-dependencies"]["apps"]


def test_module_main_helpers_dispatch_to_package_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.textual_workbench")
    main_module = importlib.import_module("llm_tools.apps.textual_workbench.__main__")

    monkeypatch.setattr(package, "main", lambda: 7)

    assert main_module._main() == 7
    assert main_module.main() == 7


def test_module_entrypoint_raises_system_exit_with_main_return_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.textual_workbench")
    called: list[str] = []
    monkeypatch.setattr(package, "main", lambda: called.append("main") or 0)

    sys.modules.pop("llm_tools.apps.textual_workbench.__main__", None)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(
            "llm_tools.apps.textual_workbench.__main__", run_name="__main__"
        )

    assert exc.value.code == 0
    assert called == ["main"]


def test_app_module_run_helpers_dispatch_to_textual_app_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_module = import_textual_workbench_modules().app
    called: list[str] = []

    monkeypatch.setattr(
        "llm_tools.apps.textual_workbench.app.TextualWorkbenchApp.run",
        lambda self: called.append("run"),
    )

    app_module.run_workbench_app()
    assert app_module.main() == 0
    assert called == ["run", "run"]

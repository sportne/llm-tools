"""Launcher and packaging tests for the remaining app entrypoints."""

from __future__ import annotations

import importlib
import runpy
import sys
import tomllib
from pathlib import Path

import pytest


def test_assistant_app_package_exports_main_and_runner() -> None:
    module = importlib.import_module("llm_tools.apps.assistant_app")
    main_module = importlib.import_module("llm_tools.apps.assistant_app.__main__")

    assert hasattr(module, "main")
    assert hasattr(module, "run_assistant_app")
    assert hasattr(main_module, "main")


def test_assistant_module_main_helper_reexports_package_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.assistant_app")
    main_module = importlib.import_module("llm_tools.apps.assistant_app.__main__")

    monkeypatch.setattr(package, "main", lambda: 7)

    assert main_module.main is not None


def test_assistant_module_entrypoint_calls_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        "llm_tools.apps.assistant_app.main", lambda: called.append("main")
    )

    sys.modules.pop("llm_tools.apps.assistant_app.__main__", None)
    runpy.run_module(
        "llm_tools.apps.assistant_app.__main__",
        run_name="__main__",
    )

    assert called == ["main"]


def test_remaining_console_scripts_are_declared_and_streamlit_scripts_are_gone() -> (
    None
):
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts["llm-tools-harness"] == "llm_tools.apps.harness_cli:main"
    assert scripts["llm-tools-assistant"] == "llm_tools.apps.assistant_app:main"
    assert "llm-tools-streamlit-assistant" not in scripts
    assert "llm-tools-chat" not in scripts
    assert "llm-tools-workbench" not in scripts


def test_dependency_groups_keep_nicegui_in_base_install_and_streamlit_removed() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    dependencies = project["dependencies"]
    optional_dependencies = project["optional-dependencies"]
    dev_dependencies = optional_dependencies["dev"]

    assert any(dependency.startswith("nicegui>=") for dependency in dependencies)
    assert any(dependency.startswith("PyYAML>=") for dependency in dependencies)
    assert all(not dependency.startswith("streamlit") for dependency in dependencies)
    assert "streamlit" not in optional_dependencies
    assert "apps" not in optional_dependencies
    assert all("textual" not in dependency for dependency in dev_dependencies)

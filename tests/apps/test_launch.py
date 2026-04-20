"""Launcher and packaging tests for the remaining app entrypoints."""

from __future__ import annotations

import importlib
import runpy
import sys
import tomllib
from pathlib import Path

import pytest
from tests.apps._imports import import_streamlit_assistant_modules


def test_streamlit_assistant_package_exports_main_and_runner() -> None:
    module = importlib.import_module("llm_tools.apps.streamlit_assistant")
    main_module = importlib.import_module("llm_tools.apps.streamlit_assistant.__main__")

    assert hasattr(module, "main")
    assert hasattr(module, "run_streamlit_assistant_app")
    assert hasattr(main_module, "main")


def test_assistant_package_main_and_runner_dispatch_to_app_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_assistant")
    called: list[str] = []

    monkeypatch.setattr(
        "llm_tools.apps.streamlit_assistant.app.run_streamlit_assistant_app",
        lambda **kwargs: called.append(f"run:{kwargs!r}"),
    )
    monkeypatch.setattr(
        "llm_tools.apps.streamlit_assistant.app.main",
        lambda argv=None: called.append("main") or 0,
    )

    package.run_streamlit_assistant_app(root_path=None, config=None)
    assert package.main() == 0
    assert len(called) == 2
    assert called[1] == "main"


def test_remaining_console_scripts_are_declared_and_textual_scripts_are_gone() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts["llm-tools-harness"] == "llm_tools.apps.harness_cli:main"
    assert (
        scripts["llm-tools-streamlit-assistant"]
        == "llm_tools.apps.streamlit_assistant:main"
    )
    assert "llm-tools-chat" not in scripts
    assert "llm-tools-workbench" not in scripts
    assert "llm-tools-streamlit-chat" not in scripts


def test_dependency_groups_keep_streamlit_in_base_install_and_textual_removed() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    dependencies = project["dependencies"]
    optional_dependencies = project["optional-dependencies"]
    dev_dependencies = optional_dependencies["dev"]

    assert any(dependency.startswith("streamlit>=") for dependency in dependencies)
    assert any(dependency.startswith("PyYAML>=") for dependency in dependencies)
    assert "streamlit" not in optional_dependencies
    assert "apps" not in optional_dependencies
    assert all("textual" not in dependency for dependency in dev_dependencies)


def test_assistant_module_main_helpers_dispatch_to_package_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_assistant")
    main_module = importlib.import_module("llm_tools.apps.streamlit_assistant.__main__")

    monkeypatch.setattr(package, "main", lambda: 7)

    assert main_module._main() == 7
    assert main_module.main() == 7


def test_assistant_module_entrypoint_raises_system_exit_with_main_return_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_assistant")
    called: list[str] = []
    monkeypatch.setattr(package, "main", lambda: called.append("main") or 0)

    sys.modules.pop("llm_tools.apps.streamlit_assistant.__main__", None)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(
            "llm_tools.apps.streamlit_assistant.__main__",
            run_name="__main__",
        )

    assert exc.value.code == 0
    assert called == ["main"]


def test_assistant_app_module_run_helpers_exist() -> None:
    app_module = import_streamlit_assistant_modules().app

    assert hasattr(app_module, "run_streamlit_assistant_app")
    assert hasattr(app_module, "main")

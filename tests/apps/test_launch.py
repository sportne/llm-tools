"""Launcher and packaging tests for the remaining app entrypoints."""

from __future__ import annotations

import importlib
import runpy
import sys
import tomllib
from pathlib import Path

import pytest
from tests.apps._imports import (
    import_streamlit_assistant_modules,
    import_streamlit_chat_modules,
)


@pytest.mark.parametrize(
    ("module_name", "main_module_name", "runner_name"),
    [
        (
            "llm_tools.apps.streamlit_chat",
            "llm_tools.apps.streamlit_chat.__main__",
            "run_streamlit_chat_app",
        ),
        (
            "llm_tools.apps.streamlit_assistant",
            "llm_tools.apps.streamlit_assistant.__main__",
            "run_streamlit_assistant_app",
        ),
    ],
)
def test_streamlit_packages_export_main_and_runner(
    module_name: str,
    main_module_name: str,
    runner_name: str,
) -> None:
    module = importlib.import_module(module_name)
    main_module = importlib.import_module(main_module_name)

    assert hasattr(module, "main")
    assert hasattr(module, runner_name)
    assert hasattr(main_module, "main")


@pytest.mark.parametrize(
    ("package_name", "runner_path", "main_path"),
    [
        (
            "llm_tools.apps.streamlit_chat",
            "llm_tools.apps.streamlit_chat.app.run_streamlit_chat_app",
            "llm_tools.apps.streamlit_chat.app.main",
        ),
        (
            "llm_tools.apps.streamlit_assistant",
            "llm_tools.apps.streamlit_assistant.app.run_streamlit_assistant_app",
            "llm_tools.apps.streamlit_assistant.app.main",
        ),
    ],
)
def test_package_main_and_runner_dispatch_to_app_layer(
    package_name: str,
    runner_path: str,
    main_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module(package_name)
    called: list[str] = []

    monkeypatch.setattr(runner_path, lambda **kwargs: called.append(f"run:{kwargs!r}"))
    monkeypatch.setattr(main_path, lambda argv=None: called.append("main") or 0)

    getattr(package, next(name for name in dir(package) if name.startswith("run_")))(
        root_path=None,
        config=None,
    )
    assert package.main() == 0
    assert len(called) == 2
    assert called[1] == "main"


def test_remaining_console_scripts_are_declared_and_textual_scripts_are_gone() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts["llm-tools-harness"] == "llm_tools.apps.harness_cli:main"
    assert scripts["llm-tools-streamlit-chat"] == "llm_tools.apps.streamlit_chat:main"
    assert (
        scripts["llm-tools-streamlit-assistant"]
        == "llm_tools.apps.streamlit_assistant:main"
    )
    assert "llm-tools-chat" not in scripts
    assert "llm-tools-workbench" not in scripts


def test_textual_dependency_hooks_are_removed() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    optional_dependencies = pyproject["project"]["optional-dependencies"]
    dev_dependencies = optional_dependencies["dev"]

    assert "apps" not in optional_dependencies
    assert all("textual" not in dependency for dependency in dev_dependencies)


@pytest.mark.parametrize(
    ("package_name", "main_module_name"),
    [
        ("llm_tools.apps.streamlit_chat", "llm_tools.apps.streamlit_chat.__main__"),
        (
            "llm_tools.apps.streamlit_assistant",
            "llm_tools.apps.streamlit_assistant.__main__",
        ),
    ],
)
def test_module_main_helpers_dispatch_to_package_main(
    package_name: str,
    main_module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module(package_name)
    main_module = importlib.import_module(main_module_name)

    monkeypatch.setattr(package, "main", lambda: 7)

    assert main_module._main() == 7
    assert main_module.main() == 7


@pytest.mark.parametrize(
    ("package_name", "main_module_name"),
    [
        ("llm_tools.apps.streamlit_chat", "llm_tools.apps.streamlit_chat.__main__"),
        (
            "llm_tools.apps.streamlit_assistant",
            "llm_tools.apps.streamlit_assistant.__main__",
        ),
    ],
)
def test_module_entrypoint_raises_system_exit_with_main_return_code(
    package_name: str,
    main_module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module(package_name)
    called: list[str] = []
    monkeypatch.setattr(package, "main", lambda: called.append("main") or 0)

    sys.modules.pop(main_module_name, None)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(main_module_name, run_name="__main__")

    assert exc.value.code == 0
    assert called == ["main"]


@pytest.mark.parametrize(
    ("importer", "app_module_path", "runner_class_path"),
    [
        (
            import_streamlit_chat_modules,
            "llm_tools.apps.streamlit_chat.app",
            "run_streamlit_chat_app",
        ),
        (
            import_streamlit_assistant_modules,
            "llm_tools.apps.streamlit_assistant.app",
            "run_streamlit_assistant_app",
        ),
    ],
)
def test_app_module_run_helpers_exist(
    importer: object,
    app_module_path: str,
    runner_class_path: str,
) -> None:
    del app_module_path
    app_module = importer().app

    assert hasattr(app_module, runner_class_path)
    assert hasattr(app_module, "main")

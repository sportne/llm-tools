"""Smoke tests for the package scaffold."""

from __future__ import annotations

import importlib
import pkgutil
import tomllib
from pathlib import Path

import llm_tools


def test_top_level_package_exports_version() -> None:
    assert llm_tools.__version__


def test_pyproject_uses_package_version_as_single_source_of_truth() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"] == (
        "llm_tools.__version__"
    )


def test_pyproject_includes_current_app_runtime_dependencies_in_base_install() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("nicegui>=") for dep in dependencies)
    assert any(dep.startswith("pathspec>=") for dep in dependencies)
    assert any(dep.startswith("sqlcipher3-wheels>=") for dep in dependencies)
    assert all(not dep.startswith("sqlcipher3-binary") for dep in dependencies)
    assert all(not dep.startswith("streamlit") for dep in dependencies)
    assert any(dep.startswith("PyYAML>=") for dep in dependencies)


def test_scaffolded_subpackages_are_importable() -> None:
    for module_name in (
        "llm_tools.apps",
        "llm_tools.tool_api",
        "llm_tools.llm_adapters",
        "llm_tools.llm_providers",
        "llm_tools.tools",
        "llm_tools.workflow_api",
        "llm_tools.harness_api",
        "llm_tools.harness_api.context",
        "llm_tools.harness_api.planning",
        "llm_tools.harness_api.replay",
        "llm_tools.harness_api.session",
        "llm_tools.harness_api.verification",
    ):
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_verification_submodule_imports_directly() -> None:
    module = importlib.import_module("llm_tools.harness_api.verification")

    assert module.__name__ == "llm_tools.harness_api.verification"


def test_context_planning_replay_and_session_submodules_import_directly() -> None:
    context_module = importlib.import_module("llm_tools.harness_api.context")
    planning_module = importlib.import_module("llm_tools.harness_api.planning")
    replay_module = importlib.import_module("llm_tools.harness_api.replay")
    session_module = importlib.import_module("llm_tools.harness_api.session")

    assert context_module.__name__ == "llm_tools.harness_api.context"
    assert planning_module.__name__ == "llm_tools.harness_api.planning"
    assert replay_module.__name__ == "llm_tools.harness_api.replay"
    assert session_module.__name__ == "llm_tools.harness_api.session"


def test_expected_subpackages_exist_under_llm_tools() -> None:
    discovered = {module.name for module in pkgutil.iter_modules(llm_tools.__path__)}
    assert {
        "apps",
        "llm_adapters",
        "llm_providers",
        "tool_api",
        "tools",
        "workflow_api",
        "harness_api",
    } <= discovered

"""Smoke tests for the package scaffold."""

from __future__ import annotations

import importlib
import pkgutil

import llm_tools


def test_top_level_package_exports_version() -> None:
    assert llm_tools.__version__ == "0.1.0"


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
        "llm_tools.harness_api.verification",
    ):
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_verification_submodule_imports_directly() -> None:
    module = importlib.import_module("llm_tools.harness_api.verification")

    assert module.__name__ == "llm_tools.harness_api.verification"


def test_context_and_planning_submodules_import_directly() -> None:
    context_module = importlib.import_module("llm_tools.harness_api.context")
    planning_module = importlib.import_module("llm_tools.harness_api.planning")

    assert context_module.__name__ == "llm_tools.harness_api.context"
    assert planning_module.__name__ == "llm_tools.harness_api.planning"


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

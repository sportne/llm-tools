"""Smoke tests for the Step 0 package scaffold."""

from __future__ import annotations

import importlib
import pkgutil

import llm_tools


def test_top_level_package_exports_version() -> None:
    assert llm_tools.__version__ == "0.1.0"


def test_scaffolded_subpackages_are_importable() -> None:
    for module_name in (
        "llm_tools.tool_api",
        "llm_tools.llm_adapters",
        "llm_tools.tools",
        "llm_tools.workflow_api",
    ):
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_expected_subpackages_exist_under_llm_tools() -> None:
    discovered = {module.name for module in pkgutil.iter_modules(llm_tools.__path__)}
    assert {"llm_adapters", "tool_api", "tools", "workflow_api"} <= discovered

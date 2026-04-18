"""Static AST checks for architectural layering boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

FORBIDDEN_IMPORTS_BY_LAYER = {
    "tool_api": (
        "llm_tools.llm_adapters",
        "llm_tools.llm_providers",
        "llm_tools.workflow_api",
        "llm_tools.harness_api",
        "llm_tools.tools",
        "llm_tools.apps",
    ),
    "llm_adapters": (
        "llm_tools.llm_providers",
        "llm_tools.workflow_api",
        "llm_tools.harness_api",
        "llm_tools.tools",
        "llm_tools.apps",
    ),
    "llm_providers": (
        "llm_tools.workflow_api",
        "llm_tools.harness_api",
        "llm_tools.tools",
        "llm_tools.apps",
    ),
    "tools": (
        "llm_tools.llm_adapters",
        "llm_tools.llm_providers",
        "llm_tools.workflow_api",
        "llm_tools.harness_api",
        "llm_tools.apps",
    ),
    "workflow_api": (
        "llm_tools.llm_providers",
        "llm_tools.harness_api",
        "llm_tools.apps",
    ),
    "harness_api": (
        "llm_tools.tools",
        "llm_tools.apps",
    ),
}


@pytest.mark.parametrize(
    ("layer_name", "forbidden_prefixes"),
    list(FORBIDDEN_IMPORTS_BY_LAYER.items()),
)
def test_layer_import_boundaries(
    layer_name: str, forbidden_prefixes: tuple[str, ...]
) -> None:
    from . import _helpers as helpers

    layer_root = helpers.PACKAGE_DIR / layer_name
    violations: list[str] = []

    for source_file in helpers.iter_python_files(layer_root):
        for reference in helpers.iter_import_references(source_file):
            for forbidden_prefix in forbidden_prefixes:
                if helpers.is_forbidden_module(reference.module, forbidden_prefix):
                    violations.append(
                        helpers.format_reference(
                            reference.path,
                            reference.lineno,
                            f"{layer_name} must not import '{reference.module}'",
                        )
                    )
                    break

    assert not violations, (
        f"Forbidden imports found in layer '{layer_name}':\n" + "\n".join(violations)
    )


def test_from_imports_resolve_full_module_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from . import _helpers as helpers

    src_dir = tmp_path / "src"
    module_path = src_dir / "llm_tools" / "example.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "\n".join(
            (
                "from llm_tools import harness_api",
                "from . import harness_api as harness_api_local",
                "from .. import tools",
                "from llm_tools.harness_api import models",
                "import llm_tools.tools as tools_module",
            )
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(helpers, "SRC_DIR", src_dir)

    modules = {
        reference.module for reference in helpers.iter_import_references(module_path)
    }

    assert {
        "llm_tools.harness_api",
        "llm_tools.tools",
        "llm_tools.harness_api.models",
    } <= modules

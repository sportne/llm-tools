"""Static AST checks for architectural layering boundaries."""

from __future__ import annotations

import pytest
from tests.architecture._helpers import (
    PACKAGE_DIR,
    format_reference,
    is_forbidden_module,
    iter_import_references,
    iter_python_files,
)

FORBIDDEN_IMPORTS_BY_LAYER = {
    "tool_api": (
        "llm_tools.llm_adapters",
        "llm_tools.llm_providers",
        "llm_tools.workflow_api",
        "llm_tools.tools",
        "llm_tools.apps",
    ),
    "llm_adapters": (
        "llm_tools.llm_providers",
        "llm_tools.workflow_api",
        "llm_tools.tools",
        "llm_tools.apps",
    ),
    "llm_providers": (
        "llm_tools.workflow_api",
        "llm_tools.tools",
        "llm_tools.apps",
    ),
    "tools": (
        "llm_tools.llm_adapters",
        "llm_tools.llm_providers",
        "llm_tools.workflow_api",
        "llm_tools.apps",
    ),
    "workflow_api": (
        "llm_tools.llm_providers",
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
    layer_root = PACKAGE_DIR / layer_name
    violations: list[str] = []

    for source_file in iter_python_files(layer_root):
        for reference in iter_import_references(source_file):
            for forbidden_prefix in forbidden_prefixes:
                if is_forbidden_module(reference.module, forbidden_prefix):
                    violations.append(
                        format_reference(
                            reference.path,
                            reference.lineno,
                            f"{layer_name} must not import '{reference.module}'",
                        )
                    )
                    break

    assert not violations, (
        f"Forbidden imports found in layer '{layer_name}':\n" + "\n".join(violations)
    )

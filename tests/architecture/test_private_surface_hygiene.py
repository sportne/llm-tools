"""Static AST checks for private cross-layer surface use."""

from __future__ import annotations

import ast

import pytest

from . import _helpers as helpers

LOWER_LAYERS_BY_LAYER = {
    "apps": (
        "tool_api",
        "llm_adapters",
        "llm_providers",
        "tools",
        "workflow_api",
        "harness_api",
    ),
    "workflow_api": (
        "tool_api",
        "llm_adapters",
        "tools",
    ),
    "harness_api": (
        "tool_api",
        "llm_adapters",
        "llm_providers",
        "workflow_api",
    ),
}


def _imported_lower_layer_aliases(
    path, allowed_prefixes: tuple[str, ...]
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    tree = helpers.parse_module(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if any(module.startswith(prefix) for prefix in allowed_prefixes):
                    aliases[alias.asname or module.split(".")[-1]] = module
        elif isinstance(node, ast.ImportFrom):
            resolved = helpers._resolve_from_import(path, node)
            if resolved is None:
                continue
            for alias in node.names:
                target = resolved if alias.name == "*" else f"{resolved}.{alias.name}"
                if any(target.startswith(prefix) for prefix in allowed_prefixes):
                    aliases[alias.asname or alias.name] = target
    return aliases


def _private_symbol_import_violations(layer_name: str) -> list[str]:
    layer_root = helpers.PACKAGE_DIR / layer_name
    allowed_prefixes = tuple(
        f"llm_tools.{lower_layer}" for lower_layer in LOWER_LAYERS_BY_LAYER[layer_name]
    )
    violations: list[str] = []
    for source_file in helpers.iter_python_files(layer_root):
        tree = helpers.parse_module(source_file)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            resolved = helpers._resolve_from_import(source_file, node)
            if resolved is None or not any(
                resolved.startswith(prefix) for prefix in allowed_prefixes
            ):
                continue
            for alias in node.names:
                if not alias.name.startswith("_") or alias.name.startswith("__"):
                    continue
                imported_as = (
                    f" as {alias.asname}"
                    if alias.asname and alias.asname != alias.name
                    else ""
                )
                violations.append(
                    helpers.format_reference(
                        source_file,
                        node.lineno,
                        (
                            f"{layer_name} must not import private symbol "
                            f"'{resolved}.{alias.name}{imported_as}'"
                        ),
                    )
                )
    return violations


def _private_module_import_violations(layer_name: str) -> list[str]:
    layer_root = helpers.PACKAGE_DIR / layer_name
    violations: list[str] = []
    for source_file in helpers.iter_python_files(layer_root):
        tree = helpers.parse_module(source_file)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if any(part.startswith("_") for part in parts[1:]):
                        violations.append(
                            helpers.format_reference(
                                source_file,
                                node.lineno,
                                f"{layer_name} must not import private module '{alias.name}'",
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                resolved = helpers._resolve_from_import(source_file, node)
                if resolved is None:
                    continue
                parts = resolved.split(".")
                if any(part.startswith("_") for part in parts[1:]):
                    violations.append(
                        helpers.format_reference(
                            source_file,
                            node.lineno,
                            f"{layer_name} must not import private module '{resolved}'",
                        )
                    )
    return violations


@pytest.mark.parametrize("layer_name", tuple(LOWER_LAYERS_BY_LAYER))
def test_layers_do_not_import_private_modules(layer_name: str) -> None:
    violations = _private_module_import_violations(layer_name)
    assert not violations, (
        f"Private module imports found in layer '{layer_name}':\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize("layer_name", tuple(LOWER_LAYERS_BY_LAYER))
def test_layers_do_not_import_lower_layer_private_symbols(layer_name: str) -> None:
    violations = _private_symbol_import_violations(layer_name)
    assert not violations, (
        f"Private lower-layer symbol imports found in layer '{layer_name}':\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize("layer_name", tuple(LOWER_LAYERS_BY_LAYER))
def test_layers_do_not_access_lower_layer_private_attributes(layer_name: str) -> None:
    layer_root = helpers.PACKAGE_DIR / layer_name
    allowed_prefixes = tuple(
        f"llm_tools.{lower_layer}" for lower_layer in LOWER_LAYERS_BY_LAYER[layer_name]
    )
    violations: list[str] = []

    for source_file in helpers.iter_python_files(layer_root):
        aliases = _imported_lower_layer_aliases(source_file, allowed_prefixes)
        tree = helpers.parse_module(source_file)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if not node.attr.startswith("_") or node.attr.startswith("__"):
                continue
            if not isinstance(node.value, ast.Name):
                continue
            module = aliases.get(node.value.id)
            if module is None:
                continue
            violations.append(
                helpers.format_reference(
                    source_file,
                    node.lineno,
                    (
                        f"{layer_name} must not access private attribute "
                        f"'{node.value.id}.{node.attr}' from '{module}'"
                    ),
                )
            )

    assert not violations, (
        f"Private lower-layer attribute access found in layer '{layer_name}':\n"
        + "\n".join(violations)
    )

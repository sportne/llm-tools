"""Static checks for Pydantic model file boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

from . import _helpers as helpers


def _is_model_file(path: Path) -> bool:
    return path.name == "models.py" or path.name.endswith("_models.py")


def _base_name(base: ast.expr) -> str | None:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def _base_model_aliases(tree: ast.AST) -> set[str]:
    aliases = {"BaseModel"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != "pydantic":
            continue
        for alias in node.names:
            if alias.name == "BaseModel":
                aliases.add(alias.asname or alias.name)
    return aliases


def _pydantic_model_names() -> set[str]:
    names: set[str] = set()
    for source_file in helpers.iter_python_files(helpers.PACKAGE_DIR):
        if not _is_model_file(source_file):
            continue
        tree = helpers.parse_module(source_file)
        base_model_aliases = _base_model_aliases(tree)
        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef) or node.name in names:
                    continue
                if any(
                    _base_name(base) in base_model_aliases or _base_name(base) in names
                    for base in node.bases
                ):
                    names.add(node.name)
                    changed = True
    return names


def test_pydantic_models_live_in_model_files() -> None:
    """Keep Vulture's model-file exclusions from hiding runtime code."""
    violations: list[str] = []
    pydantic_model_names = _pydantic_model_names()
    for source_file in helpers.iter_python_files(helpers.PACKAGE_DIR):
        if _is_model_file(source_file):
            continue
        tree = helpers.parse_module(source_file)
        base_model_aliases = _base_model_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(
                _base_name(base) in base_model_aliases
                or _base_name(base) in pydantic_model_names
                for base in node.bases
            ):
                violations.append(
                    helpers.format_reference(
                        source_file,
                        node.lineno,
                        f"Pydantic model '{node.name}' must live in a model file",
                    )
                )

    assert not violations, "Pydantic models outside model files:\n" + "\n".join(
        violations
    )

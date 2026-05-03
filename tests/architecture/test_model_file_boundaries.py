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


def _imported_aliases(tree: ast.AST, *, module: str, names: set[str]) -> set[str]:
    aliases = set(names)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != module:
            continue
        for alias in node.names:
            if alias.name in names:
                aliases.add(alias.asname or alias.name)
    return aliases


def _decorator_name(decorator: ast.expr) -> str | None:
    target = decorator
    if isinstance(target, ast.Call):
        target = target.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _has_decorator(node: ast.FunctionDef, names: set[str]) -> bool:
    return any(_decorator_name(decorator) in names for decorator in node.decorator_list)


def _is_type_checking_block(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    )


def _runtime_class_body(node: ast.ClassDef) -> list[ast.stmt]:
    return [
        statement for statement in node.body if not _is_type_checking_block(statement)
    ]


def _is_enum_class(node: ast.ClassDef, enum_aliases: set[str]) -> bool:
    return any(_base_name(base) in enum_aliases for base in node.bases)


def _is_protocol_class(node: ast.ClassDef, protocol_aliases: set[str]) -> bool:
    return any(_base_name(base) in protocol_aliases for base in node.bases)


def _is_model_class(
    node: ast.ClassDef,
    *,
    base_model_aliases: set[str],
    pydantic_model_names: set[str],
) -> bool:
    return any(
        _base_name(base) in base_model_aliases
        or _base_name(base) in pydantic_model_names
        for base in node.bases
    )


def _is_allowed_model_method(node: ast.FunctionDef) -> bool:
    if _has_decorator(
        node,
        {
            "computed_field",
            "field_serializer",
            "field_validator",
            "model_serializer",
            "model_validator",
            "property",
        },
    ):
        return True
    if _has_decorator(node, {"classmethod"}) and node.name.startswith(("from_", "to_")):
        return True
    if node.name.startswith(
        (
            "coerce_",
            "model_",
            "populate_",
            "serialize_",
            "validate_",
            "_populate_",
            "_serialize_",
            "_validate_",
        )
    ):
        return True

    positional_args = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    return (
        len(positional_args) <= 1
        and node.args.vararg is None
        and node.args.kwarg is None
    )


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


def test_model_files_contain_only_model_layer_code() -> None:
    """Keep whole-file Vulture exclusions from hiding executable behavior."""
    violations: list[str] = []
    pydantic_model_names = _pydantic_model_names()

    for source_file in helpers.iter_python_files(helpers.PACKAGE_DIR):
        if not _is_model_file(source_file):
            continue

        tree = helpers.parse_module(source_file)
        base_model_aliases = _base_model_aliases(tree)
        enum_aliases = _imported_aliases(tree, module="enum", names={"Enum", "StrEnum"})
        protocol_aliases = _imported_aliases(tree, module="typing", names={"Protocol"})
        protocol_aliases.update(
            _imported_aliases(tree, module="typing_extensions", names={"Protocol"})
        )

        for node in tree.body:
            if _is_type_checking_block(node):
                continue

            if isinstance(node, ast.ClassDef):
                is_model = _is_model_class(
                    node,
                    base_model_aliases=base_model_aliases,
                    pydantic_model_names=pydantic_model_names,
                )
                if not (
                    is_model
                    or _is_enum_class(node, enum_aliases)
                    or _is_protocol_class(node, protocol_aliases)
                ):
                    violations.append(
                        helpers.format_reference(
                            source_file,
                            node.lineno,
                            f"Non-model class '{node.name}' must not live in a model file",
                        )
                    )
                    continue

                if not is_model:
                    continue

                for statement in _runtime_class_body(node):
                    if not isinstance(statement, ast.FunctionDef):
                        continue
                    if not _is_allowed_model_method(statement):
                        violations.append(
                            helpers.format_reference(
                                source_file,
                                statement.lineno,
                                (
                                    f"Runtime method '{node.name}.{statement.name}' "
                                    "must not live in a model file"
                                ),
                            )
                        )

            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                violations.append(
                    helpers.format_reference(
                        source_file,
                        node.lineno,
                        f"Public function '{node.name}' must not live in a model file",
                    )
                )

    assert not violations, "Non-model code in model files:\n" + "\n".join(violations)


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

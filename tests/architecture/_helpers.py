"""Shared helpers for architecture-focused test suites."""

from __future__ import annotations

import ast
import os
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
PACKAGE_DIR = SRC_DIR / "llm_tools"
TESTS_DIR = ROOT_DIR / "tests"
APPROVED_DIRECT_INVOCATION_PATHS = {
    PACKAGE_DIR / "tool_api" / "runtime.py",
}
_SKIPPED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
}


@dataclass(frozen=True, slots=True)
class ImportReference:
    """One statically parsed import reference."""

    path: Path
    lineno: int
    module: str


@dataclass(frozen=True, slots=True)
class InvocationReference:
    """One direct `.invoke(...)` or `.ainvoke(...)` call site."""

    path: Path
    lineno: int
    method_name: str


def iter_python_files(root: Path) -> Iterator[Path]:
    """Yield repository Python files in a stable order."""
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIPPED_PARTS)
        current_dir = Path(directory)
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            yield current_dir / filename


@cache
def parse_module(path: Path) -> ast.AST:
    """Parse one Python file into an AST."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def module_name_for_path(path: Path) -> str:
    """Return the importable module path for a source file."""
    relative = path.relative_to(SRC_DIR).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_name_for_path(path: Path) -> str:
    """Return the package context used to resolve relative imports."""
    module_name = module_name_for_path(path)
    if path.name == "__init__.py":
        return module_name
    package_parts = module_name.split(".")[:-1]
    return ".".join(package_parts)


def iter_import_references(path: Path) -> Iterator[ImportReference]:
    """Yield static import references for one source file."""
    tree = parse_module(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield ImportReference(path=path, lineno=node.lineno, module=alias.name)
            continue

        if isinstance(node, ast.ImportFrom):
            resolved = _resolve_from_import(path, node)
            if resolved is None:
                continue
            for alias in node.names:
                module = resolved if alias.name == "*" else f"{resolved}.{alias.name}"
                yield ImportReference(path=path, lineno=node.lineno, module=module)


def iter_direct_invocations(root: Path = ROOT_DIR) -> Iterator[InvocationReference]:
    """Yield direct `.invoke(...)` and `.ainvoke(...)` call sites."""
    for path in iter_python_files(root):
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"invoke", "ainvoke"}:
                continue
            yield InvocationReference(
                path=path,
                lineno=node.lineno,
                method_name=node.func.attr,
            )


def is_repo_test_path(path: Path) -> bool:
    """Whether a file lives under the repository test tree."""
    return _is_relative_to(path, TESTS_DIR)


def is_approved_direct_invocation_path(path: Path) -> bool:
    """Whether a direct tool invocation is allowed in this file."""
    return path in APPROVED_DIRECT_INVOCATION_PATHS or is_repo_test_path(path)


def build_builtin_registry() -> Any:
    """Register the current built-in tool suites in one registry."""
    from llm_tools.tool_api import ToolRegistry
    from llm_tools.tools import (
        register_atlassian_tools,
        register_filesystem_tools,
        register_git_tools,
        register_gitlab_tools,
        register_text_tools,
    )

    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_git_tools(registry)
    register_gitlab_tools(registry)
    register_atlassian_tools(registry)
    register_text_tools(registry)
    return registry


def builtin_tools() -> list[Any]:
    """Return all built-in tools registered by the repository helpers."""
    return build_builtin_registry().list_registered_tools()


def builtin_runtime(
    *,
    allowed_side_effects: set[Any] | None = None,
    allow_filesystem: bool = True,
    allow_network: bool = True,
    allow_subprocess: bool = True,
) -> Any:
    """Return a runtime wired to the current built-in registry."""
    from llm_tools.tool_api import SideEffectClass, ToolPolicy, ToolRuntime

    policy = ToolPolicy(
        allowed_side_effects=allowed_side_effects
        or {
            SideEffectClass.NONE,
            SideEffectClass.LOCAL_READ,
            SideEffectClass.LOCAL_WRITE,
            SideEffectClass.EXTERNAL_READ,
        },
        allow_filesystem=allow_filesystem,
        allow_network=allow_network,
        allow_subprocess=allow_subprocess,
    )
    return ToolRuntime(build_builtin_registry(), policy=policy)


def filesystem_context(
    workspace: Path,
    *,
    invocation_id: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Build a filesystem-friendly tool context using current repo conventions."""
    from llm_tools.tool_api import ToolContext

    return ToolContext(
        invocation_id=invocation_id,
        workspace=str(workspace),
        metadata=metadata or {},
    )


def format_reference(path: Path, lineno: int, detail: str) -> str:
    """Return a stable, user-friendly failure reference."""
    return f"{path.relative_to(ROOT_DIR)}:{lineno}: {detail}"


def is_forbidden_module(module: str, forbidden_prefix: str) -> bool:
    """Whether an import reference targets a forbidden module prefix."""
    return module == forbidden_prefix or module.startswith(f"{forbidden_prefix}.")


def _resolve_from_import(path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module

    package_name = package_name_for_path(path)
    if package_name == "":
        return None

    package_parts = package_name.split(".")
    if node.level - 1 > len(package_parts):
        return None

    base_parts = package_parts[: len(package_parts) - (node.level - 1)]
    if node.module is not None:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True

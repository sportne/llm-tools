"""Shared helpers for architecture-focused test suites."""

from __future__ import annotations

import ast
import os
from collections.abc import Iterable, Iterator
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
_TOOL_METHOD_NAMES = frozenset({"invoke", "ainvoke"})
_UNBOUND = object()


@dataclass(frozen=True, slots=True)
class ImportReference:
    """One statically parsed import reference."""

    path: Path
    lineno: int
    module: str


@dataclass(frozen=True, slots=True)
class InvocationReference:
    """One invoke-like call site that bypasses runtime mediation."""

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
    """Yield invoke-like `.invoke(...)` and `.ainvoke(...)` bypass call sites."""
    for path in iter_python_files(root):
        tree = parse_module(path)
        yield from _InvocationScanner(path).scan(tree)


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
    return list(build_builtin_registry()._iter_registered_tools())


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


class _InvocationScanner(ast.NodeVisitor):
    """Collect invoke-like calls, including aliased method bypasses."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._references: list[InvocationReference] = []
        self._scopes: list[dict[str, object]] = [{}]

    def scan(self, tree: ast.AST) -> list[InvocationReference]:
        self.visit(tree)
        return self._references

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_body(node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind_name(node.name, _UNBOUND)
        self._visit_exprs(node.decorator_list)
        self._visit_exprs(node.bases)
        self._visit_exprs(keyword.value for keyword in node.keywords)
        self._push_scope()
        self._visit_body(node.body)
        self._pop_scope()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._push_scope()
        self._bind_arguments(node.args)
        self.visit(node.body)
        self._pop_scope()

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        method_name = self._extract_method_name(node.value)
        for target in node.targets:
            self._bind_target(target, method_name)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        method_name: str | None = None
        if node.value is not None:
            self.visit(node.value)
            method_name = self._extract_method_name(node.value)
        self._bind_target(node.target, method_name)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.target)
        self.visit(node.value)
        self._bind_target(node.target, None)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_target(node.target, self._extract_method_name(node.value))

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._bind_target(node.target, None)
        self._visit_body(node.body)
        self._visit_body(node.orelse)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with_items(node.items)
        self._visit_body(node.body)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with_items(node.items)
        self._visit_body(node.body)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._bind_name(alias.asname or alias.name.split(".")[0], _UNBOUND)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            self._bind_name(alias.asname or alias.name, _UNBOUND)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self._bind_name(node.name, _UNBOUND)
        self._visit_body(node.body)

    def visit_Call(self, node: ast.Call) -> None:
        method_name = self._extract_method_name(node.func)
        if method_name is not None:
            self._references.append(
                InvocationReference(
                    path=self._path,
                    lineno=node.lineno,
                    method_name=method_name,
                )
            )
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._bind_name(node.name, _UNBOUND)
        self._visit_exprs(node.decorator_list)
        self._visit_exprs(default for default in node.args.defaults)
        self._visit_exprs(default for default in node.args.kw_defaults if default)
        self._visit_exprs(arg.annotation for arg in self._iter_arguments(node.args))
        if node.returns is not None:
            self.visit(node.returns)
        self._push_scope()
        self._bind_arguments(node.args)
        self._visit_body(node.body)
        self._pop_scope()

    def _visit_with_items(self, items: list[ast.withitem]) -> None:
        for item in items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars, None)

    def _visit_body(self, body: list[ast.stmt]) -> None:
        for statement in body:
            self.visit(statement)

    def _visit_exprs(self, expressions: Iterable[ast.expr | None]) -> None:
        for expression in expressions:
            if expression is not None:
                self.visit(expression)

    def _push_scope(self) -> None:
        self._scopes.append({})

    def _pop_scope(self) -> None:
        self._scopes.pop()

    def _bind_arguments(self, arguments: ast.arguments) -> None:
        for argument in self._iter_arguments(arguments):
            self._bind_name(argument.arg, _UNBOUND)

    def _iter_arguments(self, arguments: ast.arguments) -> Iterator[ast.arg]:
        yield from arguments.posonlyargs
        yield from arguments.args
        if arguments.vararg is not None:
            yield arguments.vararg
        yield from arguments.kwonlyargs
        if arguments.kwarg is not None:
            yield arguments.kwarg

    def _bind_target(self, target: ast.expr, method_name: str | None) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target.id, method_name or _UNBOUND)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_target(element, method_name)
            return
        if isinstance(target, ast.Starred):
            self._bind_target(target.value, method_name)

    def _bind_name(self, name: str, value: object) -> None:
        self._scopes[-1][name] = value

    def _lookup_name(self, name: str) -> str | None:
        for scope in reversed(self._scopes):
            if name not in scope:
                continue
            value = scope[name]
            if value is _UNBOUND:
                return None
            return value if isinstance(value, str) else None
        return None

    def _extract_method_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute) and node.attr in _TOOL_METHOD_NAMES:
            return node.attr
        if isinstance(node, ast.Call) and _is_getattr_call(node):
            method_name = _string_literal_value(node.args[1])
            if method_name in _TOOL_METHOD_NAMES:
                return method_name
        if isinstance(node, ast.Name):
            return self._lookup_name(node.id)
        return None


def _is_getattr_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
    )


def _string_literal_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None

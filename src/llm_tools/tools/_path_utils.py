"""Shared path helpers for built-in tools."""

from __future__ import annotations

from pathlib import Path

from llm_tools.tool_api import ToolContext


def get_workspace_root(context: ToolContext) -> Path:
    """Return the resolved workspace root for a tool invocation."""
    workspace = (context.workspace or "").strip()
    if workspace == "":
        raise ValueError("No workspace configured for local tool execution.")
    root = Path(workspace).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(
            f"Workspace root '{root}' does not exist or is not a directory."
        )
    return root


def resolve_workspace_path(
    context: ToolContext,
    path: str,
    *,
    expect_directory: bool | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a user path relative to the workspace and enforce the root boundary."""
    root = get_workspace_root(context)
    candidate = Path(path).expanduser()
    resolved = (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (root / candidate).resolve(strict=False)
    )

    if not resolved.is_relative_to(root):
        raise ValueError(f"Path '{path}' resolves outside the workspace root '{root}'.")

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path '{resolved}' does not exist.")

    if expect_directory is True and not resolved.is_dir():
        raise NotADirectoryError(f"Path '{resolved}' is not a directory.")

    if expect_directory is False and resolved.exists() and resolved.is_dir():
        raise IsADirectoryError(f"Path '{resolved}' is a directory, not a file.")

    return resolved


def relative_display_path(root: Path, target: Path) -> str:
    """Return a stable display path relative to the workspace root."""
    if target == root:
        return "."
    return target.relative_to(root).as_posix()


def is_hidden_path(root: Path, target: Path) -> bool:
    """Return whether a path is hidden relative to the workspace root."""
    relative = target.relative_to(root)
    return any(part.startswith(".") for part in relative.parts)

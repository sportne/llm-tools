"""Unit tests for shared built-in tool path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_tools.tool_api import ToolContext
from llm_tools.tools._path_utils import (
    get_workspace_root,
    is_hidden_path,
    relative_display_path,
    resolve_workspace_path,
)


def test_get_workspace_root_rejects_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing-workspace"

    with pytest.raises(ValueError, match="does not exist"):
        get_workspace_root(ToolContext(invocation_id="inv-1", workspace=str(missing)))


def test_resolve_workspace_path_rejects_missing_paths(tmp_path: Path) -> None:
    context = ToolContext(invocation_id="inv-2", workspace=str(tmp_path))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_workspace_path(context, "missing.txt")


def test_resolve_workspace_path_rejects_file_when_directory_expected(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")
    context = ToolContext(invocation_id="inv-3", workspace=str(tmp_path))

    with pytest.raises(NotADirectoryError, match="not a directory"):
        resolve_workspace_path(context, "note.txt", expect_directory=True)


def test_resolve_workspace_path_rejects_directory_when_file_expected(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "docs"
    directory.mkdir()
    context = ToolContext(invocation_id="inv-4", workspace=str(tmp_path))

    with pytest.raises(IsADirectoryError, match="directory, not a file"):
        resolve_workspace_path(context, "docs", expect_directory=False)


def test_relative_display_path_returns_dot_for_workspace_root(tmp_path: Path) -> None:
    assert relative_display_path(tmp_path.resolve(), tmp_path.resolve()) == "."


def test_is_hidden_path_detects_hidden_relative_parts(tmp_path: Path) -> None:
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    hidden_file = hidden_dir / "note.txt"
    hidden_file.write_text("secret", encoding="utf-8")

    assert is_hidden_path(tmp_path.resolve(), hidden_file.resolve()) is True
    assert (
        is_hidden_path(tmp_path.resolve(), (tmp_path / "visible.txt").resolve())
        is False
    )


def test_get_workspace_root_returns_resolved_directory(tmp_path: Path) -> None:
    context = ToolContext(invocation_id="inv-ok", workspace=f"  {tmp_path}  ")

    assert get_workspace_root(context) == tmp_path.resolve()


def test_resolve_workspace_path_accepts_relative_and_absolute_paths(
    tmp_path: Path,
) -> None:
    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")
    context = ToolContext(invocation_id="inv-abs", workspace=str(tmp_path))

    assert resolve_workspace_path(context, "note.txt") == note.resolve()
    assert resolve_workspace_path(context, str(note.resolve())) == note.resolve()

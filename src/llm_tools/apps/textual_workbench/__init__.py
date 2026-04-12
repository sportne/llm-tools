"""Developer-facing Textual workbench for llm-tools."""

from __future__ import annotations


def run_workbench_app() -> None:
    """Launch the Textual workbench."""
    from llm_tools.apps.textual_workbench.app import run_workbench_app as _run

    _run()


def main() -> int:
    """Console entrypoint for the Textual workbench."""
    from llm_tools.apps.textual_workbench.app import main as _main

    return _main()


__all__ = [
    "main",
    "run_workbench_app",
]

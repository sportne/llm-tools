"""Interactive Streamlit assistant app built on top of llm-tools."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llm_tools.apps.assistant_config import StreamlitAssistantConfig
from llm_tools.apps.chat_config import ProviderPreset


def run_streamlit_assistant_app(
    *, root_path: Path | None, config: StreamlitAssistantConfig
) -> None:
    """Render the Streamlit assistant app for one Streamlit script execution."""
    from llm_tools.apps.streamlit_assistant.app import (
        run_streamlit_assistant_app as _run,
    )

    _run(root_path=root_path, config=config)


def main(argv: Sequence[str] | None = None) -> int:
    """Console entrypoint for the Streamlit assistant app."""
    from llm_tools.apps.streamlit_assistant.app import main as _main

    return _main(argv)


__all__ = [
    "ProviderPreset",
    "StreamlitAssistantConfig",
    "main",
    "run_streamlit_assistant_app",
]

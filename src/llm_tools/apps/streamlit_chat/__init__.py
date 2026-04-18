"""Interactive Streamlit repository chat app built on top of llm-tools."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llm_tools.apps.chat_config import ProviderPreset, TextualChatConfig


def run_streamlit_chat_app(*, root_path: Path, config: TextualChatConfig) -> None:
    """Render the Streamlit chat app for one Streamlit script execution."""
    from llm_tools.apps.streamlit_chat.app import run_streamlit_chat_app as _run

    _run(root_path=root_path, config=config)


def main(argv: Sequence[str] | None = None) -> int:
    """Console entrypoint for the Streamlit chat app."""
    from llm_tools.apps.streamlit_chat.app import main as _main

    return _main(argv)


__all__ = ["ProviderPreset", "TextualChatConfig", "main", "run_streamlit_chat_app"]

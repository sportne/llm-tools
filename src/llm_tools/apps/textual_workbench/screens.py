"""Dedicated Textual editor widgets for the workbench."""

from __future__ import annotations

from textual.widgets import TextArea


class PromptComposerTextArea(TextArea):
    """Prompt editor used for model-turn execution."""


class JsonArgumentsTextArea(TextArea):
    """JSON editor used for direct tool invocation arguments."""

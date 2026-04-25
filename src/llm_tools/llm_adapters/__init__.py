"""Adapters that translate model output into canonical turn outcomes."""

from llm_tools.llm_adapters.action_envelope import ActionEnvelopeAdapter
from llm_tools.llm_adapters.base import ParsedModelResponse
from llm_tools.llm_adapters.prompt_tools import (
    PromptToolAdapter,
    PromptToolCategory,
    PromptToolCategoryDecision,
    PromptToolDecision,
    PromptToolProtocolError,
)

__all__ = [
    "ActionEnvelopeAdapter",
    "ParsedModelResponse",
    "PromptToolAdapter",
    "PromptToolCategory",
    "PromptToolCategoryDecision",
    "PromptToolDecision",
    "PromptToolProtocolError",
]

"""Adapters that translate model output into canonical turn outcomes."""

from llm_tools.llm_adapters.action_envelope import ActionEnvelopeAdapter
from llm_tools.llm_adapters.base import ParsedModelResponse

__all__ = [
    "ActionEnvelopeAdapter",
    "ParsedModelResponse",
]

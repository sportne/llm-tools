"""Adapters that translate LLM-facing I/O into canonical model-turn outcomes."""

from llm_tools.llm_adapters.base import (
    LLMAdapter,
    ModelOutputParsingAdapter,
    ParsedModelResponse,
    ToolExposureAdapter,
)
from llm_tools.llm_adapters.openai_tool_calling import OpenAIToolCallingAdapter
from llm_tools.llm_adapters.prompt_schema import PromptSchemaAdapter
from llm_tools.llm_adapters.structured_responses import (
    StructuredModelEnvelope,
    StructuredResponseAdapter,
    StructuredToolAction,
)

__all__ = [
    "LLMAdapter",
    "ModelOutputParsingAdapter",
    "OpenAIToolCallingAdapter",
    "ParsedModelResponse",
    "PromptSchemaAdapter",
    "StructuredModelEnvelope",
    "StructuredResponseAdapter",
    "StructuredToolAction",
    "ToolExposureAdapter",
]

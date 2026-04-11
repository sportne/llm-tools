"""Adapters that translate LLM-facing I/O into canonical model-turn outcomes."""

from llm_tools.llm_adapters.base import (
    LLMAdapter,
    ModelOutputParsingAdapter,
    ParsedModelResponse,
    ToolExposureAdapter,
)
from llm_tools.llm_adapters.native_tool_calling import NativeToolCallingAdapter
from llm_tools.llm_adapters.prompt_schema import PromptSchemaAdapter
from llm_tools.llm_adapters.structured_output import (
    StructuredOutputAdapter,
    StructuredOutputEnvelope,
    StructuredToolAction,
)

__all__ = [
    "LLMAdapter",
    "ModelOutputParsingAdapter",
    "NativeToolCallingAdapter",
    "ParsedModelResponse",
    "PromptSchemaAdapter",
    "StructuredOutputAdapter",
    "StructuredOutputEnvelope",
    "StructuredToolAction",
    "ToolExposureAdapter",
]

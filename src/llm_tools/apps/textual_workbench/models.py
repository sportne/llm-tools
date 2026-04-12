"""App-local models for the Textual workbench."""

from __future__ import annotations

from enum import Enum
from os import getcwd
from typing import Any

from pydantic import BaseModel, Field

from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolResult
from llm_tools.workflow_api import WorkflowTurnResult


class WorkbenchMode(str, Enum):  # noqa: UP042
    """Supported interaction modes in the workbench."""

    NATIVE_TOOL_CALLING = "native_tool_calling"
    STRUCTURED_OUTPUT = "structured_output"
    PROMPT_SCHEMA = "prompt_schema"


class ProviderPreset(str, Enum):  # noqa: UP042
    """OpenAI-compatible provider presets available in the workbench."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM_OPENAI_COMPATIBLE = "custom_openai_compatible"


class WorkbenchConfigState(BaseModel):
    """Mutable configuration edited through the workbench controls."""

    workspace: str = Field(default_factory=getcwd)
    provider_preset: ProviderPreset = ProviderPreset.OLLAMA
    base_url: str = "http://localhost:11434/v1"
    model: str = "gemma4:26b"
    api_key: str = "ollama"
    mode: WorkbenchMode = WorkbenchMode.NATIVE_TOOL_CALLING

    enable_filesystem_tools: bool = True
    enable_git_tools: bool = True
    enable_text_tools: bool = True
    enable_atlassian_tools: bool = False

    allow_local_write: bool = False
    allow_external_read: bool = False
    allow_external_write: bool = False
    allow_network: bool = True
    allow_filesystem: bool = True
    allow_subprocess: bool = True

    execute_after_parse: bool = True


class WorkbenchRunState(BaseModel):
    """Last-run state rendered in the workbench output panes."""

    busy: bool = False
    last_exported_tools: Any = None
    last_parsed_response: ParsedModelResponse | None = None
    last_workflow_result: WorkflowTurnResult | None = None
    last_direct_tool_result: ToolResult | None = None
    current_status_text: str = ""


class ModelTurnExecutionResult(BaseModel):
    """Result of one provider-backed model turn in the workbench."""

    exported_tools: Any
    parsed_response: ParsedModelResponse
    workflow_result: WorkflowTurnResult | None = None


class DirectExecutionResult(BaseModel):
    """Result of one direct tool execution in the workbench."""

    tool_result: ToolResult

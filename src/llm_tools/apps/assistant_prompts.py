"""Prompt builders for the assistant-focused app surfaces."""

from __future__ import annotations

import json

from llm_tools.tool_api import ToolRegistry
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import ChatFinalResponse

ASSISTANT_SYSTEM_PROMPT_PREAMBLE = """
You are a general-purpose assistant with optional access to local and proprietary data.
Answer directly when no tools are needed.
Use tools only when the user asks for information that depends on workspace files,
Git history, GitLab, Atlassian resources, or other tool-visible data.
Do not invent file contents, repository state, tickets, merge requests, or remote documents.
If tool evidence is incomplete, say so in uncertainty or missing_information.
""".strip()

RESEARCH_SYSTEM_PROMPT_PREAMBLE = """
You are a durable research assistant operating inside a task-oriented harness session.
Use the selected task context and recent session state to decide whether to answer or use tools.
If more evidence is needed, return tool actions. If the selected task is sufficiently addressed
for this turn, return a concise final_response string.
Do not invent local or remote facts.
""".strip()

FIELD_GUIDANCE: dict[str, str] = {
    "answer": "Main answer. It may be tool-grounded or direct model knowledge when no tool evidence is needed.",
    "citations": "Support data-dependent claims with source identifiers when available. source_path may be local or remote.",
    "confidence": "Optional 0.0-1.0 confidence based on evidence quality and completeness.",
    "uncertainty": "List caveats, ambiguity, or places where evidence is incomplete.",
    "missing_information": "List explicit gaps that prevented a stronger answer.",
    "follow_up_suggestions": "Suggest useful next questions or source-driven follow-up steps.",
}
DEFAULT_FIELD_GUIDANCE = "Return a valid value that matches the response schema."


def _strip_schema_titles(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_schema_titles(item)
            for key, item in value.items()
            if key != "title"
        }
    if isinstance(value, list):
        return [_strip_schema_titles(item) for item in value]
    return value


def _tool_catalog(
    *,
    tool_registry: ToolRegistry,
    enabled_tool_names: set[str] | None,
) -> str:
    return json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": tool.spec.name,
                    "description": tool.spec.description,
                    "parameters": _strip_schema_titles(
                        tool.input_model.model_json_schema()
                    ),
                },
            }
            for tool in tool_registry.list_bindings()
            if enabled_tool_names is None or tool.spec.name in enabled_tool_names
        ],
        indent=2,
        sort_keys=True,
    )


def _compact_tool_catalog(
    *,
    tool_registry: ToolRegistry,
    enabled_tool_names: set[str] | None,
) -> str:
    return json.dumps(
        [
            {
                "name": tool.spec.name,
                "description": tool.spec.description,
            }
            for tool in tool_registry.list_bindings()
            if enabled_tool_names is None or tool.spec.name in enabled_tool_names
        ],
        indent=2,
        sort_keys=True,
    )


def build_assistant_system_prompt(
    *,
    tool_registry: ToolRegistry,
    tool_limits: ToolLimits,
    enabled_tool_names: set[str] | None = None,
    workspace_enabled: bool = True,
    staged_schema_protocol: bool = False,
    interaction_protocol: str | None = None,
) -> str:
    """Return the interactive assistant system prompt."""
    workspace_rules = (
        "- A workspace root is configured for this session. Local path-based tools may use it.\n"
        if workspace_enabled
        else "- No workspace root is configured for this session. Do not use local workspace tools unless a root is selected later.\n"
    )
    protocol = interaction_protocol or (
        "staged_json" if staged_schema_protocol else "action_envelope"
    )
    if protocol == "prompt_tools":
        tool_catalog = _compact_tool_catalog(
            tool_registry=tool_registry,
            enabled_tool_names=enabled_tool_names,
        )
        return (
            f"{ASSISTANT_SYSTEM_PROMPT_PREAMBLE}\n\n"
            "Available tools:\n"
            "- Use these compact summaries to choose tools when local evidence is needed.\n"
            f"{tool_catalog}\n\n"
            "Prompt-tool protocol:\n"
            "- The client will ask each step using fenced prompt-tool blocks.\n"
            "- Follow only the fenced block format requested by the latest system message.\n"
            "- Do not emit JSON tool calls, native function calls, or action envelopes.\n"
            "- Use one tool at a time, then wait for the tool result before choosing another step.\n"
            "- Finalize only when available evidence is sufficient for the requested answer.\n\n"
            "Operational rules:\n"
            f"{workspace_rules}"
            "- Use tools only when the user request depends on tool-visible data or verification.\n"
            "- If the user asks about this repository, workspace, source files, app/runtime wiring, or git state, inspect relevant local files or git data before finalizing.\n"
            "- If the user explicitly asks you to use tools or cite local files, you must do that before finalizing.\n"
            "- Prefer narrow reads and specific searches before broad file or remote fetches.\n"
            "- If a tool call fails because of missing access, missing credentials, or bad arguments, do not repeat the same failing call.\n"
            "- Answer conservatively and cite the evidence you actually have.\n\n"
            "Relevant limits:\n"
            "- search-style tools may cap result counts per call.\n"
            f"- read-oriented content is bounded by configured readable-character limits near {tool_limits.max_file_size_characters} characters.\n"
            f"- any single tool result may be truncated near {tool_limits.max_tool_result_chars} characters.\n"
        )
    if protocol == "native_tools":
        response_fields = "\n".join(
            f"- {field_name}: {FIELD_GUIDANCE.get(field_name, DEFAULT_FIELD_GUIDANCE)}"
            for field_name in ChatFinalResponse.model_fields
        )
        return (
            f"{ASSISTANT_SYSTEM_PROMPT_PREAMBLE}\n\n"
            "Native tool protocol:\n"
            "- Use the provider's native tool-calling mechanism when local evidence is needed.\n"
            "- Use at most the tools needed to answer; stop using tools once enough evidence is available.\n"
            "- When you have enough evidence, return the final answer instead of another tool call.\n"
            "- Do not describe or invent tool-call JSON in message text.\n\n"
            "Operational rules:\n"
            f"{workspace_rules}"
            "- Use tools only when the user request depends on tool-visible data or verification.\n"
            "- If the user asks about this repository, workspace, source files, app/runtime wiring, or git state, inspect relevant local files or git data before finalizing.\n"
            "- If the user explicitly asks you to use tools or cite local files, you must do that before finalizing.\n"
            "- Prefer narrow reads and specific searches before broad file or remote fetches.\n"
            "- If a tool call fails because of missing access, missing credentials, or bad arguments, do not repeat the same failing call.\n"
            "- Answer conservatively and cite the evidence you actually have.\n\n"
            "Relevant limits:\n"
            "- search-style tools may cap result counts per call.\n"
            f"- read-oriented content is bounded by configured readable-character limits near {tool_limits.max_file_size_characters} characters.\n"
            f"- any single tool result may be truncated near {tool_limits.max_tool_result_chars} characters.\n\n"
            "Final answer content should include:\n"
            f"{response_fields}\n"
        )
    if protocol == "staged_json":
        tool_catalog = _compact_tool_catalog(
            tool_registry=tool_registry,
            enabled_tool_names=enabled_tool_names,
        )
        return (
            f"{ASSISTANT_SYSTEM_PROMPT_PREAMBLE}\n\n"
            "Available tools:\n"
            "- Use the compact summaries below to choose the next step.\n"
            "- Do not invent tool arguments until the client sends the selected tool schema.\n"
            f"{tool_catalog}\n\n"
            "Structured interaction protocol:\n"
            "- Each step uses a small strict schema.\n"
            "- First choose either one tool or finalize.\n"
            "- If a tool is chosen, the client will reply with that tool's exact argument schema and usage guidance.\n"
            "- Return exactly one tool invocation for that selected tool.\n"
            "- After tool results arrive, choose the next step again.\n"
            "- Only finalize when you have enough evidence to satisfy the final strict response schema.\n\n"
            "Operational rules:\n"
            f"{workspace_rules}"
            "- Use tools only when the user request depends on tool-visible data or verification.\n"
            "- If the user asks about this repository, workspace, source files, app/runtime wiring, or git state, inspect relevant local files or git data before finalizing.\n"
            "- If the user explicitly asks you to use tools or cite local files, you must do that before finalizing.\n"
            "- Prefer narrow reads and specific searches before broad file or remote fetches.\n"
            "- If a tool call fails because of missing access, missing credentials, or bad arguments, do not repeat the same failing call.\n"
            "- Answer conservatively and cite the evidence you actually have.\n\n"
            "Relevant limits:\n"
            "- search-style tools may cap result counts per call.\n"
            f"- read-oriented content is bounded by configured readable-character limits near {tool_limits.max_file_size_characters} characters.\n"
            f"- any single tool result may be truncated near {tool_limits.max_tool_result_chars} characters.\n"
        )
    tool_catalog = _tool_catalog(
        tool_registry=tool_registry,
        enabled_tool_names=enabled_tool_names,
    )
    response_fields = "\n".join(
        f"- {field_name}: {FIELD_GUIDANCE.get(field_name, DEFAULT_FIELD_GUIDANCE)}"
        for field_name in ChatFinalResponse.model_fields
    )
    return (
        f"{ASSISTANT_SYSTEM_PROMPT_PREAMBLE}\n\n"
        "Available tools:\n"
        "- Treat the following tool catalog like an OpenAI native function-tools array.\n"
        "- If no tool is needed, answer with final_response and no actions.\n"
        f"{tool_catalog}\n\n"
        "Required action format:\n"
        "- Return exactly one structured action envelope.\n"
        "- If you need tools, return one or more actions and no final_response.\n"
        "- If you can answer, return final_response matching the final-answer schema and no actions.\n"
        "- Tool results will be supplied back as plain conversation messages.\n\n"
        "Operational rules:\n"
        f"{workspace_rules}"
        "- Use tools only when the user request depends on tool-visible data or verification.\n"
        "- If the user asks about this repository, workspace, source files, app/runtime wiring, or git state, inspect relevant local files or git data before returning final_response.\n"
        "- If the user explicitly asks you to use tools or cite local files, you must do that before returning final_response.\n"
        "- Prefer narrow reads and specific searches before broad file or remote fetches.\n"
        "- Local write tools are higher-risk than read tools and should not be used unless clearly needed.\n"
        "- Git tools require a workspace and subprocess access.\n"
        "- Atlassian and GitLab tools require network access plus the documented credentials.\n"
        "- If a tool call fails because of missing access, missing credentials, or bad arguments, do not repeat the same failing call.\n"
        "- Answer conservatively and cite the evidence you actually have.\n\n"
        "Relevant limits:\n"
        "- search-style tools may cap result counts per call.\n"
        f"- read-oriented content is bounded by configured readable-character limits near {tool_limits.max_file_size_characters} characters.\n"
        f"- any single tool result may be truncated near {tool_limits.max_tool_result_chars} characters.\n\n"
        "Final response fields:\n"
        f"{response_fields}\n"
    )


def build_research_system_prompt(
    *,
    tool_registry: ToolRegistry,
    tool_limits: ToolLimits,
    enabled_tool_names: set[str] | None = None,
    workspace_enabled: bool = True,
    staged_schema_protocol: bool = False,
) -> str:
    """Return the harness-research system prompt."""
    workspace_rules = (
        "- A workspace root is configured for this research session.\n"
        if workspace_enabled
        else "- No workspace root is configured for this research session.\n"
    )
    if staged_schema_protocol:
        tool_catalog = _compact_tool_catalog(
            tool_registry=tool_registry,
            enabled_tool_names=enabled_tool_names,
        )
        return (
            f"{RESEARCH_SYSTEM_PROMPT_PREAMBLE}\n\n"
            "Available tools:\n"
            f"{tool_catalog}\n\n"
            "Structured interaction protocol:\n"
            "- Each step uses a small strict schema.\n"
            "- First choose either one tool or finalize.\n"
            "- If a tool is chosen, the client will reply with that tool's exact argument schema and usage guidance.\n"
            "- Return exactly one tool invocation for that selected tool.\n"
            "- After tool results arrive, choose the next step again.\n"
            "- Finalize only when the selected task has reached a durable stopping point for this turn.\n\n"
            "Operational rules:\n"
            f"{workspace_rules}"
            "- Use the task and context JSON from the user message as the current source of work.\n"
            "- Prefer precise searches and targeted reads over broad scans.\n"
            "- If execution is blocked by missing access, credentials, or missing context, say so in the final response.\n"
            f"- Tool outputs may be truncated near {tool_limits.max_tool_result_chars} characters.\n"
        )
    tool_catalog = _tool_catalog(
        tool_registry=tool_registry,
        enabled_tool_names=enabled_tool_names,
    )
    return (
        f"{RESEARCH_SYSTEM_PROMPT_PREAMBLE}\n\n"
        "Available tools:\n"
        f"{tool_catalog}\n\n"
        "Required action format:\n"
        "- Return exactly one structured action envelope.\n"
        "- Use actions when you need more evidence.\n"
        "- Return final_response as a short text summary when the selected task has reached a durable stopping point for this turn.\n\n"
        "Operational rules:\n"
        f"{workspace_rules}"
        "- Use the task and context JSON from the user message as the current source of work.\n"
        "- Prefer precise searches and targeted reads over broad scans.\n"
        "- If execution is blocked by missing access, credentials, or missing context, say so in the final_response summary.\n"
        f"- Tool outputs may be truncated near {tool_limits.max_tool_result_chars} characters.\n"
    )


__all__ = [
    "build_assistant_system_prompt",
    "build_research_system_prompt",
]

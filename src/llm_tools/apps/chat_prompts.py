"""Shared prompt builders for repository chat app layers."""

from __future__ import annotations

import json

from llm_tools.tool_api import ToolRegistry
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import ChatFinalResponse

CHAT_SYSTEM_PROMPT_PREAMBLE = """
You are a repository chat assistant for a single configured project root.
Use only tool results and prior grounded chat messages when making factual claims.
Do not invent file contents, paths, line ranges, or repository behavior.
If tool evidence is incomplete, say so in uncertainty or missing_information.
All tool paths are relative to the configured root.
Tool outputs are authoritative over guesses.
""".strip()

FIELD_GUIDANCE: dict[str, str] = {
    "answer": "Main grounded answer. Keep it concise and avoid unsupported claims.",
    "citations": "Support material claims with file paths and line ranges when available.",
    "confidence": "Optional 0.0-1.0 confidence based on evidence quality and completeness.",
    "uncertainty": "List caveats, ambiguity, or places where the evidence is incomplete.",
    "missing_information": "List explicit gaps or TBD items that blocked a stronger answer.",
    "follow_up_suggestions": "Suggest useful next questions or tool-driven follow-up steps.",
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


def build_chat_system_prompt(
    *,
    tool_registry: ToolRegistry,
    tool_limits: ToolLimits,
    enabled_tool_names: set[str] | None = None,
    workspace_enabled: bool = True,
) -> str:
    """Return the interactive repository-chat system prompt."""
    tool_catalog = json.dumps(
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
            for tool in tool_registry.list_registered_tools()
            if enabled_tool_names is None or tool.spec.name in enabled_tool_names
        ],
        indent=2,
        sort_keys=True,
    )
    usage_examples = "\n".join(
        (
            "- To search the entire repo by file contents: search_text(path='.', query='provider')",
            "- To locate candidate files first: find_files(path='src', pattern='**/*.py')",
            "- To inspect a directory tree before reading: list_directory(path='src', recursive=true)",
            "- To inspect a possibly large file first: get_file_info(path='src/app.py')",
            "- To compare several candidate files first: get_file_info(paths=['src/app.py', 'src/config.py'])",
            "- To read only part of a file: read_file(path='src/app.py', start_char=0, end_char=4000)",
        )
    )
    visible_limits = "\n".join(
        (
            f"- search_text returns at most {tool_limits.max_search_matches} matches per call.",
            (
                "- read-only file tools reject readable content over "
                f"{tool_limits.max_file_size_characters} characters."
            ),
            (
                "- full-file reads are further limited by max_read_file_chars when "
                "it is set; otherwise orchestration derives a limit from the "
                "session context window."
            ),
            f"- any single tool result may be truncated near {tool_limits.max_tool_result_chars} characters.",
        )
    )
    response_fields = "\n".join(
        f"- {field_name}: {FIELD_GUIDANCE.get(field_name, DEFAULT_FIELD_GUIDANCE)}"
        for field_name in ChatFinalResponse.model_fields
    )
    workspace_rules = (
        "- A workspace root is configured for this session. Workspace-relative paths must stay inside it.\n"
        if workspace_enabled
        else "- No workspace root is configured for this session. Do not call local workspace tools unless a root is selected later.\n"
    )
    return (
        f"{CHAT_SYSTEM_PROMPT_PREAMBLE}\n\n"
        "Available tools:\n"
        "- Treat the following tool catalog like an OpenAI native function-tools array.\n"
        "- Each tool has a function name, a description, and a JSON-schema parameters object.\n"
        f"{tool_catalog}\n\n"
        "Required action format:\n"
        "- On every turn, return exactly one structured action envelope.\n"
        "- If you need more evidence, return actions with one or more tool requests and no final_response.\n"
        "- If you can answer, return final_response matching the final-answer schema and no actions.\n"
        "- Tool results will be supplied back to you as plain conversation messages; use them before deciding the next action.\n\n"
        "Tool usage examples:\n"
        f"{usage_examples}\n\n"
        "Operational rules:\n"
        f"{workspace_rules}"
        "- path must never be blank. Use path='.' to operate on the entire configured root when a workspace root is configured.\n"
        "- list_directory expects a directory path.\n"
        "- Use recursive=true and optional max_depth for tree-style listings.\n"
        "- search_text accepts either a directory path or a single file path.\n"
        "- get_file_info accepts either one file path or a list of file paths.\n"
        "- read_file expects one file path.\n"
        "- Use find_files for matching file paths or file names.\n"
        "- Use search_text for searching inside file contents.\n"
        "- Prefer find_files or search_text before broad file reads.\n"
        "- Use get_file_info before read_file when a file may be large.\n"
        "- Use start_char and end_char for partial reads when a full file is unnecessary.\n"
        "- Git tools operate on a workspace directory and require subprocess access.\n"
        "- Jira tools require the configured Jira environment variables when enabled.\n"
        "- If a tool call fails because of its arguments or target path, do not repeat the same failing call.\n"
        "- Answer conservatively and cite the evidence you actually have.\n\n"
        "Relevant limits:\n"
        f"{visible_limits}\n\n"
        "Final response fields:\n"
        f"{response_fields}\n"
    )

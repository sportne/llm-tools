"""Prompt-emitted tool protocol for plain chat-completion providers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from llm_tools.llm_adapters.base import ParsedModelResponse
from llm_tools.tool_api import ToolInvocationRequest, ToolSpec


class PromptToolProtocolError(ValueError):
    """Prompt-tool response could not be parsed or validated."""

    def __init__(self, message: str, *, invalid_payload: object | None = None) -> None:
        super().__init__(message)
        self.invalid_payload = invalid_payload


@dataclass(frozen=True, slots=True)
class PromptToolDecision:
    """Parsed decision step from a prompt-tool response."""

    mode: str
    tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class PromptToolCategoryDecision:
    """Parsed category step from a prompt-tool response."""

    mode: str
    category: str | None = None


@dataclass(frozen=True, slots=True)
class PromptToolCategory:
    """Model-facing group of related tools."""

    name: str
    description: str
    tool_specs: list[ToolSpec]


class PromptToolAdapter:
    """Format and parse the fenced prompt-tool protocol."""

    CATEGORY_ORDER = ("filesystem", "text", "git", "gitlab", "atlassian", "other")
    CATEGORY_DESCRIPTIONS = {
        "filesystem": "Read, list, inspect, or write local workspace files.",
        "text": "Search readable local file contents for literal text.",
        "git": "Inspect local git status, diffs, and history.",
        "gitlab": "Search and read GitLab project data.",
        "atlassian": "Search and read Jira, Confluence, and Bitbucket data.",
        "other": "Tools that do not fit another category.",
    }

    _BLOCK_RE = re.compile(
        r"```(?P<kind>[A-Za-z_][A-Za-z0-9_-]*)\s*\n(?P<body>.*?)```",
        re.DOTALL,
    )

    def single_action_stage_messages(
        self,
        *,
        base_messages: list[dict[str, Any]],
        tool_specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
        final_response_model: object,
        decision_context: str | None = None,
        selected_category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the default prompt-tool round prompt."""
        context_text = (
            f"\n\nCurrent turn tool-use context:\n{decision_context}\n"
            if decision_context
            else ""
        )
        category_text = (
            f"Current tool category: {selected_category}\n"
            if selected_category is not None
            else ""
        )
        if tool_specs:
            action_formats = (
                "Required tool format:\n"
                "```tool\n"
                "TOOL_NAME: exact_tool_name\n"
                "BEGIN_ARG: argument_name\n"
                "argument value\n"
                "END_ARG\n"
                "```\n"
            )
        else:
            action_formats = "No tools are available in this step.\n"
        protocol_override = (
            "Prompt-tool output contract:\n"
            "- The only valid output is exactly one fenced ```tool block or exactly "
            "one fenced ```final block.\n"
            "- In a tool block, the first non-empty line must be exactly "
            "TOOL_NAME: one_exact_available_tool_name.\n"
            "- Do not write name:, library_name:, function:, tool:, tool_name in "
            "lowercase, a bare tool name, or the tool name followed by a colon.\n"
            "- Each argument marker must be one line: BEGIN_ARG: argument_name.\n"
            "- Do not put the argument name on the next line after BEGIN_ARG:.\n"
        )
        return [
            *base_messages,
            {
                "role": "system",
                "content": (
                    "Current step: choose exactly one next action.\n"
                    f"{protocol_override}"
                    "Return exactly one fenced block and no prose outside it.\n"
                    "Use a ```tool block when another tool call is necessary.\n"
                    "Use a ```final block only when the available evidence is sufficient.\n"
                    "One model round may call at most one tool.\n"
                    "The fenced block must be the final substantive content.\n"
                    f"{category_text}"
                    f"{context_text}\n"
                    f"Available tools and argument schemas:\n"
                    f"{self._tool_action_catalog(tool_specs, input_models)}\n\n"
                    f"{action_formats}"
                    "Required final format:\n"
                    "```final\nANSWER:\nPlain markdown answer here.\n```\n"
                    f"{self._final_schema_text(final_response_model)}"
                ),
            },
        ]

    def category_decision_stage_messages(
        self,
        *,
        base_messages: list[dict[str, Any]],
        categories: list[PromptToolCategory],
        final_response_model: object,
        decision_context: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the optional prompt-tool category selection prompt."""
        context_text = (
            f"\n\nCurrent turn tool-use context:\n{decision_context}\n"
            if decision_context
            else ""
        )
        return [
            *base_messages,
            {
                "role": "system",
                "content": (
                    "Current step: choose a tool category or finalize.\n"
                    "Use the prompt-tool category output contract for this step.\n"
                    "The only valid output is one fenced ```category block.\n"
                    "MODE is not a category name; MODE must be exactly category or "
                    "finalize.\n"
                    "Return exactly one fenced category block and no prose.\n"
                    "Choose a category only when another tool call is necessary.\n"
                    "Do not provide a tool name, tool arguments, or final answer in this step.\n"
                    f"{context_text}\n"
                    f"Available categories:\n{self._category_catalog(categories)}\n\n"
                    "Required formats:\n"
                    "```category\nMODE: category\nCATEGORY: category_name\n```\n"
                    "or\n"
                    "```category\nMODE: finalize\n```\n"
                    f"{self._final_schema_text(final_response_model)}"
                ),
            },
        ]

    def decision_stage_messages(
        self,
        *,
        base_messages: list[dict[str, Any]],
        tool_specs: list[ToolSpec],
        decision_context: str | None = None,
    ) -> list[dict[str, Any]]:
        context_text = (
            f"\n\nCurrent turn tool-use context:\n{decision_context}\n\n"
            if decision_context
            else ""
        )
        return [
            *base_messages,
            {
                "role": "system",
                "content": (
                    "Current step: choose the next action.\n"
                    "Return exactly one fenced decision block and no prose.\n"
                    "Inside the block, MODE must be exactly tool or finalize.\n"
                    "When MODE is tool, TOOL_NAME must be one exact available tool name.\n"
                    "Do not provide tool arguments or a final answer in this step.\n\n"
                    f"Available tools:\n{self._tool_catalog(tool_specs)}\n\n"
                    f"{context_text}"
                    "Required formats:\n"
                    "```decision\nMODE: tool\nTOOL_NAME: tool_name\n```\n"
                    "or\n"
                    "```decision\nMODE: finalize\n```"
                ),
            },
        ]

    def tool_invocation_stage_messages(
        self,
        *,
        base_messages: list[dict[str, Any]],
        tool_spec: ToolSpec,
        input_model: type[BaseModel],
    ) -> list[dict[str, Any]]:
        return [
            *base_messages,
            {
                "role": "system",
                "content": (
                    "Current step: fill arguments for the already-selected tool.\n"
                    f"The selected tool is fixed: {tool_spec.name}\n"
                    "This is not a tool-selection step.\n"
                    f"Tool description: {tool_spec.description}\n"
                    "Return exactly one fenced tool block for the fixed tool and no prose.\n"
                    "The tool block must be the final substantive content.\n"
                    "Do not choose another tool and do not finalize in this step.\n"
                    "Changing TOOL_NAME is invalid even if another available tool seems better.\n\n"
                    "Hard requirements:\n"
                    f"- Copy this literal line as the first non-empty line inside the block: TOOL_NAME: {tool_spec.name}\n"
                    "- TOOL_NAME is not a choice field in this step.\n"
                    f"- Do not write '{tool_spec.name}:' as a field name.\n"
                    "- Every argument must start with BEGIN_ARG: argument_name on one line.\n"
                    "- Do not put the argument name on the next line after BEGIN_ARG:.\n"
                    "- Every argument must end with END_ARG on its own line.\n\n"
                    "Tool argument schema:\n"
                    f"{json.dumps(input_model.model_json_schema(), indent=2, sort_keys=True, default=str)}\n\n"
                    "Required format:\n"
                    "```tool\n"
                    f"TOOL_NAME: {tool_spec.name}\n"
                    "BEGIN_ARG: argument_name\n"
                    "argument value\n"
                    "END_ARG\n"
                    "```"
                ),
            },
        ]

    def final_response_stage_messages(
        self,
        *,
        base_messages: list[dict[str, Any]],
        final_response_model: object,
    ) -> list[dict[str, Any]]:
        schema_text = self._final_schema_text(final_response_model)
        return [
            *base_messages,
            {
                "role": "system",
                "content": (
                    "Current step: finalize the answer.\n"
                    "Return exactly one fenced final block and no prose outside it.\n"
                    "Do not include a tool invocation.\n"
                    "Answer the user's request using the available tool result content "
                    "as evidence.\n"
                    "Tool call audit metadata may show which tools and arguments were "
                    "used, but it is not evidence and must not be repeated as the "
                    "answer.\n"
                    "If evidence is incomplete, provide the best supported answer and "
                    "state what is uncertain or missing.\n"
                    "Use plain markdown after ANSWER unless a JSON object is required.\n"
                    f"{schema_text}\n\n"
                    "Required format:\n"
                    "```final\nANSWER:\nPlain markdown answer here.\n```"
                ),
            },
        ]

    def repair_stage_message(
        self,
        *,
        stage_name: str,
        error: Exception,
        invalid_payload: object | None,
        tool_specs: list[ToolSpec] | None = None,
        input_models: dict[str, type[BaseModel]] | None = None,
        categories: list[PromptToolCategory] | None = None,
        selected_tool: ToolSpec | None = None,
        input_model: type[BaseModel] | None = None,
        final_response_model: object | None = None,
    ) -> str:
        guidance = {
            "action": (
                "Action rules: return exactly one ```tool block or one ```final block. "
                "A tool block must include TOOL_NAME and all required arguments. "
                "The first non-empty line inside a tool block must be exactly "
                "TOOL_NAME: one_exact_available_tool_name. "
                "Do not use name:, library_name:, function:, a bare tool name, "
                "or the tool name followed by a colon. "
                "Use BEGIN_ARG: argument_name on one line. "
                "Do not add prose outside the block."
            ),
            "category": (
                "Category rules: return exactly one ```category block. "
                "MODE must be exactly category or finalize. "
                "When MODE is category, CATEGORY must be one exact available category name. "
                "Do not include tool arguments or a final answer."
            ),
            "decision": (
                "Decision rules: return exactly one ```decision block. "
                "MODE must be exactly tool or finalize. "
                "When MODE is tool, TOOL_NAME must be one exact available tool name. "
                "Do not add prose outside the block."
            ),
            "final_response": (
                "Finalization rules: return exactly one ```final block with ANSWER:. "
                "Do not include a tool block or prose outside the block. "
                "Use tool result content as evidence; do not answer by repeating "
                "tool call audit metadata."
            ),
        }.get(stage_name)
        if guidance is None and stage_name.startswith("tool:"):
            selected_tool_name = (
                selected_tool.name if selected_tool is not None else None
            )
            tool_line = (
                f"TOOL_NAME: {selected_tool_name}"
                if selected_tool_name is not None
                else "TOOL_NAME: <selected tool>"
            )
            guidance = (
                "Tool rules: this is not a tool-selection step. "
                "Return exactly one ```tool block for the already-selected tool. "
                f"The first non-empty line inside the block must be exactly {tool_line}. "
                "Do not change TOOL_NAME even if another available tool seems better. "
                "Do not use the tool name itself as a field name. "
                "Use BEGIN_ARG: argument_name on one line, then the value, then END_ARG."
            )
        if guidance is None:
            guidance = "Return only the fields required for this stage."

        details = []
        if categories is not None:
            details.append(
                f"Available categories:\n{self._category_catalog(categories)}"
            )
        if tool_specs is not None:
            if input_models is None:
                details.append(f"Available tools:\n{self._tool_catalog(tool_specs)}")
            else:
                details.append(
                    "Available tools and argument schemas:\n"
                    f"{self._tool_action_catalog(tool_specs, input_models)}"
                )
                details.append(
                    "Exact tool block shape:\n"
                    "```tool\n"
                    "TOOL_NAME: exact_tool_name\n"
                    "BEGIN_ARG: argument_name\n"
                    "argument value\n"
                    "END_ARG\n"
                    "```"
                )
        if selected_tool is not None and input_model is not None:
            details.append(
                "Selected tool schema:\n"
                f"{json.dumps(input_model.model_json_schema(), indent=2, sort_keys=True, default=str)}"
            )
            details.append(
                "Exact tool block shape:\n"
                "```tool\n"
                f"TOOL_NAME: {selected_tool.name}\n"
                "BEGIN_ARG: argument_name\n"
                "argument value\n"
                "END_ARG\n"
                "```"
            )
        if final_response_model is not None:
            details.append(self._final_schema_text(final_response_model))

        return (
            f"The previous {stage_name} response was invalid.\n"
            "Correct the response for the same stage only.\n"
            f"{guidance}\n"
            f"Validation summary: {self._validation_error_summary(error)}\n"
            "Previous invalid payload:\n"
            f"{self._format_invalid_payload(invalid_payload)}\n"
            + ("\n".join(details) + "\n" if details else "")
            + "Return the corrected fenced block now."
        )

    def parse_decision(
        self, text: str, *, tool_specs: list[ToolSpec]
    ) -> PromptToolDecision:
        body = self._single_block_body(text, "decision")
        fields = self._parse_key_lines(body)
        mode = fields.get("MODE", "").strip().lower()
        allowed_tools = {spec.name for spec in tool_specs}
        if mode == "finalize":
            if "TOOL_NAME" in fields and fields["TOOL_NAME"].strip():
                raise PromptToolProtocolError(
                    "Decision with MODE: finalize must not include TOOL_NAME.",
                    invalid_payload=text,
                )
            return PromptToolDecision(mode="finalize")
        if mode != "tool":
            raise PromptToolProtocolError(
                "Decision MODE must be 'tool' or 'finalize'.",
                invalid_payload=text,
            )
        tool_name = fields.get("TOOL_NAME", "").strip()
        if tool_name not in allowed_tools:
            raise PromptToolProtocolError(
                f"Decision selected unknown tool: {tool_name or '(missing)'}.",
                invalid_payload=text,
            )
        return PromptToolDecision(mode="tool", tool_name=tool_name)

    def parse_category_decision(
        self,
        text: str,
        *,
        categories: list[PromptToolCategory],
    ) -> PromptToolCategoryDecision:
        body = self._single_block_body(text, "category")
        fields = self._parse_key_lines(body)
        mode = fields.get("MODE", "").strip().lower()
        allowed_categories = {category.name for category in categories}
        if mode == "finalize":
            if "CATEGORY" in fields and fields["CATEGORY"].strip():
                raise PromptToolProtocolError(
                    "Category decision with MODE: finalize must not include CATEGORY.",
                    invalid_payload=text,
                )
            return PromptToolCategoryDecision(mode="finalize")
        if mode != "category":
            raise PromptToolProtocolError(
                "Category MODE must be 'category' or 'finalize'.",
                invalid_payload=text,
            )
        category_name = fields.get("CATEGORY", "").strip()
        if category_name not in allowed_categories:
            raise PromptToolProtocolError(
                f"Category decision selected unknown category: {category_name or '(missing)'}.",
                invalid_payload=text,
            )
        return PromptToolCategoryDecision(mode="category", category=category_name)

    def parse_single_action(
        self,
        text: str,
        *,
        tool_specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
        final_response_model: object,
    ) -> ParsedModelResponse:
        kind, body = self._single_block(
            text,
            allowed_kinds={"tool", "final"},
            require_final_block=True,
        )
        if kind == "final":
            answer = self._final_answer_from_body(body)
            return ParsedModelResponse(
                final_response=self._validate_final_response(
                    answer,
                    final_response_model=final_response_model,
                    invalid_payload=text,
                )
            )
        parsed_name = self._tool_name_from_body(body, invalid_payload=text)
        allowed_tools = {spec.name for spec in tool_specs}
        if parsed_name not in allowed_tools:
            raise PromptToolProtocolError(
                f"Tool block selected unknown tool: {parsed_name or '(missing)'}.",
                invalid_payload=text,
            )
        input_model = input_models.get(parsed_name)
        if input_model is None:
            raise ValueError(
                f"Tool '{parsed_name}' was not prepared for this interaction.",
            )
        raw_arguments = self._parse_argument_blocks(body, invalid_payload=text)
        arguments = self._validate_arguments(
            raw_arguments,
            input_model=input_model,
            invalid_payload=text,
        )
        return ParsedModelResponse(
            invocations=[
                ToolInvocationRequest(
                    tool_name=parsed_name,
                    arguments=arguments,
                    tool_call_id=f"prompt-tool-{uuid4().hex}",
                )
            ]
        )

    def parse_tool_invocation(
        self,
        text: str,
        *,
        tool_name: str,
        input_model: type[BaseModel],
    ) -> ParsedModelResponse:
        body = self._single_block_body(text, "tool", require_final_block=True)
        parsed_name = self._tool_name_from_body(body, invalid_payload=text)
        if parsed_name != tool_name:
            raise PromptToolProtocolError(
                f"Tool block selected '{parsed_name}', expected '{tool_name}'.",
                invalid_payload=text,
            )
        raw_arguments = self._parse_argument_blocks(body, invalid_payload=text)
        arguments = self._validate_arguments(
            raw_arguments,
            input_model=input_model,
            invalid_payload=text,
        )
        return ParsedModelResponse(
            invocations=[
                ToolInvocationRequest(
                    tool_name=tool_name,
                    arguments=arguments,
                    tool_call_id=f"prompt-tool-{uuid4().hex}",
                )
            ]
        )

    def parse_final_response(
        self,
        text: str,
        *,
        final_response_model: object,
    ) -> ParsedModelResponse:
        body = self._single_block_body(text, "final", allow_plain_text=True)
        answer = self._final_answer_from_body(body)
        return ParsedModelResponse(
            final_response=self._validate_final_response(
                answer,
                final_response_model=final_response_model,
                invalid_payload=text,
            )
        )

    def _single_block_body(
        self,
        text: str,
        kind: str,
        *,
        allow_plain_text: bool = False,
        require_final_block: bool = False,
    ) -> str:
        blocks = [
            (match.group("kind").strip().lower(), match.group("body"))
            for match in self._BLOCK_RE.finditer(text)
        ]
        matching = [body for block_kind, body in blocks if block_kind == kind]
        if not matching:
            if allow_plain_text and text.strip():
                return f"ANSWER:\n{text.strip()}"
            raise PromptToolProtocolError(
                f"Missing fenced {kind} block.", invalid_payload=text
            )
        if len(matching) != 1:
            raise PromptToolProtocolError(
                f"Expected exactly one fenced {kind} block.", invalid_payload=text
            )
        if kind in {"decision", "final"} and len(blocks) != 1:
            raise PromptToolProtocolError(
                f"Expected only one fenced {kind} block.", invalid_payload=text
            )
        if kind == "tool" and len(blocks) != 1:
            raise PromptToolProtocolError(
                "Tool response must contain only one fenced tool block.",
                invalid_payload=text,
            )
        if require_final_block:
            final_match = list(self._BLOCK_RE.finditer(text))[-1]
            if final_match.group("kind").strip().lower() != kind:
                raise PromptToolProtocolError(
                    f"The fenced {kind} block must be the final substantive content.",
                    invalid_payload=text,
                )
            trailing = text[final_match.end() :].strip()
            if trailing:
                raise PromptToolProtocolError(
                    f"The fenced {kind} block must be the final substantive content.",
                    invalid_payload=text,
                )
        return matching[0].strip()

    def _single_block(
        self,
        text: str,
        *,
        allowed_kinds: set[str],
        require_final_block: bool = False,
    ) -> tuple[str, str]:
        blocks = [
            (match.group("kind").strip().lower(), match.group("body"), match)
            for match in self._BLOCK_RE.finditer(text)
        ]
        if len(blocks) != 1:
            expected = ", ".join(sorted(allowed_kinds))
            raise PromptToolProtocolError(
                f"Expected exactly one fenced block of type: {expected}.",
                invalid_payload=text,
            )
        kind, body, match = blocks[0]
        if kind not in allowed_kinds:
            expected = ", ".join(sorted(allowed_kinds))
            raise PromptToolProtocolError(
                f"Expected fenced block of type: {expected}.",
                invalid_payload=text,
            )
        if require_final_block:
            leading = text[: match.start()].strip()
            if leading:
                raise PromptToolProtocolError(
                    f"The fenced {kind} block must be the only substantive content.",
                    invalid_payload=text,
                )
            trailing = text[match.end() :].strip()
            if trailing:
                raise PromptToolProtocolError(
                    f"The fenced {kind} block must be the final substantive content.",
                    invalid_payload=text,
                )
        return kind, body.strip()

    @staticmethod
    def _parse_key_lines(body: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for line in body.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip().upper()] = value.strip()
        return fields

    def _tool_name_from_body(self, body: str, *, invalid_payload: object) -> str:
        for line in body.splitlines():
            if line.strip().lower().startswith("tool_name:"):
                return line.split(":", 1)[1].strip()
        raise PromptToolProtocolError(
            "Tool block is missing TOOL_NAME.",
            invalid_payload=invalid_payload,
        )

    def _parse_argument_blocks(
        self, body: str, *, invalid_payload: object
    ) -> dict[str, Any]:
        lines = body.splitlines()
        arguments: dict[str, Any] = {}
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if not line.lower().startswith("begin_arg:"):
                index += 1
                continue
            arg_name = line.split(":", 1)[1].strip()
            if not arg_name:
                raise PromptToolProtocolError(
                    "BEGIN_ARG is missing an argument name.",
                    invalid_payload=invalid_payload,
                )
            if arg_name in arguments:
                raise PromptToolProtocolError(
                    f"Duplicate argument: {arg_name}.",
                    invalid_payload=invalid_payload,
                )
            index += 1
            value_lines: list[str] = []
            while index < len(lines) and lines[index].strip().lower() != "end_arg":
                value_lines.append(lines[index])
                index += 1
            if index >= len(lines):
                raise PromptToolProtocolError(
                    f"Argument '{arg_name}' is missing END_ARG.",
                    invalid_payload=invalid_payload,
                )
            arguments[arg_name] = self._parse_argument_value(
                "\n".join(value_lines).strip()
            )
            index += 1
        return arguments

    @staticmethod
    def _parse_argument_value(value: str) -> Any:
        if value == "":
            return ""
        if value in {"true", "false", "null"}:
            return json.loads(value)
        if value[0] in '[{"' or re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", value):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def _validate_arguments(
        arguments: dict[str, Any],
        *,
        input_model: type[BaseModel],
        invalid_payload: object,
    ) -> dict[str, Any]:
        extra_args = set(arguments) - set(input_model.model_fields)
        if extra_args:
            raise PromptToolProtocolError(
                f"Tool arguments included unknown fields: {', '.join(sorted(extra_args))}.",
                invalid_payload=invalid_payload,
            )
        try:
            return input_model.model_validate(arguments).model_dump(mode="json")
        except ValidationError as exc:
            raise PromptToolProtocolError(
                str(exc), invalid_payload=invalid_payload
            ) from exc

    @staticmethod
    def _final_answer_from_body(body: str) -> str:
        lines = body.splitlines()
        for index, line in enumerate(lines):
            if line.strip().upper() == "ANSWER:":
                return "\n".join(lines[index + 1 :]).strip()
            if line.strip().upper().startswith("ANSWER:"):
                first = line.split(":", 1)[1].strip()
                rest = "\n".join(lines[index + 1 :]).strip()
                return "\n".join(part for part in [first, rest] if part).strip()
        return body.strip()

    @staticmethod
    def _validate_final_response(
        value: str,
        *,
        final_response_model: object,
        invalid_payload: object,
    ) -> object:
        if final_response_model is str:
            if not value:
                raise PromptToolProtocolError(
                    "Final answer must not be empty.", invalid_payload=invalid_payload
                )
            PromptToolAdapter._reject_audit_metadata_answer(
                value,
                invalid_payload=invalid_payload,
            )
            return value
        if isinstance(final_response_model, type) and issubclass(
            final_response_model, BaseModel
        ):
            payload: object
            if value.startswith("{"):
                try:
                    payload = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise PromptToolProtocolError(
                        str(exc), invalid_payload=invalid_payload
                    ) from exc
            elif "answer" in final_response_model.model_fields:
                PromptToolAdapter._reject_audit_metadata_answer(
                    value,
                    invalid_payload=invalid_payload,
                )
                payload = {"answer": value}
            else:
                raise PromptToolProtocolError(
                    "Final response must be a JSON object for this response model.",
                    invalid_payload=invalid_payload,
                )
            try:
                if isinstance(payload, dict):
                    answer = payload.get("answer")
                    if isinstance(answer, str):
                        PromptToolAdapter._reject_audit_metadata_answer(
                            answer,
                            invalid_payload=invalid_payload,
                        )
                return final_response_model.model_validate(payload).model_dump(
                    mode="json"
                )
            except ValidationError as exc:
                raise PromptToolProtocolError(
                    str(exc), invalid_payload=invalid_payload
                ) from exc
        return value

    @staticmethod
    def _reject_audit_metadata_answer(
        value: str,
        *,
        invalid_payload: object,
    ) -> None:
        normalized = " ".join(value.strip().lower().split())
        if not normalized:
            return
        audit_markers = (
            "prior tool call record:",
            "tool call audit metadata",
            "name=read_file,",
            "name=search_text,",
            "name=list_directory,",
            "read_file({",
            "search_text({",
            "list_directory({",
        )
        if normalized.startswith(audit_markers):
            raise PromptToolProtocolError(
                (
                    "Final answer repeated tool-call audit metadata instead of "
                    "answering from tool result evidence."
                ),
                invalid_payload=invalid_payload,
            )

    @staticmethod
    def _tool_catalog(tool_specs: list[ToolSpec]) -> str:
        return json.dumps(
            [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "tags": list(spec.tags),
                }
                for spec in tool_specs
            ],
            indent=2,
            sort_keys=True,
            default=str,
        )

    @classmethod
    def derive_tool_categories(
        cls,
        tool_specs: list[ToolSpec],
    ) -> list[PromptToolCategory]:
        """Group tools into stable model-facing categories from ToolSpec tags."""
        grouped: dict[str, list[ToolSpec]] = {name: [] for name in cls.CATEGORY_ORDER}
        for spec in tool_specs:
            tags = set(spec.tags)
            category = "other"
            for candidate in cls.CATEGORY_ORDER:
                if candidate != "other" and candidate in tags:
                    category = candidate
                    break
            grouped[category].append(spec)
        categories: list[PromptToolCategory] = []
        for name in cls.CATEGORY_ORDER:
            specs = grouped[name]
            if not specs:
                continue
            categories.append(
                PromptToolCategory(
                    name=name,
                    description=cls.CATEGORY_DESCRIPTIONS[name],
                    tool_specs=sorted(specs, key=lambda spec: spec.name),
                )
            )
        return categories

    @staticmethod
    def category_tool_specs(
        categories: list[PromptToolCategory],
        category_name: str,
    ) -> list[ToolSpec]:
        """Return the tool specs in one selected category."""
        for category in categories:
            if category.name == category_name:
                return list(category.tool_specs)
        raise PromptToolProtocolError(f"Unknown prompt-tool category: {category_name}.")

    @staticmethod
    def _category_catalog(categories: list[PromptToolCategory]) -> str:
        return json.dumps(
            [
                {
                    "name": category.name,
                    "description": category.description,
                    "tool_count": len(category.tool_specs),
                    "tools": [spec.name for spec in category.tool_specs],
                }
                for category in categories
            ],
            indent=2,
            sort_keys=True,
            default=str,
        )

    @staticmethod
    def _tool_action_catalog(
        tool_specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
    ) -> str:
        entries: list[dict[str, Any]] = []
        for spec in tool_specs:
            input_model = input_models.get(spec.name)
            entry: dict[str, Any] = {
                "name": spec.name,
                "description": spec.description,
                "tags": list(spec.tags),
            }
            if input_model is not None:
                entry["arguments_schema"] = input_model.model_json_schema()
            entries.append(entry)
        return json.dumps(entries, indent=2, sort_keys=True, default=str)

    @staticmethod
    def _final_schema_text(final_response_model: object) -> str:
        if isinstance(final_response_model, type) and issubclass(
            final_response_model, BaseModel
        ):
            return (
                "Final response schema:\n"
                f"{json.dumps(final_response_model.model_json_schema(), indent=2, sort_keys=True, default=str)}"
            )
        return "Final response may be plain text."

    @staticmethod
    def _validation_error_summary(error: Exception) -> str:
        message = str(error).strip()
        return message or type(error).__name__

    @staticmethod
    def _format_invalid_payload(invalid_payload: object | None) -> str:
        if invalid_payload is None:
            return "(unavailable)"
        if isinstance(invalid_payload, str):
            return PromptToolAdapter._summarize_invalid_payload(invalid_payload)
        try:
            rendered = json.dumps(
                invalid_payload, indent=2, sort_keys=True, default=str
            )
        except TypeError:
            rendered = str(invalid_payload)
        return PromptToolAdapter._summarize_invalid_payload(rendered)

    @staticmethod
    def _summarize_invalid_payload(payload: str) -> str:
        lowered = payload.lower()
        if '"actions"' in lowered or '"final_response"' in lowered:
            return (
                "Previous output used a structured JSON action-envelope shape. "
                "The current prompt-tool stage requires the requested fenced block only."
            )
        if "```" in payload and len(payload) > 800:
            return "Previous output included fenced text that did not match this stage."
        if len(payload) > 800:
            return payload[:800] + "...(truncated)"
        return payload


__all__ = [
    "PromptToolAdapter",
    "PromptToolCategory",
    "PromptToolCategoryDecision",
    "PromptToolDecision",
    "PromptToolProtocolError",
]

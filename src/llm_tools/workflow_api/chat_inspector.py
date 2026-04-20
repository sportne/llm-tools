"""Inspector and payload-redaction helpers for chat sessions."""

from __future__ import annotations

import json

from llm_tools.tool_api import ToolResult
from llm_tools.tool_api.redaction import RedactionConfig, RedactionTarget, Redactor
from llm_tools.workflow_api.chat_models import (
    ChatMessage,
    ChatWorkflowApprovalState,
)
from llm_tools.workflow_api.models import ApprovalRequest


def _serialize_chat_message(message: ChatMessage) -> dict[str, str]:
    if message.role == "tool":
        return {"role": "user", "content": f"Tool result:\n{message.content}"}
    return {"role": message.role, "content": message.content}


def _tool_status_label(tool_name: str) -> str:
    if tool_name in {"list_directory", "find_files"}:
        return "listing files"
    if tool_name == "search_text":
        return "searching text"
    if tool_name in {"get_file_info", "read_file"}:
        return "reading file"
    return "thinking"


def _sanitize_tool_result_message(tool_result: ToolResult) -> dict[str, object]:
    if tool_result.ok:
        return {
            "tool_name": tool_result.tool_name,
            "status": "ok",
            "output": tool_result.output,
        }
    return {
        "tool_name": tool_result.tool_name,
        "status": "error",
        "error": (
            tool_result.error.model_dump(mode="json")
            if tool_result.error is not None
            else {"message": "Unknown tool error"}
        ),
    }


def _dump_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _format_tool_result_for_model(result: dict[str, object], *, max_chars: int) -> str:
    rendered = _dump_json(result)
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[:max_chars]}...(truncated)"


def _sanitize_tool_result_for_chat(tool_result: ToolResult) -> ToolResult:
    metadata = dict(tool_result.metadata)
    execution_record = _sanitize_execution_record(metadata.get("execution_record"))
    if execution_record is not None:
        metadata["execution_record"] = execution_record
    return tool_result.model_copy(update={"metadata": metadata})


def _sanitize_execution_record(record: object) -> dict[str, object] | None:
    if not isinstance(record, dict):
        return None
    sanitized = dict(record)
    sanitized.pop("validated_input", None)
    sanitized.pop("validated_output", None)
    request = sanitized.get("request")
    redacted_input = sanitized.get("redacted_input")
    if isinstance(request, dict) and isinstance(redacted_input, dict):
        updated_request = dict(request)
        updated_request["arguments"] = redacted_input
        sanitized["request"] = updated_request
    return sanitized


def _sanitize_parsed_response_for_inspector(
    parsed_response: object,
    *,
    redaction_config: RedactionConfig,
) -> object:
    if not hasattr(parsed_response, "model_dump"):
        return parsed_response
    payload = parsed_response.model_dump(mode="json")
    actions = payload.get("invocations", [])
    if isinstance(actions, list):
        payload["invocations"] = [
            _sanitize_invocation_payload(action, redaction_config=redaction_config)
            for action in actions
        ]
    return payload


def _sanitize_invocation_payload(
    payload: object,
    *,
    redaction_config: RedactionConfig,
) -> object:
    if not isinstance(payload, dict):
        return payload
    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments")
    if not isinstance(tool_name, str) or not isinstance(arguments, dict):
        return payload
    sanitized = dict(payload)
    sanitized["arguments"] = _redact_tool_payload(
        tool_name,
        arguments,
        target=RedactionTarget.INPUT,
        redaction_config=redaction_config,
    )
    return sanitized


def _build_approval_state(
    approval_request: ApprovalRequest,
    *,
    redaction_config: RedactionConfig,
) -> ChatWorkflowApprovalState:
    return ChatWorkflowApprovalState(
        approval_request=approval_request,
        tool_name=approval_request.tool_name,
        redacted_arguments=_redact_tool_payload(
            approval_request.tool_name,
            approval_request.request.arguments,
            target=RedactionTarget.INPUT,
            redaction_config=redaction_config,
        ),
        policy_reason=approval_request.policy_reason,
        policy_metadata=dict(approval_request.policy_metadata),
    )


def _redact_tool_payload(
    tool_name: str,
    payload: object,
    *,
    target: RedactionTarget,
    redaction_config: RedactionConfig,
) -> dict[str, object]:
    redactor = Redactor(redaction_config, tool_name=tool_name)
    redacted = redactor.redact_structured(payload, target=target)
    if not isinstance(redacted, dict):
        return {}
    return redacted


__all__ = [
    "_format_tool_result_for_model",
    "_redact_tool_payload",
    "_sanitize_execution_record",
    "_sanitize_invocation_payload",
    "_sanitize_parsed_response_for_inspector",
    "_sanitize_tool_result_for_chat",
    "_sanitize_tool_result_message",
    "_serialize_chat_message",
    "_tool_status_label",
]

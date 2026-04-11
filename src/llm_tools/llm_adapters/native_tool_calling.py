"""Provider-agnostic native tool-calling adapter."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from llm_tools.llm_adapters.base import LLMAdapter, ParsedModelResponse
from llm_tools.tool_api import ToolInvocationRequest, ToolSpec


class NativeToolCallingAdapter(LLMAdapter):
    """Translate native tool definitions and tool-call payloads."""

    def export_tool_descriptions(
        self,
        specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
    ) -> list[dict[str, Any]]:
        """Return canonical native tool descriptions."""
        tools: list[dict[str, Any]] = []
        for spec in specs:
            input_model = input_models[spec.name]
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": input_model.model_json_schema(),
                    },
                }
            )
        return tools

    def parse_model_output(self, payload: object) -> ParsedModelResponse:
        """Parse native tool-calling output into a canonical turn result."""
        normalized = self._normalize_payload(payload)

        if isinstance(normalized, list):
            return ParsedModelResponse(
                invocations=[self._parse_tool_call(item) for item in normalized]
            )

        if not isinstance(normalized, dict):
            raise ValueError(
                "Native tool-calling payload must normalize to an object or list."
            )

        tool_calls = normalized.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return ParsedModelResponse(
                invocations=[self._parse_tool_call(item) for item in tool_calls]
            )

        if self._looks_like_tool_call(normalized):
            return ParsedModelResponse(invocations=[self._parse_tool_call(normalized)])

        return ParsedModelResponse(
            final_response=self._extract_final_response(normalized)
        )

    def _parse_tool_call(self, payload: object) -> ToolInvocationRequest:
        normalized = self._normalize_payload(payload)
        if not isinstance(normalized, dict) or not self._looks_like_tool_call(
            normalized
        ):
            raise ValueError(
                "Native tool call payload is missing a function definition."
            )

        function_payload = normalized["function"]
        function_name = function_payload.get("name")
        if not isinstance(function_name, str) or function_name.strip() == "":
            raise ValueError("Native tool call is missing a valid function name.")

        arguments = self._parse_arguments(function_payload.get("arguments"))
        return ToolInvocationRequest(tool_name=function_name, arguments=arguments)

    def _parse_arguments(self, payload: object) -> dict[str, Any]:
        normalized = self._normalize_payload(payload)

        if normalized in (None, ""):
            return {}

        if isinstance(normalized, str):
            try:
                normalized = json.loads(normalized)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Native tool-call arguments are not valid JSON."
                ) from exc

        if not isinstance(normalized, dict):
            raise ValueError("Native tool-call arguments must decode to an object.")

        return normalized

    def _extract_final_response(self, payload: dict[str, Any]) -> str:
        content = payload.get("content")

        if isinstance(content, str):
            final_response = content.strip()
            if final_response == "":
                raise ValueError("Assistant response content must not be empty.")
            return final_response

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                normalized_item = self._normalize_payload(item)
                if not isinstance(normalized_item, dict):
                    raise ValueError("Assistant content parts must be objects.")
                if normalized_item.get("type") != "text":
                    raise ValueError("Assistant content parts must be text only.")
                text = normalized_item.get("text")
                if isinstance(text, dict):
                    text = text.get("value")
                if not isinstance(text, str):
                    raise ValueError("Assistant text content must be a string.")
                parts.append(text)

            final_response = "".join(parts).strip()
            if final_response == "":
                raise ValueError("Assistant response content must not be empty.")
            return final_response

        raise ValueError("Assistant response content must be plain text.")

    def _normalize_payload(self, payload: object) -> object:
        if isinstance(payload, BaseModel):
            return payload.model_dump(mode="json", exclude_none=True)

        model_dump = getattr(payload, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json", exclude_none=True)

        return payload

    def _looks_like_tool_call(self, payload: dict[str, Any]) -> bool:
        return isinstance(payload.get("function"), dict)

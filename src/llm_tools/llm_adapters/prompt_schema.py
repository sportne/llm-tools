"""Prompt-schema fallback adapter."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from llm_tools.llm_adapters.base import LLMAdapter, ParsedModelResponse
from llm_tools.llm_adapters.structured_responses import _normalize_structured_payload
from llm_tools.tool_api import ToolSpec

_FENCED_JSON_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class PromptSchemaAdapter(LLMAdapter):
    """Render prompt guidance and parse prompt-returned JSON."""

    def export_tool_descriptions(
        self,
        specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
    ) -> str:
        """Render prompt instructions for the canonical response envelope."""
        tool_sections: list[str] = []
        for spec in specs:
            input_model = input_models[spec.name]
            schema = json.dumps(
                input_model.model_json_schema(), indent=2, sort_keys=True
            )
            tool_sections.append(
                "\n".join(
                    [
                        f"- Tool: {spec.name}",
                        f"  Description: {spec.description}",
                        "  Input schema:",
                        schema,
                    ]
                )
            )

        tool_block = (
            "\n\n".join(tool_sections) if tool_sections else "- No tools available."
        )
        return "\n".join(
            [
                "Return JSON only.",
                "Choose exactly one of the following response modes:",
                '1. {"actions": [{"tool_name": "...", "arguments": {...}}], "final_response": null}',
                '2. {"actions": [], "final_response": "..."}',
                "Do not include both actions and final_response.",
                "Available tools:",
                tool_block,
            ]
        )

    def parse_model_output(self, payload: object) -> ParsedModelResponse:
        """Parse prompt-returned payloads with one deterministic repair pass."""
        normalized = _normalize_structured_payload(
            self._normalize_prompt_payload(payload)
        )
        from llm_tools.llm_adapters.structured_responses import (
            StructuredResponseAdapter,
        )

        return StructuredResponseAdapter().parse_model_output(normalized)

    def _normalize_prompt_payload(self, payload: object) -> object:
        if not isinstance(payload, str):
            return payload

        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            match = _FENCED_JSON_PATTERN.search(payload)
            if match is None:
                raise ValueError("Prompt-schema payload is not valid JSON.") from None

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError("Prompt-schema fenced JSON is not valid.") from exc

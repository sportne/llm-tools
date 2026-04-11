"""Tests for shared LLM adapter abstractions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolInvocationRequest


def test_parsed_model_response_accepts_invocations_only() -> None:
    response = ParsedModelResponse(
        invocations=[ToolInvocationRequest(tool_name="read_file", arguments={})]
    )

    assert response.final_response is None
    assert len(response.invocations) == 1


def test_parsed_model_response_accepts_final_response_only() -> None:
    response = ParsedModelResponse(final_response="All done.")

    assert response.invocations == []
    assert response.final_response == "All done."


def test_parsed_model_response_rejects_both_modes() -> None:
    with pytest.raises(ValidationError):
        ParsedModelResponse(
            invocations=[ToolInvocationRequest(tool_name="read_file", arguments={})],
            final_response="done",
        )


def test_parsed_model_response_rejects_empty_mode() -> None:
    with pytest.raises(ValidationError):
        ParsedModelResponse()


def test_parsed_model_response_rejects_blank_final_response() -> None:
    with pytest.raises(ValidationError):
        ParsedModelResponse(final_response="   ")

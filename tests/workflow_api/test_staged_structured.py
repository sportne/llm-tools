"""Tests for the shared staged structured tool runner."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.tool_api import ToolSpec
from llm_tools.workflow_api.executor import PreparedModelInteraction
from llm_tools.workflow_api.staged_structured import (
    StagedStructuredToolRunner,
    format_invalid_payload,
    repair_stage_guidance,
    tool_spec_by_name,
    validation_error_summary,
)


class _LookupInput(BaseModel):
    path: str


class _FinalAnswer(BaseModel):
    answer: str


class _FakeProvider:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.messages: list[list[dict[str, object]]] = []
        self.response_model_names: list[str] = []

    def run_structured(self, **kwargs: object) -> object:
        return self._next(kwargs)

    async def run_structured_async(self, **kwargs: object) -> object:
        return self._next(kwargs)

    def _next(self, kwargs: dict[str, object]) -> object:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.messages.append([dict(message) for message in messages])
        response_model = kwargs["response_model"]
        assert isinstance(response_model, type)
        self.response_model_names.append(response_model.__name__)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _prepared(adapter: ActionEnvelopeAdapter) -> PreparedModelInteraction:
    spec = ToolSpec(name="lookup", description="Read one path.")
    response_model = adapter.build_response_model(
        [spec],
        {"lookup": _LookupInput},
        final_response_model=_FinalAnswer,
    )
    return PreparedModelInteraction(
        response_model=response_model,
        schema=adapter.export_schema(response_model),
        tool_names=["lookup"],
        tool_specs=[spec],
        input_models={"lookup": _LookupInput},
    )


def test_staged_runner_runs_tool_stage_with_small_models() -> None:
    adapter = ActionEnvelopeAdapter()
    provider = _FakeProvider(
        [
            {"mode": "tool", "tool_name": "lookup"},
            {
                "mode": "tool",
                "tool_name": "lookup",
                "arguments": {"path": "README.md"},
            },
        ]
    )

    parsed = StagedStructuredToolRunner(adapter=adapter, temperature=0).run_round(
        provider=provider,
        messages=[{"role": "user", "content": "inspect"}],
        prepared=_prepared(adapter),
        final_response_model=_FinalAnswer,
    )

    assert parsed.invocations[0].tool_name == "lookup"
    assert parsed.invocations[0].arguments == {"path": "README.md"}
    assert provider.response_model_names == ["DecisionStep", "LookupInvocationStep"]
    assert "ActionEnvelope" not in provider.response_model_names
    assert "Current step: choose the next action." in str(
        provider.messages[0][-1]["content"]
    )
    assert "invoke the selected tool 'lookup'" in str(
        provider.messages[1][-1]["content"]
    )


def test_staged_runner_runs_final_stage_async() -> None:
    adapter = ActionEnvelopeAdapter()
    provider = _FakeProvider(
        [
            {"mode": "finalize"},
            {"mode": "finalize", "final_response": {"answer": "done"}},
        ]
    )

    parsed = asyncio.run(
        StagedStructuredToolRunner(adapter=adapter, temperature=0).run_round_async(
            provider=provider,
            messages=[{"role": "user", "content": "answer"}],
            prepared=_prepared(adapter),
            final_response_model=_FinalAnswer,
        )
    )

    assert parsed.final_response == {"answer": "done"}
    assert provider.response_model_names == ["DecisionStep", "FinalResponseStep"]
    assert "Current step: finalize the answer." in str(provider.messages[1][-1])


def test_staged_runner_repairs_twice_then_raises() -> None:
    adapter = ActionEnvelopeAdapter()
    provider = _FakeProvider(
        [
            RuntimeError("first"),
            RuntimeError("second"),
            RuntimeError("third"),
        ]
    )
    runner = StagedStructuredToolRunner(adapter=adapter, temperature=0)

    with pytest.raises(RuntimeError, match="third"):
        runner.run_round(
            provider=provider,
            messages=[{"role": "user", "content": "answer"}],
            prepared=_prepared(adapter),
            final_response_model=_FinalAnswer,
        )

    assert len(provider.messages) == 3
    assert provider.messages[1][-1]["content"].startswith(
        "The previous decision response was invalid."
    )
    assert provider.messages[2][-1]["content"].startswith(
        "The previous decision response was invalid."
    )


def test_staged_runner_helpers_cover_edge_cases() -> None:
    adapter = ActionEnvelopeAdapter()
    runner = StagedStructuredToolRunner(adapter=adapter, temperature=0)
    spec = ToolSpec(name="lookup", description="Read one path.")

    assert validation_error_summary(RuntimeError("")) == "RuntimeError"
    assert repair_stage_guidance("tool:lookup").startswith("Tool stage rules:")
    assert repair_stage_guidance("final_response").startswith(
        "Finalization stage rules:"
    )
    assert (
        repair_stage_guidance("other")
        == "Return only the fields required for this stage."
    )
    assert format_invalid_payload(None) == "(unavailable)"
    assert format_invalid_payload("bad") == "bad"
    bad_keys = {object(): "value"}
    assert format_invalid_payload(bad_keys) == str(bad_keys)
    assert tool_spec_by_name([spec], "lookup") is spec
    assert runner.selected_tool(
        tool_specs=[spec],
        input_models={"lookup": _LookupInput},
        tool_name="lookup",
    ) == (spec, _LookupInput)
    with pytest.raises(ValueError, match="Unknown tool selected"):
        tool_spec_by_name([], "missing")
    with pytest.raises(ValueError, match="valid tool name"):
        runner.selected_tool(tool_specs=[spec], input_models={}, tool_name="")
    with pytest.raises(ValueError, match="was not prepared"):
        runner.selected_tool(tool_specs=[spec], input_models={}, tool_name="lookup")

"""Canonical adapter for structured action-envelope interactions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class _LooseAction(BaseModel):
    """Fallback action model for local/offline payload parsing."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class _SimplifiedEnvelopeMixin(BaseModel):
    """Lightweight envelope for JSON-schema-constrained providers."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["actions", "final"]

    @model_validator(mode="after")
    def _validate_mode(self) -> _SimplifiedEnvelopeMixin:
        actions = getattr(self, "actions", [])
        final_response = getattr(self, "final_response", None)
        if self.mode == "actions":
            if len(actions) == 0 or final_response is not None:
                raise ValueError(
                    "Simplified action envelope requires actions-only payloads in actions mode."
                )
            return self
        if len(actions) != 0 or final_response is None:
            raise ValueError(
                "Simplified action envelope requires final_response-only payloads in final mode."
            )
        return self


class _DecisionStepMixin(BaseModel):
    """Strict decision step for staged structured interaction."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["tool", "finalize"]
    tool_name: str | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> _DecisionStepMixin:
        if self.mode == "tool":
            if self.tool_name is None:
                raise ValueError("tool_name is required when mode='tool'.")
            return self
        if self.tool_name is not None:
            raise ValueError("tool_name is only allowed when mode='tool'.")
        return self


class _ToolInvocationStepMixin(BaseModel):
    """Strict tool-invocation step for staged structured interaction."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["tool"] = "tool"


class _FinalResponseStepMixin(BaseModel):
    """Strict finalization step for staged structured interaction."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["finalize"] = "finalize"


class _SingleActionStepMixin(BaseModel):
    """Single-stage structured agent step."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["tool", "finalize"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    final_response: Any | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> _SingleActionStepMixin:
        if self.mode == "tool":
            if self.tool_name is None:
                raise ValueError("tool_name is required when mode='tool'.")
            if self.final_response is not None:
                raise ValueError("final_response is not allowed when mode='tool'.")
            return self
        if self.tool_name is not None:
            raise ValueError("tool_name is only allowed when mode='tool'.")
        if self.arguments:
            raise ValueError("arguments are only allowed when mode='tool'.")
        if self.final_response is None:
            raise ValueError("final_response is required when mode='finalize'.")
        return self


class _EnvelopeModeMixin(BaseModel):
    """Enforce one-turn output mode consistency."""

    model_config = ConfigDict(extra="forbid")
    final_response: Any | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> _EnvelopeModeMixin:
        actions = getattr(self, "actions", [])
        has_actions = len(actions) > 0
        has_final_response = self.final_response is not None

        if isinstance(self.final_response, str) and self.final_response.strip() == "":
            raise ValueError("final_response must not be empty.")
        if has_actions == has_final_response:
            raise ValueError(
                "Action envelope must contain either actions or final_response."
            )
        return self


class _LooseEnvelope(_EnvelopeModeMixin):
    """Lax envelope used when no typed response model is provided."""

    actions: list[_LooseAction] = Field(default_factory=list)
    final_response: Any | None = None


class _NoActionsEnvelopeMixin(_EnvelopeModeMixin):
    """Reject action payloads when no tools are exposed."""

    @model_validator(mode="after")
    def _validate_no_actions(self) -> _NoActionsEnvelopeMixin:
        if getattr(self, "actions", []):
            raise ValueError("No tools are available; actions must be empty.")
        return self


__all__ = [
    "_DecisionStepMixin",
    "_EnvelopeModeMixin",
    "_FinalResponseStepMixin",
    "_LooseAction",
    "_SimplifiedEnvelopeMixin",
    "_SingleActionStepMixin",
    "_ToolInvocationStepMixin",
    "_LooseEnvelope",
    "_NoActionsEnvelopeMixin",
]

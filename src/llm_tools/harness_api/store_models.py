"""Durable storage abstractions for canonical harness state."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, model_validator

from llm_tools.harness_api.models import HarnessState
from llm_tools.harness_api.replay import StoredHarnessArtifacts


class StoredHarnessState(BaseModel):
    """A versioned store snapshot for one persisted harness session."""

    session_id: str
    revision: str
    saved_at: str = Field(
        default_factory=lambda: _timestamp(datetime.now(UTC)),
        min_length=1,
    )
    state: HarnessState
    artifacts: StoredHarnessArtifacts = Field(default_factory=StoredHarnessArtifacts)

    @model_validator(mode="after")
    def validate_session_binding(self) -> StoredHarnessState:
        """Bind the store snapshot identity to the canonical inner state."""
        if self.session_id != self.state.session.session_id:
            raise ValueError(
                "StoredHarnessState session_id must match state.session.session_id."
            )
        if (
            self.artifacts.trace is not None
            and self.artifacts.trace.session_id != self.session_id
        ):
            raise ValueError(
                "StoredHarnessState trace.session_id must match session_id."
            )
        if (
            self.artifacts.summary is not None
            and self.artifacts.summary.session_id != self.session_id
        ):
            raise ValueError(
                "StoredHarnessState summary.session_id must match session_id."
            )
        return self


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "StoredHarnessState",
    "_timestamp",
]

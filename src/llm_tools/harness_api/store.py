"""Durable storage abstractions for canonical harness state."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ValidationError, model_validator

from llm_tools.harness_api.models import HarnessState

CURRENT_HARNESS_STATE_SCHEMA_VERSION = "3"
SUPPORTED_HARNESS_STATE_SCHEMA_VERSIONS = frozenset(
    {CURRENT_HARNESS_STATE_SCHEMA_VERSION}
)


class UnsupportedHarnessStateVersionError(ValueError):
    """Raised when persisted state uses an unsupported schema version."""


class HarnessStateConflictError(ValueError):
    """Raised when optimistic concurrency checks fail during save."""


class StoredHarnessState(BaseModel):
    """A versioned store snapshot for one persisted harness session."""

    session_id: str
    revision: str
    state: HarnessState

    @model_validator(mode="after")
    def validate_session_binding(self) -> StoredHarnessState:
        """Bind the store snapshot identity to the canonical inner state."""
        if self.session_id != self.state.session.session_id:
            raise ValueError(
                "StoredHarnessState session_id must match state.session.session_id."
            )
        return self


class HarnessStateStore(Protocol):
    """Store contract for durable harness session persistence."""

    def load_session(self, session_id: str) -> StoredHarnessState | None:
        """Return the current stored snapshot for one session, when present."""

    def save_session(
        self,
        state: HarnessState,
        *,
        expected_revision: str | None = None,
    ) -> StoredHarnessState:
        """Persist a new canonical snapshot, optionally with optimistic locking."""

    def delete_session(self, session_id: str) -> None:
        """Delete one persisted session snapshot when present."""


class InMemoryHarnessStateStore:
    """Minimal in-memory store that exercises the canonical persistence contract."""

    def __init__(self) -> None:
        self._snapshots: dict[str, StoredHarnessState] = {}

    def load_session(self, session_id: str) -> StoredHarnessState | None:
        snapshot = self._snapshots.get(session_id)
        if snapshot is None:
            return None
        return snapshot.model_copy(deep=True)

    def save_session(
        self,
        state: HarnessState,
        *,
        expected_revision: str | None = None,
    ) -> StoredHarnessState:
        ensure_supported_schema_version(state.schema_version)
        session_id = state.session.session_id
        current = self._snapshots.get(session_id)
        if expected_revision is not None:
            current_revision = None if current is None else current.revision
            if current_revision != expected_revision:
                raise HarnessStateConflictError(
                    f"Session '{session_id}' revision mismatch."
                )

        next_revision = "1" if current is None else str(int(current.revision) + 1)
        snapshot = StoredHarnessState(
            session_id=session_id,
            revision=next_revision,
            state=state.model_copy(deep=True),
        )
        self._snapshots[session_id] = snapshot
        return snapshot.model_copy(deep=True)

    def delete_session(self, session_id: str) -> None:
        self._snapshots.pop(session_id, None)


def ensure_supported_schema_version(schema_version: str) -> None:
    """Reject persisted state that requires unsupported migration logic."""
    if schema_version not in SUPPORTED_HARNESS_STATE_SCHEMA_VERSIONS:
        raise UnsupportedHarnessStateVersionError(
            "Unsupported harness state schema_version: "
            f"'{schema_version}'. Supported versions: "
            + ", ".join(sorted(SUPPORTED_HARNESS_STATE_SCHEMA_VERSIONS))
            + "."
        )


def serialize_harness_state(state: HarnessState) -> str:
    """Serialize canonical harness state for durable storage."""
    ensure_supported_schema_version(state.schema_version)
    return state.model_dump_json()


def deserialize_harness_state(payload: str) -> HarnessState:
    """Deserialize and validate canonical harness state from persisted JSON."""
    try:
        state = HarnessState.model_validate_json(payload)
    except ValidationError:
        raise
    ensure_supported_schema_version(state.schema_version)
    return state

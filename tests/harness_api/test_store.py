"""Persistence contract tests for harness state storage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_tools.harness_api import (
    CURRENT_HARNESS_STATE_SCHEMA_VERSION,
    BudgetPolicy,
    HarnessSession,
    HarnessState,
    HarnessStateConflictError,
    InMemoryHarnessStateStore,
    StoredHarnessState,
    TaskOrigin,
    TaskRecord,
    UnsupportedHarnessStateVersionError,
    deserialize_harness_state,
    serialize_harness_state,
)


def _state(schema_version: str = CURRENT_HARNESS_STATE_SCHEMA_VERSION) -> HarnessState:
    return HarnessState(
        schema_version=schema_version,
        session=HarnessSession(
            session_id="session-1",
            root_task_id="task-1",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:00Z",
        ),
        tasks=[
            TaskRecord(
                task_id="task-1",
                title="Root task",
                intent="Complete the request.",
                origin=TaskOrigin.USER_REQUESTED,
            )
        ],
    )


def test_store_round_trips_typed_snapshots() -> None:
    store = InMemoryHarnessStateStore()
    first = store.save_session(_state())

    assert isinstance(first, StoredHarnessState)
    assert first.session_id == "session-1"
    assert first.revision == "1"

    loaded = store.load_session("session-1")
    assert loaded == first
    assert loaded is not first
    assert loaded.state == first.state


def test_store_enforces_optimistic_concurrency() -> None:
    store = InMemoryHarnessStateStore()
    first = store.save_session(_state())

    updated_state = first.state.model_copy(
        update={
            "session": first.state.session.model_copy(update={"current_turn_index": 0})
        }
    )
    second = store.save_session(updated_state, expected_revision=first.revision)
    assert second.revision == "2"

    with pytest.raises(HarnessStateConflictError, match="revision mismatch"):
        store.save_session(updated_state, expected_revision="1")


def test_store_delete_is_idempotent() -> None:
    store = InMemoryHarnessStateStore()
    store.save_session(_state())

    store.delete_session("session-1")
    store.delete_session("session-1")

    assert store.load_session("session-1") is None


def test_serialization_rejects_unsupported_schema_versions() -> None:
    state = _state(schema_version="99")

    with pytest.raises(UnsupportedHarnessStateVersionError, match="Unsupported"):
        serialize_harness_state(state)

    payload = state.model_dump_json()
    with pytest.raises(UnsupportedHarnessStateVersionError, match="Unsupported"):
        deserialize_harness_state(payload)


def test_deserialize_rejects_corrupt_payloads() -> None:
    with pytest.raises(ValidationError):
        deserialize_harness_state('{"schema_version":"2","session":{"session_id":"s"}}')

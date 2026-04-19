"""Persistence contract tests for harness state storage."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_tools.harness_api import (
    CURRENT_HARNESS_STATE_SCHEMA_VERSION,
    BudgetPolicy,
    FileHarnessStateStore,
    HarnessSession,
    HarnessSessionSummary,
    HarnessState,
    HarnessStateConflictError,
    HarnessStateCorruptionError,
    InMemoryHarnessStateStore,
    StoredHarnessArtifacts,
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
        deserialize_harness_state(
            f'{{"schema_version":"{CURRENT_HARNESS_STATE_SCHEMA_VERSION}","session":{{"session_id":"s"}}}}'
        )


def test_store_round_trips_saved_at_and_artifacts() -> None:
    store = InMemoryHarnessStateStore()
    snapshot = store.save_session(
        _state(),
        artifacts=StoredHarnessArtifacts(
            summary=HarnessSessionSummary(
                session_id="session-1",
                current_turn_index=0,
                total_turns=0,
            )
        ),
    )

    assert snapshot.saved_at.endswith("Z")
    assert snapshot.artifacts.summary is not None
    assert snapshot.artifacts.summary.session_id == "session-1"


def test_store_lists_sessions_newest_first() -> None:
    store = InMemoryHarnessStateStore()
    first_state = _state()
    first = store.save_session(first_state)
    second_state = HarnessState(
        schema_version=CURRENT_HARNESS_STATE_SCHEMA_VERSION,
        session=HarnessSession(
            session_id="session-2",
            root_task_id="task-2",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:01Z",
        ),
        tasks=[
            TaskRecord(
                task_id="task-2",
                title="Second task",
                intent="Second request.",
                origin=TaskOrigin.USER_REQUESTED,
            )
        ],
    )
    second = store.save_session(second_state)

    snapshots = store.list_sessions()

    assert [snapshot.session_id for snapshot in snapshots] == [
        second.session_id,
        first.session_id,
    ]


def test_file_store_round_trips_snapshots(tmp_path) -> None:
    store = FileHarnessStateStore(tmp_path)
    saved = store.save_session(_state())

    loaded = store.load_session(saved.session_id)

    assert loaded == saved
    assert store.list_sessions(limit=1)[0].session_id == saved.session_id


def test_store_list_limit_and_successful_round_trip_serialization() -> None:
    store = InMemoryHarnessStateStore()
    store.save_session(_state())
    second_state = _state().model_copy(
        update={
            "session": HarnessSession(
                session_id="session-2",
                root_task_id="task-2",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:01Z",
            ),
            "tasks": [
                TaskRecord(
                    task_id="task-2",
                    title="Second task",
                    intent="Second request.",
                    origin=TaskOrigin.USER_REQUESTED,
                )
            ],
        }
    )
    store.save_session(second_state)

    limited = store.list_sessions(limit=1)
    assert len(limited) == 1

    payload = serialize_harness_state(_state())
    deserialized = deserialize_harness_state(payload)
    assert deserialized.session.session_id == "session-1"


def test_stored_harness_state_validates_snapshot_bindings() -> None:
    state = _state()

    with pytest.raises(ValidationError, match="session_id must match"):
        StoredHarnessState(session_id="other", revision="1", state=state)

    with pytest.raises(ValidationError, match="trace.session_id must match"):
        StoredHarnessState(
            session_id="session-1",
            revision="1",
            state=state,
            artifacts=StoredHarnessArtifacts(
                trace={"session_id": "other", "turns": [], "final_stop_reason": None}
            ),
        )

    with pytest.raises(ValidationError, match="summary.session_id must match"):
        StoredHarnessState(
            session_id="session-1",
            revision="1",
            state=state,
            artifacts=StoredHarnessArtifacts(
                summary=HarnessSessionSummary(
                    session_id="other",
                    current_turn_index=0,
                    total_turns=0,
                )
            ),
        )


def test_file_store_conflict_and_delete_existing_file(tmp_path) -> None:
    store = FileHarnessStateStore(tmp_path)
    saved = store.save_session(_state())

    with pytest.raises(HarnessStateConflictError, match="revision mismatch"):
        store.save_session(_state(), expected_revision="999")

    store.delete_session(saved.session_id)
    assert store.load_session(saved.session_id) is None


def test_file_store_raises_typed_corruption_for_invalid_session_file(tmp_path) -> None:
    store = FileHarnessStateStore(tmp_path)
    path = store._path_for_session("session-1")
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(HarnessStateCorruptionError, match="Corrupt persisted"):
        store.load_session("session-1")


def test_file_store_list_sessions_skips_corrupt_files(tmp_path) -> None:
    store = FileHarnessStateStore(tmp_path)
    saved = store.save_session(_state())
    store._path_for_session("broken").write_text("{not json", encoding="utf-8")

    snapshots = store.list_sessions()

    assert [snapshot.session_id for snapshot in snapshots] == [saved.session_id]


def test_file_store_atomic_save_preserves_existing_snapshot_on_replace_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileHarnessStateStore(tmp_path)
    initial = store.save_session(_state())
    original_replace = Path.replace

    def _fail_replace(self: Path, target: Path) -> Path:
        if self.suffix == ".tmp":
            raise OSError("replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", _fail_replace)

    updated_state = initial.state.model_copy(
        update={
            "tasks": [
                initial.state.tasks[0].model_copy(update={"status_summary": "updated"})
            ]
        },
        deep=True,
    )
    with pytest.raises(OSError, match="replace failed"):
        store.save_session(updated_state, expected_revision=initial.revision)

    loaded = store.load_session(initial.session_id)
    assert loaded == initial
    assert list(tmp_path.glob("*.tmp")) == []

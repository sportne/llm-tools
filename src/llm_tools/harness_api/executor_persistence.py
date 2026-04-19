"""Persistence helpers for harness execution."""

from __future__ import annotations

from llm_tools.harness_api.models import HarnessState
from llm_tools.harness_api.replay import StoredHarnessArtifacts
from llm_tools.harness_api.store import (
    HarnessStateConflictError,
    HarnessStateStore,
    StoredHarnessState,
)


def load_required_snapshot(
    store: HarnessStateStore, session_id: str
) -> StoredHarnessState:
    """Load one stored session snapshot or raise for an unknown id."""
    snapshot = store.load_session(session_id)
    if snapshot is None:
        raise ValueError(f"Unknown session id: {session_id}")
    return snapshot


def save_session_with_conflict_retry(
    *,
    store: HarnessStateStore,
    base_snapshot: StoredHarnessState,
    new_state: HarnessState,
    artifacts: StoredHarnessArtifacts | None,
    max_persistence_retries: int,
) -> StoredHarnessState:
    """Save state with optimistic-lock retry when no external divergence occurred."""
    expected_revision = base_snapshot.revision
    attempts = 0
    while True:
        try:
            return store.save_session(
                new_state,
                expected_revision=expected_revision,
                artifacts=artifacts,
            )
        except HarnessStateConflictError:
            if attempts >= max_persistence_retries:
                raise
            latest = load_required_snapshot(store, base_snapshot.session_id)
            if latest.state != base_snapshot.state:
                raise
            attempts += 1
            expected_revision = latest.revision


__all__ = ["load_required_snapshot", "save_session_with_conflict_retry"]

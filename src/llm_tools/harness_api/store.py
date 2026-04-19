"""Durable storage abstractions for canonical harness state."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import quote
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, model_validator

from llm_tools.harness_api.models import HarnessState
from llm_tools.harness_api.replay import StoredHarnessArtifacts

CURRENT_HARNESS_STATE_SCHEMA_VERSION = "3"
SUPPORTED_HARNESS_STATE_SCHEMA_VERSIONS = frozenset(
    {CURRENT_HARNESS_STATE_SCHEMA_VERSION}
)


class UnsupportedHarnessStateVersionError(ValueError):
    """Raised when persisted state uses an unsupported schema version."""


class HarnessStateConflictError(ValueError):
    """Raised when optimistic concurrency checks fail during save."""


class HarnessStateCorruptionError(ValueError):
    """Raised when a persisted harness session file cannot be trusted."""

    def __init__(self, path: Path, message: str) -> None:
        self.path = path
        super().__init__(f"{message}: {path}")


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


class HarnessStateStore(Protocol):
    """Store contract for durable harness session persistence."""

    def load_session(self, session_id: str) -> StoredHarnessState | None:
        """Return the current stored snapshot for one session, when present."""

    def save_session(
        self,
        state: HarnessState,
        *,
        expected_revision: str | None = None,
        artifacts: StoredHarnessArtifacts | None = None,
    ) -> StoredHarnessState:
        """Persist a new canonical snapshot, optionally with optimistic locking."""

    def delete_session(self, session_id: str) -> None:
        """Delete one persisted session snapshot when present."""

    def list_sessions(self, *, limit: int | None = None) -> list[StoredHarnessState]:
        """Return stored sessions in newest-first order."""


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
        artifacts: StoredHarnessArtifacts | None = None,
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
            saved_at=_timestamp(datetime.now(UTC)),
            state=state.model_copy(deep=True),
            artifacts=_resolved_artifacts(current=current, artifacts=artifacts),
        )
        self._snapshots[session_id] = snapshot
        return snapshot.model_copy(deep=True)

    def delete_session(self, session_id: str) -> None:
        self._snapshots.pop(session_id, None)

    def list_sessions(self, *, limit: int | None = None) -> list[StoredHarnessState]:
        snapshots = sorted(
            (snapshot.model_copy(deep=True) for snapshot in self._snapshots.values()),
            key=lambda item: (item.saved_at, item.session_id),
            reverse=True,
        )
        if limit is not None:
            return snapshots[:limit]
        return snapshots


class FileHarnessStateStore:
    """JSON-file-backed harness store suitable for CLI and manual inspection."""

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def load_session(self, session_id: str) -> StoredHarnessState | None:
        path = self._path_for_session(session_id)
        if not path.exists():
            return None
        return self._load_snapshot_file(path)

    def save_session(
        self,
        state: HarnessState,
        *,
        expected_revision: str | None = None,
        artifacts: StoredHarnessArtifacts | None = None,
    ) -> StoredHarnessState:
        ensure_supported_schema_version(state.schema_version)
        session_id = state.session.session_id
        current = self.load_session(session_id)
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
            saved_at=_timestamp(datetime.now(UTC)),
            state=state.model_copy(deep=True),
            artifacts=_resolved_artifacts(current=current, artifacts=artifacts),
        )
        path = self._path_for_session(session_id)
        temp_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            temp_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
            temp_path.replace(path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
        return snapshot.model_copy(deep=True)

    def delete_session(self, session_id: str) -> None:
        path = self._path_for_session(session_id)
        if path.exists():
            path.unlink()

    def list_sessions(self, *, limit: int | None = None) -> list[StoredHarnessState]:
        snapshots: list[StoredHarnessState] = []
        for path in self._directory.glob("*.json"):
            try:
                snapshots.append(self._load_snapshot_file(path))
            except HarnessStateCorruptionError:
                continue
        snapshots.sort(
            key=lambda item: (item.saved_at, item.session_id),
            reverse=True,
        )
        if limit is not None:
            return snapshots[:limit]
        return snapshots

    def _path_for_session(self, session_id: str) -> Path:
        return self._directory / f"{quote(session_id, safe='')}.json"

    @staticmethod
    def _load_snapshot_file(path: Path) -> StoredHarnessState:
        try:
            payload = path.read_text(encoding="utf-8")
            return StoredHarnessState.model_validate_json(payload)
        except (OSError, UnicodeDecodeError, ValidationError, ValueError) as exc:
            raise HarnessStateCorruptionError(
                path,
                "Corrupt persisted harness session file",
            ) from exc


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


def _resolved_artifacts(
    *,
    current: StoredHarnessState | None,
    artifacts: StoredHarnessArtifacts | None,
) -> StoredHarnessArtifacts:
    if artifacts is not None:
        return artifacts.model_copy(deep=True)
    if current is None:
        return StoredHarnessArtifacts()
    return current.artifacts.model_copy(deep=True)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

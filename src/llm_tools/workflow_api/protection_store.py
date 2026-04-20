"""Persistence helpers for workflow protection state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml  # type: ignore[import-untyped]

from llm_tools.workflow_api.protection_models import (
    ProtectionConfig,
    ProtectionCorpus,
    ProtectionDocument,
    ProtectionFeedbackEntry,
    ProtectionFeedbackFile,
)


class ProtectionFeedbackStore:
    """Read and write the structured corrections sidecar file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load_entries(self) -> list[ProtectionFeedbackEntry]:
        if not self._path.exists():
            return []
        raw = self._read_payload()
        data = raw if isinstance(raw, dict) else {"entries": raw}
        return ProtectionFeedbackFile.model_validate(data).entries

    def save_entries(self, entries: list[ProtectionFeedbackEntry]) -> None:
        payload = ProtectionFeedbackFile(entries=entries)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_payload(payload.model_dump(mode="json"))

    def append_entry(
        self, entry: ProtectionFeedbackEntry
    ) -> list[ProtectionFeedbackEntry]:
        entries = self.load_entries()
        entries.append(entry)
        self.save_entries(entries)
        return entries

    def _read_payload(self) -> Any:
        suffix = self._path.suffix.lower()
        text = self._path.read_text(encoding="utf-8")
        if suffix == ".json":
            return json.loads(text)
        if suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(text)
            return {} if loaded is None else loaded
        raise ValueError(f"Unsupported protection feedback file type: {self._path}")

    def _write_payload(self, payload: dict[str, Any]) -> None:
        suffix = self._path.suffix.lower()
        if suffix == ".json":
            self._path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            return
        if suffix in {".yaml", ".yml"}:
            self._path.write_text(
                yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
            )
            return
        raise ValueError(f"Unsupported protection feedback file type: {self._path}")


def load_protection_corpus(
    config: ProtectionConfig,
    *,
    feedback_store: ProtectionFeedbackStore | None = None,
) -> ProtectionCorpus:
    """Load the configured unstructured documents and structured feedback entries."""
    documents = [
        ProtectionDocument(
            document_id=_document_id_for_path(path),
            path=path,
            content=Path(path).read_text(encoding="utf-8"),
        )
        for path in config.document_paths
    ]
    feedback_entries: list[ProtectionFeedbackEntry] = []
    if feedback_store is not None:
        feedback_entries = feedback_store.load_entries()
    return ProtectionCorpus(documents=documents, feedback_entries=feedback_entries)


def _document_id_for_path(path: str) -> str:
    return Path(path).name or f"document-{uuid4().hex}"


__all__ = ["ProtectionFeedbackStore", "load_protection_corpus"]

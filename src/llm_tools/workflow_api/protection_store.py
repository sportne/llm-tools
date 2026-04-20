"""Persistence helpers for workflow protection state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from llm_tools.workflow_api.protection_models import (
    ProtectionConfig,
    ProtectionCorpus,
    ProtectionDocument,
    ProtectionFeedbackEntry,
    ProtectionFeedbackFile,
)

_ALLOWED_PROTECTION_DOCUMENT_SUFFIXES = {
    ".adoc",
    ".csv",
    ".json",
    ".log",
    ".md",
    ".rst",
    ".text",
    ".txt",
    ".yaml",
    ".yml",
}


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


class ProtectionCorpusLoadIssue(BaseModel):
    """One issue discovered while loading the protection corpus."""

    path: str
    message: str


class ProtectionCorpusLoadReport(BaseModel):
    """Protection corpus plus any load-time issues."""

    corpus: ProtectionCorpus
    issues: list[ProtectionCorpusLoadIssue] = Field(default_factory=list)

    @property
    def usable_document_count(self) -> int:
        return len(self.corpus.documents)


def inspect_protection_corpus(
    config: ProtectionConfig,
    *,
    feedback_store: ProtectionFeedbackStore | None = None,
) -> ProtectionCorpusLoadReport:
    """Load the configured documents and report skipped or invalid entries."""
    documents: list[ProtectionDocument] = []
    issues: list[ProtectionCorpusLoadIssue] = []
    for configured_path in config.document_paths:
        resolved = Path(configured_path).expanduser()
        if not resolved.exists():
            issues.append(
                ProtectionCorpusLoadIssue(
                    path=str(configured_path),
                    message="Path does not exist.",
                )
            )
            continue
        if resolved.is_dir():
            for candidate in sorted(resolved.rglob("*")):
                if not candidate.is_file():
                    continue
                suffix = candidate.suffix.lower()
                if suffix not in _ALLOWED_PROTECTION_DOCUMENT_SUFFIXES:
                    issues.append(
                        ProtectionCorpusLoadIssue(
                            path=str(candidate),
                            message=(
                                "Skipped unsupported file type for protection corpus."
                            ),
                        )
                    )
                    continue
                try:
                    content = candidate.read_text(encoding="utf-8")
                except Exception as exc:
                    issues.append(
                        ProtectionCorpusLoadIssue(
                            path=str(candidate),
                            message=(
                                "Unable to read protection document: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )
                    )
                    continue
                documents.append(
                    ProtectionDocument(
                        document_id=_document_id_for_path(str(candidate)),
                        path=str(candidate),
                        content=content,
                    )
                )
            continue
        suffix = resolved.suffix.lower()
        if suffix not in _ALLOWED_PROTECTION_DOCUMENT_SUFFIXES:
            issues.append(
                ProtectionCorpusLoadIssue(
                    path=str(resolved),
                    message="Unsupported file type for protection corpus.",
                )
            )
            continue
        try:
            content = resolved.read_text(encoding="utf-8")
        except Exception as exc:
            issues.append(
                ProtectionCorpusLoadIssue(
                    path=str(resolved),
                    message=(
                        f"Unable to read protection document: {type(exc).__name__}: {exc}"
                    ),
                )
            )
            continue
        documents.append(
            ProtectionDocument(
                document_id=_document_id_for_path(str(resolved)),
                path=str(resolved),
                content=content,
            )
        )
    feedback_entries: list[ProtectionFeedbackEntry] = []
    if feedback_store is not None:
        feedback_entries = feedback_store.load_entries()
    return ProtectionCorpusLoadReport(
        corpus=ProtectionCorpus(
            documents=documents,
            feedback_entries=feedback_entries,
        ),
        issues=issues,
    )


def load_protection_corpus(
    config: ProtectionConfig,
    *,
    feedback_store: ProtectionFeedbackStore | None = None,
) -> ProtectionCorpus:
    """Load the configured unstructured documents and structured feedback entries."""
    return inspect_protection_corpus(
        config,
        feedback_store=feedback_store,
    ).corpus


def _document_id_for_path(path: str) -> str:
    return Path(path).name or f"document-{uuid4().hex}"


__all__ = [
    "ProtectionCorpusLoadIssue",
    "ProtectionCorpusLoadReport",
    "ProtectionFeedbackStore",
    "inspect_protection_corpus",
    "load_protection_corpus",
]

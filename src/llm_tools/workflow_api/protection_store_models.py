"""Persistence helpers for workflow protection state."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.workflow_api.protection_models import (
    ProtectionCorpus,
)


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

    @property
    def converted_document_count(self) -> int:
        return sum(
            1 for document in self.corpus.documents if document.read_kind != "text"
        )

    @property
    def uncategorized_document_count(self) -> int:
        return sum(
            1 for document in self.corpus.documents if not document.sensitivity_label
        )


__all__ = [
    "ProtectionCorpusLoadIssue",
    "ProtectionCorpusLoadReport",
]

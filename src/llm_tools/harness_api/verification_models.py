"""Typed verification contracts for durable harness execution."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class VerificationTrigger(str, Enum):  # noqa: UP042
    """How verification is scheduled for a task expectation."""

    AUTOMATIC = "automatic"
    MANUAL = "manual"


class VerificationTiming(str, Enum):  # noqa: UP042
    """When verification should run relative to task completion."""

    BEFORE_COMPLETION = "before_completion"
    AFTER_COMPLETION = "after_completion"


class VerificationFailureMode(str, Enum):  # noqa: UP042
    """Reason a verification run failed or could not pass cleanly."""

    EXPECTATION_UNMET = "expectation_unmet"
    MISSING_EVIDENCE = "missing_evidence"
    VERIFIER_ERROR = "verifier_error"
    INCONCLUSIVE_CHECK = "inconclusive_check"


class VerificationStatus(str, Enum):  # noqa: UP042
    """Verification status recorded against a task."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class NoProgressSignalKind(str, Enum):  # noqa: UP042
    """Structured no-progress classifications for stalled sessions."""

    STALLED_TASK = "stalled_task"
    REPEATED_RETRY = "repeated_retry"
    REPEATED_OUTCOME = "repeated_outcome"


class VerificationExpectation(BaseModel):
    """Task-level declaration of what should be verified."""

    expectation_id: str
    description: str
    required_for_completion: bool = False
    trigger: VerificationTrigger = VerificationTrigger.AUTOMATIC
    timing: VerificationTiming = VerificationTiming.AFTER_COMPLETION
    verifier_name: str | None = None

    @model_validator(mode="after")
    def validate_expectation(self) -> VerificationExpectation:
        """Require stable identity and a non-empty description."""
        if self.expectation_id.strip() == "":
            raise ValueError("expectation_id must not be empty.")
        if self.description.strip() == "":
            raise ValueError("description must not be empty.")
        return self


class VerificationEvidenceRecord(BaseModel):
    """Persisted evidence captured during verification."""

    evidence_id: str
    task_id: str | None = None
    recorded_at: str | None = None
    summary: str | None = None
    artifact_ref: str | None = None
    verifier_name: str | None = None

    @model_validator(mode="after")
    def validate_record(self) -> VerificationEvidenceRecord:
        """Require stable evidence identity."""
        if self.evidence_id.strip() == "":
            raise ValueError("evidence_id must not be empty.")
        return self


class VerificationResult(BaseModel):
    """First-class result returned by a verifier run."""

    task_id: str
    status: VerificationStatus = VerificationStatus.NOT_RUN
    checked_at: str | None = None
    summary: str | None = None
    expectation_ids: list[str] = Field(default_factory=list)
    evidence: list[VerificationEvidenceRecord] = Field(default_factory=list)
    failure_mode: VerificationFailureMode | None = None

    @model_validator(mode="after")
    def validate_result(self) -> VerificationResult:
        """Require timestamps, stable references, and coherent failure state."""
        if self.task_id.strip() == "":
            raise ValueError("task_id must not be empty.")
        if self.status is not VerificationStatus.NOT_RUN and (
            self.checked_at is None or self.checked_at.strip() == ""
        ):
            raise ValueError("checked_at is required once verification has run.")
        if len(set(self.expectation_ids)) != len(self.expectation_ids):
            raise ValueError("VerificationResult expectation_ids must be unique.")
        evidence_ids = [record.evidence_id for record in self.evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("VerificationResult evidence ids must be unique.")
        if self.status is VerificationStatus.PASSED and self.failure_mode is not None:
            raise ValueError("failure_mode is not allowed for passed verification.")
        return self


class NoProgressSignal(BaseModel):
    """Persisted signal describing stalled harness execution."""

    signal_id: str
    kind: NoProgressSignalKind
    task_id: str | None = None
    detected_at: str | None = None
    summary: str | None = None

    @model_validator(mode="after")
    def validate_signal(self) -> NoProgressSignal:
        """Require stable signal identity."""
        if self.signal_id.strip() == "":
            raise ValueError("signal_id must not be empty.")
        return self


__all__ = [
    "NoProgressSignal",
    "NoProgressSignalKind",
    "VerificationEvidenceRecord",
    "VerificationExpectation",
    "VerificationFailureMode",
    "VerificationResult",
    "VerificationStatus",
    "VerificationTiming",
    "VerificationTrigger",
]

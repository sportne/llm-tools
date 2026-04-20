"""Protection models and protocols for workflow layers."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from llm_tools.tool_api import ProtectionProvenanceSnapshot

ProtectionEnvironment = dict[str, Any]


class ProtectionAction(str, Enum):  # noqa: UP042
    """Supported protection actions for prompts and responses."""

    ALLOW = "allow"
    CONSTRAIN = "constrain"
    CHALLENGE = "challenge"
    SANITIZE = "sanitize"
    BLOCK = "block"


class ProtectionConfig(BaseModel):
    """Config used to enable and load workflow protection state."""

    enabled: bool = False
    document_paths: list[str] = Field(default_factory=list)
    corrections_path: str | None = None
    challenge_interactively: bool = True
    review_final_answers: bool = True
    purge_library_state_on_violation: bool = True
    cache_documents: bool = True

    @field_validator("document_paths")
    @classmethod
    def validate_document_paths(cls, value: list[str]) -> list[str]:
        cleaned = [entry.strip() for entry in value if entry.strip()]
        return cleaned

    @field_validator("corrections_path")
    @classmethod
    def validate_corrections_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ProtectionDocument(BaseModel):
    """One unstructured document used for sensitivity classification."""

    document_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    content: str


class ProtectionFeedbackEntry(BaseModel):
    """One user-confirmed sensitivity correction stored alongside the corpus."""

    entry_id: str = Field(
        default_factory=lambda: f"feedback-{uuid4().hex}", min_length=1
    )
    example_type: str = Field(default="prompt", min_length=1)
    example_text: str = Field(min_length=1)
    expected_sensitivity_label: str = Field(min_length=1)
    rationale: str | None = None
    session_id: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        min_length=1,
    )
    referenced_document_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("example_type", "expected_sensitivity_label")
    @classmethod
    def validate_non_empty_fields(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("feedback fields must not be empty")
        return cleaned

    @field_validator("rationale")
    @classmethod
    def validate_optional_rationale(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ProtectionFeedbackFile(BaseModel):
    """Serializable corrections sidecar payload."""

    entries: list[ProtectionFeedbackEntry] = Field(default_factory=list)


class ProtectionCorpus(BaseModel):
    """Loaded sensitivity documents plus structured feedback entries."""

    documents: list[ProtectionDocument] = Field(default_factory=list)
    feedback_entries: list[ProtectionFeedbackEntry] = Field(default_factory=list)


class ProtectionAssessment(BaseModel):
    """Classifier assessment prior to environment comparison."""

    sensitivity_label: str | None = None
    reasoning: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    referenced_document_ids: list[str] = Field(default_factory=list)
    recommended_action: ProtectionAction | None = None
    guard_text: str | None = None
    sanitized_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptProtectionDecision(BaseModel):
    """Final controller decision for one model-bound prompt."""

    action: ProtectionAction = ProtectionAction.ALLOW
    sensitivity_label: str | None = None
    reasoning: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    referenced_document_ids: list[str] = Field(default_factory=list)
    guard_text: str | None = None
    challenge_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResponseProtectionDecision(BaseModel):
    """Final controller decision for one candidate final response."""

    action: ProtectionAction = ProtectionAction.ALLOW
    sensitivity_label: str | None = None
    reasoning: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    referenced_document_ids: list[str] = Field(default_factory=list)
    sanitized_payload: Any | None = None
    safe_message: str | None = None
    should_purge: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProtectionFeedbackPrompt(BaseModel):
    """Structured feedback reply when a user disputes a prompt challenge."""

    analysis_is_correct: bool
    expected_sensitivity_label: str | None = None
    rationale: str | None = None

    @field_validator("expected_sensitivity_label")
    @classmethod
    def validate_expected_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def validate_required_label(self) -> ProtectionFeedbackPrompt:
        if not self.analysis_is_correct and self.expected_sensitivity_label is None:
            raise ValueError(
                "expected_sensitivity_label is required when analysis_is_correct is false."
            )
        return self


class ProtectionPendingPrompt(BaseModel):
    """Durable state for one unresolved prompt-side protection challenge."""

    original_user_message: str = Field(min_length=1)
    serialized_messages: list[dict[str, Any]] = Field(default_factory=list)
    reasoning: str = Field(min_length=1)
    predicted_sensitivity_label: str | None = None
    referenced_document_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        min_length=1,
    )
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class SensitivityClassifier(Protocol):
    """Classifier contract used by the protection controller."""

    def assess_prompt(
        self,
        *,
        corpus: ProtectionCorpus,
        environment: ProtectionEnvironment,
        messages: list[dict[str, Any]],
        provenance: ProtectionProvenanceSnapshot,
    ) -> ProtectionAssessment:
        """Return a sensitivity assessment for a model-bound prompt."""

    def assess_response(
        self,
        *,
        corpus: ProtectionCorpus,
        environment: ProtectionEnvironment,
        response_payload: Any,
        provenance: ProtectionProvenanceSnapshot,
    ) -> ProtectionAssessment:
        """Return a sensitivity assessment for a candidate final response."""


@runtime_checkable
class EnvironmentComparator(Protocol):
    """Map classifier assessments onto environment-specific actions."""

    def resolve_prompt_action(
        self,
        *,
        assessment: ProtectionAssessment,
        environment: ProtectionEnvironment,
    ) -> ProtectionAction:
        """Return the effective prompt action for the environment."""

    def resolve_response_action(
        self,
        *,
        assessment: ProtectionAssessment,
        environment: ProtectionEnvironment,
    ) -> ProtectionAction:
        """Return the effective response action for the environment."""


__all__ = [
    "EnvironmentComparator",
    "PromptProtectionDecision",
    "ProtectionAction",
    "ProtectionAssessment",
    "ProtectionConfig",
    "ProtectionCorpus",
    "ProtectionDocument",
    "ProtectionEnvironment",
    "ProtectionFeedbackEntry",
    "ProtectionFeedbackFile",
    "ProtectionFeedbackPrompt",
    "ProtectionPendingPrompt",
    "ResponseProtectionDecision",
    "SensitivityClassifier",
]

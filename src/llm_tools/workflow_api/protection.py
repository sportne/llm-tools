"""Optional prompt and response protection contracts for workflow layers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from llm_tools.tool_api import (
    ProtectionProvenanceSnapshot,
    SourceProvenanceRef,
    ToolResult,
)

try:
    import yaml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None

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


class DefaultEnvironmentComparator:
    """Simple comparator using recommended actions and optional label filters."""

    def resolve_prompt_action(
        self,
        *,
        assessment: ProtectionAssessment,
        environment: ProtectionEnvironment,
    ) -> ProtectionAction:
        return self._resolve_action(
            assessment=assessment,
            environment=environment,
            challenge_default=ProtectionAction.CHALLENGE,
        )

    def resolve_response_action(
        self,
        *,
        assessment: ProtectionAssessment,
        environment: ProtectionEnvironment,
    ) -> ProtectionAction:
        action = self._resolve_action(
            assessment=assessment,
            environment=environment,
            challenge_default=ProtectionAction.BLOCK,
        )
        if action is ProtectionAction.CHALLENGE:
            return ProtectionAction.BLOCK
        return action

    def _resolve_action(
        self,
        *,
        assessment: ProtectionAssessment,
        environment: ProtectionEnvironment,
        challenge_default: ProtectionAction,
    ) -> ProtectionAction:
        if assessment.recommended_action is not None:
            return assessment.recommended_action
        sensitivity_label = assessment.sensitivity_label
        if sensitivity_label is None:
            return ProtectionAction.ALLOW
        blocked_labels = {
            str(label).strip()
            for label in environment.get("blocked_sensitivity_labels", [])
            if str(label).strip()
        }
        allowed_labels = {
            str(label).strip()
            for label in environment.get("allowed_sensitivity_labels", [])
            if str(label).strip()
        }
        constrained_labels = {
            str(label).strip()
            for label in environment.get("constrained_sensitivity_labels", [])
            if str(label).strip()
        }
        sanitized_labels = {
            str(label).strip()
            for label in environment.get("sanitized_sensitivity_labels", [])
            if str(label).strip()
        }
        if sensitivity_label in blocked_labels:
            return challenge_default
        if allowed_labels and sensitivity_label not in allowed_labels:
            return challenge_default
        if sensitivity_label in constrained_labels:
            return ProtectionAction.CONSTRAIN
        if sensitivity_label in sanitized_labels:
            return ProtectionAction.SANITIZE
        return ProtectionAction.ALLOW


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
            if yaml is None:
                raise ValueError(
                    "PyYAML is required for YAML protection feedback files."
                )
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
            if yaml is None:
                raise ValueError(
                    "PyYAML is required for YAML protection feedback files."
                )
            self._path.write_text(
                yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
            )
            return
        raise ValueError(f"Unsupported protection feedback file type: {self._path}")


class ProtectionController:
    """Evaluate prompt and response sensitivity against a workflow environment."""

    def __init__(
        self,
        *,
        config: ProtectionConfig,
        classifier: SensitivityClassifier,
        environment: ProtectionEnvironment | None = None,
        comparator: EnvironmentComparator | None = None,
        corpus: ProtectionCorpus | None = None,
        feedback_store: ProtectionFeedbackStore | None = None,
    ) -> None:
        self._config = config
        self._classifier = classifier
        self._environment = dict(environment or {})
        self._comparator = comparator or DefaultEnvironmentComparator()
        self._feedback_store = feedback_store
        if self._feedback_store is None and config.corrections_path is not None:
            self._feedback_store = ProtectionFeedbackStore(config.corrections_path)
        self._corpus = corpus or load_protection_corpus(
            config,
            feedback_store=self._feedback_store,
        )

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def corpus(self) -> ProtectionCorpus:
        return self._corpus.model_copy(deep=True)

    def assess_prompt(
        self,
        *,
        messages: list[dict[str, Any]],
        provenance: ProtectionProvenanceSnapshot | None = None,
    ) -> PromptProtectionDecision:
        if not self.enabled:
            return PromptProtectionDecision(action=ProtectionAction.ALLOW)
        assessment = self._classifier.assess_prompt(
            corpus=self._corpus,
            environment=dict(self._environment),
            messages=list(messages),
            provenance=provenance or ProtectionProvenanceSnapshot(),
        )
        action = self._comparator.resolve_prompt_action(
            assessment=assessment,
            environment=dict(self._environment),
        )
        return PromptProtectionDecision(
            action=action,
            sensitivity_label=assessment.sensitivity_label,
            reasoning=assessment.reasoning,
            confidence=assessment.confidence,
            referenced_document_ids=list(assessment.referenced_document_ids),
            guard_text=assessment.guard_text or self._default_guard_text(assessment),
            challenge_message=self._challenge_message(assessment),
            metadata=dict(assessment.metadata),
        )

    async def assess_prompt_async(
        self,
        *,
        messages: list[dict[str, Any]],
        provenance: ProtectionProvenanceSnapshot | None = None,
    ) -> PromptProtectionDecision:
        assessor = getattr(self._classifier, "assess_prompt_async", None)
        if callable(assessor):
            assessment = await assessor(
                corpus=self._corpus,
                environment=dict(self._environment),
                messages=list(messages),
                provenance=provenance or ProtectionProvenanceSnapshot(),
            )
            action = self._comparator.resolve_prompt_action(
                assessment=assessment,
                environment=dict(self._environment),
            )
            return PromptProtectionDecision(
                action=action,
                sensitivity_label=assessment.sensitivity_label,
                reasoning=assessment.reasoning,
                confidence=assessment.confidence,
                referenced_document_ids=list(assessment.referenced_document_ids),
                guard_text=assessment.guard_text
                or self._default_guard_text(assessment),
                challenge_message=self._challenge_message(assessment),
                metadata=dict(assessment.metadata),
            )
        return self.assess_prompt(messages=messages, provenance=provenance)

    def review_response(
        self,
        *,
        response_payload: Any,
        provenance: ProtectionProvenanceSnapshot | None = None,
    ) -> ResponseProtectionDecision:
        if not self.enabled or not self._config.review_final_answers:
            return ResponseProtectionDecision(action=ProtectionAction.ALLOW)
        assessment = self._classifier.assess_response(
            corpus=self._corpus,
            environment=dict(self._environment),
            response_payload=response_payload,
            provenance=provenance or ProtectionProvenanceSnapshot(),
        )
        action = self._comparator.resolve_response_action(
            assessment=assessment,
            environment=dict(self._environment),
        )
        sanitized_payload = None
        if action is ProtectionAction.SANITIZE:
            sanitized_payload = _sanitize_payload(
                response_payload,
                assessment.sanitized_text,
            )
        return ResponseProtectionDecision(
            action=action,
            sensitivity_label=assessment.sensitivity_label,
            reasoning=assessment.reasoning,
            confidence=assessment.confidence,
            referenced_document_ids=list(assessment.referenced_document_ids),
            sanitized_payload=sanitized_payload,
            safe_message=self._blocked_response_message(assessment),
            should_purge=(
                action in {ProtectionAction.SANITIZE, ProtectionAction.BLOCK}
                and self._config.purge_library_state_on_violation
            ),
            metadata=dict(assessment.metadata),
        )

    async def review_response_async(
        self,
        *,
        response_payload: Any,
        provenance: ProtectionProvenanceSnapshot | None = None,
    ) -> ResponseProtectionDecision:
        assessor = getattr(self._classifier, "assess_response_async", None)
        if callable(assessor):
            assessment = await assessor(
                corpus=self._corpus,
                environment=dict(self._environment),
                response_payload=response_payload,
                provenance=provenance or ProtectionProvenanceSnapshot(),
            )
            action = self._comparator.resolve_response_action(
                assessment=assessment,
                environment=dict(self._environment),
            )
            sanitized_payload = None
            if action is ProtectionAction.SANITIZE:
                sanitized_payload = _sanitize_payload(
                    response_payload,
                    assessment.sanitized_text,
                )
            return ResponseProtectionDecision(
                action=action,
                sensitivity_label=assessment.sensitivity_label,
                reasoning=assessment.reasoning,
                confidence=assessment.confidence,
                referenced_document_ids=list(assessment.referenced_document_ids),
                sanitized_payload=sanitized_payload,
                safe_message=self._blocked_response_message(assessment),
                should_purge=(
                    action in {ProtectionAction.SANITIZE, ProtectionAction.BLOCK}
                    and self._config.purge_library_state_on_violation
                ),
                metadata=dict(assessment.metadata),
            )
        return self.review_response(
            response_payload=response_payload,
            provenance=provenance,
        )

    def build_pending_prompt(
        self,
        *,
        original_user_message: str,
        serialized_messages: list[dict[str, Any]],
        decision: PromptProtectionDecision,
        session_id: str | None = None,
    ) -> ProtectionPendingPrompt:
        return ProtectionPendingPrompt(
            original_user_message=original_user_message,
            serialized_messages=list(serialized_messages),
            reasoning=decision.reasoning
            or "Prompt sensitivity review blocked model execution.",
            predicted_sensitivity_label=decision.sensitivity_label,
            referenced_document_ids=list(decision.referenced_document_ids),
            session_id=session_id,
            metadata=dict(decision.metadata),
        )

    def parse_feedback_message(
        self,
        message: str,
    ) -> ProtectionFeedbackPrompt | None:
        stripped = message.strip()
        if not stripped:
            return None
        try:
            return ProtectionFeedbackPrompt.model_validate_json(stripped)
        except (ValidationError, json.JSONDecodeError):
            pass

        normalized = stripped.casefold()
        if normalized in {
            "yes",
            "y",
            "correct",
            "analysis correct",
            "analysis is correct",
        }:
            return ProtectionFeedbackPrompt(analysis_is_correct=True)
        if normalized in {
            "no",
            "n",
            "incorrect",
            "analysis incorrect",
            "analysis is incorrect",
        }:
            return None

        parsed_pairs: dict[str, str] = {}
        for line in stripped.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed_pairs[key.strip().casefold().replace(" ", "_")] = value.strip()
        if parsed_pairs:
            prompt_payload: dict[str, Any] = {}
            try:
                analysis_value = parsed_pairs.get(
                    "analysis_is_correct"
                ) or parsed_pairs.get("correct")
                if analysis_value is not None:
                    prompt_payload["analysis_is_correct"] = _parse_bool(analysis_value)
                expected_label = (
                    parsed_pairs.get("expected_sensitivity_label")
                    or parsed_pairs.get("sensitivity_label")
                    or parsed_pairs.get("label")
                )
                if expected_label is not None:
                    prompt_payload["expected_sensitivity_label"] = expected_label
                rationale = parsed_pairs.get("rationale")
                if rationale is not None:
                    prompt_payload["rationale"] = rationale
                return ProtectionFeedbackPrompt.model_validate(prompt_payload)
            except (ValidationError, ValueError):
                return None
        return None

    def record_feedback(
        self,
        *,
        pending_prompt: ProtectionPendingPrompt,
        feedback_prompt: ProtectionFeedbackPrompt,
    ) -> ProtectionFeedbackEntry:
        expected_label = feedback_prompt.expected_sensitivity_label
        if expected_label is None:
            raise ValueError(
                "expected_sensitivity_label is required for stored corrections."
            )
        entry = ProtectionFeedbackEntry(
            example_type="prompt",
            example_text=pending_prompt.original_user_message,
            expected_sensitivity_label=expected_label,
            rationale=feedback_prompt.rationale,
            session_id=pending_prompt.session_id,
            referenced_document_ids=list(pending_prompt.referenced_document_ids),
            metadata={
                "predicted_sensitivity_label": pending_prompt.predicted_sensitivity_label,
                "serialized_messages": pending_prompt.serialized_messages,
                **dict(pending_prompt.metadata),
            },
        )
        if self._feedback_store is not None:
            updated_entries = self._feedback_store.append_entry(entry)
            self._corpus = self._corpus.model_copy(
                update={"feedback_entries": updated_entries},
                deep=True,
            )
        else:
            self._corpus = self._corpus.model_copy(
                update={"feedback_entries": [*self._corpus.feedback_entries, entry]},
                deep=True,
            )
        return entry

    def feedback_required_message(self) -> str:
        return (
            "If the sensitivity analysis is incorrect, reply with a structured correction such as "
            '"analysis_is_correct: false\nexpected_sensitivity_label: <label>".'
        )

    def confirmation_block_message(self) -> str:
        return "The request remains blocked for this environment. Reformulate it or change the environment policy."

    def correction_recorded_message(self, *, expected_sensitivity_label: str) -> str:
        return (
            "Recorded your correction with expected sensitivity label "
            f"'{expected_sensitivity_label}'. Please resubmit the original request."
        )

    def _challenge_message(self, assessment: ProtectionAssessment) -> str:
        reasoning = (
            assessment.reasoning
            or "The response may exceed the environment sensitivity allowed here."
        )
        return (
            f"Potential sensitivity issue: {reasoning} "
            "Please confirm whether this analysis is correct. If it is incorrect, reply with "
            '"analysis_is_correct: false" and an explicit "expected_sensitivity_label".'
        )

    def _blocked_response_message(self, assessment: ProtectionAssessment) -> str:
        reasoning = (
            assessment.reasoning
            or "The candidate response was not appropriate for the current environment."
        )
        return f"The response was withheld because it may exceed the permitted sensitivity for this environment. {reasoning}"

    def _default_guard_text(self, assessment: ProtectionAssessment) -> str | None:
        if assessment.recommended_action is not ProtectionAction.CONSTRAIN:
            return None
        return (
            "Answer only within the lowest sensitivity that is acceptable for the current environment. "
            "Do not reveal or synthesize more sensitive information."
        )


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


def collect_provenance_from_tool_results(
    tool_results: list[ToolResult],
) -> ProtectionProvenanceSnapshot:
    """Collect unique source provenance refs from tool results."""
    deduped: dict[tuple[str, str, str], SourceProvenanceRef] = {}
    for tool_result in tool_results:
        for entry in tool_result.source_provenance:
            key = (entry.source_kind, entry.source_id, entry.content_hash)
            if key not in deduped:
                deduped[key] = entry.model_copy(deep=True)
    return ProtectionProvenanceSnapshot(sources=list(deduped.values()))


def _document_id_for_path(path: str) -> str:
    return Path(path).name or f"document-{uuid4().hex}"


def _parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"Unrecognized boolean value: {value}")


def _sanitize_payload(payload: Any, sanitized_text: str | None) -> Any:
    replacement = sanitized_text or "[REDACTED]"
    if isinstance(payload, str):
        return replacement
    if isinstance(payload, dict):
        sanitized = dict(payload)
        if "answer" in sanitized and isinstance(sanitized["answer"], str):
            sanitized["answer"] = replacement
            return sanitized
        sanitized["content"] = replacement
        return sanitized
    return replacement


__all__ = [
    "DefaultEnvironmentComparator",
    "EnvironmentComparator",
    "PromptProtectionDecision",
    "ProtectionAction",
    "ProtectionAssessment",
    "ProtectionConfig",
    "ProtectionController",
    "ProtectionCorpus",
    "ProtectionDocument",
    "ProtectionEnvironment",
    "ProtectionFeedbackEntry",
    "ProtectionFeedbackPrompt",
    "ProtectionFeedbackStore",
    "ProtectionPendingPrompt",
    "ResponseProtectionDecision",
    "SensitivityClassifier",
    "collect_provenance_from_tool_results",
    "load_protection_corpus",
]

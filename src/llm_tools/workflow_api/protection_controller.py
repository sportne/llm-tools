"""Workflow protection controller implementation."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from llm_tools.tool_api import ProtectionProvenanceSnapshot
from llm_tools.workflow_api.protection_models import (
    EnvironmentComparator,
    PromptProtectionDecision,
    ProtectionAction,
    ProtectionAssessment,
    ProtectionConfig,
    ProtectionCorpus,
    ProtectionEnvironment,
    ProtectionFeedbackEntry,
    ProtectionFeedbackPrompt,
    ProtectionPendingPrompt,
    ResponseProtectionDecision,
    SensitivityClassifier,
)
from llm_tools.workflow_api.protection_store import (
    ProtectionFeedbackStore,
    load_protection_corpus,
)


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
        source_policy_allow = self._single_allowed_source_extractively_used(assessment)
        action = (
            ProtectionAction.ALLOW
            if source_policy_allow
            else self._comparator.resolve_response_action(
                assessment=assessment,
                environment=dict(self._environment),
            )
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
            metadata={
                **dict(assessment.metadata),
                **(
                    {"source_policy": "single_allowed_source"}
                    if source_policy_allow
                    else {}
                ),
            },
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
            source_policy_allow = self._single_allowed_source_extractively_used(
                assessment
            )
            action = (
                ProtectionAction.ALLOW
                if source_policy_allow
                else self._comparator.resolve_response_action(
                    assessment=assessment,
                    environment=dict(self._environment),
                )
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
                metadata={
                    **dict(assessment.metadata),
                    **(
                        {"source_policy": "single_allowed_source"}
                        if source_policy_allow
                        else {}
                    ),
                },
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

    def _single_allowed_source_extractively_used(
        self,
        assessment: ProtectionAssessment,
    ) -> bool:
        source_ids = set(
            assessment.source_document_ids_used or assessment.referenced_document_ids
        )
        if len(source_ids) != 1:
            return False
        if assessment.requires_cross_source_synthesis is not False:
            return False
        if assessment.requires_inference_beyond_source is not False:
            return False
        allowed_labels = {
            str(label).strip().casefold()
            for label in self._environment.get("allowed_sensitivity_labels", [])
            if str(label).strip()
        }
        if not allowed_labels:
            return False
        source_id = next(iter(source_ids))
        document = next(
            (
                item
                for item in self._corpus.documents
                if source_id
                in {
                    item.document_id,
                    item.path,
                    item.display_name or "",
                }
            ),
            None,
        )
        if document is None or not document.sensitivity_label:
            return False
        return document.sensitivity_label.casefold() in allowed_labels


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


__all__ = ["DefaultEnvironmentComparator", "ProtectionController"]

"""App-layer helpers for building protection controllers and classifiers."""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel, Field

from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import ProtectionProvenanceSnapshot
from llm_tools.workflow_api import (
    DefaultEnvironmentComparator,
    ProtectionAction,
    ProtectionAssessment,
    ProtectionConfig,
    ProtectionController,
    ProtectionCorpus,
    SensitivityClassifier,
    load_protection_corpus,
)
from llm_tools.workflow_api.protection import ProtectionFeedbackStore


class _LLMProtectionAssessmentModel(BaseModel):
    sensitivity_label: str | None = None
    reasoning: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    referenced_document_ids: list[str] = Field(default_factory=list)
    recommended_action: ProtectionAction | None = None
    guard_text: str | None = None
    sanitized_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMProtectionClassifier(SensitivityClassifier):
    """First-party classifier backed by the OpenAI-compatible provider layer."""

    def __init__(self, *, provider: OpenAICompatibleProvider) -> None:
        self._provider = provider

    def assess_prompt(
        self,
        *,
        corpus: ProtectionCorpus,
        environment: dict[str, Any],
        messages: list[dict[str, Any]],
        provenance: ProtectionProvenanceSnapshot,
    ) -> ProtectionAssessment:
        payload = self._provider._run_with_fallback(
            messages=_build_classifier_messages(
                task="prompt",
                corpus=corpus,
                environment=environment,
                payload={
                    "messages": messages,
                    "provenance": provenance.model_dump(mode="json"),
                },
            ),
            response_model=_LLMProtectionAssessmentModel,
            request_params={"temperature": 0.0},
        )
        return ProtectionAssessment.model_validate(_model_payload(payload))

    async def assess_prompt_async(
        self,
        *,
        corpus: ProtectionCorpus,
        environment: dict[str, Any],
        messages: list[dict[str, Any]],
        provenance: ProtectionProvenanceSnapshot,
    ) -> ProtectionAssessment:
        payload = await self._provider._run_with_fallback_async(
            messages=_build_classifier_messages(
                task="prompt",
                corpus=corpus,
                environment=environment,
                payload={
                    "messages": messages,
                    "provenance": provenance.model_dump(mode="json"),
                },
            ),
            response_model=_LLMProtectionAssessmentModel,
            request_params={"temperature": 0.0},
        )
        return ProtectionAssessment.model_validate(_model_payload(payload))

    def assess_response(
        self,
        *,
        corpus: ProtectionCorpus,
        environment: dict[str, Any],
        response_payload: Any,
        provenance: ProtectionProvenanceSnapshot,
    ) -> ProtectionAssessment:
        payload = self._provider._run_with_fallback(
            messages=_build_classifier_messages(
                task="response",
                corpus=corpus,
                environment=environment,
                payload={
                    "response_payload": response_payload,
                    "provenance": provenance.model_dump(mode="json"),
                },
            ),
            response_model=_LLMProtectionAssessmentModel,
            request_params={"temperature": 0.0},
        )
        return ProtectionAssessment.model_validate(_model_payload(payload))

    async def assess_response_async(
        self,
        *,
        corpus: ProtectionCorpus,
        environment: dict[str, Any],
        response_payload: Any,
        provenance: ProtectionProvenanceSnapshot,
    ) -> ProtectionAssessment:
        payload = await self._provider._run_with_fallback_async(
            messages=_build_classifier_messages(
                task="response",
                corpus=corpus,
                environment=environment,
                payload={
                    "response_payload": response_payload,
                    "provenance": provenance.model_dump(mode="json"),
                },
            ),
            response_model=_LLMProtectionAssessmentModel,
            request_params={"temperature": 0.0},
        )
        return ProtectionAssessment.model_validate(_model_payload(payload))


def build_protection_environment(
    *,
    app_name: str,
    model_name: str,
    workspace: str | None,
    enabled_tools: list[str] | set[str],
    allow_network: bool,
    allow_filesystem: bool,
    allow_subprocess: bool,
) -> dict[str, Any]:
    """Return the default environment payload passed to protection checks."""
    return {
        "app_name": app_name,
        "model_name": model_name,
        "workspace": workspace,
        "enabled_tools": sorted(enabled_tools),
        "allow_network": allow_network,
        "allow_filesystem": allow_filesystem,
        "allow_subprocess": allow_subprocess,
    }


def build_protection_controller(
    *,
    config: ProtectionConfig,
    provider: object,
    environment: dict[str, Any],
) -> ProtectionController | None:
    """Build the first-party protection controller when the feature is enabled."""
    if not config.enabled:
        return None
    if not isinstance(provider, OpenAICompatibleProvider):
        return None
    feedback_store = (
        ProtectionFeedbackStore(config.corrections_path)
        if config.corrections_path is not None
        else None
    )
    corpus = load_protection_corpus(config, feedback_store=feedback_store)
    return ProtectionController(
        config=config,
        classifier=LLMProtectionClassifier(provider=provider),
        environment=environment,
        comparator=DefaultEnvironmentComparator(),
        corpus=corpus,
        feedback_store=feedback_store,
    )


def _build_classifier_messages(
    *,
    task: str,
    corpus: ProtectionCorpus,
    environment: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    documents = [
        {
            "document_id": document.document_id,
            "path": document.path,
            "content": document.content[:12000],
        }
        for document in corpus.documents
    ]
    feedback_entries = [
        entry.model_dump(mode="json") for entry in corpus.feedback_entries[-50:]
    ]
    return [
        {
            "role": "system",
            "content": (
                "You classify sensitivity for either a model-bound prompt or a candidate response. "
                "Use the provided proprietary sensitivity corpus and confirmed feedback entries. "
                "Return a structured assessment with reasoning, a sensitivity label when you can infer one, "
                "and a recommended action. Use CHALLENGE for prompts that should be reviewed with the user, "
                "CONSTRAIN when a safer lower-sensitivity answer appears possible, SANITIZE when the response "
                "can be safely rewritten, and BLOCK when it should be withheld."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": task,
                    "environment": environment,
                    "documents": documents,
                    "feedback_entries": feedback_entries,
                    "payload": payload,
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
        },
    ]


def _model_payload(payload: object) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return cast(dict[str, Any], payload.model_dump(mode="json"))
    if isinstance(payload, dict):
        return payload
    raise TypeError(
        f"Unsupported protection assessment payload: {type(payload).__name__}"
    )


__all__ = [
    "LLMProtectionClassifier",
    "build_protection_controller",
    "build_protection_environment",
]

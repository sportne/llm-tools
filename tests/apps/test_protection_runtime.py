"""Focused tests for app-layer protection runtime helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llm_tools.apps import protection_runtime
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import ProtectionProvenanceSnapshot
from llm_tools.workflow_api import (
    ProtectionAction,
    ProtectionConfig,
    ProtectionCorpus,
    ProtectionDocument,
    ProtectionFeedbackEntry,
)


class _Provider(OpenAICompatibleProvider):
    pass


def test_llm_protection_classifier_sync_and_async_paths() -> None:
    provider = _Provider(model="demo")
    sync_calls: list[tuple[list[dict[str, str]], type[object], dict[str, float]]] = []
    async_calls: list[tuple[list[dict[str, str]], type[object], dict[str, float]]] = []

    def fake_run_with_fallback(*, messages, response_model, request_params):
        sync_calls.append((messages, response_model, request_params))
        return response_model(
            reasoning="prompt ok",
            recommended_action=ProtectionAction.CONSTRAIN,
        )

    async def fake_run_with_fallback_async(*, messages, response_model, request_params):
        async_calls.append((messages, response_model, request_params))
        return {
            "reasoning": "response unsafe",
            "recommended_action": ProtectionAction.BLOCK.value,
        }

    provider._run_with_fallback = fake_run_with_fallback  # type: ignore[attr-defined]
    provider._run_with_fallback_async = fake_run_with_fallback_async  # type: ignore[attr-defined]
    classifier = protection_runtime.LLMProtectionClassifier(provider=provider)
    corpus = ProtectionCorpus(
        documents=[
            ProtectionDocument(
                document_id="doc-1",
                path="/policy.txt",
                content="keep this protected",
            )
        ],
        feedback_entries=[
            ProtectionFeedbackEntry(
                example_text="What is the sensitivity?",
                expected_sensitivity_label="internal",
            )
        ],
    )
    environment = {"app_name": "assistant", "allowed_sensitivity_labels": ["public"]}

    prompt_assessment = classifier.assess_prompt(
        corpus=corpus,
        environment=environment,
        messages=[{"role": "user", "content": "hello"}],
        provenance=ProtectionProvenanceSnapshot(),
    )
    response_assessment = asyncio.run(
        classifier.assess_response_async(
            corpus=corpus,
            environment=environment,
            response_payload={"answer": "secret"},
            provenance=ProtectionProvenanceSnapshot(),
        )
    )

    assert prompt_assessment.recommended_action is ProtectionAction.CONSTRAIN
    assert response_assessment.recommended_action is ProtectionAction.BLOCK
    assert sync_calls[0][2] == {"temperature": 0.0}
    assert async_calls[0][2] == {"temperature": 0.0}
    sync_payload = json.loads(sync_calls[0][0][1]["content"])
    async_payload = json.loads(async_calls[0][0][1]["content"])
    assert sync_payload["task"] == "prompt"
    assert async_payload["task"] == "response"
    assert sync_payload["documents"][0]["document_id"] == "doc-1"
    assert (
        async_payload["feedback_entries"][0]["expected_sensitivity_label"] == "internal"
    )


def test_build_protection_environment_and_controller(tmp_path: Path) -> None:
    document = tmp_path / "policy.txt"
    document.write_text("never expose this", encoding="utf-8")
    provider = _Provider(model="demo")

    environment = protection_runtime.build_protection_environment(
        app_name="assistant",
        model_name="demo",
        workspace=str(tmp_path),
        enabled_tools={"search_text", "read_file"},
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
    )
    controller = protection_runtime.build_protection_controller(
        config=ProtectionConfig(
            enabled=True,
            document_paths=[str(document)],
            corrections_path=str(tmp_path / "corrections.json"),
        ),
        provider=provider,
        environment=environment,
    )

    assert environment["enabled_tools"] == ["read_file", "search_text"]
    assert controller is not None
    assert controller.enabled is True
    assert controller.corpus.documents[0].path == str(document)
    assert (
        protection_runtime.build_protection_controller(
            config=ProtectionConfig(enabled=False),
            provider=provider,
            environment=environment,
        )
        is None
    )
    assert (
        protection_runtime.build_protection_controller(
            config=ProtectionConfig(enabled=True),
            provider=object(),
            environment=environment,
        )
        is None
    )


def test_model_payload_accepts_dicts_and_rejects_unknown_objects() -> None:
    assert protection_runtime._model_payload({"reasoning": "ok"}) == {"reasoning": "ok"}

    with pytest.raises(TypeError, match="Unsupported protection assessment payload"):
        protection_runtime._model_payload(object())

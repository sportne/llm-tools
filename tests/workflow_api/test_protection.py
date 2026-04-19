"""Direct tests for workflow protection contracts and helpers."""

from __future__ import annotations

import asyncio
import json

import pytest

import llm_tools.workflow_api.protection as protection_module
from llm_tools.tool_api import SourceProvenanceRef, ToolResult
from llm_tools.workflow_api import (
    DefaultEnvironmentComparator,
    ProtectionAction,
    ProtectionAssessment,
    ProtectionConfig,
    ProtectionController,
    ProtectionFeedbackEntry,
    ProtectionFeedbackPrompt,
    ProtectionFeedbackStore,
    ProtectionPendingPrompt,
    collect_provenance_from_tool_results,
    load_protection_corpus,
)


class _Classifier:
    def __init__(self, *, prompt: ProtectionAssessment, response: ProtectionAssessment):
        self._prompt = prompt
        self._response = response

    def assess_prompt(self, **kwargs) -> ProtectionAssessment:
        del kwargs
        return self._prompt

    def assess_response(self, **kwargs) -> ProtectionAssessment:
        del kwargs
        return self._response


def test_feedback_store_appends_json_entries(tmp_path) -> None:
    store = ProtectionFeedbackStore(tmp_path / "corrections.json")
    entries = store.append_entry(
        ProtectionController(
            config=ProtectionConfig(enabled=True),
            classifier=_Classifier(
                prompt=ProtectionAssessment(reasoning="ok"),
                response=ProtectionAssessment(reasoning="ok"),
            ),
        ).record_feedback(
            pending_prompt=ProtectionPendingPrompt(
                original_user_message="What is the sensitivity?",
                serialized_messages=[
                    {"role": "user", "content": "What is the sensitivity?"}
                ],
                reasoning="Potential issue.",
            ),
            feedback_prompt=ProtectionFeedbackPrompt(
                analysis_is_correct=False,
                expected_sensitivity_label="public",
                rationale="This is already approved.",
            ),
        )
    )

    payload = json.loads((tmp_path / "corrections.json").read_text(encoding="utf-8"))
    assert payload["entries"][0]["expected_sensitivity_label"] == "public"
    assert payload["entries"][0]["entry_id"] == entries[0].entry_id


def test_collect_provenance_deduplicates_sources() -> None:
    source = SourceProvenanceRef(
        source_kind="local_file",
        source_id="/workspace/example.txt",
        content_hash="abc",
        whole_source_reproduction_allowed=True,
    )
    snapshot = collect_provenance_from_tool_results(
        [
            ToolResult(
                ok=True,
                tool_name="read_file",
                tool_version="0.1.0",
                source_provenance=[source],
            ),
            ToolResult(
                ok=True,
                tool_name="read_file",
                tool_version="0.1.0",
                source_provenance=[source],
            ),
        ]
    )

    assert len(snapshot.sources) == 1
    assert snapshot.sources[0].source_id == "/workspace/example.txt"


def test_controller_sanitizes_dict_answer_payload() -> None:
    controller = ProtectionController(
        config=ProtectionConfig(enabled=True),
        classifier=_Classifier(
            prompt=ProtectionAssessment(reasoning="ok"),
            response=ProtectionAssessment(
                reasoning="Needs sanitization.",
                recommended_action=ProtectionAction.SANITIZE,
                sanitized_text="Safe answer",
            ),
        ),
    )

    decision = controller.review_response(
        response_payload={"answer": "Sensitive answer"}
    )

    assert decision.action is ProtectionAction.SANITIZE
    assert decision.sanitized_payload == {"answer": "Safe answer"}


class _AsyncClassifier(_Classifier):
    async def assess_prompt_async(self, **kwargs) -> ProtectionAssessment:
        del kwargs
        return self._prompt

    async def assess_response_async(self, **kwargs) -> ProtectionAssessment:
        del kwargs
        return self._response


def test_default_environment_comparator_uses_environment_label_sets() -> None:
    comparator = DefaultEnvironmentComparator()
    assessment = ProtectionAssessment(sensitivity_label="secret")

    assert (
        comparator.resolve_prompt_action(
            assessment=assessment,
            environment={"blocked_sensitivity_labels": ["secret"]},
        )
        is ProtectionAction.CHALLENGE
    )
    assert (
        comparator.resolve_response_action(
            assessment=assessment,
            environment={"blocked_sensitivity_labels": ["secret"]},
        )
        is ProtectionAction.BLOCK
    )
    assert (
        comparator.resolve_prompt_action(
            assessment=assessment,
            environment={"constrained_sensitivity_labels": ["secret"]},
        )
        is ProtectionAction.CONSTRAIN
    )
    assert (
        comparator.resolve_response_action(
            assessment=assessment,
            environment={"sanitized_sensitivity_labels": ["secret"]},
        )
        is ProtectionAction.SANITIZE
    )


def test_feedback_store_yaml_round_trip_and_unsupported_extension(tmp_path) -> None:
    yaml_store = ProtectionFeedbackStore(tmp_path / "corrections.yaml")
    entry = ProtectionFeedbackEntry(
        example_text="What is the sensitivity?",
        expected_sensitivity_label="internal",
    )

    assert ProtectionFeedbackStore(tmp_path / "missing.json").load_entries() == []
    yaml_store.save_entries([entry])
    loaded = yaml_store.load_entries()

    assert loaded[0].expected_sensitivity_label == "internal"
    with pytest.raises(ValueError, match="Unsupported protection feedback file type"):
        ProtectionFeedbackStore(tmp_path / "corrections.txt").save_entries([])


def test_controller_short_circuits_when_disabled_or_review_is_off() -> None:
    disabled = ProtectionController(
        config=ProtectionConfig(enabled=False),
        classifier=_Classifier(
            prompt=ProtectionAssessment(reasoning="prompt"),
            response=ProtectionAssessment(reasoning="response"),
        ),
    )
    review_off = ProtectionController(
        config=ProtectionConfig(enabled=True, review_final_answers=False),
        classifier=_Classifier(
            prompt=ProtectionAssessment(reasoning="prompt"),
            response=ProtectionAssessment(reasoning="response"),
        ),
    )

    assert (
        disabled.assess_prompt(messages=[{"role": "user", "content": "hi"}]).action
        is ProtectionAction.ALLOW
    )
    assert (
        review_off.review_response(response_payload="secret").action
        is ProtectionAction.ALLOW
    )


def test_controller_async_paths_cover_default_messages() -> None:
    controller = ProtectionController(
        config=ProtectionConfig(enabled=True),
        classifier=_AsyncClassifier(
            prompt=ProtectionAssessment(
                reasoning="Need a constrained answer.",
                recommended_action=ProtectionAction.CONSTRAIN,
            ),
            response=ProtectionAssessment(
                reasoning="Too sensitive.",
                recommended_action=ProtectionAction.BLOCK,
            ),
        ),
    )

    prompt_decision = asyncio.run(
        controller.assess_prompt_async(messages=[{"role": "user", "content": "hi"}])
    )
    response_decision = asyncio.run(
        controller.review_response_async(response_payload="secret")
    )

    assert prompt_decision.action is ProtectionAction.CONSTRAIN
    assert prompt_decision.guard_text is not None
    assert response_decision.action is ProtectionAction.BLOCK
    assert "withheld" in (response_decision.safe_message or "")


def test_controller_parses_feedback_and_updates_in_memory_corpus() -> None:
    controller = ProtectionController(
        config=ProtectionConfig(enabled=True),
        classifier=_Classifier(
            prompt=ProtectionAssessment(reasoning="prompt"),
            response=ProtectionAssessment(reasoning="response"),
        ),
    )
    pending_prompt = ProtectionPendingPrompt(
        original_user_message="How sensitive is this?",
        serialized_messages=[{"role": "user", "content": "How sensitive is this?"}],
        reasoning="Potential issue.",
        predicted_sensitivity_label="secret",
        metadata={"source": "test"},
    )

    assert controller.parse_feedback_message("yes") == ProtectionFeedbackPrompt(
        analysis_is_correct=True
    )
    assert controller.parse_feedback_message("no") is None
    assert controller.parse_feedback_message("analysis_is_correct: maybe") is None
    parsed = controller.parse_feedback_message(
        "analysis_is_correct: false\nlabel: public\nrationale: approved"
    )

    assert parsed is not None
    assert parsed.expected_sensitivity_label == "public"
    entry = controller.record_feedback(
        pending_prompt=pending_prompt,
        feedback_prompt=parsed,
    )
    assert controller.corpus.feedback_entries[-1].entry_id == entry.entry_id
    with pytest.raises(ValueError, match="expected_sensitivity_label is required"):
        controller.record_feedback(
            pending_prompt=pending_prompt,
            feedback_prompt=ProtectionFeedbackPrompt(analysis_is_correct=True),
        )


def test_controller_messages_and_helper_functions_cover_private_paths(tmp_path) -> None:
    document = tmp_path / "policy.txt"
    document.write_text("Never combine these facts.", encoding="utf-8")
    store = ProtectionFeedbackStore(tmp_path / "corrections.json")
    store.save_entries(
        [
            ProtectionFeedbackEntry(
                example_text="prompt",
                expected_sensitivity_label="restricted",
            )
        ]
    )
    corpus = load_protection_corpus(
        ProtectionConfig(enabled=True, document_paths=[str(document)]),
        feedback_store=store,
    )

    assert corpus.documents[0].document_id == "policy.txt"
    assert corpus.feedback_entries[0].expected_sensitivity_label == "restricted"
    assert protection_module._parse_bool("yes") is True
    assert protection_module._parse_bool("0") is False
    with pytest.raises(ValueError, match="Unrecognized boolean value"):
        protection_module._parse_bool("maybe")
    assert protection_module._sanitize_payload("secret", "clean") == "clean"
    assert protection_module._sanitize_payload({"answer": "secret"}, "clean") == {
        "answer": "clean"
    }
    assert protection_module._sanitize_payload({"body": "secret"}, "clean") == {
        "body": "secret",
        "content": "clean",
    }
    assert protection_module._sanitize_payload(42, None) == "[REDACTED]"


def test_feedback_prompt_validators_trim_and_require_label() -> None:
    prompt = ProtectionFeedbackPrompt(
        analysis_is_correct=False,
        expected_sensitivity_label=" public ",
        rationale=" approved ",
    )

    assert prompt.expected_sensitivity_label == "public"
    assert prompt.rationale == "approved"
    with pytest.raises(ValueError, match="expected_sensitivity_label is required"):
        ProtectionFeedbackPrompt(analysis_is_correct=False)

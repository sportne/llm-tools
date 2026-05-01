"""Direct tests for workflow protection contracts and helpers."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import llm_tools.workflow_api.protection as protection_module
import llm_tools.workflow_api.protection_controller as protection_controller_module
import llm_tools.workflow_api.protection_store as protection_store_module
from llm_tools.tool_api import SourceProvenanceRef, ToolResult
from llm_tools.workflow_api import (
    DefaultEnvironmentComparator,
    ProtectionAction,
    ProtectionAssessment,
    ProtectionCategory,
    ProtectionConfig,
    ProtectionController,
    ProtectionFeedbackEntry,
    ProtectionFeedbackPrompt,
    ProtectionFeedbackStore,
    ProtectionPendingPrompt,
    collect_provenance_from_tool_results,
    inspect_protection_corpus,
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


def test_controller_allows_single_allowed_extractively_used_source() -> None:
    controller = ProtectionController(
        config=ProtectionConfig(enabled=True),
        environment={
            "allowed_sensitivity_labels": ["MINOR"],
            "sensitivity_categories": [
                ProtectionCategory(label="MINOR", aliases=["low"]).model_dump(
                    mode="json"
                )
            ],
        },
        corpus=protection_module.ProtectionCorpus(
            documents=[
                protection_module.ProtectionDocument(
                    document_id="policy.pdf",
                    path="/corpus/policy.pdf",
                    content="allowed source",
                    display_name="policy.pdf",
                    sensitivity_label="low",
                )
            ]
        ),
        classifier=_Classifier(
            prompt=ProtectionAssessment(reasoning="ok"),
            response=ProtectionAssessment(
                reasoning="Classifier would otherwise block.",
                recommended_action=ProtectionAction.BLOCK,
                source_document_ids_used=["policy.pdf"],
                requires_cross_source_synthesis=False,
                requires_inference_beyond_source=False,
            ),
        ),
    )

    decision = controller.review_response(response_payload={"answer": "summary"})

    assert decision.action is ProtectionAction.ALLOW
    assert decision.metadata["source_policy"] == "single_allowed_source"


def test_controller_does_not_single_source_allow_inferential_answers() -> None:
    controller = ProtectionController(
        config=ProtectionConfig(enabled=True),
        environment={"allowed_sensitivity_labels": ["MINOR"]},
        corpus=protection_module.ProtectionCorpus(
            documents=[
                protection_module.ProtectionDocument(
                    document_id="policy.pdf",
                    path="/corpus/policy.pdf",
                    content="allowed source",
                    sensitivity_label="MINOR",
                )
            ]
        ),
        classifier=_Classifier(
            prompt=ProtectionAssessment(reasoning="ok"),
            response=ProtectionAssessment(
                reasoning="Needs inference.",
                recommended_action=ProtectionAction.BLOCK,
                source_document_ids_used=["policy.pdf"],
                requires_cross_source_synthesis=False,
                requires_inference_beyond_source=True,
            ),
        ),
    )

    decision = controller.review_response(response_payload={"answer": "analysis"})

    assert decision.action is ProtectionAction.BLOCK


def test_controller_does_not_single_source_allow_uncategorized_sources() -> None:
    controller = ProtectionController(
        config=ProtectionConfig(enabled=True),
        environment={"allowed_sensitivity_labels": ["MINOR"]},
        corpus=protection_module.ProtectionCorpus(
            documents=[
                protection_module.ProtectionDocument(
                    document_id="policy.pdf",
                    path="/corpus/policy.pdf",
                    content="uncategorized source",
                )
            ]
        ),
        classifier=_Classifier(
            prompt=ProtectionAssessment(reasoning="ok"),
            response=ProtectionAssessment(
                reasoning="Source lacks category.",
                recommended_action=ProtectionAction.BLOCK,
                source_document_ids_used=["policy.pdf"],
                requires_cross_source_synthesis=False,
                requires_inference_beyond_source=False,
            ),
        ),
    )

    decision = controller.review_response(response_payload={"answer": "summary"})

    assert decision.action is ProtectionAction.BLOCK


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


def test_default_environment_comparator_canonicalizes_category_aliases() -> None:
    comparator = DefaultEnvironmentComparator()
    category_catalog = [
        ProtectionCategory(label="MINOR", aliases=["low", "limited"]).model_dump(
            mode="json"
        )
    ]

    assert (
        comparator.resolve_response_action(
            assessment=ProtectionAssessment(sensitivity_label="low"),
            environment={
                "allowed_sensitivity_labels": ["MINOR"],
                "sensitivity_categories": category_catalog,
            },
        )
        is ProtectionAction.ALLOW
    )
    assert (
        comparator.resolve_prompt_action(
            assessment=ProtectionAssessment(sensitivity_label="MINOR"),
            environment={
                "blocked_sensitivity_labels": ["limited"],
                "sensitivity_categories": category_catalog,
            },
        )
        is ProtectionAction.CHALLENGE
    )


def test_protection_category_canonicalization_helper_edges() -> None:
    environment = {
        "allowed_sensitivity_labels": ["low", "", None],
        "sensitivity_categories": [
            ProtectionCategory(label="MINOR", aliases=["low"]),
            {"label": "MAJOR", "aliases": ["high"]},
            {"aliases": ["invalid"]},
            "not-a-category",
        ],
    }

    assert (
        protection_controller_module._canonical_sensitivity_label(
            " low ", environment=environment
        )
        == "MINOR"
    )
    assert (
        protection_controller_module._canonical_sensitivity_label(
            "high", environment=environment
        )
        == "MAJOR"
    )
    assert (
        protection_controller_module._canonical_sensitivity_label(
            "unknown", environment=environment
        )
        == "unknown"
    )
    assert (
        protection_controller_module._canonical_sensitivity_label(
            "", environment=environment
        )
        is None
    )
    assert (
        protection_controller_module._canonical_sensitivity_label(
            None, environment=environment
        )
        is None
    )
    assert protection_controller_module._canonical_label_set(
        "allowed_sensitivity_labels", environment
    ) == {"minor"}


def test_feedback_store_path_and_load_rejects_unsupported_existing_extension(
    tmp_path: Path,
) -> None:
    yaml_store = ProtectionFeedbackStore(tmp_path / "empty.yaml")
    yaml_store.path.write_text("", encoding="utf-8")

    assert yaml_store.path == tmp_path / "empty.yaml"
    assert yaml_store.load_entries() == []

    unsupported = ProtectionFeedbackStore(tmp_path / "corrections.txt")
    unsupported.path.write_text("entries: []", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported protection feedback file type"):
        unsupported.load_entries()


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


def test_inspect_protection_corpus_reports_read_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "policies"
    corpus_dir.mkdir()
    good_file = corpus_dir / "policy.txt"
    bad_dir_file = corpus_dir / "broken.txt"
    standalone_file = tmp_path / "standalone.txt"
    unsupported_file = tmp_path / "binary.bin"
    good_file.write_text("internal guidance", encoding="utf-8")
    bad_dir_file.write_text("broken", encoding="utf-8")
    standalone_file.write_text("also broken", encoding="utf-8")
    unsupported_file.write_bytes(b"bin")

    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self in {bad_dir_file, standalone_file}:
            raise OSError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    report = inspect_protection_corpus(
        ProtectionConfig(
            enabled=True,
            document_paths=[
                str(corpus_dir),
                str(standalone_file),
                str(unsupported_file),
            ],
        )
    )

    assert [document.path for document in report.corpus.documents] == [str(good_file)]
    assert any(
        issue.path == str(bad_dir_file) and "Unable to read" in issue.message
        for issue in report.issues
    )
    assert any(
        issue.path == str(standalone_file) and "Unable to read" in issue.message
        for issue in report.issues
    )
    assert any(
        issue.path == str(unsupported_file) and "Unsupported file type" in issue.message
        for issue in report.issues
    )


def test_inspect_protection_corpus_accepts_directory_entries(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "policies"
    corpus_dir.mkdir()
    (corpus_dir / "policy.txt").write_text("internal guidance", encoding="utf-8")
    (corpus_dir / "notes.md").write_text("more guidance", encoding="utf-8")
    (corpus_dir / "skip.png").write_bytes(b"png")

    report = inspect_protection_corpus(
        ProtectionConfig(
            enabled=True,
            document_paths=[str(corpus_dir), str(tmp_path / "missing.txt")],
        )
    )

    assert sorted(document.path for document in report.corpus.documents) == [
        str(corpus_dir / "notes.md"),
        str(corpus_dir / "policy.txt"),
    ]
    assert any(issue.path == str(tmp_path / "missing.txt") for issue in report.issues)
    assert any(issue.path == str(corpus_dir / "skip.png") for issue in report.issues)


def test_inspect_protection_corpus_converts_markitdown_documents_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "policies"
    corpus_dir.mkdir()
    source = corpus_dir / "report.pdf"
    source.write_bytes(b"%PDF")
    fake_module = ModuleType("markitdown")
    convert_calls: list[str] = []

    class FakeMarkItDown:
        def convert(self, path: str) -> SimpleNamespace:
            convert_calls.append(path)
            return SimpleNamespace(text_content=f"converted:{path}")

    fake_module.MarkItDown = FakeMarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)

    config = ProtectionConfig(enabled=True, document_paths=[str(corpus_dir)])
    first = inspect_protection_corpus(config)
    second = inspect_protection_corpus(config)

    assert first.converted_document_count == 1
    assert first.corpus.documents[0].read_kind == "markitdown"
    assert first.corpus.documents[0].display_name == "report.pdf"
    assert first.corpus.documents[0].content == f"converted:{source}"
    assert first.corpus.documents[0].content_hash
    assert second.corpus.documents[0].content == f"converted:{source}"
    assert convert_calls == [str(source)]


def test_inspect_protection_corpus_reports_conversion_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "policies"
    corpus_dir.mkdir()
    source = corpus_dir / "report.docx"
    source.write_bytes(b"docx")
    fake_module = ModuleType("markitdown")

    class FakeMarkItDown:
        def convert(self, path: str) -> SimpleNamespace:
            del path
            raise RuntimeError("conversion failed")

    fake_module.MarkItDown = FakeMarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)

    report = inspect_protection_corpus(
        ProtectionConfig(enabled=True, document_paths=[str(corpus_dir)])
    )

    assert report.usable_document_count == 0
    assert any(
        issue.path == str(source) and "Unable to convert" in issue.message
        for issue in report.issues
    )


def test_inspect_protection_corpus_reads_category_metadata_conventions(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "policies"
    minor_dir = corpus_dir / "minor"
    minor_dir.mkdir(parents=True)
    front_matter_file = corpus_dir / "front.md"
    manifest_file = corpus_dir / ".llm-tools-protection-sources.yaml"
    folder_file = minor_dir / "folder.txt"
    manifest_target = corpus_dir / "manifest.txt"
    front_matter_file.write_text(
        "---\nsensitivity: low\n---\nGuidance text.",
        encoding="utf-8",
    )
    manifest_file.write_text(
        "sources:\n  - path: manifest.txt\n    category: MAJOR\n",
        encoding="utf-8",
    )
    manifest_target.write_text("Manifest categorized.", encoding="utf-8")
    folder_file.write_text("Folder categorized.", encoding="utf-8")

    report = inspect_protection_corpus(
        ProtectionConfig(
            enabled=True,
            document_paths=[str(corpus_dir)],
            allowed_sensitivity_labels=["MINOR", "MAJOR"],
            sensitivity_categories=[
                ProtectionCategory(label="MINOR", aliases=["low"]),
                ProtectionCategory(label="MAJOR"),
            ],
        )
    )
    by_id = {document.document_id: document for document in report.corpus.documents}

    assert by_id["front.md"].sensitivity_label == "MINOR"
    assert by_id["front.md"].sensitivity_label_source == "front_matter"
    assert by_id["manifest.txt"].sensitivity_label == "MAJOR"
    assert by_id["manifest.txt"].sensitivity_label_source == (
        "manifest:.llm-tools-protection-sources.yaml"
    )
    assert by_id["minor/folder.txt"].sensitivity_label == "MINOR"
    assert by_id["minor/folder.txt"].sensitivity_label_source == "folder"
    assert ".llm-tools-protection-sources.yaml" not in by_id


def test_inspect_protection_corpus_uses_unique_relative_document_ids(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "policies"
    left = corpus_dir / "left"
    right = corpus_dir / "right"
    left.mkdir(parents=True)
    right.mkdir()
    (left / "policy.txt").write_text("left", encoding="utf-8")
    (right / "policy.txt").write_text("right", encoding="utf-8")

    report = inspect_protection_corpus(
        ProtectionConfig(enabled=True, document_paths=[str(corpus_dir)])
    )

    assert sorted(document.document_id for document in report.corpus.documents) == [
        "left/policy.txt",
        "right/policy.txt",
    ]


def test_inspect_protection_corpus_handles_source_manifest_variants(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "policies"
    corpus_dir.mkdir()
    manifest = corpus_dir / ".llm-tools-protection-sources.json"
    alpha = corpus_dir / "alpha.txt"
    beta = corpus_dir / "beta.md"
    alpha.write_text("alpha", encoding="utf-8")
    beta.write_text("beta", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "sources": {
                    "alpha.txt": "MINOR",
                    "*.md": {"sensitivity_label": "MAJOR"},
                    "missing-category.txt": {},
                    "": "MINOR",
                }
            }
        ),
        encoding="utf-8",
    )

    report = inspect_protection_corpus(
        ProtectionConfig(
            enabled=True,
            document_paths=[str(corpus_dir)],
            allowed_sensitivity_labels=["MINOR", "MAJOR"],
        )
    )
    by_id = {document.document_id: document for document in report.corpus.documents}

    assert by_id["alpha.txt"].sensitivity_label == "MINOR"
    assert by_id["beta.md"].sensitivity_label == "MAJOR"
    assert any(
        "invalid protection source metadata" in issue.message for issue in report.issues
    )


def test_inspect_protection_corpus_reports_bad_source_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "policies"
    corpus_dir.mkdir()
    manifest = corpus_dir / ".llm-tools-protection-sources.yaml"
    manifest.write_text("sources: []", encoding="utf-8")
    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == manifest:
            raise OSError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    report = inspect_protection_corpus(
        ProtectionConfig(enabled=True, document_paths=[str(corpus_dir)])
    )

    assert any(
        issue.path == str(manifest)
        and "Unable to read protection source metadata" in issue.message
        for issue in report.issues
    )


def test_inspect_protection_corpus_skips_internal_metadata_files(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "policies"
    cache_dir = corpus_dir / ".llm_tools" / "cache"
    cache_dir.mkdir(parents=True)
    corrections = corpus_dir / ".llm-tools-protection-corrections.json"
    source_metadata = corpus_dir / ".llm-tools-protection-sources.yaml"
    cache_file = cache_dir / "content.md"
    normal = corpus_dir / "policy.txt"
    corrections.write_text('{"entries": []}', encoding="utf-8")
    source_metadata.write_text("sources: []", encoding="utf-8")
    cache_file.write_text("cached conversion", encoding="utf-8")
    normal.write_text("real policy", encoding="utf-8")

    report = inspect_protection_corpus(
        ProtectionConfig(
            enabled=True,
            document_paths=[str(corpus_dir)],
            corrections_path=str(corrections),
        )
    )

    assert [document.document_id for document in report.corpus.documents] == [
        "policy.txt"
    ]


def test_protection_store_helper_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outside_rule = protection_store_module._SourceCategoryRule(
        base_dir=tmp_path / "root",
        pattern="*.txt",
        category="MINOR",
        source="test",
    )
    assert outside_rule.matches(tmp_path / "other" / "file.txt") is False
    assert protection_store_module._document_id_for_path("").startswith("document-")
    assert (
        protection_store_module._document_id_for_path_with_root(
            tmp_path / "file.txt", corpus_root=tmp_path / "root"
        )
        == "file.txt"
    )
    assert protection_store_module._front_matter_category("---\n-\n---\nbody") is None
    assert (
        protection_store_module._front_matter_category(
            "---\ncategory: [not-a-string]\n---\nbody"
        )
        is None
    )
    assert (
        protection_store_module._front_matter_category("---\ncategory: [\n---\nbody")
        is None
    )

    source = tmp_path / "plain.pdf"
    source.write_text("plain utf8", encoding="utf-8")
    monkeypatch.setattr(
        protection_store_module, "get_conversion_backend", lambda path: None
    )
    loaded = protection_store_module._load_protection_readable_content(
        source,
        suffix=".pdf",
        cache_root=None,
    )

    assert loaded.read_kind == "text"
    assert loaded.content == "plain utf8"


def test_inspect_protection_corpus_reports_too_large_converted_documents(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "policies"
    corpus_dir.mkdir()
    source = corpus_dir / "large.pdf"
    source.write_bytes(b"x" * (1024 * 1024 + 1))

    report = inspect_protection_corpus(
        ProtectionConfig(enabled=True, document_paths=[str(corpus_dir)])
    )

    assert report.usable_document_count == 0
    assert any("readable byte limit" in issue.message for issue in report.issues)


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

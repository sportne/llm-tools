"""Direct tests for harness verification contracts."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from llm_tools.harness_api import (
    BudgetPolicy,
    HarnessSession,
    HarnessState,
    TaskRecord,
    VerificationEvidenceRecord,
    VerificationExpectation,
    VerificationFailureMode,
    VerificationResult,
    VerificationStatus,
    VerificationTiming,
    VerificationTrigger,
    Verifier,
)


def test_verification_expectation_defaults_and_validation() -> None:
    expectation = VerificationExpectation(
        expectation_id="expectation-1",
        description="Confirm tests passed.",
    )

    assert expectation.required_for_completion is False
    assert expectation.trigger is VerificationTrigger.AUTOMATIC
    assert expectation.timing is VerificationTiming.AFTER_COMPLETION

    with pytest.raises(ValidationError, match="expectation_id must not be empty"):
        VerificationExpectation(expectation_id=" ", description="Check output")

    with pytest.raises(ValidationError, match="description must not be empty"):
        VerificationExpectation(expectation_id="expectation-1", description=" ")


def test_verification_result_enforces_checked_at_and_unique_refs() -> None:
    with pytest.raises(ValidationError, match="checked_at is required"):
        VerificationResult(task_id="task-1", status=VerificationStatus.PASSED)

    with pytest.raises(ValidationError, match="expectation_ids must be unique"):
        VerificationResult(
            task_id="task-1",
            status=VerificationStatus.FAILED,
            checked_at="2026-01-01T00:00:00Z",
            expectation_ids=["expectation-1", "expectation-1"],
        )

    with pytest.raises(ValidationError, match="evidence ids must be unique"):
        VerificationResult(
            task_id="task-1",
            status=VerificationStatus.FAILED,
            checked_at="2026-01-01T00:00:00Z",
            evidence=[
                VerificationEvidenceRecord(evidence_id="evidence-1"),
                VerificationEvidenceRecord(evidence_id="evidence-1"),
            ],
        )

    with pytest.raises(ValidationError, match="not allowed for passed"):
        VerificationResult(
            task_id="task-1",
            status=VerificationStatus.PASSED,
            checked_at="2026-01-01T00:00:00Z",
            failure_mode=VerificationFailureMode.EXPECTATION_UNMET,
        )


def test_stub_verifier_smoke_test() -> None:
    task = TaskRecord(
        task_id="task-1",
        title="Implement verification",
        verification_expectations=[
            VerificationExpectation(
                expectation_id="expectation-1",
                description="Confirm verification evidence exists.",
                required_for_completion=True,
            )
        ],
    )
    state = HarnessState(
        schema_version="1",
        session=HarnessSession(
            session_id="session-1",
            root_task_id="task-1",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:00Z",
        ),
        tasks=[task],
    )

    class StubVerifier:
        def verify(
            self,
            *,
            task: TaskRecord,
            state: HarnessState,
            expectations: Sequence[VerificationExpectation],
        ) -> VerificationResult:
            return VerificationResult(
                task_id=task.task_id,
                status=VerificationStatus.PASSED,
                checked_at="2026-01-01T00:00:05Z",
                expectation_ids=[
                    expectation.expectation_id for expectation in expectations
                ],
                evidence=[
                    VerificationEvidenceRecord(
                        evidence_id="evidence-1",
                        task_id=task.task_id,
                        recorded_at="2026-01-01T00:00:04Z",
                        summary=f"verified in {state.session.session_id}",
                        artifact_ref="artifacts/test-report.json",
                    )
                ],
            )

    verifier = StubVerifier()

    assert isinstance(verifier, Verifier)

    result = verifier.verify(
        task=task,
        state=state,
        expectations=task.verification_expectations,
    )

    assert result.status is VerificationStatus.PASSED
    assert result.expectation_ids == ["expectation-1"]
    assert result.evidence[0].task_id == "task-1"
    assert result.evidence[0].artifact_ref == "artifacts/test-report.json"

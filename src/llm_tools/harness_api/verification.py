"""Typed verification contracts for durable harness execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from llm_tools.harness_api.verification_models import (
    NoProgressSignal as NoProgressSignal,
)
from llm_tools.harness_api.verification_models import (
    NoProgressSignalKind as NoProgressSignalKind,
)
from llm_tools.harness_api.verification_models import (
    VerificationEvidenceRecord as VerificationEvidenceRecord,
)
from llm_tools.harness_api.verification_models import (
    VerificationExpectation as VerificationExpectation,
)
from llm_tools.harness_api.verification_models import (
    VerificationFailureMode as VerificationFailureMode,
)
from llm_tools.harness_api.verification_models import (
    VerificationResult as VerificationResult,
)
from llm_tools.harness_api.verification_models import (
    VerificationStatus as VerificationStatus,
)
from llm_tools.harness_api.verification_models import (
    VerificationTiming as VerificationTiming,
)
from llm_tools.harness_api.verification_models import (
    VerificationTrigger as VerificationTrigger,
)

if TYPE_CHECKING:
    from llm_tools.harness_api.models import HarnessState, TaskRecord


@runtime_checkable
class Verifier(Protocol):
    """Callable verifier surface consumed by the harness."""

    def verify(
        self,
        *,
        task: TaskRecord,
        state: HarnessState,
        expectations: Sequence[VerificationExpectation],
    ) -> VerificationResult:
        """Run verification for one task against the durable harness state."""

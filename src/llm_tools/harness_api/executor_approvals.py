"""Approval and retry models for harness execution."""

from __future__ import annotations

from enum import Enum

from llm_tools.harness_api.executor_approvals_models import (
    HarnessRetryPolicy as HarnessRetryPolicy,
)


class ApprovalResolution(str, Enum):  # noqa: UP042
    """Explicit operator resolution for a persisted approval wait."""

    APPROVE = "approve"
    DENY = "deny"
    CANCEL = "cancel"
    EXPIRE = "expire"


__all__ = ["ApprovalResolution", "HarnessRetryPolicy"]

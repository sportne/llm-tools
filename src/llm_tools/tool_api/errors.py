"""Domain-specific errors for the tool API."""

from __future__ import annotations


class ToolRegistryError(Exception):
    """Base class for registry-specific errors."""


class DuplicateToolError(ToolRegistryError):
    """Raised when registering a tool whose canonical name already exists."""


class ToolNotRegisteredError(ToolRegistryError):
    """Raised when looking up a tool name that has not been registered."""


class RetryableToolExecutionError(RuntimeError):
    """Execution failure that should be surfaced as retryable."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.retryable = True

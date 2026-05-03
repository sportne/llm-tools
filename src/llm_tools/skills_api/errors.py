"""Runtime errors for skill resolution and loading."""

from __future__ import annotations


class SkillResolutionError(ValueError):
    """Base class for skill resolution failures."""


class SkillNotFoundError(SkillResolutionError):
    """Raised when a skill invocation does not match any enabled skill."""


class SkillAmbiguousError(SkillResolutionError):
    """Raised when a plain-name invocation matches multiple effective skills."""


class SkillDisabledError(SkillResolutionError):
    """Raised when an explicit invocation targets a disabled skill."""


class SkillLoadError(OSError):
    """Raised when a resolved skill body cannot be loaded."""

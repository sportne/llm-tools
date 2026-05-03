# ADR 0017: Keep Runtime Behavior Out of Model Files

## Status

Accepted

## Context

The repository uses Vulture as a long-term maintenance tool for finding
possibly unused code. Pydantic models, enums, protocols, validators, and
serializers often look unused to static dead-code analysis because they are
referenced indirectly through schemas, validation, serialization, public API
exports, or framework conventions. To keep Vulture useful, model files are
excluded from its normal dead-code scan.

## Decision

Keep Pydantic models and model-adjacent declarations in `models.py` or
`*_models.py`, and keep normal runtime behavior out of those files.

Model files may contain models, enums, protocols, validators, serializers,
computed fields, and narrowly model-shaped helpers. Public functions, runtime
services, non-model classes, and behavioral orchestration should live in
non-model modules. Architecture tests should continue to enforce both sides of
that boundary: Pydantic models live in model files, and model files do not hide
runtime code.

## Consequences

Vulture exclusions stay narrow enough to support long-term maintenance instead
of becoming places where unused runtime behavior can accumulate unseen.

The cost is additional modules and imports, especially for small feature slices.
That friction is accepted because it keeps typed contracts visually separate
from execution logic and makes dead-code review findings more trustworthy.

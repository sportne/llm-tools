# ADR 0026: Separate Hidden-Path Discovery from Explicit Access

## Status

Accepted

## Context

Filesystem discovery and search can easily surface noisy, generated, ignored, or
sensitive paths such as dot-hidden directories and `.gitignore`-ignored files.
At the same time, users may explicitly ask the assistant to inspect a hidden or
ignored file by path.

## Decision

Exclude dot-hidden and `.gitignore`-ignored paths from filesystem discovery and
search by default, while preserving explicit direct file access when a caller
names the path.

Discovery and search tools may expose an `include_hidden` option to reveal
hidden and ignored paths. Direct reads remain governed by normal workspace path
validation, permissions, and policy rather than by discovery visibility.

## Consequences

Default list and search results stay focused and avoid casually exposing hidden
or ignored files to the model.

The cost is a nuanced rule: a path can be hidden from discovery but still
readable when explicitly requested and policy-allowed. That distinction is
accepted because discovery is a broad surfacing operation, while direct reads
are narrower and user- or model-specified.

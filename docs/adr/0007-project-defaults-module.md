# ADR 0007: Use an In-Package Project Defaults Module

## Status

Accepted

## Context

The assistant app needs a source-controlled way for a delivered project or app
variant to set its operational starting point: assistant configuration, provider
connection presets, provider URL help text, and initial administrator settings.
YAML config and CLI flags already work for startup overrides, but they are not a
good fit for defaults that should be reviewed, versioned, tested, and swapped by
branch or build variant.

## Decision

Load a conventional in-package **Project Defaults Module** at
`llm_tools.apps.assistant_app.project_defaults` and read one typed
`PROJECT_DEFAULTS` object from it.

The defaults object owns the project-shipped baseline for:

- assistant startup configuration;
- **Provider Connection Presets**;
- provider Base URL help text;
- initial administrator settings.

Optional YAML config deeply overlays mapping fields from the project baseline,
list fields replace rather than concatenate, and CLI arguments override both.
Initial administrator settings from the module are used as an absent-row fallback
only; they are not written to persistence until an administrator saves settings.

## Consequences

Project-specific defaults are part of the source-controlled deliverable instead
of an operator-selected Python hook or workspace-discovered file. Different
operational variants can use branches or build-time file replacement.

The app still keeps process deployment settings such as host, port, database
paths, encryption key paths, TLS, auth mode, and browser launch behavior outside
the **Project Defaults Module**.

User- or administrator-managed provider preset persistence remains a future
layer; the module defines the starting catalog without preventing later
additions or overrides.

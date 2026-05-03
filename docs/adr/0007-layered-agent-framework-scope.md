# ADR 0007: Treat llm-tools as a Layered Agent Framework

## Status

Accepted

## Context

`llm-tools` began around typed tool definition, validation, registration, and
execution, but the implemented repository now includes reusable skills,
workflow, harness, provider, and assistant app layers. Treating those higher
layers as accidental would make the architecture docs, import-boundary tests,
assistant app, and durable harness APIs fight the actual direction of the
project.

## Decision

Keep `tool_api` as the lowest reusable substrate, while treating `skills_api`,
`workflow_api`, `harness_api`, and `apps` as supported in-repository surfaces
for an agent and assistant framework.

The lower layers must remain reusable and dependency-directed, and downstream
extensions should still prefer library surfaces over app glue when they do not
need product-specific behavior. The higher layers are not out of scope merely
because they involve planning, durable sessions, protection, prompt management,
or assistant product flows.

## Consequences

The repository accepts a broader scope than a minimal tool-core package, which
increases dependency and boundary pressure. In exchange, the framework can keep
tested integration points for assistant chat, deep tasks, skills, and durable
harness execution instead of forcing those concerns into downstream forks or
unreviewed app-local glue.

Future refactors should split concentrated modules and protect layering rather
than deleting supported behavior solely to make the package look smaller.

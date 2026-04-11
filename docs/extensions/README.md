# Extension Docs

This section is for future-facing guidance on extending `llm-tools`.

Today, the most useful extension entry points are:

- defining new `Tool` subclasses
- adding registration helpers for tool groups
- adding new LLM adapters
- building higher-level composition on top of `workflow_api`

Start with the usage guides for the concrete contracts:

- [Defining Tools](../usage/defining-tools.md)
- [Registry and Runtime](../usage/registry-and-runtime.md)
- [Policy](../usage/policy.md)
- [Adapters](../usage/adapters.md)

This section should stay high-level and future-oriented rather than duplicating
the practical how-to material in `docs/usage/`.


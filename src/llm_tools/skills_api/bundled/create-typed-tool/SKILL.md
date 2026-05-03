---
name: create-typed-tool
description: Create or update an llm-tools typed Tool using the repository's tool_api conventions.
---

# Create Typed Tool

Use this skill when adding a new typed tool to `llm-tools`.

Follow the existing `tool_api` patterns:

- define Pydantic input and output models in a `*_models.py` file;
- keep executable behavior outside model files;
- register the tool through the relevant tool-family registration entrypoint;
- route execution through `ToolRuntime` rather than direct tool invocation;
- add focused tests for schema, registration, policy, and execution behavior.

Prefer small explicit models and preserve the package's layer boundaries.

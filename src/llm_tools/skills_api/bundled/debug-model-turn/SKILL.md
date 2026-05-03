---
name: debug-model-turn
description: Diagnose model-turn protocol failures across provider output, parsing, fallback, and tool execution.
---

# Debug Model Turn

Use this skill when a model turn fails, loops, or returns malformed tool output.

Work through the turn boundary in order:

- identify the selected model and provider protocol;
- check the response mode and fallback path;
- inspect redacted model-turn events before raw provider details;
- reproduce the parsed model turn with the smallest provider or adapter fixture;
- verify whether the failure belongs to provider transport, parsing, tool policy,
  tool execution, or response protection.

Keep raw provider responses out of durable events unless an existing debug path
explicitly allows them.

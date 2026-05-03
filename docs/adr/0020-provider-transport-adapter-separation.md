# ADR 0020: Separate Provider Transport from Model-Output Adapters

## Status

Accepted

## Context

`llm-tools` needs to work with multiple model-service API shapes and multiple
ways of expressing model actions: native tool calls, structured JSON, and
prompt-emitted tool decisions. If provider transport and model-output
normalization are implemented as one provider-specific blob, each new endpoint
shape risks duplicating parsing, action-envelope, fallback, and workflow
handoff behavior.

## Decision

Keep provider transport in `llm_providers` and canonical model-output
normalization in `llm_adapters`.

Provider modules own typed calls to model services and provider-specific request
or response mechanics. Adapter modules own conversion from model output into
canonical parsed responses, tool invocations, and final responses that workflow
and harness code can consume. Native provider protocols may add new transport
code, but they should still map into the shared model-turn and adapter contracts
instead of bypassing them with provider-shaped workflow results.

## Consequences

Model-output normalization stays reusable across OpenAI-compatible endpoints,
native provider protocols, prompt-tool fallback, workflow execution, assistant
chat, and durable harness sessions.

The cost is additional abstraction between provider calls and workflow
execution. That separation is accepted because it keeps provider-specific API
details from leaking into the canonical execution path.

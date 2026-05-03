# ADR 0018: Support Prompt-Emitted Tool Calls as a First-Class Protocol

## Status

Accepted

## Context

Some model endpoints support native tool calls or structured JSON schema
responses, but proprietary, enterprise, local, or otherwise constrained LLM
service endpoints may only expose plain chat-completion text. Assuming native
tools or structured responses are always available would make the assistant and
harness layers unusable with those endpoints or force each caller to invent its
own text protocol.

## Decision

Treat prompt-emitted tool calls as a supported Model-Turn Protocol path, not as
a legacy compatibility hack.

`prompt_tools` should parse fenced text decisions into the same canonical parsed
model response and `ToolInvocationRequest` path used by native tools and
structured JSON responses. Protocol fallback, parsing errors, repair prompts,
and final-answer handling belong in the model-turn protocol layer so workflow,
harness, and assistant callers share the same behavior.

## Consequences

`llm-tools` can support endpoints that lack native tool or structured-response
APIs while preserving shared policy, approval, redaction, execution, and result
normalization through the existing runtime path.

The cost is a more complex protocol parser and repair loop, plus less provider
enforcement than native tool or schema modes. That cost is accepted because
plain-text endpoints remain important for local and proprietary deployments.

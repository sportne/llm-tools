# ADR 0023: Use Staged Structured JSON Tool Use

## Status

Accepted

## Context

Structured JSON response mode can ask a model to return a complete multi-action
envelope in one response, or it can ask the model to choose one next action,
validate that action's arguments, execute it, and repeat until finalization.
Local and constrained model endpoints are more likely to produce invalid or
overconfident multi-action envelopes when asked to plan and serialize an entire
tool batch at once.

## Decision

Use staged structured JSON tool use for JSON response mode.

The model should choose one action at a time: select a tool or finalize,
validate the selected tool's arguments against that tool's schema, execute the
tool through the normal runtime path, then repeat until a final response is
produced.

## Consequences

JSON mode gets narrower validation targets, better repair prompts, and less
brittle behavior for local or weaker structured-output endpoints while still
normalizing into the same tool invocation and workflow execution path.

The cost is additional model turns compared with a single multi-action
structured envelope. That cost is accepted because correctness and recoverable
validation failures matter more than minimizing turn count for agentic tool use.

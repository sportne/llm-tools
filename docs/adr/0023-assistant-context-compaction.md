# ADR 0023: Compact Provider Context Without Truncating Transcript

## Status

Accepted

## Context

Assistant Chat needs a complete user-visible transcript for persistence,
inspection, retry, and review, but provider calls have finite context windows.
When conversations grow long, sending the full transcript can fail or crowd out
new task context.

## Decision

Keep the full transcript for UI and persistence, while compacting only the
model-visible provider context.

Older completed turns may be summarized into durable context summaries when the
configured token budget is exceeded, and provider calls should include those
summaries instead of the exact older messages. If a provider rejects a request as
too large, the chat runner may compact more aggressively and retry once.

## Consequences

Users retain a complete transcript while model calls stay within practical
context limits.

The cost is that the model may reason from summaries rather than verbatim older
turns. That trade-off is accepted because mutating or truncating the durable
transcript would make the app less inspectable and less trustworthy.

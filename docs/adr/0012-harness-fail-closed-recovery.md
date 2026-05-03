# ADR 0012: Fail Closed for Harness Recovery and Approval Pauses

## Status

Accepted

## Context

Harness sessions can pause with pending approvals, resume after process
restart, or recover from a crash after an incomplete turn checkpoint. In those
states, automatically continuing work can execute model-selected actions after
the operator denied consent, failed to respond, canceled the pause, or after the
process lost the exact execution boundary of a partial turn.

## Decision

Harness recovery and approval handling fail closed.

Approval denial, expiration, and operator cancel record the blocked invocation
but do not continue later invocations from the same paused model turn.
Interrupted non-approval tail turns are not replayed automatically; callers or
operators must explicitly acknowledge and drop the incomplete tail before rerun.
Pending approval records preserve only scrubbed context and are rebuilt from
current execution context on resume. Durable approval history should keep
minimal approval records and audit metadata rather than raw approval request
payloads.

## Consequences

The harness avoids inferring consent or safe replay from ambiguous durable
state. This is more conservative than automatic continuation, and it may require
operator action after crashes or approval pauses, but it keeps resumable
execution aligned with policy and human approval boundaries.

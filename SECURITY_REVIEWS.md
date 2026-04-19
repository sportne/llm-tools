# SECURITY_REVIEWS.md

## Purpose

This document records completed security reviews for `llm-tools`.

`SEC_TASKS.md` remains the canonical backlog for review execution and follow-up
remediation tracking. This file keeps the durable review notes, evidence, and
residual risk summaries produced by completed review passes.

## 2026-04-19: `harness_api` persistence, resume, replay, and summary review

### Scope

- `src/llm_tools/harness_api/store.py`
- `src/llm_tools/harness_api/resume.py`
- `src/llm_tools/harness_api/replay.py`
- `src/llm_tools/harness_api/session.py`
- `src/llm_tools/harness_api/models.py`
- `src/llm_tools/harness_api/protection.py`
- workflow approval resume path used by harness resume:
  `src/llm_tools/workflow_api/executor.py`

### Evidence

- Reviewed code paths:
  `src/llm_tools/harness_api/store.py`,
  `src/llm_tools/harness_api/resume.py`,
  `src/llm_tools/harness_api/replay.py`,
  `src/llm_tools/harness_api/session.py`,
  `src/llm_tools/harness_api/models.py`,
  `src/llm_tools/harness_api/protection.py`,
  `src/llm_tools/workflow_api/executor.py`
- Relevant tests inspected and run:
  `tests/harness_api/test_store.py`,
  `tests/harness_api/test_resume.py`,
  `tests/harness_api/test_replay_golden.py`,
  `tests/harness_api/test_protection_scrub.py`,
  `tests/harness_api/test_session_api.py`,
  `tests/harness_api/test_harness_executor.py`
- Test execution result:
  focused harness test set passed via the repo venv: 66 passed
- Direct repro evidence:
  denied approval still allowed a later `write_file` invocation from the same
  persisted parsed response to execute
  tampered persisted approval payload could redirect approved execution to a
  different tool invocation

### Findings

- High: denied, canceled, or expired approvals still execute later invocations
  from the same persisted parsed response.
  Preconditions: an approval-gated invocation is followed by additional
  invocations in the same parsed response, and resume is resolved with a
  non-approve outcome.
  Impact: approval denial does not fully stop execution of the persisted action
  sequence, so later side effects can still occur.
- Medium: persisted approval records are structurally validated but
  unauthenticated, so tampered stored state can redirect approved execution.
  Preconditions: an attacker or lower-trust principal can modify persisted
  harness session JSON before approval resume.
  Impact: operator approval may be applied to a modified persisted action rather
  than the originally reviewed action.
- Medium: stored summaries and traces are trusted for replay and inspection
  without consistency checks against canonical state.
  Preconditions: a persisted artifact can be modified independently of the
  canonical state payload.
  Impact: operator or audit views can be spoofed even when the main stored state
  is unchanged.
- Medium: persisted approvals, traces, metadata, logs, and summaries can retain
  sensitive content.
  Preconditions: secrets or sensitive data appear in tool arguments, policy
  metadata, logs, artifact names, or free-form summary fields.
  Impact: file-backed session storage and inspector surfaces may retain data
  beyond the scrubbed pending-approval base context.
- Low: corrupt file-backed state is not handled defensively and can cause load
  or list failures.
  Preconditions: malformed or partially written session artifacts exist on disk.
  Impact: corrupted persisted files become an availability issue for inspection
  and session enumeration.

### Recommended Remediation

- Prevent post-denial execution of trailing invocations after approval
  rejection, cancellation, or expiry.
- Add integrity binding for persisted approval payloads used during resume so
  the approved payload can be authenticated before execution.
- Validate or recompute persisted trace and summary artifacts before trusting
  them for replay or inspection.
- Reduce sensitive data retention in persisted approval, trace, and summary
  artifacts by default, including arguments, policy metadata, logs, artifact
  names, and free-form summary content.
- Handle malformed file-backed session artifacts without breaking load and list
  flows.

### Residual Risk

Persisted harness state should currently be treated as trusted storage. Until
approval replay stops on non-approved outcomes and persisted approval payloads
and observability artifacts gain integrity protection, a principal who can
modify stored session JSON can influence future execution after approval and can
falsify replay or summary views.

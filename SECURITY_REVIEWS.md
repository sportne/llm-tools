# SECURITY_REVIEWS.md

## Purpose

This document is the cumulative artifact for completed security assessments in
`llm-tools`.

Use it to record completed review outcomes in one place after the corresponding
backlog work in `SEC_TASKS.md` is done. `SEC_TASKS.md` remains the canonical
backlog; this file is the durable review log.

## Entry conventions

Each completed review entry should include:

- review date
- component or scope reviewed
- code paths reviewed
- tests inspected and run
- findings summary with severity
- recommended remediation summary
- residual risk summary
- related `SEC_TASKS.md` phase or checklist reference

## Completed Reviews

### 2026-04-19: `harness_api` lifecycle and control-flow review

Related backlog: Phase 4 in `SEC_TASKS.md`

Scope:

- `src/llm_tools/harness_api/models.py`
- `src/llm_tools/harness_api/executor.py`
- `src/llm_tools/harness_api/tasks.py`
- `src/llm_tools/harness_api/session.py`
- directly related helpers: `resume.py`, `replay.py`, `store.py`, `planning.py`,
  `context.py`, `verification.py`, `protection.py`
- relevant tests under `tests/harness_api/`

Code paths reviewed:

- session, task, turn, and pending-approval state models
- executor run, resume, retry, approval, budget, and persistence-conflict paths
- task lifecycle transition helpers
- public session stop/resume/inspection behavior
- resume-time corruption detection and replay artifact construction

Tests:

- inspected: `tests/harness_api/test_harness_models.py`,
  `test_resume.py`, `test_harness_executor.py`, `test_task_lifecycle.py`,
  `test_session_api.py`, `test_session_additional_coverage.py`,
  `test_planning.py`
- run: `~/.venvs/llm-tools/bin/python -m pytest -q tests/harness_api/test_harness_models.py tests/harness_api/test_resume.py tests/harness_api/test_harness_executor.py tests/harness_api/test_task_lifecycle.py tests/harness_api/test_session_api.py tests/harness_api/test_session_additional_coverage.py`
- result: `94 passed`
- environment note: `make install-dev` could not refresh dependencies because
  network access was unavailable, but the existing shared virtual environment
  already contained the required test packages

Findings summary:

- `Medium`: executor can persist `stop_reason=completed` when the planner
  returns no actionable tasks even though blocked non-terminal tasks remain
- `Medium`: non-approval turns have at-least-once semantics, so crash-before-save
  can replay non-idempotent side effects
- `Medium`: `max_tool_invocations` is enforced after turn execution, so one turn
  can overshoot the configured cap
- `Medium`: approval resume rehydrates from the full current process
  environment, which can widen privilege across turns
- `Low`: `NO_PROGRESS` is modeled and validated but not enforced by executor
  control flow

Recommended remediation summary:

- prevent `completed` terminalization when non-terminal blocked work remains
- add durable checkpointing or idempotency safeguards for executed side effects
- enforce tool-invocation budget before or during dispatch
- constrain approval resume environment rehydration to a reviewed scope
- add executor-level no-progress detection and negative coverage for repeated
  non-progress loops

Residual risk:

Model validation and resume-time corruption checks are strong, but control-flow
integrity still depends on follow-up hardening in stop semantics, recovery
behavior, budget enforcement timing, and approval resume privilege boundaries.

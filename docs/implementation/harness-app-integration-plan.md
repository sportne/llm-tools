# Harness App Integration Plan

This document captures the Phase 10.3 plan for integrating the persisted
`harness_api` session surface into the existing apps without moving harness
contracts into app-specific modules. The goal is to reuse the new session API,
trace artifacts, summaries, and replay views as shared seams.

## Textual Chat

- add an optional persisted session sidebar that lists recent harness sessions
  through `HarnessSessionService.list_sessions(...)`
- surface waiting approvals with explicit resume affordances that map to
  `resume_session(..., approval_resolution=...)`
- keep repository-chat interaction logic in the app layer while storing harness
  state, traces, and summaries through `harness_api`

## Streamlit Chat

- add persisted session selection and inspection controls driven by the same
  session list and inspect APIs
- reuse trace and summary artifacts for operator-visible history rather than
  rebuilding presentation state in Streamlit
- keep provider and app-specific configuration local to the Streamlit app

## Textual Workbench

- add harness-backed session launch, inspect, replay, and approval-resume flows
  as workbench tools over the public session service
- treat the workbench as a client of `HarnessSessionService` rather than a new
  orchestration surface
- reuse persisted artifacts for debugging panels so trace rendering stays
  library-owned and app presentation remains thin

## Shared Seams

- `HarnessSessionService` remains the public orchestration entrypoint
- `StoredHarnessState.saved_at` and `list_sessions(...)` support recent-session
  views without app-side sorting contracts
- `snapshot.artifacts.trace`, `snapshot.artifacts.summary`, and
  `replay_session(...)` remain the only supported observability and replay
  inspection contracts for apps

## Out of Scope

- converting the existing apps to harness-backed execution in this change set
- moving app configuration or presentation state into `harness_api`
- adding app-specific persistence formats beyond the shared session store

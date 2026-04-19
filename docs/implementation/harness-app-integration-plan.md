# Harness App Integration Plan

This document captures the older Phase 10.3 plan for integrating the persisted
`harness_api` session surface into the app layer without moving harness
contracts into app-specific modules.

Textual clients have since been removed. The remaining interactive surfaces are
the Streamlit apps, and `streamlit_chat` is now a deprecated migration surface.

## Streamlit Chat

- add persisted session selection and inspection controls driven by the same
  session list and inspect APIs
- reuse trace and summary artifacts for operator-visible history rather than
  rebuilding presentation state in Streamlit
- keep provider and app-specific configuration local to the Streamlit app

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

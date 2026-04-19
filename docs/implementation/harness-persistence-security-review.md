# Harness API Persistence Security Review

Date: 2026-04-19

## Scope

- `src/llm_tools/harness_api/store.py`
- `src/llm_tools/harness_api/resume.py`
- `src/llm_tools/harness_api/replay.py`
- `src/llm_tools/harness_api/protection.py`
- supporting persistence and resume paths in:
  - `src/llm_tools/harness_api/models.py`
  - `src/llm_tools/harness_api/executor.py`
  - `src/llm_tools/harness_api/session.py`
  - `src/llm_tools/workflow_api/executor.py`
  - `src/llm_tools/workflow_api/chat_session.py`
  - `src/llm_tools/tool_api/runtime.py`
- reviewed tests:
  - `tests/harness_api/test_store.py`
  - `tests/harness_api/test_resume.py`
  - `tests/harness_api/test_replay_golden.py`
  - `tests/harness_api/test_protection_scrub.py`
  - `tests/harness_api/test_session_api.py`
  - `tests/harness_api/test_harness_executor.py`

## Evidence

- Focused test run completed successfully:
  - `$HOME/.venvs/llm-tools/bin/python -m pytest -q tests/harness_api/test_store.py tests/harness_api/test_resume.py tests/harness_api/test_replay_golden.py tests/harness_api/test_protection_scrub.py tests/harness_api/test_session_api.py tests/harness_api/test_harness_executor.py`
  - Result: `66 passed in 7.92s`
- Pending approval persistence is partially scrubbed:
  - env, logs, artifacts, and source provenance are removed before persistence
  - metadata is retained
- Resume-time validation is materially stronger than replay/inspection validation:
  - canonical state is revalidated
  - incomplete-turn and approval linkage are rechecked
  - stored summary and trace artifacts are not recomputed before use

## Findings

### High: persisted harness state and derived artifacts can retain sensitive data

Affected paths:

- `src/llm_tools/harness_api/store.py`
- `src/llm_tools/harness_api/executor.py`
- `src/llm_tools/harness_api/replay.py`
- `src/llm_tools/tool_api/runtime.py`
- `src/llm_tools/workflow_api/chat_session.py`

Details:

- Canonical session persistence stores the full `HarnessState` JSON snapshot.
- Completed turns persist `workflow_result` records, including parsed invocations and outcome payloads.
- Protection scrubbing only replaces the tail turn `final_response`; it does not scrub prior turns, tool invocation arguments, tool outputs, error details, verification evidence, or retained task summaries.
- Replay trace generation persists request payloads, logs, artifacts, and policy metadata from execution records.
- Runtime redaction does apply to logs and artifacts before execution records are stored, but request arguments and outputs may still remain in persisted state and execution metadata depending on tool/runtime configuration.

Impact:

- Secrets or sensitive operational data can remain in persisted snapshots, replay artifacts, and operator-visible inspection output.
- The storage layer should be treated as confidentiality-sensitive, not merely operational metadata.

Current coverage:

- Existing tests confirm pending approval context strips env/logs/artifacts/source provenance.
- Existing tests confirm final-response protection scrubbing.
- No tests currently assert that secret-bearing invocation args, outputs, logs, artifacts, metadata, error details, or verification evidence are absent from persisted state.

### Medium: stored summaries are trusted without recomputation

Affected paths:

- `src/llm_tools/harness_api/session.py`
- `src/llm_tools/harness_api/store.py`

Details:

- `inspect_session()` and `list_sessions()` prefer `snapshot.artifacts.summary` over recomputing from canonical state.
- Validation only binds `summary.session_id` to the session id; it does not verify the summary matches the stored state.

Impact:

- A tampered persisted summary can mislead operators about completion state, active tasks, pending approvals, or latest decision text.
- This does not directly change execution behavior, but it weakens operational trust in inspection surfaces.

Current coverage:

- Existing tests validate session-id binding only.
- No tests cover tampered or stale summaries.

### Medium: replay trusts stored trace artifacts as authoritative

Affected paths:

- `src/llm_tools/harness_api/replay.py`
- `src/llm_tools/harness_api/session.py`

Details:

- `replay_session()` uses `snapshot.artifacts.trace` whenever present and falls back to canonical turns only when trace is missing.
- Trace validation checks internal shape, but not consistency with `state.turns`.

Impact:

- A tampered trace can fabricate approval flow, workflow statuses, decision summaries, or selected tasks in operator-visible replay output.
- Replay is therefore an unauthenticated cache, not a trustworthy reconstruction.

Current coverage:

- Golden tests cover only successful expected traces and replay output.
- No negative tests cover stale, injected, or mismatched traces.

### Low: corrupted on-disk session files are not isolated into typed corruption handling

Affected paths:

- `src/llm_tools/harness_api/store.py`
- `src/llm_tools/harness_api/resume.py`

Details:

- `FileHarnessStateStore.load_session()` and `list_sessions()` directly deserialize JSON and will raise on malformed or partially written files.
- `resume_session()` only classifies corruption after a valid `StoredHarnessState` is already loaded.
- File writes are in-place rather than temp-file-plus-rename.

Impact:

- Partial writes or manual file corruption can cause raw load failures instead of a typed corrupt-session result.
- This is primarily an availability and operability issue, with some integrity ambiguity during manual recovery.

Current coverage:

- Existing tests cover invalid inner state and approval mismatches after successful load.
- No tests cover malformed JSON files, torn writes, or list behavior when one file is corrupt.

## Recommended remediation

1. Minimize persisted data before save.
   - Add an explicit harness-level persistence scrub for completed turns and derived artifacts.
   - Default to dropping or hashing tool args, outputs, error details, logs, artifacts, and metadata unless required for resume semantics.

2. Treat trace and summary as cache, not authority.
   - Recompute summary on read, or validate it against canonical state before use.
   - Rebuild replay from canonical turns by default, or attach and verify a digest over trace content.

3. Harden file-store integrity behavior.
   - Write snapshots atomically with temp-file plus rename.
   - Catch JSON and validation failures in `load_session()` and `list_sessions()` and surface typed corruption outcomes.

4. Add negative regression tests.
   - Tampered summary accepted by inspect/list.
   - Tampered trace accepted by replay.
   - Corrupted JSON session file.
   - Secret-bearing invocation args, outputs, logs, artifacts, metadata, and error details persisting across save/inspect/replay.

## Residual risk

- Approval resume integrity is in materially better shape than replay and operator-summary integrity.
- The main unresolved risk is confidentiality and operator trust in persisted artifacts, not approval bypass.
- Until persistence minimization and artifact revalidation land, stored harness sessions should be treated as sensitive and attacker-modifiable.

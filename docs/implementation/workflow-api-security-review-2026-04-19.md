# `workflow_api` Security Review (2026-04-19)

## Scope

Reviewed code paths:

- `src/llm_tools/workflow_api/executor.py`
- `src/llm_tools/workflow_api/chat_session.py`
- `src/llm_tools/workflow_api/protection.py`
- `src/llm_tools/workflow_api/models.py`
- `src/llm_tools/llm_adapters/action_envelope.py`
- `src/llm_tools/tool_api/runtime.py`
- `src/llm_tools/tool_api/policy.py`
- `src/llm_tools/tool_api/execution.py`
- `src/llm_tools/tool_api/models.py`
- `src/llm_tools/harness_api/executor.py`
- `src/llm_tools/harness_api/models.py`
- `src/llm_tools/harness_api/replay.py`
- `src/llm_tools/harness_api/protection.py`

## Validation

Targeted suites run:

```bash
"$HOME/.venvs/llm-tools/bin/python" -m pytest -q   tests/workflow_api/test_executor.py   tests/workflow_api/test_executor_additional.py   tests/workflow_api/test_chat_session.py   tests/workflow_api/test_protection.py   tests/tool_api/test_runtime.py   tests/tool_api/test_policy.py   tests/architecture/test_runtime_mediation.py   tests/architecture/test_no_direct_tool_invocation.py   tests/harness_api/test_harness_executor.py   tests/harness_api/test_replay_golden.py   tests/harness_api/test_protection_scrub.py
```

Result: `137 passed in 12.39s`

Additional tests were inspected for coverage of approval, replay, mediation, and redaction behavior.

## Findings

### High: Approval denial and timeout are not fail-closed in raw `WorkflowExecutor`

Affected paths:

- `src/llm_tools/workflow_api/executor.py`
- `tests/workflow_api/test_executor.py`
- `tests/workflow_api/test_executor_additional.py`

Issue:

- `WorkflowExecutor._resume_persisted_approval()` and `_resume_persisted_approval_async()` mark the blocked invocation as `approval_denied` or `approval_timed_out`, then continue executing later model-selected invocations from `pending_index + 1`.
- This is currently expected by tests in `test_workflow_executor_deny_marks_outcome_and_continues()` and `test_workflow_executor_finalizes_expired_approvals()`.

Security impact:

- A model can place an approval-gated action first and a later side-effecting action second. If the first action is denied or expires, the later action still runs unless a higher layer wraps `WorkflowExecutor` and force-stops.
- The lower-level workflow API therefore does not preserve a fail-closed approval boundary by itself.

Evidence:

- `src/llm_tools/workflow_api/executor.py:337-350`
- `src/llm_tools/workflow_api/executor.py:378-390`
- `tests/workflow_api/test_executor.py:675-757`
- `tests/workflow_api/test_executor_additional.py:162-174`

Recommended remediation:

- Make non-approved resolutions terminal by default in `WorkflowExecutor`.
- If continuation is still needed for a specific caller, require an explicit opt-in policy at the workflow boundary.
- Add regression tests asserting that later invocations do not execute on `deny`, `expire`, or `cancel` unless that opt-in is enabled.

### High: Persisted approval resume is not cryptographically or structurally bound to the originally approved request

Affected paths:

- `src/llm_tools/workflow_api/executor.py`
- `src/llm_tools/harness_api/models.py`
- `src/llm_tools/harness_api/executor.py`

Issue:

- `PendingApprovalRecord` validates only index consistency.
- `resume_persisted_approval()` trusts `record.parsed_response`, `record.pending_index`, and `record.base_context`, then re-executes the indexed invocation with `approval_override=True`.
- There is no integrity check that `approval_request.request` still matches the invocation at `pending_index`, or that the resumed execution context is materially identical to what was approved.

Security impact:

- If persisted approval records are tampered with, an approval granted for one request can be redirected to a different tool invocation or materially different arguments/context during resume.
- This is especially relevant because `HarnessExecutor` rehydrates environment at resume time, intentionally changing execution context.

Evidence:

- `src/llm_tools/harness_api/models.py:229-248`
- `src/llm_tools/workflow_api/executor.py:221-253`
- `src/llm_tools/workflow_api/executor.py:321-327`
- `src/llm_tools/harness_api/executor.py:664-674`
- `src/llm_tools/harness_api/executor.py:908-910`

Recommended remediation:

- Persist and verify a stable digest covering the approved request, tool name/version, invocation index, and any approval-relevant context fields.
- Reject resume if the persisted invocation no longer matches `approval_request.request` exactly.
- Add negative tests for tampered approval records and mismatched approval-request payloads.

### Medium: Prompt protection retains raw blocked prompt material in session state and correction storage

Affected paths:

- `src/llm_tools/workflow_api/protection.py`
- `src/llm_tools/workflow_api/chat_session.py`

Issue:

- `ProtectionPendingPrompt` stores full `serialized_messages`.
- `build_pending_prompt()` persists them into chat session state.
- `record_feedback()` writes those serialized messages into correction metadata.
- `ChatSessionTurnRunner` also emits raw `provider_messages` inspector payloads before provider execution.

Security impact:

- A prompt blocked before model execution can still be retained in memory, surfaced in inspector/debug flows, and written to correction files.
- Those snapshots can include raw user content, prior tool result content, and system/context material that protection just decided should not proceed unchanged.

Evidence:

- `src/llm_tools/workflow_api/protection.py:192-205`
- `src/llm_tools/workflow_api/protection.py:565-582`
- `src/llm_tools/workflow_api/protection.py:643-678`
- `src/llm_tools/workflow_api/chat_session.py:202-206`
- `tests/workflow_api/test_chat_session.py:616-734`

Recommended remediation:

- Store a minimized or redacted prompt snapshot instead of full serialized messages.
- Apply the same redaction policy used for tool payloads to inspector payloads and correction-file metadata.
- Add negative tests asserting that blocked prompt contents are not persisted verbatim.

### Medium: Replay and trace artifacts preserve raw request arguments outside the runtime redaction path

Affected paths:

- `src/llm_tools/harness_api/replay.py`
- `tests/harness_api/test_replay_golden.py`

Issue:

- Replay traces store `outcome.request` directly, not the redacted request from the execution record.
- Approval-path policy snapshots also copy `approval_request.policy_metadata` directly.
- Current golden tests expect raw request arguments to remain present in trace output.

Security impact:

- Sensitive tool arguments can survive into persisted replay and trace metadata even when runtime execution records were redacted.
- The observability surface therefore has weaker confidentiality properties than the runtime result surface.

Evidence:

- `src/llm_tools/harness_api/replay.py:401-428`
- `src/llm_tools/harness_api/replay.py:431-460`
- `tests/harness_api/test_replay_golden.py:114-121`

Recommended remediation:

- Build replay request payloads from the redacted execution record when available.
- Redact or minimize approval metadata before storing it in traces.
- Add regression tests for secret-bearing tool arguments to ensure replay output stays redacted.

## Missing Negative Tests

- Tampered `PendingApprovalRecord` where `approval_request.request` differs from `parsed_response.invocations[pending_index - 1]`.
- Denied, expired, and cancelled approval paths that assert later invocations do not run.
- Redaction coverage for `pending_protection_prompt.serialized_messages`, correction metadata, and `provider_messages` inspector payloads.
- Replay redaction coverage for raw request arguments and approval metadata.
- Provenance propagation coverage across workflow-mediated tool execution.

## Residual Risk Summary

- Runtime mediation remains the strongest control boundary. Direct tool invocation is architecture-tested, runtime-issued execution contexts are enforced, and normal policy checks precede execution.
- The largest remaining risks are above the runtime boundary: approval resumption, observability/replay persistence, and protection-related data retention are not consistently fail-closed or redaction-preserving.
- Existing tests provide good behavior coverage, but some current expected behavior encodes the security gaps above rather than preventing them.

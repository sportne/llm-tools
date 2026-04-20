# Product Evaluation: Streamlit Assistant After Initial Product Hardening

## Purpose

This document reevaluates the current Streamlit assistant after the first round of
product fixes. The goal is no longer to identify the original obvious UX gaps;
those were already addressed in the app and tests. The goal now is to capture
what has improved, what still feels product-incomplete, and which follow-up work
should be prioritized next.

This evaluation remains centered on the three shipped product experiences:

1. normal Streamlit assistant chat
2. Streamlit assistant with enterprise or private-network sources
3. Streamlit research sessions backed by `harness_api`

The assessment below is grounded in the current repository behavior, updated
Streamlit UI flow, provider fallback behavior, and focused product-oriented test
coverage.

## Current Assessment

The product is in materially better shape than it was in the initial evaluation.
The Streamlit assistant now guides the user through setup in a defensible order,
explains blocked tools in product language, gives much better feedback when an
OpenAI-compatible endpoint cannot satisfy structured output requirements, and
maintains better continuity when a chat task turns into durable research.

The main remaining risks are no longer basic usability failures. They are now
mostly about end-to-end validation, status visibility, and smoothing the last
remaining split between direct chat and durable research.

## Status Of The Original Ranked Fixes

### Completed in this pass

1. Guided setup sequence
   The sidebar now leads the user through `Connect model`, `Choose workspace`,
   `Allow access`, and `Choose sources` in that order, with explicit readiness
   and next-step guidance.

2. Product-language tool readiness
   Tool availability is now described in user-facing terms such as missing
   workspace, missing credentials, missing network access, or approval-gated
   execution rather than raw capability labels.

3. Better provider failure feedback
   When the OpenAI-compatible provider exhausts fallback modes, the surfaced
   error now explains that it is a provider compatibility problem, identifies
   the overall failure type, and lists the attempted modes.

4. Better chat-to-research transition
   The research launch surface now explains when to stay in chat versus start a
   research task, seeds the research prompt from current chat state, and writes
   a continuity note back into the chat transcript.

### Still open

5. More browser-level validation
   The product is much better exercised through tests and stateful harnesses,
   but we still do not have a real browser-driven validation pass for the exact
   visual/status experience.

## Walkthrough Matrix Revisited

### 1. Local-only chat with no workspace root

Config: `examples/assistant_configs/local-only-chat.yaml`

Prompt used:
`How does this assistant behave with chat only?`

Current behavior:
- The first-run experience is clearer because the setup panel is now ordered.
- The UI makes it easier to understand that chat can work before any workspace
  or local-tool setup exists.
- The source-readiness area gives a better explanation of what is and is not
  available yet.

What improved:
- The product no longer presents setup as an unstructured block of controls.
- A new user is much less likely to think they must configure local access just
  to try the assistant.

Residual risk:
- This flow is well covered in app-level tests, but it still needs occasional
  browser-level checking to validate the actual presentation and pacing.

### 2. Local-only chat with workspace root and filesystem enabled

Config: `examples/assistant_configs/local-only-chat.yaml`

Prompt used:
`Read plan.txt and summarize it.`

Current behavior:
- The separation between selecting a workspace root and enabling local access is
  still strict and safe.
- The difference is now much easier to understand because Step 2 and Step 3 are
  separated intentionally and tool blockers say exactly what is missing.
- Persisted sessions continue to retain runtime state and transcript history.

What improved:
- The biggest source of confusion from the original review was largely removed.
- Approval-gated tools are also explained more clearly before the user runs
  them.

Residual risk:
- The product still relies on several related concepts that the user must learn:
  workspace scope, filesystem access, subprocess access, and approval behavior.
  The UI is much better, but this remains an area worth watching in real manual
  use.

### 3. Enterprise config with private-network endpoint and remote tools

Config: `examples/assistant_configs/enterprise-data-chat.yaml`

Prompt used:
`Search enterprise systems for release notes.`

Current behavior:
- The custom OpenAI-compatible provider path remains the correct internal
  deployment path.
- Remote tool gating is much easier to follow because readiness copy now says
  whether the missing prerequisite is credentials, network access, or both.
- Provider failure reporting is substantially better when an internal endpoint
  cannot satisfy any structured-output fallback mode.

What improved:
- The operator path is more linear.
- Private-network failures are less opaque because all-mode fallback failures
  now surface as compatibility problems instead of undifferentiated runtime
  errors.

Residual risk:
- Failure visibility is much better than before, but normal successful fallback
  behavior is still mostly invisible. The user still cannot easily tell which
  mode succeeded or whether the endpoint is operating in a degraded-but-working
  compatibility path.

### 4. Harness-backed research flow with approval and resume

Config: `examples/assistant_configs/harness-research-chat.yaml`

Prompt used:
`Read research-note.txt then summarize it.`

Current behavior:
- Research launch still correctly maps Streamlit runtime state into a harness
  session.
- Approval-required sessions still persist durably and can be resumed
  explicitly.
- Research launch now feels more connected to the surrounding conversation
  because the prompt can seed from current chat state and the transcript records
  the transition.

What improved:
- The shift from chat to research is less abrupt.
- The product now better communicates that research is an extension of the same
  conversation rather than a disconnected subsystem.

Residual risk:
- Research still has its own lifecycle, status surfaces, and detail views. That
  is appropriate architecturally, but product-wise it still creates a secondary
  mental model the user must track.

## Current Product Findings

### UX findings

1. The major setup and readiness blockers from the first evaluation are largely
   resolved.
   The assistant is now much more intuitive on first use.

2. Research is better integrated, but not fully unified with chat.
   The product now explains the handoff well, yet users still need to understand
   separate concepts such as active chat, research task status, pending
   approvals, and resumable sessions.

3. Provider compatibility feedback is strong on failure but still weak on
   success-path visibility.
   An operator can now understand why the endpoint failed, but not yet which
   provider mode is actively succeeding in the background.

4. The product still needs validation in a real browser session.
   Current tests do a good job exercising state, persistence, and controller
   wiring, but they do not fully replace actual end-user observation of layout,
   timing, and status perception.

### Backend and provider findings

1. The OpenAI-compatible provider path is in better product shape.
   Fallback behavior is tested and failure messages are clearer.

2. The remaining provider gap is observability rather than basic correctness.
   The system should eventually make the active or last-successful mode easier
   to inspect during normal operation, not only after an error.

### State-machine and data-flow findings

1. The direct-chat flow is now better explained at the UI level.
   The link between runtime setup, permissions, tool readiness, and transcript
   persistence is covered more directly in tests.

2. The research flow is also in better shape.
   Launch, approval, resume, and summary return paths are now represented more
   continuously.

3. The remaining state-flow risk is breadth rather than a known major defect.
   More coverage is still warranted around session switching, draft
   preservation, research detail navigation, and longer-lived continuity across
   reloads.

## Coverage Status

The product is now backed by stronger integration-oriented coverage than it had
in the initial evaluation.

Current focused coverage includes:
- config-driven product checks in `tests/examples/test_assistant_config_examples.py`
- higher-level Streamlit product-journey checks in
  `tests/apps/test_streamlit_assistant.py`
- OpenAI-compatible fallback-path checks in
  `tests/llm_providers/test_openai_compatible.py`
- additional provider error-shaping checks in
  `tests/llm_providers/test_openai_compatible_extra.py`

The current focused regression slice passes:
- `tests/examples/test_assistant_config_examples.py`
- `tests/apps/test_streamlit_assistant.py`
- `tests/llm_providers/test_openai_compatible.py`
- `tests/llm_providers/test_openai_compatible_extra.py`

Result at the time of this update: `85 passed`

## Revised Ranked Fix List

1. Add browser-level validation when the environment allows it.
   The remaining highest-priority uncertainty is now the real visual and status
   experience, not the core app wiring.

2. Surface active provider compatibility mode during normal operation.
   Make it easier to inspect whether an endpoint is succeeding via tools, JSON,
   or markdown-JSON fallback without waiting for a full failure.

3. Continue unifying research status with the surrounding chat experience.
   Keep the durable harness model, but reduce the amount of status interpretation
   the user must do across separate surfaces.

4. Expand persistence-oriented product tests.
   Add more coverage around session switching, draft preservation, research
   detail navigation, interrupted turns, and resumed work after reload.

5. Validate the private-network operator path against a realistic internal
   OpenAI-compatible endpoint implementation.
   The fallback model is well covered in fake-client tests, but a realistic
   deployment check would better expose compatibility gaps before rollout.

## Bottom Line

The project is now meaningfully closer to a usable internal product. The most
important earlier UX issues have been addressed in code and tests. The next
phase should focus less on basic flow cleanup and more on observability,
end-to-end validation, and the remaining chat-versus-research integration edge
cases.

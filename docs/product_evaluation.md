# Product Evaluation: Existing Streamlit Assistant

## Purpose

This document evaluates the current product as shipped before making product
changes. The focus is the existing Streamlit assistant and its harness-backed
research flow, using the shipped example configs and the current OpenAI-
compatible provider fallback behavior.

The walkthroughs below were executed through the current repository's runtime,
state, and test harnesses rather than a full browser-automation stack. The goal
was to validate actual product behavior, data flow boundaries, and state
transitions without redesigning the system first.

## Walkthrough Matrix

### 1. Local-only chat with no workspace root

Config: `examples/assistant_configs/local-only-chat.yaml`

Prompt used:
`How does this assistant behave with chat only?`

Artifacts inspected:
- visible empty-state copy
- runtime/provider defaults
- transcript output
- persisted session record and session index

Observed behavior:
- The product can answer directly without a workspace.
- First launch creates a usable session automatically.
- The empty state makes it clear that the assistant can answer without tools.
- Session persistence works for a simple chat-only exchange.

User confusion points:
- The distinction between "chat only" and "workspace-enabled" is clear in copy,
  but not strongly reflected in the surrounding controls.
- The product still exposes multiple setup controls before the user has formed a
  first successful interaction.

### 2. Local-only chat with workspace root and filesystem enabled

Config: `examples/assistant_configs/local-only-chat.yaml`

Prompt used:
`Read plan.txt and summarize it.`

Artifacts inspected:
- sidebar runtime settings
- workspace-root selection
- permission toggles
- transcript output after a tool turn
- persisted session record after reload

Observed behavior:
- Selecting a workspace root and enabling filesystem access is enough to unlock
  a real `read_file` turn.
- The persisted session keeps both the runtime state and the transcript after a
  reload.
- The split between root selection and permission enablement is implemented
  correctly in code and persistence.

User confusion points:
- Selecting a workspace root does not enable filesystem access, which is safe
  but easy to misunderstand.
- Workspace scoping, filesystem enablement, and subprocess enablement are three
  separate concepts that the UI presents in one pass without a strong setup
  sequence.

### 3. Enterprise config with private-network endpoint and remote tools

Config: `examples/assistant_configs/enterprise-data-chat.yaml`

Prompt used:
`Search enterprise systems for release notes.`

Artifacts inspected:
- provider/base URL configuration
- enabled remote tool set
- tool exposure before and after network permission enablement
- transcript output for a direct-answer turn

Observed behavior:
- The config correctly selects the custom OpenAI-compatible provider path.
- Remote tools stay blocked until both credentials and network permission are
  available.
- Once credentials and the network gate are present, the expected Jira,
  Confluence, Bitbucket, and GitLab search/read tools are exposed to the model.

User confusion points:
- Remote-tool readiness depends on a compound condition: enabled tool,
  credentials present, and network access enabled.
- The product has the right safety model, but the operator path is not linear.
  A new user has to infer which prerequisite is missing.

### 4. Harness-backed research flow with approval and resume

Config: `examples/assistant_configs/harness-research-chat.yaml`

Prompt used:
`Read research-note.txt then summarize it.`

Artifacts inspected:
- research controller creation from runtime state
- `HarnessSessionService` launch flow
- approval-required intermediate inspection
- explicit approval resume
- completed session state

Observed behavior:
- Research launch correctly maps Streamlit runtime state into a harness session.
- Approval-required turns persist as pending sessions and can be resumed
  explicitly.
- Approved resumes complete the session and clear pending approval state.

User confusion points:
- Research behaves like a second product surface adjacent to chat rather than a
  deeper mode of the same assistant.
- The transition from chat state to research state is implicit in the sidebar.
- Approval waiting, resumable research, and completed research are represented
  accurately, but the user has to interpret multiple status surfaces.

## Top Findings

### UX failures

1. Setup is safe but not strongly sequenced.
   The product asks the user to reason about provider selection, base URL,
   workspace root, filesystem access, subprocess access, network access, and
   tool enablement as parallel concerns. The code enforces them correctly, but
   the UI does not clearly lead the user through them.

2. Workspace selection and permission enablement are easy to conflate.
   The system correctly treats them separately, but many users will assume that
   selecting a root enables local tools immediately.

3. Research feels like a separate subsystem.
   Durable research is powerful, but the product currently presents it as an
   adjacent control surface rather than a natural continuation of normal chat.

4. Tool readiness is precise but hard to read.
   Remote tools depend on credentials plus network permission, and local tools
   depend on root selection plus filesystem/subprocess permissions. The model is
   defensible; the presentation is not yet intuitive.

### Backend/provider failures

1. The OpenAI-compatible fallback logic is valuable but not product-visible
   enough.
   The provider layer can fall back across `TOOLS`, `JSON`, and `MD_JSON`, but a
   private-network operator still sees mostly error outcomes rather than a clear
   story about what was attempted and what failed.

2. Private-network provider setup is operationally correct but still operator-
   oriented.
   Custom endpoint support requires the right base URL, model, API-key path, and
   permission settings. These are all supported, but the product does not yet
   collapse them into a guided setup.

### State-machine and data-flow failures

1. Chat and research use different persistence and lifecycle models.
   That is architecturally reasonable, but product-wise it creates a split
   mental model.

2. Queued follow-up prompts and approval state live partly in transient UI turn
   state and partly in persisted transcript/session state.
   The implementation is coherent, but the user-facing behavior can feel
   inconsistent when a turn is interrupted or waiting.

3. Approval resolution is durable and fail-closed, which is correct.
   The product cost is that users need clearer cues about whether the assistant
   is blocked, waiting, resuming, or done.

## Test Gaps Revealed

Before this evaluation, the repository already had strong unit-level and
reducer-level coverage, but it lacked enough product-journey tests that:

- start from the shipped example configs
- exercise a full direct chat turn
- exercise a real local tool turn with persisted Streamlit state
- verify enterprise tool exposure under network and credential gating
- run the harness-backed research controller through approval and resume
- cover the provider failure path when all structured-output modes fail

## Ranked Fix List

1. Turn setup into a guided sequence.
   Group endpoint setup, workspace setup, permission enablement, and source
   enablement so the user is never guessing which prerequisite is missing.

2. Surface tool readiness in product language.
   Make it explicit whether a tool is unavailable because of missing workspace,
   missing credentials, missing network permission, or an approval gate.

3. Improve provider failure feedback.
   When all OpenAI-compatible fallback modes fail, show which modes were tried
   and whether the failure was schema/parse-related or transport-related.

4. Tighten the chat-to-research transition.
   Clarify when the user should stay in chat versus launch a durable research
   task, and preserve more continuity between those two surfaces.

5. Add more browser-level validation when the environment allows it.
   The current test harness is enough to expose state-flow and persistence
   issues, but a future pass should verify the exact visual/status experience in
   a real browser session.

## New Coverage Added In This Pass

The evaluation is backed by new automated coverage in:

- `tests/examples/test_assistant_config_examples.py`
- `tests/apps/test_streamlit_assistant.py`
- `tests/llm_providers/test_openai_compatible.py`

Those tests now cover the three shipped experiences plus the missing provider
all-modes-failed path.

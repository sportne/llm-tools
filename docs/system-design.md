# llm-tools System Design

## Purpose, Scope, And Truth Model

`llm-tools` is a typed Python library and application stack for defining,
validating, exposing, executing, and inspecting tools used by LLM-backed and
non-LLM workflows. The lower layers provide the reusable tool and model-turn
substrate. The upper layers add workflow execution, durable harness sessions,
local skills, assistant runtime assembly, a NiceGUI assistant app, and a harness
CLI.

This document describes the current implemented system. It is architectural,
not a source reference. It documents subsystem responsibilities, public
contracts, runtime flows, dependency direction, security posture, persistence
shape, LLM interaction design, built-in capabilities, and dependency purpose.
Exact class fields and function signatures remain owned by source code and
tests.

The documentation truth model is:

1. Tested implementation is the primary source of truth.
2. Public package exports and product entrypoints break ties when older docs
   lag.
3. ADRs explain selected decision rationale but do not structure this document.
4. Feature-gated capabilities are documented as implemented behavior, even when
   disabled by default.
5. Active backlog and dated review logs are intentionally excluded.

This document is written for maintainers, contributors, security reviewers, and
advanced integrators. End-user operation of the assistant app belongs in the
future Assistant User Guide.

## System Architecture

### Package Shape

The repository uses a `src` layout rooted at `src/llm_tools/`.

The practical layers are:

- `tool_api`
- `llm_adapters`
- `llm_providers`
- `tools`
- `skills_api`
- `workflow_api`
- `harness_api`
- `apps`

The supported library surfaces are `llm_tools.tool_api`,
`llm_tools.llm_adapters`, `llm_tools.llm_providers`, `llm_tools.tools`,
`llm_tools.skills_api`, `llm_tools.workflow_api`, and
`llm_tools.harness_api`. The supported product entrypoints are
`llm_tools.apps.assistant_app` / `llm-tools-assistant` and
`llm_tools.apps.harness_cli` / `llm-tools-harness`.

### Layering

The dependency direction is intentionally one-way:

```text
apps
  -> harness_api
  -> workflow_api
  -> skills_api
  -> llm_providers
  -> tool_api
  -> tools

harness_api
  -> workflow_api
  -> skills_api
  -> llm_adapters
  -> llm_providers
  -> tool_api

workflow_api
  -> skills_api
  -> llm_adapters
  -> tool_api

skills_api
  -> tool_api

llm_providers
  -> llm_adapters

tools
  -> tool_api

llm_adapters
  -> no higher internal layer

tool_api
  -> no higher internal layer
```

`tests/architecture/test_layering.py` enforces these rules. The important
constraints are that `tool_api` stays independent, `workflow_api` does not
import `harness_api`, `harness_api` does not import `apps`, `tools` do not
import orchestration or app layers, and `apps` compose lower layers without
becoming the default extension surface.

### Runtime Flow

The common execution path has four nested boundaries:

1. An app, harness service, script, or library consumer assembles a provider,
   tool registry, tool policy, context, and prompts.
2. `workflow_api` prepares a model-facing contract, obtains a parsed model turn,
   and routes tool invocations through `WorkflowExecutor`.
3. `tool_api.ToolRuntime` resolves the registered tool, evaluates policy,
   validates input, supplies mediated execution services, invokes the tool,
   validates output, redacts observability records, and returns a normalized
   `ToolResult`.
4. `harness_api`, when used, persists each orchestration turn as canonical
   `HarnessState`, classifies resume state, and controls durable task,
   approval, retry, verification, and stop behavior.

Assistant Chat uses the same workflow boundary for interactive chat turns.
Deep Task uses the same workflow boundary through a durable harness session. In
the assistant app, Deep Task is implemented but hidden by the default-disabled
`deep_task_mode_enabled` feature gate. This keeps model parsing and tool
execution shared while allowing the harness to own persistence and resume
semantics.

### Model And Tool Boundary

The model is never allowed to invoke Python callables directly. It can only
emit a provider response that is parsed into a canonical `ParsedModelResponse`
and then executed through `WorkflowExecutor` and `ToolRuntime`. Tool exposure is
separate from tool registration: a registered tool may be hidden from the model
because feature flags, credentials, workspace readiness, session permissions,
side-effect policy, or approval requirements make it unavailable.

## Subsystem Designs

### `tool_api`

`tool_api` is the typed execution substrate. It owns:

- `ToolSpec`, `ToolInvocationRequest`, `ToolResult`, `ToolError`,
  `ExecutionRecord`, policy models, source provenance models, and redaction
  models.
- `Tool`, the typed implementation base.
- `ToolRegistry`, which registers and resolves tools by name.
- `ToolPolicy`, which gates exposure and execution by tool name, side-effect
  class, required services, required secrets, and approval rules.
- `ToolRuntime`, which mediates every invocation.
- `ToolExecutionContext`, `ExecutionServices`, and mediated service brokers for
  filesystem, subprocess, and remote gateways.

The runtime invocation sequence is deliberately strict:

1. Resolve the tool by name.
2. Evaluate policy against the `ToolSpec` and host context.
3. Reject denied calls before validation or execution.
4. Validate input with the tool's Pydantic input model.
5. Redact input for execution records unless policy permits retaining
   unredacted input.
6. Build a scoped `ToolExecutionContext` containing only permitted services and
   scoped secrets required by the tool.
7. Invoke the sync or async implementation under timeout handling.
8. Validate output with the tool's Pydantic output model.
9. Redact output for execution records unless policy permits retaining
   unredacted output.
10. Return a normalized success or failure `ToolResult`.

`ToolRuntime` wires execution services centrally. Tools receive mediated
brokers; they should not construct host filesystem access, subprocess access,
or remote-service clients for themselves. This design keeps policy enforcement
and credentials at the runtime boundary rather than scattered through tool
implementations.

#### Tool Abstractions

The core tool abstractions are intentionally separate:

- `ToolSpec` describes the model-visible and policy-visible contract: name,
  version, description, tags, side-effect class, required capabilities,
  required secrets, timeout, output-retention preference, and cache-write
  behavior.
- `Tool` binds a `ToolSpec` to Pydantic input/output models and a sync or async
  implementation.
- `ToolRegistry` stores named tool bindings and resolves invocations by tool
  name.
- `ToolPolicy` evaluates whether a tool may be exposed or executed in a
  particular host context.
- `ToolRuntime` is the only supported path for invoking registered tools.

This separation is security-relevant. A tool can be installed in the registry
without being exposed to a model; a tool can be exposed for model selection but
still require approval at execution time; and an approved invocation still runs
through input validation, scoped services, output validation, and redaction.

#### Tool Policy Evaluation Algorithm

Policy evaluation is deterministic and stops at the first denial or approval
requirement:

1. Resolve the tool name and tags from the `ToolSpec`.
2. Deny if the tool name is explicitly denied.
3. Deny if an allowlist exists and the tool name is not on it.
4. Deny if any tool tag is explicitly denied.
5. Deny if an allowed-tag set exists and the tool has no matching allowed tag.
6. Deny if the tool's side-effect class is not allowed.
7. Deny if the tool requires filesystem, subprocess, or network access and the
   policy does not allow that capability.
8. Deny internal workspace cache writes unless policy permits them.
9. Deny if any required secret is absent from the host context.
10. If the side-effect class is configured to require approval, return an
    approval-required decision.
11. Otherwise allow the tool.

The same policy verdicts are used when preparing model-visible tool schemas and
when executing an invocation. This prevents a mismatch where the model sees a
tool that the runtime would categorically deny, unless the caller deliberately
asks to include approval-required tools so the model can request an operation
that pauses for approval.

#### Tool Runtime Execution Algorithm

`ToolRuntime` normalizes every invocation through the same sequence:

1. Resolve the tool from the registry.
2. Evaluate policy before input validation or execution.
3. If policy denies, return a normalized `ToolResult` with a policy error.
4. Validate model-supplied arguments with the tool's Pydantic input model.
5. Redact the validated input for execution records unless the redaction policy
   allows retaining unredacted input.
6. Build a runtime-issued `ToolExecutionContext` containing only scoped secrets
   and mediated services required by the tool.
7. Issue an opaque execution permit and invoke the sync or async tool
   implementation under the tool timeout.
8. Validate raw tool output with the tool's Pydantic output model.
9. Redact output, logs, artifacts, and error details according to redaction
   policy and tool output-retention settings.
10. Capture logs, artifacts, and source provenance emitted through the
    execution context.
11. Return a normalized `ToolResult` with an embedded `ExecutionRecord`.

Execution failures are shaped into stable error codes such as tool-not-found,
policy-denied, input-validation, timeout, execution-failed, and
output-validation failures. Network-dependent timeout failures can be marked
retryable. Successful and failed invocations both produce execution records so
audit and inspector surfaces can explain what happened without needing raw
provider payloads.

#### Scoped Execution Services

Tools do not receive the host process environment or raw filesystem/subprocess
functions. The runtime constructs `ExecutionServices` based on the tool spec
and current policy:

- Filesystem access is provided through a workspace-aware `FilesystemBroker`.
- Subprocess access is provided through a workspace-confined
  `SubprocessBroker`.
- GitLab, Jira, Bitbucket, and Confluence access is provided through
  runtime-built gateways using only the secrets named by the tool spec.
- Secrets are exposed through a read-only `SecretView` containing only granted
  secret keys.

If a tool asks for a service it was not granted, the broker raises a stable
runtime error. This makes capability scoping enforceable even when a tool
implementation is incorrect or incomplete.

### `llm_adapters`

`llm_adapters` normalizes model output into canonical internal responses.
`ActionEnvelopeAdapter` builds structured response models from exposed tool
specs and parses provider payloads into `ParsedModelResponse`. The prompt-tool
adapter parses fenced text protocols for endpoints that cannot return native
tools or structured JSON.

The adapter layer does not call providers and does not execute tools. It is the
translation boundary between provider-shaped output and workflow-shaped parsed
turns.

#### Action Envelope Contract

`ActionEnvelopeAdapter` builds dynamic Pydantic response models from the tool
specs currently exposed for one model turn. In native structured mode, the
response model contains either:

- a list of actions whose `tool_name` values are restricted to exposed tools and
  whose `arguments` are validated against the selected tool's input model, or
- a final response shaped by the caller-provided final-response model.

When no tools are exposed, the generated response model admits only final
responses. In staged structured mode, the same adapter builds narrower models:
a decision model, a fixed-tool invocation model, a final-response model, or a
single-action model. This lets the model-turn protocol ask for the smallest
valid schema at each stage.

The adapter's output is always `ParsedModelResponse`: a final response, a list
of canonical `ToolInvocationRequest` values, or both when a provider returns
that shape. The adapter does not judge whether an invocation is allowed. It only
normalizes provider output into the workflow contract.

#### Prompt-Tool Parsing Contract

`PromptToolAdapter` supports plain text providers by rendering a fenced-block
protocol. It can ask the model to choose a category, choose a tool, provide
arguments for a fixed tool, or produce a final response. Parser output still
becomes `ParsedModelResponse`, and any tool arguments still go through
Pydantic validation in the workflow/runtime path.

Prompt-tool parsing is intentionally stricter than natural-language parsing:
the model must produce the expected fenced block and field markers. Malformed
prompt-tool output is a protocol parse failure, not an invitation to guess tool
arguments from prose.

#### Adapter Trust Boundary

Adapters handle untrusted model/provider output. They validate shape, normalize
payloads from provider SDK objects or JSON strings, and reject unknown or
malformed action envelopes. They do not provide authorization, do not grant
services, do not access secrets, and do not call tool implementations.
Authorization and side effects start only after `WorkflowExecutor` passes a
canonical invocation to `ToolRuntime`.

### `llm_providers`

`llm_providers` owns model-service transport. Implemented provider protocols
are:

- OpenAI API compatible transport, backed by `openai` and `instructor`.
- Native Ollama transport, backed by the `ollama` Python package.
- Native Ask Sage transport, backed by direct HTTPS JSON requests.

In the assistant app, the native Ollama and native Ask Sage protocol choices
and presets are implemented but hidden by default. They are exposed only when
the administrator enables `ollama_native_provider_enabled` or
`ask_sage_native_provider_enabled`, respectively. OpenAI-compatible transport
is not hidden by one of these native-provider feature gates.

Provider connections are split into protocol, endpoint/auth settings, request
settings, response mode strategy, and selected model. A provider connection may
exist without a selected model, but executable model turns require one.

`ResponseModeStrategy` values are:

- `tools`: native tool/function calling where supported.
- `json`: native structured JSON/schema responses.
- `prompt_tools`: text responses parsed through the prompt-tool protocol.
- `auto`: try the protocol's preferred structured path and fall back where the
  provider reports a repairable capability or parse failure.

OpenAI API `auto` prefers native tool/schema paths and then prompt tools.
Native Ollama `auto` can use native tools, JSON-schema structured output, then
prompt tools. Native Ask Sage supports JSON and prompt tools, but not native
tools mode.

Providers expose model discovery and preflight behavior where supported.
Discovery belongs to a provider connection identity and does not create a
selected model by itself.

#### Provider Connection Model

Provider configuration is split so that security and capability decisions are
explicit:

- **Protocol** identifies the transport family: OpenAI-compatible, native
  Ollama, or native Ask Sage.
- **Connection settings** describe endpoint and authentication shape.
- **Request settings** carry provider-specific non-secret controls such as
  timeout-like or provider-native request parameters.
- **Response mode strategy** chooses native tools, structured JSON,
  prompt-tools, or `auto`.
- **Selected model** identifies the concrete model for executable turns.

Provider presets may populate non-secret connection fields and selected model
defaults, but provider credentials are not part of persisted presets. The
assistant supplies provider secrets from session memory when building the
runtime provider. Lower-level library users may construct providers directly
with credentials supplied by their own host environment.

#### Provider Support Matrix

| Protocol | Transport | Auth Shape | Discovery / Preflight | Response Modes | Assistant Visibility |
| --- | --- | --- | --- | --- | --- |
| OpenAI-compatible | `openai` client plus Instructor/native schema helpers | No secret or bearer-style API key depending on endpoint settings | Lists models through the compatible models endpoint when available; preflights selected mode or resolved `auto` mode. | Native tools/schema where supported, JSON/schema, prompt tools, and `auto` fallback. | Visible by default. |
| Native Ollama | `ollama` Python client against Ollama chat/list APIs | Usually no secret; host URL identifies the local or remote Ollama service. | Lists models from the Ollama host and preflights native tools, JSON, prompt tools, or resolved `auto`. | Native Ollama tools, JSON-schema structured output, prompt tools, and `auto` fallback. | Hidden by default behind `ollama_native_provider_enabled`. |
| Native Ask Sage | Direct HTTPS JSON requests to Ask Sage server endpoints | `x-access-tokens` style access token. | Lists models through `/get-models` and preflights JSON or prompt-tool behavior. | JSON and prompt tools; native tools are rejected for this protocol. | Hidden by default behind `ask_sage_native_provider_enabled`. |

The support matrix is product- and protocol-level behavior. Individual
endpoints can still fail preflight because a model rejects a mode, an endpoint
lacks model listing, credentials are wrong, or the server returns incompatible
payloads.

#### Provider Preflight Algorithm

Preflight is an advisory readiness probe for a configured provider connection:

1. Try model discovery when the protocol supports it.
2. Record whether connection and model-listing succeeded.
3. For `auto`, attempt the protocol's preferred structured mode and resolve the
   effective mode from the successful call.
4. If a structured attempt fails with a retryable mode/capability failure and
   the protocol permits prompt-tool fallback, try the prompt-tool probe.
5. For explicit modes, run only the selected mode and report whether it worked.
6. Return a typed result containing connection status, model acceptance,
   selected-mode support, discovered model ids, resolved mode, actionable
   message, and a redacted error summary.

Preflight does not persist credentials or create a selected model by itself. It
is used by app and CLI surfaces to make provider readiness visible before a
workflow turn depends on the configuration.

#### Provider Trust Boundaries And Controls

Provider calls cross the boundary from local trusted code into an external or
local model service. The main trust inputs are prompt messages, tool schemas,
final-response schemas, provider request settings, credentials, and remote
provider responses. Provider implementations normalize returned payloads through
Pydantic models and `ActionEnvelopeAdapter` before the workflow layer executes
anything.

The provider layer does not execute tools, does not decide tool policy, and does
not persist raw provider responses. Response-mode fallback is explicit and
bounded by provider capability hooks. When a provider returns malformed JSON or
schema-incompatible structured output, the failure is treated as a validation or
mode-support error rather than as partially trusted data.

### `skills_api`

`skills_api` provides reusable support for local `SKILL.md` instruction
packages. It owns:

- bounded local discovery under caller-provided skill roots
- parsing and validation of required `name` and `description` frontmatter
- scope-aware metadata records
- deterministic resolution by name or explicit path
- enablement filtering supplied by callers
- loading selected skill instructions
- rendering available-skill and loaded-skill model context
- skill usage records

Skill scopes are `enterprise`, `user`, `project`, and `bundled`, with plain-name
resolution precedence in that order. Same-name skills at the same scope are
ambiguous unless an explicit path is supplied.

The skills API does not execute scripts, install dependencies, persist
enablement, own UI, or perform heuristic skill selection. Those decisions belong
to workflow, harness, or app consumers.

Assistant integration for local and bundled skills is implemented but hidden by
the default-disabled `skills_enabled` feature gate. When that gate is off, the
library skill APIs remain available to direct consumers, but the assistant does
not discover skills for session context or expose explicit `$skill` invocation.

#### Skill Discovery Algorithm

Skill discovery is local-filesystem metadata discovery, not code execution:

1. Resolve each caller-provided skill root to a canonical local path.
2. Skip roots that do not exist or are not directories.
3. Walk directories with bounded depth and bounded directory count.
4. Skip symlinked directories unless the caller explicitly enables symlink
   following.
5. Skip hidden directories by default.
6. Treat files named `SKILL.md` as candidate skills.
7. Read only the YAML front matter needed for metadata.
8. Require single-line `name` and `description` fields within configured
   lengths.
9. Accept only skill names made from letters, digits, `_`, `-`, `.`, and `:`.
10. Record invalid skill files as discovery errors rather than importing,
    executing, or partially trusting them.

Discovered skill metadata is sorted by scope precedence, name, and path so the
model-visible available-skill list is deterministic.

#### Skill Resolution And Prompt Context Algorithm

Skill resolution supports explicit path or name invocation. Path resolution
matches a discovered canonical `SKILL.md` path. Name resolution applies
enablement filtering, then chooses the highest-precedence scope:
`enterprise`, `user`, `project`, then `bundled`. If more than one enabled skill
with the same name exists at the same best scope, resolution fails as
ambiguous. Disabled skills fail unless the caller explicitly asks to allow
disabled resolution.

Assistant skill context is built only when `skills_enabled` is on:

1. Build roots from the selected workspace as `project` scope plus packaged
   bundled skills as `bundled` scope.
2. Discover metadata under those roots.
3. Apply session disabled-name and disabled-path settings.
4. Render a bounded available-skills context containing names, descriptions,
   and file paths.
5. Parse explicit `$skill` names from the user message.
6. Resolve each requested skill through the deterministic resolver.
7. Load only the selected skill's `SKILL.md` contents.
8. Add a loaded-skill context envelope to the prompt and record skill usage
   metadata, including a content hash.

The skills API does not do semantic auto-selection. A model can see available
skill metadata and may decide to use a skill according to prompt instructions,
but the assistant only loads skill bodies for explicit `$skill` invocations in
the current implementation.

#### Skills Trust Model

Skills are prompt instructions from local files. They can influence model
behavior, but they do not grant capabilities. Loading a skill does not enable a
tool, bypass feature gates, provide filesystem/network/subprocess permission,
inject credentials, override approval policy, or change runtime protection.

The main risks are prompt-injection through local skill content, stale or
misleading instructions, and accidental exposure of local paths in
model-visible skill metadata. Controls are bounded discovery, metadata
validation, deterministic resolution, feature-gated assistant integration,
explicit invocation, budgeted rendering, and lower-layer enforcement of all
tool and credential boundaries.

### `workflow_api`

`workflow_api` is the reusable one-turn workflow layer above tools and
providers. It owns:

- `WorkflowExecutor`, which prepares model-facing tool contracts and executes
  parsed responses through `ToolRuntime`.
- `WorkflowTurnResult`, invocation outcomes, and transient approval requests.
- `ModelTurnProtocolRunner`, which obtains one parsed model turn through native,
  staged JSON, prompt-tool, or protection-mediated paths.
- interactive chat state, compaction, inspector, and protection helpers used by
  the assistant app.

The one-turn contract is the key reusable boundary:

1. Build provider messages and a prepared model interaction.
2. Run the model-turn protocol.
3. Parse exactly one model turn.
4. Execute zero or more parsed tool invocations through the runtime.
5. Return a `WorkflowTurnResult`.

Workflow approvals are transient unless a harness session persists them as a
pending approval. Workflow-level protection may evaluate prompt content before
the provider call and final response content before storage or display.
Protection decisions can allow, challenge, sanitize, or block content.
In the assistant app this protection path is implemented but inactive by
default unless `information_protection_enabled` and runtime protection
configuration both allow controller assembly.

#### Prepared Model Interaction Algorithm

Before a provider call, `WorkflowExecutor` prepares the model-facing contract:

1. Read registered tools from `ToolRegistry`.
2. If a host context is available, evaluate each tool through `ToolPolicy`.
3. Exclude denied tools from the model-facing schema.
4. Include approval-required tools only when the caller asks to expose tools
   that may pause for approval.
5. Collect the remaining `ToolSpec` values and input models.
6. Ask `ActionEnvelopeAdapter` to build a response model and JSON schema for
   the exposed tools and final-response model.
7. Return the prepared schema, tool names, specs, input models, and response
   model as a `PreparedModelInteraction`.

This is the point where registration becomes exposure. A registered tool that
is hidden by policy, feature gates, missing credentials, or workspace readiness
does not appear in the provider-facing schema.

#### Parsed Response Execution Algorithm

After the model-turn protocol returns a `ParsedModelResponse`,
`WorkflowExecutor` converts it into workflow outcomes:

1. If the parsed response contains a final response, return a final-response
   `WorkflowTurnResult` without tool execution.
2. Otherwise execute parsed tool invocations in order.
3. For each invocation, create a per-invocation `ToolContext` with a stable
   invocation id, workspace, environment, metadata, logs, and artifacts.
4. Inspect policy through `ToolRuntime.inspect_invocation`.
5. If the invocation requires approval and no approval override applies, build
   an `ApprovalRequest`, store pending workflow approval state, append an
   `APPROVAL_REQUESTED` outcome, and stop executing later invocations from that
   parsed response.
6. Otherwise execute the invocation through `ToolRuntime`.
7. Append an `EXECUTED` outcome containing the normalized `ToolResult`.
8. Return a `WorkflowTurnResult` containing the parsed response and ordered
   invocation outcomes.

The sequence stops at the first approval requirement. Later tool calls from the
same parsed model response do not run while a previous invocation is waiting
for user or harness approval.

#### Workflow Approval Semantics

Workflow approvals are in-memory by default. A pending approval records the
approval request, a deep copy of the parsed response, the base tool context, the
blocked invocation index, and an expiration time. Approving the request resumes
execution from the blocked index with an approval override for that invocation
only. Denial, cancellation, or expiration returns a workflow result containing
only the corresponding approval outcome.

Harness sessions make this same approval pause durable by storing a
`PendingApprovalRecord` next to the incomplete harness turn. Assistant Chat
uses transient workflow approval state; Deep Task uses the harness-backed
durable path.

#### Workflow Failure And Observability Shape

Workflow execution does not raise ordinary tool failures as raw exceptions to
the app. Tool lookup failures, policy denials, validation errors, timeouts,
execution failures, and output-validation failures are normalized into
`ToolResult` values by `ToolRuntime`, then wrapped in ordered
`WorkflowInvocationOutcome` records.

Inspector and workbench surfaces are derived from these normalized workflow
events and execution records. They can show provider messages, parsed
responses, tool outcomes, and approval/protection state without bypassing
redaction or requiring direct access to model provider payloads.

### `harness_api`

`harness_api` is the durable orchestration layer. It owns:

- canonical persisted `HarnessState`
- `HarnessSession`, `HarnessTurn`, `TaskRecord`, pending approvals, budget
  policy, verification records, no-progress signals, and turn decisions
- state stores and schema-version handling
- resume classification
- durable approval pauses
- deterministic planning and selected-task control
- retry, stop, verification, replay, inspect, list, and session-service flows

`HarnessState` is the durable source of truth. Stored summaries, traces, replay
views, and workbench projections are derived observability artifacts. A valid
state has exactly one root user-requested task, contiguous turn indices, at most
one incomplete tail turn, and at most one active pending approval.

Harness execution checkpoints an incomplete turn before provider or tool
execution begins. If execution completes, the completed turn, decision, and
state changes are persisted. If a policy-gated invocation requires approval,
the harness persists one incomplete tail turn plus one scrubbed
`PendingApprovalRecord`. If a process crashes after the checkpoint and before
completion, resume classifies the tail as interrupted and fails closed by
default.

Harness state is provider-neutral. The harness may embed workflow results, but
provider transport details and app presentation state are not the durable
contract.

#### Harness State Model

The harness stores durable work as canonical `HarnessState`. The core state
objects are:

- `HarnessSession`: session identity, current turn index, budget policy,
  start/end timestamps, and stop reason.
- `TaskRecord`: durable task graph node with lifecycle status, parent and
  dependency links, retry count, verification expectations, verification
  outcome, artifacts, and status summary.
- `HarnessTurn`: one persisted orchestration turn, including selected task ids,
  workflow result, approval audit metadata, verification snapshot, turn
  decision, no-progress signals, and timestamps.
- `PendingApprovalRecord`: the exact parsed response, approval request,
  sanitized base context, and invocation index needed to resume a paused tool
  invocation.

State validation enforces invariants such as unique task ids, valid dependency
relationships, valid task verification state, contiguous turn indexes, one
incomplete tail turn at most, and matching pending approval metadata. These
invariants make stored state inspectable and replayable without trusting
derived summaries.

#### Harness Session Lifecycle Algorithm

The public harness service is a session-level API around the executor:

1. `create_session` creates a root user-requested task, assigns schema version,
   initializes budget policy, and persists the first canonical state snapshot.
2. `run_session` loads the snapshot, checks the expected revision, and enters
   the executor loop.
3. The executor classifies persisted state through resume inspection before
   doing any work.
4. Terminal, corrupt, or incompatible-schema states stop immediately.
5. Waiting approvals require an explicit approval resolution.
6. Interrupted non-approval turns stop by default unless the caller explicitly
   allows interrupted-turn replay.
7. Runnable states execute one turn at a time until the applier returns a stop
   decision, budget/no-progress controls stop execution, or approval is needed.
8. `inspect_session` reconstructs canonical artifacts and optionally replay
   text from stored state.
9. `stop_session` records an operator stop decision, clears pending approvals,
   and persists the terminal session state.

This design makes resume classification part of every run, not a separate
maintenance operation. A stored snapshot has to be judged safe to continue
before a provider or tool call can happen.

#### Harness Turn Checkpoint And Commit Algorithm

Each runnable turn follows a strict checkpoint/commit sequence:

1. Check pre-turn budgets such as maximum turns, tool invocations, or elapsed
   time.
2. Ask the driver for deterministic selected task ids.
3. If no task is selected and active tasks remain, stop with a no-progress
   signal.
4. Build the tool context for the selected tasks.
5. Append an incomplete `HarnessTurn` with selected tasks and verification
   status.
6. Persist that incomplete turn before calling the provider.
7. Run the provider through the driver to obtain a parsed model response.
8. Reject the turn if executing the parsed response would exceed remaining tool
   budget.
9. Execute the parsed response through `WorkflowExecutor`.
10. If approval is requested, persist the incomplete turn, approval audit
    record, and one sanitized `PendingApprovalRecord`, then stop.
11. If no approval is needed, apply the completed workflow result to harness
    state through the applier.
12. Check post-turn retry exhaustion, budget exhaustion, and no-progress
    signals.
13. Persist the completed turn, turn decision, updated state, and derived
    artifacts.

The incomplete-turn checkpoint is the crash boundary. After a crash, resume can
distinguish a safe runnable state from an interrupted provider/tool turn that
requires operator review.

#### Resume, Approval, Retry, And Stop Behavior

Resume classification can return runnable, terminal, waiting for approval,
approval expired, interrupted, corrupt, or incompatible schema. Waiting
approval is valid only when exactly one pending approval matches the incomplete
tail turn's parsed response, workflow outcome, approval id, and invocation
index. Missing, dangling, mismatched, or multiple pending approvals are corrupt
state.

Approvals are durable. A pending approval stores enough sanitized context to
resume the blocked invocation without re-asking the model. Denial, expiration,
cancelation, and operator stop clear pending approvals and produce terminal or
stopped state according to the resolved action.

Provider failures and retryable tool errors use bounded retry budgets. Retry
attempts increment task retry counts. Exhausted provider retries stop with
no-progress; exhausted retryable tool errors stop with error. Budget exhaustion
and repeated non-terminal turns without progress force stop decisions even if
the model would otherwise continue.

#### Replay, Summaries, And Verification

Harness replay and summaries are derived artifacts. They can be rebuilt from
canonical state and are not the contract for resuming execution. Turn traces
capture selected tasks, workflow outcomes, decisions, and sanitized context
needed for inspection.

Verification is represented on tasks as expectations and outcomes. Turn records
snapshot verification status for selected tasks so an operator can see what the
harness believed at the time of each turn. Required verification expectations
must pass before a task with those expectations can be marked completed.

#### Assistant Deep Task Wrapper

Assistant Deep Task wraps the harness service with app-local runtime assembly.
The assistant supplies the same tool registry, policy, provider, in-memory
credentials, environment overrides, workspace, and optional protection
controller that a chat turn would use. The harness state store is scoped under
the owning chat session, and pending harness approvals are restored into
transient UI state on startup.

Deep Task remains hidden by default behind `deep_task_mode_enabled`. Enabling
the feature exposes the assistant UI path; it does not bypass harness resume
classification, approval durability, budget checks, or tool runtime mediation.

### `apps`

`apps` contains supported product entrypoints and app-local composition.

The assistant app is a NiceGUI browser application. Its architectural role is
composition: it translates user-facing session state into lower-layer provider,
workflow, tool, skills, protection, and harness inputs. It owns product state
and presentation, but tool execution, model-turn parsing, protection decisions,
and harness state transitions remain mediated by lower layers.

The assistant app includes:

- local username/password authentication by default
- encrypted SQLCipher persistence
- per-user chat ownership and preferences
- persistent and temporary chat sessions
- provider configuration, model discovery, and preflight
- session-scoped tool permissions and approvals
- in-memory provider and tool credentials
- background chat turn execution with queued UI events
- workbench/inspector records
- admin settings, branding, and feature flags
- feature-gated Deep Task mode backed by `harness_api`, hidden by default behind
  `deep_task_mode_enabled`

#### Assistant App Runtime Assembly Algorithm

`AssistantRuntimeBundle` is the app-layer assembly boundary shared by Assistant
Chat and Deep Task. For each turn, runtime assembly:

1. Merge source-controlled assistant config with mutable session runtime state.
2. Create the selected provider from protocol, connection settings, request
   settings, selected model, response-mode strategy, timeout, and in-memory
   credential input.
3. Build the complete assistant tool registry, then filter tool specs by
   administrator feature gates.
4. Resolve the workspace root and intersect user-enabled tools with feature-
   visible tool specs.
5. Filter exposed tools again by runtime readiness: workspace availability,
   filesystem/subprocess/network permissions, required secrets, and tool policy.
6. Build `ToolPolicy`, `ToolRegistry`, and `WorkflowExecutor`.
7. If `skills_enabled` is on, discover project and bundled skills, render the
   available-skill context, resolve explicit `$skill` invocations, and append
   loaded skill text to prompts.
8. Build separate Chat and Deep Task system prompts from the same registry,
   limits, workspace state, protocol capabilities, and skill context.
9. If information protection is enabled and configured, build separate
   protection controllers for Chat and Deep Task with app-specific environment
   names.
10. Return a bundle that can construct either a chat runner or a harness service
    without recomputing product state inside lower layers.

The assistant runtime never enables a model-facing tool merely because it is
registered. Registration, feature visibility, user enablement, session
permission, credential availability, side-effect policy, and approval rules are
separate gates.

#### Assistant Chat Flow

Assistant Chat keeps a `ChatSessionState` plus a NiceGUI transcript. When a
user submits a prompt, the controller appends the user transcript entry, creates
a per-turn event queue, builds an `AssistantRuntimeBundle`, and constructs a
`ChatSessionTurnRunner`. The runner projects the stored transcript into
provider-visible messages, applies context compaction when needed, runs the
model-turn protocol, executes any parsed tool calls through `WorkflowExecutor`,
and eventually records a structured final response or an approval/protection
pause.

The UI does not mutate workflow internals directly. Background turn execution
emits queued events such as status, inspector payloads, tool activity,
protection challenges, approvals, and completion. The controller drains those
events back into session state, transcript entries, workbench items, and
persisted records. This keeps long-running provider/tool work off the UI event
path while preserving a single controller-owned state transition point.

Temporary chats use the same runtime and workflow path but are not written as
durable chat sessions. Persistent chats are saved with transcript,
workflow-session state, token usage, inspector state, runtime settings, and
workbench items. Provider and remote-tool credentials remain session-memory
values and are not written with either temporary or persistent chats.

#### Deep Task Integration

Deep Task is the assistant-facing use of `harness_api`; it is implemented but
hidden by default behind `deep_task_mode_enabled`. When enabled, the assistant
uses the same runtime bundle and tool policy as Chat, then creates a
`HarnessSessionService` backed by the chat-scoped harness store. The harness
provider receives the Deep Task system prompt, selected model provider,
temperature, and optional Deep Task protection controller.

Deep Task persists canonical harness state under the owning chat session. The
assistant restores pending harness approvals on startup by resuming stored
harness snapshots and projecting a waiting approval into transient UI state.
Approval decisions then resume the harness service rather than replaying an
unmediated tool call.

#### Workbench And Inspector Design

The workbench is a persisted presentation surface for inspection artifacts and
future result/artifact views. Inspector events store sanitized provider
messages, parsed responses, and tool execution summaries. They are derived
observability payloads, not durable workflow contracts. Workbench item payloads
are encrypted at rest with the same per-user field encryption used for chats.

Inspector payloads are intentionally separated from transcript text and final
answers. This makes debugging and security review possible without changing the
provider-visible conversation model, and lets protection or redaction controls
scrub sensitive workflow details before they become persisted artifacts.

#### Admin Settings, Hosted Mode, And Feature Gates

Administrator settings own global feature gates, branding, and hosted-mode
controls. Feature gates are evaluated when runtime choices are loaded and again
when a turn is assembled, so a disabled gate removes hidden providers or tools
from both stored session state and model-visible exposure.

Hosted mode uses local users with admin-created accounts and server-side
browser sessions. The default auth mode is local username/password. Hosted
secret entry is disabled when the app is reachable over non-loopback HTTP
unless the operator passes explicit insecure-hosted-secrets risk acceptance.
TLS termination is normally expected outside the app through a reverse proxy,
though direct certificate files are supported for bootstrap and self-signed
testing.

Several assistant capabilities are implemented but not visible in the default
product configuration: information protection, local skills, native Ollama,
native Ask Sage, `write_file`, Atlassian tools, GitLab tools, and Deep Task.
Their gates are listed in [Feature-Gated Capabilities](#feature-gated-capabilities).

The harness CLI is a product entrypoint for durable harness sessions outside
the NiceGUI app. It composes the harness layer with provider and workflow
primitives without taking on assistant UI concerns.

## Cross-Cutting Designs

### LLM Interaction Design

The model-turn protocol exists to keep provider quirks out of workflow and app
code. Callers provide messages, a prepared interaction, a final-response model,
temperature, optional protection context, and a provider. The runner chooses or
honors the applicable protocol:

- Native provider path: use provider-native structured output or tools.
- Staged JSON path: ask for one action-selection or final-response step, then
  validate the selected tool's arguments with a narrower schema.
- Prompt-tool path: parse fenced text tool requests for plain-text endpoints.
- Protection path: mediate prompt and response content through workflow-level
  protection. In the assistant app, this path is default-hidden behind
  `information_protection_enabled`.

Repair attempts are limited. Model-turn events are redacted and describe
progress such as stage start, repair, fallback, provider completion, protection
action, and parsed response. Raw provider payloads are not part of the event
contract.

Interactive chat preserves the full transcript for UI purposes while projecting
a bounded provider-visible context. Context compaction summarizes older
completed turns when token limits are exceeded or when a provider rejects a
request as too large.

#### Model-Turn Protocol Selection Algorithm

One workflow turn produces exactly one parsed model response: either a final
response or one or more tool invocations. Protocol selection follows provider
capabilities rather than product branding:

1. Apply prompt-side protection when a protection controller is present. A block
   or challenge can return a parsed final response without calling the provider.
2. If the provider declares prompt-tool protocol preference, run the prompt-tool
   path.
3. Else if the provider declares staged-schema protocol preference, run the
   staged structured path.
4. Else run the native provider path.
5. If the chosen native or staged path raises a provider-declared fallback
   error, emit a redacted fallback event and retry through prompt tools.
6. Apply response-side protection to the parsed final response when configured.
7. Emit one redacted parsed-response event.

The runner has synchronous and asynchronous variants with the same decision
shape. The protocol boundary is provider-neutral: providers advertise callable
surfaces such as native structured output, structured JSON, or text responses,
and the runner composes the appropriate adapter and response model.

#### Native Structured Path

The native path is used when the provider can accept model-facing schemas or
tool/function definitions directly. For action turns, the runner asks the
provider to return the prepared interaction response model built by
`ActionEnvelopeAdapter`; the adapter then normalizes provider output into a
canonical `ParsedModelResponse`.

Forced finalization is a special native mode used when the workflow needs a
final answer rather than another tool decision. The runner first tries provider
structured output against the final-response model. If that fails with a
repairable structured-output error, it can fall back to text finalization when
the provider supports text responses. Text finalization is wrapped in a safe
final-response payload instead of being treated as trusted structured data.

Native fallback to prompt tools is allowed only when the provider reports the
exception as repairable for prompt-tool fallback. This keeps capability
downgrades explicit and reviewable rather than silently changing interaction
contracts for arbitrary provider failures.

#### Staged Structured JSON Path

The staged path exists for providers that can return structured JSON but do not
support provider-native tool calling. It narrows each provider request to the
minimum schema needed for that stage.

The default staged algorithm is:

1. Ask for a decision stage that chooses either `finalize` or one exposed tool.
2. If the decision is `finalize`, ask for a final-response payload using the
   final-response schema.
3. If the decision selects a tool, validate the selected tool name against the
   prepared interaction.
4. Build a tool-specific argument schema from that tool's Pydantic input model.
5. Ask for one tool-invocation payload and parse it into the canonical
   invocation shape.

A single-action staged strategy can collapse decision and tool/final-response
selection into one structured step when configured by provider preference or
runtime strategy. Both staged strategies run bounded repair attempts: when a
provider payload fails with a repairable validation error, the runner appends
repair guidance and the invalid payload summary, then retries the same stage.
Non-repairable errors and exhausted repair attempts fail the turn.

#### Prompt-Tool Path

The prompt-tool path supports plain-text endpoints. The runner renders a text
protocol that describes available tools, expected fenced response forms, and
final-answer behavior. The provider returns text, and `PromptToolAdapter`
parses the response into either a final response or tool invocation.

Prompt tools are less strongly typed at the provider boundary than native or
staged structured output, but they still converge on the same canonical
`ParsedModelResponse`. Tool input validation, policy, approvals, execution
service scoping, and output validation still happen after parsing through
`WorkflowExecutor` and `ToolRuntime`.

#### Event Redaction And Observability

Model-turn protocol events are progress records, not raw transcripts. Events
include stage starts, repair attempts, fallback decisions, provider completion,
protection actions, and parsed-response summaries. Error summaries and payloads
are redacted through the configured redaction policy before entering inspector
or workbench surfaces.

This design gives security reviewers enough observability to understand which
protocol and fallback path ran without persisting raw provider payloads as a
side channel around transcript, protection, and execution-record redaction.

### Policy, Permissions, And Approvals

Tool policy combines multiple gates:

- registered tool name
- currently exposed tool set
- side-effect class
- filesystem, subprocess, and network permissions
- required secret availability
- explicit approval requirements
- runtime feature visibility

Fresh assistant sessions default to no enabled tools and no filesystem,
subprocess, or network permission. Write-capable side effects are approval
gated by default. The UI can enable a tool only when the corresponding feature
gate, workspace, permission, and credentials allow it to be exposed.

Approval requests are one invocation at a time. In the workflow layer they are
transient. In the harness layer an approval pause becomes durable as a pending
approval record. Denial, expiration, cancel, and interrupted recovery fail
closed: the blocked invocation is recorded and later invocations from the same
paused model turn do not continue running.

#### Registration, Exposure, Execution, And Approval

The system treats registration, exposure, execution, and approval as separate
states:

| State | Meaning | Owner |
| --- | --- | --- |
| Registered | A tool implementation is available in `ToolRegistry`. | Tool registration code. |
| Feature-visible | Product/admin gates allow the tool to be considered by the assistant. | Assistant runtime feature filtering. |
| Exposed | The current context and policy allow the tool to appear in the model-facing schema. | `WorkflowExecutor.prepare_model_interaction`. |
| Approval-required | The tool may be requested by the model but execution pauses for approval. | `ToolPolicy` and `WorkflowExecutor`. |
| Executed | `ToolRuntime` has validated, scoped, invoked, validated output, redacted, and recorded the invocation. | `ToolRuntime`. |

Moving from one state to another is always narrowing. A later state cannot
re-introduce a tool or capability that an earlier gate removed.

#### Policy Inputs

Policy uses both static tool metadata and dynamic context:

- static tool name, tags, side-effect class, required services, required
  secrets, timeout, and cache-write metadata
- dynamic workspace presence, enabled service permissions, network permission,
  subprocess permission, filesystem permission, and available secrets
- product-layer feature visibility and user-enabled tool selections before the
  workflow layer prepares model-facing schemas

This design lets library callers reuse `ToolPolicy` without adopting assistant
UI concepts, while the assistant can still apply stricter product visibility
rules before lower-level policy evaluation.

### Persistence Design

Assistant persistence uses SQLCipher-backed SQLite through synchronous
SQLAlchemy. The default database location is:

```text
~/.llm-tools/assistant/nicegui/chat.sqlite3
```

The SQLCipher database key is stored outside the database. User-owned fields
are encrypted again with per-user data keys wrapped by a local server key.
Default key locations are:

```text
~/.llm-tools/assistant/nicegui/hosted/db.key
~/.llm-tools/assistant/nicegui/hosted/user-kek.key
```

#### Assistant Data Model

Assistant persistence separates queryable relational metadata from encrypted
user-owned payloads:

| Table | Stores | Security-Relevant Shape |
| --- | --- | --- |
| `chat_sessions` | Durable chat shell, runtime settings, workflow state, token usage, inspector state, owner id, temporary flag, selected model metadata. | Titles, root paths, runtime JSON, workflow state, token usage, and inspector JSON are encrypted per owner. Queryable fields are limited to ids, timestamps, owner id, protocol/model metadata, and ordering data. |
| `chat_messages` | Ordered transcript entries and optional structured final responses. | Message text and final-response JSON are encrypted; role, ordinal, completion state, visibility, and timestamps remain relational metadata. |
| `workbench_items` | Inspector/workbench artifacts associated with a chat. | Titles and payload JSON are encrypted; kind, active flag, timing, and ownership through the parent chat remain queryable. |
| `harness_sessions` | Chat-scoped Deep Task harness snapshots and artifacts. | Canonical harness state and artifacts JSON are encrypted and owner-scoped. |
| `app_preferences` | User preferences and global administrator settings. | Preference/admin payload JSON is encrypted; the key remains queryable. |
| `users` | Local hosted-mode user records. | Password hashes are stored, not plaintext passwords. User ids, usernames, roles, disabled flags, and timestamps are relational metadata. |
| `user_sessions` | Server-side browser session records. | Stores token hashes, not raw browser tokens. |
| `auth_events` | Minimal authentication audit events. | Stores user id, event type, detail JSON, and timestamp. |
| `user_key_records` | Wrapped per-owner data encryption keys. | Stores wrapped data keys and key metadata; plaintext data keys are not persisted. |

Temporary chats use the same in-memory record shape as durable chats but are not
saved through `save_session`. Deleting a temporary chat removes controller
memory state and session secrets. Deleting a durable chat removes dependent
messages, workbench items, and chat-scoped harness snapshots through relational
ownership and explicit store deletion.

#### Key Hierarchy And Encrypted Envelopes

Assistant persistence has two encryption layers:

1. SQLCipher encrypts the SQLite database file using a database key stored
   outside the database.
2. User-owned text and JSON fields are encrypted again with AES-256-GCM
   envelopes.

The field-encryption hierarchy is:

1. A server key-encryption key is stored in `user-kek.key`.
2. Each owner id has a generated 256-bit data encryption key.
3. The data encryption key is wrapped by the server key and stored in
   `user_key_records`.
4. Each encrypted field uses AES-256-GCM with a fresh nonce.
5. Additional authenticated data binds the envelope to table, row id, owner key
   id, column name, and envelope version.

Binding encrypted fields to table, row, owner, and column prevents simple
encrypted-value swapping between database cells. Decryption failures are treated
as store corruption. Admin password resets remain possible because per-user data
keys are wrapped by a server key rather than by the user's password.

#### Persistence Algorithms

Session save is full-record oriented for durable chats:

1. Filter runtime settings against currently visible feature-gated tools and
   providers.
2. Sync summary metadata from runtime state and transcript visibility.
3. If the chat is temporary, stop before writing.
4. Upsert the `chat_sessions` row.
5. Replace transcript rows for the session with the current ordered transcript.
6. Replace workbench rows for the session with the current workbench items.
7. Encrypt user-owned text and JSON fields before writing.

Session load reverses the process:

1. Select only rows visible to the requesting owner.
2. Decrypt encrypted fields with owner-bound authenticated data.
3. Validate runtime, workflow state, token usage, inspector state, transcript
   final responses, and workbench payloads with Pydantic models.
4. Treat malformed JSON, invalid models, and decryption failures as corruption
   rather than silently dropping persisted state.

Preference and admin-setting saves use the same encrypted JSON envelope pattern
through `app_preferences`. Hosted browser sessions are separate from chat
sessions: the store persists server-side session ids, token hashes, expiration,
and revocation state, while raw browser tokens remain outside the database.

#### Harness And Cache Persistence

Assistant-backed Deep Task state is stored in `harness_sessions`, scoped to the
owning chat session and owner id. Saves use optimistic revision checks so stale
writers cannot overwrite a newer harness snapshot. Loads validate supported
harness schema versions before returning stored state.

Harness CLI/session persistence outside the assistant stores canonical
`HarnessState` with explicit schema version. File-backed malformed records are
treated as corruption. Summaries and traces are derived from canonical state.

Readable-content conversion caches, read-file caches, and Confluence attachment
caches are not the source of truth. They are performance artifacts derived from
local workspace files or remote attachments. Cache entries can contain sensitive
converted or downloaded content and are therefore treated as internal tool data:
they are hidden from filesystem discovery and subject to filesystem policy when
a tool needs to create or read them.

SQLite files are local to one assistant server process. Sharing one SQLite file
through a network drive across multiple local app instances is not supported.
Hosted multi-user use should centralize database access in one assistant server
process.

### Credential Design

The assistant app does not read process environment variables as implicit
provider or tool credentials. Provider API keys and remote-tool credentials are
typed into the app, held in server memory for the active browser/app session,
and not written to SQLite, config files, browser storage, or provider presets.
Credential input fields are cleared after submission. Credentials expire after
two hours by default and must be re-entered.

Non-secret service URLs and provider base URLs are runtime configuration and may
be persisted. Provider auth schemes describe the credential shape: no secret,
bearer token, or `x-access-tokens`.

Lower-level library and CLI consumers may still pass explicit credentials or
environment-derived values into their own host contexts.

### Protection Subsystem Design

Protection is an experimental, feature-gated workflow subsystem. In the
assistant app it is hidden behind `information_protection_enabled`, disabled by
default, and only assembled for a session when administrator settings and
runtime protection configuration both permit it. The feature gate controls
product visibility and runtime assembly; it does not weaken the lower-level
policy, approval, persistence, or tool-execution controls.

#### Purpose And Scope

Protection evaluates whether model-bound prompts or candidate final responses
may exceed the sensitivity allowed for the current workflow environment. It is
designed for proprietary-information handling, environment-specific disclosure
constraints, and reviewable prompt/response decisions.

Protection is not a generic content moderation system, malware detector, DLP
appliance, or replacement for tool policy. It does not make model output
trusted. It adds a workflow-level decision point before provider calls and
before final response retention.

#### Inputs And Trust Boundaries

Protection consumes:

- `ProtectionConfig`, including enabled state, corpus document paths, allowed
  sensitivity labels, category definitions, correction-file path, challenge
  behavior, final-answer review behavior, purge behavior, and document-cache
  behavior.
- `ProtectionCorpus`, containing loaded protection documents and structured
  feedback entries.
- `ProtectionEnvironment`, containing app name, selected model, workspace,
  enabled tools, permission flags, allowed sensitivity labels, and category
  metadata.
- Provider messages or candidate response payloads.
- `ProtectionProvenanceSnapshot`, collected from tool results or harness state.
- A `SensitivityClassifier`, currently backed in the assistant by the
  OpenAI-compatible provider layer.

The main trust boundaries are user prompt text, model-generated assessments,
model-generated final responses, local corpus files, correction sidecar files,
tool-result provenance, and persisted workflow or harness state. Protection
normalizes model assessments into typed Pydantic contracts before applying
environment policy. The classifier can recommend an action, but the controller
and environment comparator produce the effective decision.

#### Protection Corpus Loading Algorithm

Corpus loading is file based and deterministic for a given configuration:

1. Resolve each configured document path.
2. If the path is missing, record a load issue and continue.
3. If the path is a directory, recursively inspect files beneath it.
4. Skip internal protection metadata files and `.llm_tools` cache paths.
5. Accept plain readable suffixes such as Markdown, text, CSV, JSON, YAML, logs,
   reStructuredText, and AsciiDoc.
6. For convertible document suffixes, use the same readable-content conversion
   infrastructure as filesystem reads.
7. Enforce readable byte limits before conversion.
8. Use cached converted content when document caching is enabled and a cache hit
   exists.
9. Convert supported documents when needed and optionally write the converted
   content cache.
10. Resolve sensitivity labels from front matter, source manifests, or
    configured category rules.
11. Assign a stable document id relative to the corpus root and compute a
    content hash.
12. Load structured feedback entries from the correction store when configured.

Unsupported file types, unreadable files, conversion failures, too-large files,
and invalid source manifests become load issues rather than silently entering
the corpus. This gives operators a reviewable corpus-inspection surface without
letting one bad document prevent other valid documents from loading.

#### Prompt Assessment And Decision Algorithm

Prompt-side protection runs before a provider call:

1. If protection is disabled, return `allow`.
2. Send the model-bound messages, corpus, environment, and provenance snapshot
   to the classifier.
3. Validate the classifier payload as `ProtectionAssessment`.
4. Resolve aliases to canonical sensitivity labels using the environment's
   category catalog.
5. If the assessment includes a recommended action, use it as the initial
   action.
6. Otherwise compare the canonical label against blocked, allowed,
   constrained, and sanitized label sets.
7. For prompts, unknown or disallowed sensitive labels default to `challenge`
   rather than automatic provider execution.
8. Build a `PromptProtectionDecision` containing action, label, reasoning,
   confidence, referenced document ids, guard text, challenge message, and
   metadata.

The effective prompt actions are `allow`, `constrain`, `challenge`, `sanitize`,
and `block`. In current prompt flow, `constrain` adds guard text directing the
model to answer within the lowest acceptable sensitivity. `challenge` pauses
for user review rather than sending the prompt directly to the provider.

#### Prompt Challenge And Feedback Algorithm

When a prompt decision requires challenge, the controller can build a
`ProtectionPendingPrompt` that records the original user message, serialized
provider messages, controller reasoning, predicted sensitivity label,
referenced document ids, session id, and decision metadata. This pending prompt
is app/workflow state for an unresolved challenge.

User feedback is parsed as either structured JSON, simple affirmative text, or
line-oriented key-value correction text. If the user confirms the analysis,
the request remains blocked for the current environment. If the user disputes
the analysis, they must provide an expected sensitivity label. The controller
then records a `ProtectionFeedbackEntry` with the original prompt, expected
label, rationale, session id, referenced documents, predicted label, serialized
messages, and metadata.

When a correction store is configured, feedback is appended to a JSON or YAML
sidecar and the in-memory corpus is refreshed with the updated feedback list.
The user is instructed to resubmit the original request after correction rather
than silently replaying a previously challenged prompt.

#### Response Review, Sanitization, And Blocking Algorithm

Response-side protection runs before a candidate final response is retained or
shown:

1. If protection is disabled or final-answer review is disabled, return
   `allow`.
2. Send the response payload, corpus, environment, and provenance snapshot to
   the classifier.
3. Validate the classifier payload as `ProtectionAssessment`.
4. Check whether the single-source allowance applies.
5. If the allowance applies, return `allow` and mark metadata with
   `source_policy=single_allowed_source`.
6. Otherwise resolve the response action through the environment comparator.
7. Convert any response-side `challenge` result into `block`; final responses
   are not interactively challenged.
8. If the action is `sanitize`, replace the response payload or its answer/body
   field with the classifier-provided safe text or `[REDACTED]`.
9. If the action is `block`, produce a safe withholding message.
10. If the action is `sanitize` or `block` and purge is enabled, set
    `should_purge`.

This separates classifier assessment from retention behavior. The classifier may
identify sensitive content; the controller determines whether the app may
persist or display it in the current environment.

#### Source Provenance And Single-Source Allowance

Protection uses source provenance to avoid over-blocking purely extractive use
of one already-allowed source. The single-source allowance applies only when:

- exactly one source document id is used or referenced
- the assessment says cross-source synthesis is not required
- the assessment says inference beyond source text is not required
- the environment has an allowed sensitivity-label set
- the referenced source resolves to a corpus document
- the source document has a sensitivity label allowed by the current
  environment

If any condition is missing or ambiguous, the allowance does not apply and the
normal response action is used. This keeps the exception narrow: it is for
extractive use of one permitted source, not for synthesis, inference, or
unlabeled material.

#### Persistence, Corrections, Caches, And Harness Scrubbing

Protection state touches several persistence surfaces:

- Corpus documents are loaded from configured local files and may use
  readable-content conversion caches when `cache_documents` is enabled.
- Feedback corrections are stored only when a correction path is configured.
- Prompt challenges can be represented as pending prompt state in chat session
  state.
- Response violations can trigger purge behavior.
- Harness-backed Deep Task state can be scrubbed before persistence when
  protection requires sanitization or blocking.

Harness scrubbing replaces protected final responses with a safe message or
`[WITHHELD BY PROTECTION]`, removes tool-result outputs, clears logs and
artifacts, removes validated and redacted execution-record outputs, and clears
pending approvals. Scrubbing is deliberately broader than replacing the final
answer because tool outputs, execution records, logs, artifacts, raw inspection
payloads, and replay views can otherwise re-expose protected content.

#### Assistant Chat And Deep Task Integration

Assistant runtime assembly builds protection controllers only when the
administrator feature gate and session runtime configuration allow them. Chat
and Deep Task receive separate controllers because they have different app names
and integration paths.

Assistant Chat can keep an unresolved prompt-side challenge in session state and
use a forced-enabled controller configuration while that challenge is pending.
Deep Task passes protection through the harness provider path and can collect
provenance from harness state. Both flows use the same workflow-level
controller contracts and model-turn protocol integration.

The first-party classifier currently requires an `OpenAICompatibleProvider`.
If the app is using another provider type, the app-layer builder does not create
the first-party protection controller. This is a current implementation
constraint and one reason the feature remains experimental and gated.

#### Security-Relevant Controls

Protection's main controls are:

- disabled-by-default feature flag in the assistant app
- explicit runtime protection configuration
- typed Pydantic assessment and decision models
- category alias canonicalization before action resolution
- environment-specific allowed, blocked, constrained, and sanitized label sets
- prompt-side challenge instead of automatic execution for disallowed labels
- response-side block instead of interactive challenge for final answers
- narrow single-source allowance conditions
- explicit purge signal for response violations
- corpus inspection reports for missing, unsupported, invalid, or failed
  documents
- correction sidecar parsing with structured validation
- harness scrubbing of prior tool payloads and execution records
- redacted model-turn events rather than raw provider payload emission

These controls are reviewable at the workflow and app boundaries. They do not
depend on built-in tool implementations trusting model text.

#### Residual Risks And Non-Goals

Protection is classifier-assisted and therefore probabilistic when backed by an
LLM. It can produce false positives, false negatives, incomplete rationales, or
incorrect labels. It depends on the quality and completeness of the configured
corpus and sensitivity categories. It does not inspect arbitrary external
systems unless their content is represented in corpus documents, prompt text,
response payloads, or provenance. It does not replace access control,
credential hygiene, provider trust decisions, or runtime tool policy.

The first-party assistant integration currently depends on the
OpenAI-compatible provider surface for classification. Native provider support
for protection classification is not assumed by this design. The correction
store improves future classification context but is not an audited approval
record and should not be treated as a policy override.

#### Test Evidence

Protection behavior is covered by direct workflow protection tests, app-layer
protection runtime tests, model-turn protocol tests, harness protection
scrubbing tests, and backend-matrix protection scenarios. The tests exercise
controller decisions, prompt feedback parsing and recording, response
sanitization, single-source allowance, category alias canonicalization, disabled
and final-review-off behavior, corpus inspection failures, document conversion
caching, source manifests, metadata-file skipping, too-large converted
documents, app-layer classifier wiring, environment construction, and harness
state scrubbing.

### Security Design

The security surface includes filesystem reads and writes, subprocess-backed Git
helpers, network-capable remote read tools, provider calls, prompt and response
content, SQLCipher persistence, local user authentication, browser sessions,
durable harness state, caches, and workbench/inspection artifacts.

#### Assets And Trust Boundaries

The main protected assets are:

- workspace files, filenames, metadata, converted document text, and write
  targets
- provider credentials and remote-tool credentials held in session memory
- prompts, model-visible context, model responses, and structured tool
  arguments
- persisted chat transcripts, workflow state, workbench/inspector payloads,
  preferences, admin settings, and harness snapshots
- SQLCipher database key, server key-encryption key, and wrapped per-owner data
  keys
- remote GitLab, Jira, Bitbucket, and Confluence data reachable by configured
  tokens
- generated caches, including readable-content conversion caches and
  Confluence attachment caches

The primary trust boundaries are browser to assistant server, assistant server
to model provider, assistant server to local filesystem, assistant server to
subprocess-backed Git, assistant server to remote enterprise services,
assistant persistence to local disk, and model output back into workflow/tool
execution.

#### Actors And Attacker Assumptions

The design assumes the following actors and failure modes:

- A legitimate local or hosted user can submit prompts that are malicious,
  confused, or prompt-injection shaped.
- A model provider can return malformed, adversarial, overlong, or
  schema-incompatible output.
- A workspace can contain malicious paths, symlinks, hidden files, large files,
  unsupported binary data, or parser-hostile documents.
- A remote service can return malformed payloads, transient failures, or content
  beyond what the user expected the token to reveal.
- A browser session token can be stolen; the server stores token hashes and
  supports revocation/expiration, but an active stolen browser token is still an
  application-session risk until revoked or expired.
- A database copy can be stolen; SQLCipher plus field encryption protect
  against offline inspection without key files, but a database copy stolen with
  the key files is a higher-impact event.
- A fully compromised assistant server can access in-memory credentials, active
  sessions, key files available to the process, and any workspace or remote
  data the process can access.

The system is designed to fail closed around tool execution, approvals, harness
resume, schema validation, and persistence corruption. It does not assume model
output is benign simply because it came from a configured provider.

#### Security Control Matrix

| Threat / Risk | Primary Controls | Implementation Area | Residual Risk |
| --- | --- | --- | --- |
| Model asks to invoke an unapproved or unavailable tool. | Tool exposure separate from registration, default-deny assistant tools, `ToolPolicy`, side-effect classes, approval gates. | `tool_api`, `workflow_api`, assistant runtime assembly. | A user or admin can still enable powerful tools and approve risky calls. |
| Path traversal or symlink escape from workspace. | Workspace-relative POSIX paths, root confinement, symlink rejection, internal cache path rejection. | Filesystem tools and path utilities. | Direct reads can still expose sensitive files intentionally located inside the workspace. |
| Destructive local write. | `write_file_tool_enabled` default-hidden gate, local-write side-effect class, approval requirement, overwrite/parent creation controls. | Filesystem `write_file`, assistant feature gates, `ToolPolicy`. | Approved writes can modify important project files. |
| Subprocess abuse through Git tools. | Fixed non-shell Git commands, workspace-contained repository root, stripped Git environment, prompt disabled, timeout/output limits. | Git tools and execution services. | Git metadata and diffs may reveal sensitive local content. |
| Remote-token overexposure. | Required scoped secrets, network permission, default-hidden remote tools, bounded result counts, timeouts, typed output shaping. | GitLab and Atlassian tools, assistant credential handling. | Tools can reveal any data allowed by the upstream token. |
| Prompt or response exceeds configured sensitivity. | Experimental protection controllers, corpus/category policy, prompt challenge/block, response sanitize/block, harness scrubbing. | `workflow_api` protection, app protection runtime, harness protection. | Classifier-assisted protection can be wrong or incomplete. |
| Malformed provider output drives execution. | Adapter parsing, Pydantic validation, staged repair limits, prompt-tool parsing into canonical responses, runtime tool input validation. | `llm_adapters`, `workflow_api`, `tool_api`. | A valid but malicious tool request can still require policy/approval judgment. |
| Persistent data exposure from database copy. | SQLCipher database encryption, per-owner AES-GCM field encryption, wrapped data keys, authenticated data binding. | Assistant SQLite store and crypto helpers. | Database plus key-file compromise defeats offline protection. |
| Browser session theft. | Server-side session records, token hashes, expiration, revocation, local auth by default. | Assistant auth/store. | Active stolen sessions remain useful until expiration or revocation. |
| Corrupt or tampered persistence. | Pydantic validation after load, decryption failure as corruption, supported harness schema checks, fail-closed resume classification. | Assistant store, harness store/resume. | Operator intervention may be required to recover useful state. |
| Caches reveal sensitive derived content. | Internal cache directories hidden from discovery, workspace confinement, cache metadata validation, tool permission checks. | Filesystem readable-content pipeline and Confluence attachment reads. | Local filesystem access to cache directories can still reveal cached content. |

#### Deployment Modes

Local single-user use runs the assistant server, browser, database, key files,
workspace, and in-memory credentials under one local operator's control. The
main risks are prompt/tool misuse, accidental workspace exposure, local malware,
and stolen local database/key files.

Hosted use adds user separation, browser session management, reverse-proxy/TLS
requirements, admin-managed users, and per-user chat ownership. Hosted mode
should centralize database access in one assistant server process. It should
not expose secret entry over non-loopback HTTP unless the operator explicitly
accepts that risk.

Library/CLI use moves credential and deployment responsibility to the caller.
The lower layers still provide typed contracts, runtime policy, execution
services, provider adapters, and harness state validation, but they do not
automatically impose assistant UI defaults such as session-memory credentials or
admin feature gates.

Security controls include:

- mechanically enforced layer boundaries
- runtime-mediated tool execution
- default-deny assistant permissions
- side-effect classes and approval gates
- scoped execution services and scoped secrets
- workspace-relative filesystem paths
- hidden/ignored discovery rules
- output limits and timeouts for remote and subprocess tools
- redacted execution records
- workflow-level protection before provider calls and before final retention
- SQLCipher database encryption
- per-user field encryption
- Argon2 password hashes
- hashed server-side browser session tokens
- fail-closed harness resume and approval behavior

The app uses local admin-created users. There is no public self-registration.
Admins can create, disable, and reset users. Users see their own chats,
preferences, workbench records, temporary sessions, and in-memory credentials.
Admin password resets remain practical because per-user data keys are wrapped by
a server key; this protects against copied database files and accidental
cross-user data exposure, not against a fully compromised web server.

Hosted deployments should use a TLS-terminating reverse proxy and set
`--public-base-url` to the HTTPS URL. Direct certificate files are supported for
bootstrap and self-signed testing. If the app is reachable over non-loopback
HTTP, secret entry is disabled unless
`--allow-insecure-hosted-secrets` is passed as explicit risk acceptance.

Filesystem discovery excludes dot-hidden and ignore-file-matched paths by
default. Direct reads of explicitly named workspace-relative paths are still
eligible when policy allows them. This separates broad discovery visibility from
explicit access.

Remote integrations apply collection limits, truncation metadata, explicit
timeouts, scoped required secrets, and source provenance. Confluence attachment
reads are intentionally different from Confluence page reads: attachment reads
download bytes into an internal cache and require filesystem permission plus a
local-write side-effect class.

The system does not claim to protect secrets after full server compromise, does
not provide cross-machine SQLite conflict resolution, does not migrate or scrub
old pre-hardening snapshots in place, and does not make model output trusted
without runtime validation and policy mediation.

### Feature-Gated Capabilities

Administrator settings gate implemented capabilities that are disabled by
default:

| Capability | Gate | Owning Area | Default | Design Effect |
| --- | --- | --- | --- | --- |
| Deep Task mode | `deep_task_mode_enabled` | Assistant app / `harness_api` | Disabled | Shows harness-backed durable task mode in the app. |
| Information protection | `information_protection_enabled` | `workflow_api` / assistant runtime | Disabled | Experimental feature that enables prompt and response protection controllers. |
| Local skills | `skills_enabled` | `skills_api` / assistant runtime | Disabled | Discovers project and bundled skills and permits explicit `$skill` invocation. |
| Native Ollama protocol | `ollama_native_provider_enabled` | `llm_providers` / assistant provider UI | Disabled | Shows native Ollama provider protocol and presets. |
| Native Ask Sage protocol | `ask_sage_native_provider_enabled` | `llm_providers` / assistant provider UI | Disabled | Shows native Ask Sage provider protocol and presets. |
| `write_file` tool | `write_file_tool_enabled` | filesystem tools / assistant runtime | Disabled | Makes the local write tool visible; execution remains policy and approval gated. |
| Atlassian tools | `atlassian_tools_enabled` | Atlassian tools / assistant runtime | Disabled | Makes Jira, Confluence, and Bitbucket tools visible when credentials and network policy allow them. |
| GitLab tools | `gitlab_tools_enabled` | GitLab tools / assistant runtime | Disabled | Makes GitLab tools visible when credentials and network policy allow them. |

Feature gates affect product visibility and available tool/provider choices.
They do not bypass runtime policy, approvals, credential checks, or workspace
requirements.

## Built-In Capabilities

### Tool Families

The built-in tool families are filesystem, Git, GitLab, and Atlassian. The
assistant registry registers all families, then filters visibility and exposure
by feature flags, session policy, credentials, and workspace readiness.

Filesystem tools use workspace-relative POSIX-style paths. Discovery/search
tools hide dot-hidden and ignored paths by default. `read_file`, `search_text`,
and `get_file_info` can use document conversion caches for readable content.

Git tools run non-interactive subprocess commands inside the resolved workspace
repository. They are read-only from the tool contract perspective and impose
timeouts and output limits.

GitLab tools are network read integrations scoped by `GITLAB_BASE_URL` and
`GITLAB_API_TOKEN`. In the assistant app they are implemented but hidden by the
default-disabled `gitlab_tools_enabled` feature gate. Atlassian tools are
network read integrations scoped by service-specific base URLs and tokens for
Jira, Bitbucket, and Confluence. In the assistant app they are implemented but
hidden by the default-disabled `atlassian_tools_enabled` feature gate.

### Individual Tool Summary

| Tool | Family | Purpose | Capability Shape | Required Secrets | Feature Gate |
| --- | --- | --- | --- | --- | --- |
| `find_files` | filesystem | Find files by workspace-relative glob. | Local read; filesystem required; hidden paths excluded unless requested. | None | None |
| `get_file_info` | filesystem | Inspect one file or a small batch before reading. | Local read; filesystem required; may use internal cache metadata. | None | None |
| `list_directory` | filesystem | List immediate or recursive directory children. | Local read; filesystem required. | None | None |
| `read_file` | filesystem | Read text or converted Markdown content. | Local read; filesystem required; writes internal conversion cache. | None | None |
| `search_text` | filesystem | Search readable file contents for literal text. | Local read; filesystem required; may use readable-content cache. | None | None |
| `write_file` | filesystem | Write text content inside the workspace. | Local write; filesystem required; approval-gated by default. | None | `write_file_tool_enabled` |
| `run_git_status` | git | Run non-interactive `git status`. | Local read; filesystem and subprocess required; timeout 10s. | None | None |
| `run_git_diff` | git | Run non-interactive `git diff`. | Local read; filesystem and subprocess required; timeout 10s. | None | None |
| `run_git_log` | git | Run non-interactive `git log`. | Local read; filesystem and subprocess required; timeout 10s. | None | None |
| `search_gitlab_code` | GitLab | Search one GitLab project for code matches. | External read; network required; timeout 30s; collection limits. | `GITLAB_BASE_URL`, `GITLAB_API_TOKEN` | `gitlab_tools_enabled` |
| `read_gitlab_file` | GitLab | Read one UTF-8 file from a GitLab project. | External read; network required; timeout 30s. | `GITLAB_BASE_URL`, `GITLAB_API_TOKEN` | `gitlab_tools_enabled` |
| `read_gitlab_merge_request` | GitLab | Read one GitLab merge request by IID. | External read; network required; timeout 30s; bounded commits/changes. | `GITLAB_BASE_URL`, `GITLAB_API_TOKEN` | `gitlab_tools_enabled` |
| `search_jira` | Atlassian | Search Jira issues with JQL. | External read; network required; timeout 30s; collection limits. | `JIRA_BASE_URL`, `JIRA_API_TOKEN` | `atlassian_tools_enabled` |
| `read_jira_issue` | Atlassian | Read one Jira issue by key. | External read; network required; timeout 30s; allowlisted fields unless requested. | `JIRA_BASE_URL`, `JIRA_API_TOKEN` | `atlassian_tools_enabled` |
| `search_bitbucket_code` | Atlassian | Search Bitbucket Server/DC code. | External read; network required; timeout 30s; collection limits. | `BITBUCKET_BASE_URL`, `BITBUCKET_API_TOKEN` | `atlassian_tools_enabled` |
| `read_bitbucket_file` | Atlassian | Read one UTF-8 Bitbucket file. | External read; network required; timeout 30s. | `BITBUCKET_BASE_URL`, `BITBUCKET_API_TOKEN` | `atlassian_tools_enabled` |
| `read_bitbucket_pull_request` | Atlassian | Read one Bitbucket pull request. | External read; network required; timeout 30s; bounded commits/changes. | `BITBUCKET_BASE_URL`, `BITBUCKET_API_TOKEN` | `atlassian_tools_enabled` |
| `search_confluence` | Atlassian | Search Confluence with CQL. | External read; network required; timeout 30s; collection limits. | `CONFLUENCE_BASE_URL`, `CONFLUENCE_API_TOKEN` | `atlassian_tools_enabled` |
| `read_confluence_page` | Atlassian | Read one Confluence page body. | External read; network required; timeout 30s. | `CONFLUENCE_BASE_URL`, `CONFLUENCE_API_TOKEN` | `atlassian_tools_enabled` |
| `read_confluence_attachment` | Atlassian | Read one Confluence attachment. | Local write plus external read; network and filesystem required; timeout 30s; internal attachment cache. | `CONFLUENCE_BASE_URL`, `CONFLUENCE_API_TOKEN` | `atlassian_tools_enabled` |

### Tool Assurance Design

This section expands the tool summary into assurance-oriented design notes. It
does not document every input field or output field. It explains what each tool
does, the algorithm it follows, the trust boundaries it crosses, the controls
that constrain it, and the residual risks a security reviewer should understand.

#### Shared Filesystem Tool Controls

Filesystem tools operate only through `ToolRuntime` and require a workspace in
the execution context. Tool-visible paths are workspace-relative POSIX paths.
Absolute paths, Windows rooted paths, empty paths, and paths targeting
`.llm_tools` internal tool-managed directories are rejected. Resolution walks
each path component beneath the resolved workspace root, rejects symlinked
parent directories, rejects symlinked final files/directories, rejects
non-directory parent components, and verifies that the final resolved path stays
inside the workspace root.

Discovery tools apply hidden and ignored path filtering. Dot-hidden path
components and `.gitignore`-matched paths are hidden unless the call explicitly
sets `include_hidden`. Internal `.llm_tools` cache directories are never exposed
through discovery. Direct file reads are separate from discovery visibility:
explicit root-confined paths can be read when policy allows, even if broad
discovery would hide them.

Readable-content loading first enforces byte limits, then tries UTF-8 text, then
document conversion for supported non-text formats. MarkItDown handles common
office/document formats. MPXJ handles Microsoft Project formats through a
single JVM-backed reader. Converted Markdown is cached under the workspace's
internal read-file cache with source size and mtime metadata; stale cache
entries are ignored. Conversion errors are normalized to stable messages that
avoid leaking local host paths.

#### `list_directory`

**Purpose**: List immediate or recursive children below one workspace-relative
directory so a model can inspect project shape before selecting precise files.

**Inputs And Trust Boundaries**: The model controls the requested directory,
recursive flag, optional depth, and source filters. The host filesystem is the
protected resource. The workspace root and tool limits come from runtime
context, not model text.

**Algorithm**: Resolve the requested directory inside the workspace. Build a
per-call `.gitignore` matcher from the workspace. Walk children in stable name
order. Apply hidden, ignored, include, exclude, and internal-cache filters.
Respect recursive depth and the configured maximum recursive depth. Stop when
the maximum entry count is reached and mark the result truncated.

**Required Capabilities**: Filesystem read access. No network, subprocess, or
secret access.

**Security-Relevant Controls**: Root confinement, symlink rejection, internal
cache hiding, hidden/ignored filtering, deterministic traversal order, maximum
depth, and maximum entry count.

**Persistence, Cache, Logs, And Provenance**: Does not persist data or write
cache. Logs only the directory listing action through the runtime path. It does
not add source provenance because it lists metadata rather than reading source
content.

**Failure Behavior**: Invalid paths, missing directories, symlink traversal, or
path escape attempts fail through normalized tool errors. Overlarge listings
return partial results with `truncated=true`.

**Residual Risks**: Directory and filename metadata can itself be sensitive.
Operators should keep filesystem permission disabled unless directory discovery
is acceptable for the workspace.

**Test Evidence**: Filesystem tests cover stable ordering, hidden path
exclusion, recursive behavior, root-confined resolution, and tool policy
mediation.

#### `find_files`

**Purpose**: Find files matching a workspace-relative glob pattern under a
root-confined search directory.

**Inputs And Trust Boundaries**: The model controls the glob pattern, search
root, and hidden-path option. The host filesystem is the protected resource.

**Algorithm**: Normalize the glob pattern to POSIX style and resolve the search
root as a workspace-confined directory. Walk the tree in stable order without
following symlinks. Prune hidden, ignored, excluded, and internal-cache
directories. Count files scanned, enforce maximum scan and result limits, match
with explicit recursive `**` semantics, and mark truncation when limits are
hit.

**Required Capabilities**: Filesystem read access. No network, subprocess, or
secret access.

**Security-Relevant Controls**: Root confinement, symlink skipping, `.gitignore`
filtering, internal-cache exclusion, bounded recursion, bounded scan count, and
bounded result count.

**Persistence, Cache, Logs, And Provenance**: Does not write cache or persist
data. It returns file metadata, not file contents.

**Failure Behavior**: Invalid paths and invalid empty patterns fail. Limit
exhaustion returns truncated search results.

**Residual Risks**: Search results reveal file names and relative paths. Hidden
paths can be revealed only when `include_hidden` is explicitly set and policy
allows filesystem access.

**Test Evidence**: Tests cover model-facing guidance for recursive globs and
filesystem discovery behavior; architecture tests enforce runtime-mediated
execution.

#### `get_file_info`

**Purpose**: Inspect one file or a small batch of files before deciding whether
to read them.

**Inputs And Trust Boundaries**: The model supplies one or more
workspace-relative file paths. The filesystem and optional readable-content
conversion stack are the protected resources.

**Algorithm**: Resolve each requested file inside the workspace. Load readable
content using the same byte-limit, text-detection, conversion, and cache
algorithm as `read_file`. Return deterministic metadata such as status, read
kind, size, character count, estimated token count, and path data.

**Required Capabilities**: Filesystem read access. It may use the internal
readable-content cache for converted documents.

**Security-Relevant Controls**: Root confinement, symlink rejection, internal
path rejection, byte limits before conversion, conversion error sanitization,
and Pydantic output validation.

**Persistence, Cache, Logs, And Provenance**: May populate the internal
read-file conversion cache when conversion is needed. It does not return full
content.

**Failure Behavior**: Unreadable, unsupported, or too-large files are reported
as structured statuses where possible. Invalid paths fail through normalized
tool errors.

**Residual Risks**: Metadata such as file size, type, and readable status can be
sensitive. Converted-document inspection may execute parser libraries against
untrusted files, so filesystem permission should remain scoped to trusted
workspaces.

**Test Evidence**: Filesystem tests cover conversion, cache reuse, cache
invalidation, and limit behavior shared with `read_file`.

#### `read_file`

**Purpose**: Read bounded text or converted Markdown content from one
workspace-confined file.

**Inputs And Trust Boundaries**: The model controls the workspace-relative path
and optional character range. The local file and conversion backends are the
protected resources.

**Algorithm**: Resolve the file under the workspace with symlink rejection.
Reject oversized inputs before conversion. Read UTF-8 text when possible. If
the file is non-text and has a supported conversion backend, reuse a fresh cache
entry or convert the file to Markdown. Enforce configured character limits,
normalize the requested range, cap the returned slice to the full-read limit,
and return content plus truncation and size metadata.

**Required Capabilities**: Filesystem read access. It writes only the internal
workspace conversion cache.

**Security-Relevant Controls**: Root confinement, symlink rejection, byte limit
before conversion, character limit after conversion, bounded range reads,
internal-cache path rejection, conversion error sanitization, output validation,
and source provenance emission for successfully read content.

**Persistence, Cache, Logs, And Provenance**: Converted content is cached under
`.llm_tools/cache/read_file` with source mtime and size metadata. Successful
reads log the resolved workspace-relative path, add a redacted artifact through
runtime redaction, and add local source provenance.

**Failure Behavior**: Unsupported binary content, too-large input, too-large
readable content, conversion failure, and invalid ranges produce structured
statuses or normalized errors depending on where validation fails.

**Residual Risks**: The tool can expose any explicitly named file inside the
workspace when filesystem permission allows it, including hidden files. This is
intentional direct access, not discovery. Conversion libraries process file
content and may have their own parser risks.

**Test Evidence**: Tests cover text reads, converted document reads, cache
reuse, cache invalidation, byte-limit rejection before conversion, bounded
character ranges, large-range truncation, invalid ranges, redacted artifacts,
and source provenance.

#### `search_text`

**Purpose**: Search readable workspace file contents for a literal substring.

**Inputs And Trust Boundaries**: The model controls the literal query, search
path, and hidden-path option. The filesystem and conversion backends are the
protected resources.

**Algorithm**: Resolve the search target as either a file or directory. For
directories, walk in stable order without following symlinks, pruning hidden,
ignored, excluded, and internal-cache paths unless hidden inclusion is
explicitly requested. Load readable text with the shared readable-content
algorithm. Search for literal matches, shape bounded match snippets, and stop
when scan or result limits are hit.

**Required Capabilities**: Filesystem read access. It may write internal
conversion cache entries for converted documents.

**Security-Relevant Controls**: Root confinement, symlink rejection/skipping,
hidden/ignored filtering for discovery, scan/result limits, byte and character
limits, internal-cache exclusion, and literal search semantics rather than shell
or regex execution.

**Persistence, Cache, Logs, And Provenance**: May populate the conversion cache.
It logs the search action but does not add source provenance for every scanned
file.

**Failure Behavior**: Invalid targets fail. Unsupported or unreadable files are
skipped or reported according to search result shaping. Limit exhaustion returns
truncated results.

**Residual Risks**: Match snippets can reveal sensitive content across many
files. The tool should be enabled only when workspace content search is
acceptable.

**Test Evidence**: Filesystem and chat-helper tests cover search behavior,
readable content handling, path utilities, and runtime integration.

#### `write_file`

**Purpose**: Write text content to one workspace-confined file.

**Inputs And Trust Boundaries**: The model controls the target path, content,
encoding, parent creation flag, and overwrite flag. The local filesystem is the
protected resource.

**Algorithm**: Resolve a writable path under the workspace without requiring the
file to exist. Reject path escapes, internal tool paths, symlinked parents,
symlinked final paths, and directory targets. If the file exists and overwrite
is false, fail. If the parent directory is missing, create it only when
`create_parents` is true. Write text with the requested encoding, count bytes
written, log the write, and add the resolved path as an artifact.

**Required Capabilities**: Filesystem access and local-write side-effect
permission. In the assistant, the tool is hidden behind
`write_file_tool_enabled` and write side effects are approval-gated by default.

**Security-Relevant Controls**: Feature-gated visibility, local-write side
effect class, approval policy, root confinement, symlink rejection, internal
path rejection, explicit overwrite behavior, explicit parent creation, and
runtime output/artifact redaction.

**Persistence, Cache, Logs, And Provenance**: Writes the requested file. Logs a
write message and records the file path as a redacted artifact. It does not add
source provenance because it creates or modifies content rather than reading
source material.

**Failure Behavior**: Existing files without overwrite, missing parents without
parent creation, directory targets, path escapes, and symlinks fail before
writing.

**Residual Risks**: This is a real local write primitive inside the workspace.
If enabled and approved, it can overwrite important project files. It should
remain feature-gated and approval-gated in security-sensitive environments.

**Test Evidence**: Tests cover successful writes, parent creation, overwrite
rejection, absolute-path rejection, symlinked path rejection, artifact
redaction, and runtime side-effect policy.

#### Shared Git Tool Controls

Git tools run fixed non-interactive `git` commands through the mediated
subprocess service. They resolve the target path inside the workspace, ask Git
for the repository root, and reject repositories whose resolved root escapes the
workspace. The subprocess environment strips `GIT_*` variables, sets
`GIT_TERMINAL_PROMPT=0`, and sets `GIT_CONFIG_NOSYSTEM=1`. Output collection
uses stdout/stderr pipes, reader threads, a ten-second timeout, byte limits, and
UTF-8 decoding with replacement. The tools mark outputs truncated instead of
returning unbounded command output. They set `retain_output_in_execution_record`
to false so command output is not duplicated into execution records.

#### `run_git_status`

**Purpose**: Return concise repository status and branch information.

**Inputs And Trust Boundaries**: The model supplies a workspace-relative path
used to find the repository. The subprocess boundary and local repository
metadata are protected resources.

**Algorithm**: Resolve the target directory, discover the workspace-contained
repository root with `git rev-parse --show-toplevel`, then run
`git status --short --branch`. Capture bounded output and return it with a
truncation flag.

**Required Capabilities**: Filesystem and subprocess access. No network or
secrets.

**Security-Relevant Controls**: Fixed command arguments, no shell invocation,
workspace-contained repository root, stripped Git environment, terminal prompt
disabled, timeout, output limit, and output not retained in execution records.

**Persistence, Cache, Logs, And Provenance**: Does not write files or cache. It
logs the command action.

**Failure Behavior**: Missing Git executable, invalid repository discovery,
nonzero Git exit, timeout, and output-limit discovery failures return
normalized errors.

**Residual Risks**: Status output can reveal local branch names and file paths.
Subprocess permission should be enabled only for trusted workspaces.

**Test Evidence**: Git tests cover environment stripping, subprocess invocation,
repository-root validation, timeout handling, output truncation, and command
arguments.

#### `run_git_diff`

**Purpose**: Return bounded Git diff output for unstaged, staged, or ref-based
comparison.

**Inputs And Trust Boundaries**: The model supplies the target path, staged
flag, and optional diff ref. The subprocess boundary and repository contents are
protected resources.

**Algorithm**: Resolve the workspace-contained repository root. Build a fixed
argument list beginning with `git diff --no-ext-diff --no-textconv`. Add
`--staged` when requested. Validate the optional ref by trimming whitespace and
rejecting empty refs, option-like refs beginning with `-`, and control
characters. Run the command without a shell and return bounded diff text.

**Required Capabilities**: Filesystem and subprocess access. No network or
secrets.

**Security-Relevant Controls**: Fixed command shape, external diff/textconv
disabled, no shell, option-like ref rejection, workspace root containment,
stripped Git environment, timeout, output limit, and no execution-record output
retention.

**Persistence, Cache, Logs, And Provenance**: Does not write files or cache.
Logs the command action.

**Failure Behavior**: Invalid refs fail before command execution. Git failures,
timeouts, and output-limit conditions become normalized errors or truncated
results.

**Residual Risks**: Diff output can contain sensitive file content. The tool is
read-only but can reveal uncommitted changes.

**Test Evidence**: Tests cover staged/ref command construction, option-like ref
rejection, repository escape rejection, output truncation, formatted failures,
and subprocess timeout behavior.

#### `run_git_log`

**Purpose**: Return concise recent commit history for a repository.

**Inputs And Trust Boundaries**: The model supplies the target path and limit.
The subprocess boundary and repository history are protected resources.

**Algorithm**: Resolve the workspace-contained repository root. Validate the
limit through the input model, capped by the maximum Git log limit. Run
`git log --max-count <limit> --oneline --decorate` without a shell and return
bounded output.

**Required Capabilities**: Filesystem and subprocess access. No network or
secrets.

**Security-Relevant Controls**: Fixed command arguments, validated maximum
limit, no shell, workspace root containment, stripped Git environment, timeout,
output limit, and no execution-record output retention.

**Persistence, Cache, Logs, And Provenance**: Does not write files or cache.
Logs the command action.

**Failure Behavior**: Invalid limits fail at input validation. Git failures,
timeouts, and output-limit conditions become normalized errors or truncated
results.

**Residual Risks**: Commit subjects, branch decorations, and commit ids can
reveal sensitive project history or naming.

**Test Evidence**: Tests cover limit validation, fixed command construction,
repository-root handling, output truncation, and failure formatting.

#### Shared Remote Integration Controls

GitLab and Atlassian tools are implemented remote integrations but hidden in
the assistant app by default. GitLab visibility requires
`gitlab_tools_enabled`; Jira, Bitbucket, and Confluence visibility requires
`atlassian_tools_enabled`. Even after a feature gate is enabled, each tool still
requires network access, service-specific required secrets, and
runtime-provided remote gateways. Missing base URLs or API tokens are policy
failures before tool execution. Collection searches fetch one extra record
(`limit + 1`) and return only the requested limit so truncation is explicit.
Remote timeouts and transient HTTP/connection failures are normalized as
retryable tool errors. Returned payloads are mapped into typed Pydantic outputs,
and remote file/page reads use the same character range and readable-size
controls as filesystem reads.

#### `search_gitlab_code`

**Purpose**: Search one GitLab project for code matches.

**Inputs And Trust Boundaries**: The model supplies project, query, optional
ref, and limit. GitLab is an external trust boundary. Credentials come from
scoped runtime secrets.

**Algorithm**: Require the GitLab gateway, resolve the project, normalize the
project name, call GitLab code search for blobs with `limit + 1`, map paths,
names, refs, start lines, and snippets into typed matches, and mark truncation
when the extra record is present.

**Required Capabilities**: Network access and `GITLAB_BASE_URL` /
`GITLAB_API_TOKEN`. Feature-gated in the assistant by `gitlab_tools_enabled`.

**Security-Relevant Controls**: Required scoped secrets, external-read
side-effect class, network permission, 30-second timeout, bounded result count,
typed output mapping, and retryable transient failure normalization.

**Persistence, Cache, Logs, And Provenance**: Does not cache or persist remote
content. Logs the search action. It returns snippets but does not add source
provenance for each match.

**Failure Behavior**: Unsupported client methods, project lookup failures,
transient remote errors, and policy denial become normalized tool errors.

**Residual Risks**: Search snippets can reveal repository content from remote
systems. The tool depends on the configured GitLab token's upstream
authorization.

**Test Evidence**: GitLab tests cover result mapping, truncation, fake client
compatibility paths, and runtime policy.

#### `read_gitlab_file`

**Purpose**: Read one UTF-8 text file from a GitLab project.

**Inputs And Trust Boundaries**: The model supplies project, file path, optional
ref, and optional character range. GitLab content and credentials cross the
external trust boundary.

**Algorithm**: Require the GitLab gateway, resolve the project, choose the
requested ref or project default branch or `HEAD`, fetch the file, decode text
from the GitLab object or base64 content, reject binary and non-UTF-8 payloads,
enforce readable character limits, normalize the range, cap the returned slice,
and return a resolved `project@ref:path` identity.

**Required Capabilities**: Network access and GitLab secrets. Feature-gated in
the assistant by `gitlab_tools_enabled`.

**Security-Relevant Controls**: Required scoped secrets, external-read side
effect, timeout, binary/UTF-8 checks, character limits, bounded range reads,
typed output validation, and source provenance for successful reads.

**Persistence, Cache, Logs, And Provenance**: Does not cache remote files.
Successful reads log the action, add an artifact identity, and add remote source
provenance.

**Failure Behavior**: Binary or non-UTF-8 content returns structured
`unsupported` output. Too-large text returns structured `too_large` output.
Remote failures become normalized errors.

**Residual Risks**: The tool can reveal any file the configured GitLab token can
read in the requested project. Ref defaults may read the current default branch
when the model omits a ref.

**Test Evidence**: Tests cover text reads, range handling, default branch
selection, binary handling, and structured unsupported output.

#### `read_gitlab_merge_request`

**Purpose**: Read metadata, commits, and changed-file summaries for one GitLab
merge request.

**Inputs And Trust Boundaries**: The model supplies project, merge request IID,
commit limit, and change limit. GitLab is the external trust boundary.

**Algorithm**: Require the GitLab gateway, resolve the project, fetch the merge
request, collect commits and changes through supported client APIs, map selected
metadata into typed output, truncate commits and changed files independently,
and limit diff excerpts to 400 characters per changed file.

**Required Capabilities**: Network access and GitLab secrets. Feature-gated in
the assistant by `gitlab_tools_enabled`.

**Security-Relevant Controls**: Required scoped secrets, external-read side
effect, timeout, bounded commit/change lists, bounded diff excerpts, typed
output validation, and retryable transient failure normalization.

**Persistence, Cache, Logs, And Provenance**: Does not cache or persist remote
content. Logs the read action. It does not add source provenance because the
output is review metadata rather than a single readable source file.

**Failure Behavior**: Unsupported client methods and remote failures become
normalized errors. Over-limit commits or changed files are returned with
truncation flags.

**Residual Risks**: Merge request descriptions, branch names, commit titles, and
diff excerpts can contain sensitive information.

**Test Evidence**: Tests cover merge request metadata mapping, commit/change
mapping, and truncation behavior.

#### `search_jira`

**Purpose**: Search Jira issues with JQL and return bounded issue summaries.

**Inputs And Trust Boundaries**: The model supplies JQL and limit. Jira is an
external trust boundary. Credentials are scoped runtime secrets.

**Algorithm**: Require the Jira gateway, call `enhanced_jql` when available or
fall back to `jql`, request `limit + 1` issues, extract an allowlisted summary
shape from each issue, return only the requested limit, and mark truncation when
an extra issue was fetched.

**Required Capabilities**: Network access and `JIRA_BASE_URL` /
`JIRA_API_TOKEN`. Feature-gated in the assistant by `atlassian_tools_enabled`.

**Security-Relevant Controls**: Required scoped secrets, external-read side
effect, timeout, bounded result count, summary-shaped output instead of raw
field maps, typed output validation, and retryable transient failure
normalization.

**Persistence, Cache, Logs, And Provenance**: Does not cache or persist remote
content. Logs the JQL search.

**Failure Behavior**: Missing credentials fail policy. Unsupported clients and
remote errors become normalized tool errors. Over-limit results return
`truncated=true`.

**Residual Risks**: JQL is user/model-controlled and can reveal issue metadata
within the token's Jira permissions.

**Test Evidence**: Atlassian tests cover enhanced JQL, fallback JQL, truncation,
missing credential policy denial, unsupported clients, and retryable transient
failures.

#### `read_jira_issue`

**Purpose**: Read one Jira issue by key with a controlled summary plus
explicitly requested extra fields.

**Inputs And Trust Boundaries**: The model supplies issue key and optional
requested field names. Jira is an external trust boundary.

**Algorithm**: Require the Jira gateway, fetch the issue through `issue` or
`get_issue`, extract key, summary, description, status, issue type, assignee,
and only requested fields that exist in the raw field map.

**Required Capabilities**: Network access and Jira secrets. Feature-gated in the
assistant by `atlassian_tools_enabled`.

**Security-Relevant Controls**: Required scoped secrets, external-read side
effect, timeout, field allowlisting by default, duplicate/blank requested-field
validation, typed output validation, and transient failure normalization.

**Persistence, Cache, Logs, And Provenance**: Does not cache or persist remote
content. Logs the issue read.

**Failure Behavior**: Missing credentials fail policy. Unsupported clients,
invalid requested fields, and remote errors become normalized errors.

**Residual Risks**: Descriptions and explicitly requested custom fields can
contain sensitive data. The upstream token's Jira permissions determine what is
readable.

**Test Evidence**: Tests cover issue mapping, fallback client method,
requested-field validation, unsupported clients, and credential policy denial.

#### `search_bitbucket_code`

**Purpose**: Search Bitbucket Server/DC code within one project.

**Inputs And Trust Boundaries**: The model supplies project key, query, and
limit. Bitbucket is an external trust boundary.

**Algorithm**: Require the Bitbucket gateway, call code search with `limit + 1`,
extract collection payloads from common response shapes, map repository slug,
path, line number, and snippet, return the requested limit, and mark
truncation.

**Required Capabilities**: Network access and `BITBUCKET_BASE_URL` /
`BITBUCKET_API_TOKEN`. Feature-gated in the assistant by
`atlassian_tools_enabled`.

**Security-Relevant Controls**: Required scoped secrets, external-read side
effect, timeout, bounded result count, typed output validation, and retryable
transient failure normalization.

**Persistence, Cache, Logs, And Provenance**: Does not cache or persist remote
content. Logs the search action.

**Failure Behavior**: Remote errors become normalized tool errors. Over-limit
results return `truncated=true`.

**Residual Risks**: Search snippets can reveal sensitive repository content
available to the configured Bitbucket token.

**Test Evidence**: Tests cover result mapping and truncation via extra fetched
matches.

#### `read_bitbucket_file`

**Purpose**: Read one UTF-8 text file from Bitbucket Server/DC.

**Inputs And Trust Boundaries**: The model supplies project key, repository
slug, path, optional ref, and optional character range. Bitbucket is the
external trust boundary.

**Algorithm**: Require the Bitbucket gateway, choose the requested ref or
`HEAD`, fetch file bytes/content, normalize bytes or text to UTF-8 text, reject
binary or non-UTF-8 content, enforce readable character limits, normalize and
cap the range, and return a resolved `project/repo@ref:path` identity.

**Required Capabilities**: Network access and Bitbucket secrets. Feature-gated
in the assistant by `atlassian_tools_enabled`.

**Security-Relevant Controls**: Required scoped secrets, external-read side
effect, timeout, binary/UTF-8 checks, character limits, bounded ranges, typed
output validation, and source provenance for successful reads.

**Persistence, Cache, Logs, And Provenance**: Does not cache remote files.
Successful reads log the action, add an artifact identity, and add remote source
provenance.

**Failure Behavior**: Binary or non-UTF-8 content returns structured
`unsupported` output. Too-large content returns `too_large`. Remote failures
become normalized errors.

**Residual Risks**: The tool can reveal files readable by the configured
Bitbucket token. Default `HEAD` may be ambiguous across repositories.

**Test Evidence**: Tests cover text reads, range truncation, binary/UTF-8
handling, and source output shaping.

#### `read_bitbucket_pull_request`

**Purpose**: Read metadata, commits, and changed-file summaries for one
Bitbucket pull request.

**Inputs And Trust Boundaries**: The model supplies project key, repository
slug, pull request id, and list limits. Bitbucket is the external trust
boundary.

**Algorithm**: Require the Bitbucket gateway, fetch pull request metadata,
commits, and changed files, extract common payload fields, return bounded commit
and changed-file lists, and mark independent truncation flags.

**Required Capabilities**: Network access and Bitbucket secrets. Feature-gated
in the assistant by `atlassian_tools_enabled`.

**Security-Relevant Controls**: Required scoped secrets, external-read side
effect, timeout, bounded commits and changed-file lists, typed output
validation, and retryable transient failure normalization.

**Persistence, Cache, Logs, And Provenance**: Does not cache or persist remote
content. Logs the pull request read.

**Failure Behavior**: Remote failures become normalized errors. Over-limit
commit/change collections return truncation flags.

**Residual Risks**: Pull request metadata, commit messages, branch names, and
changed-file names can contain sensitive content.

**Test Evidence**: Tests cover pull request mapping, commit/change mapping, and
shared helper extraction behavior.

#### `search_confluence`

**Purpose**: Search Confluence content with CQL.

**Inputs And Trust Boundaries**: The model supplies CQL and limit. Confluence is
an external trust boundary.

**Algorithm**: Require the Confluence gateway, call CQL search with `limit + 1`
and highlighted excerpts, extract content metadata from common response shapes,
build absolute web URLs from the configured base URL, return the requested
limit, and mark truncation.

**Required Capabilities**: Network access and `CONFLUENCE_BASE_URL` /
`CONFLUENCE_API_TOKEN`. Feature-gated in the assistant by
`atlassian_tools_enabled`.

**Security-Relevant Controls**: Required scoped secrets, external-read side
effect, timeout, bounded result count, typed output validation, and retryable
transient failure normalization.

**Persistence, Cache, Logs, And Provenance**: Does not cache or persist remote
content. Logs the CQL search.

**Failure Behavior**: Remote failures become normalized errors. Over-limit
results return `truncated=true`.

**Residual Risks**: CQL and excerpts can reveal Confluence content according to
the configured token's permissions.

**Test Evidence**: Tests cover result mapping, absolute URL construction, and
truncation behavior.

#### `read_confluence_page`

**Purpose**: Read the body of one Confluence page.

**Inputs And Trust Boundaries**: The model supplies page id and optional
character range. Confluence is the external trust boundary.

**Algorithm**: Require the Confluence gateway, fetch the page with storage body,
space, and links expanded, extract body content from supported representation
shapes, build an absolute web URL, apply shared text read limits and range
logic, add source provenance, and return page metadata plus bounded content.

**Required Capabilities**: Network access and Confluence secrets. Feature-gated
in the assistant by `atlassian_tools_enabled`.

**Security-Relevant Controls**: Required scoped secrets, external-read side
effect, timeout, body extraction validation, character limits, bounded ranges,
typed output validation, and source provenance.

**Persistence, Cache, Logs, And Provenance**: Does not cache page bodies.
Successful reads log the action, add an artifact identity, and add remote source
provenance.

**Failure Behavior**: Missing body content returns structured error status via
the shared text result shape. Remote failures become normalized errors.

**Residual Risks**: Page body storage can include sensitive content and markup.
The upstream token controls what is readable.

**Test Evidence**: Tests cover page body extraction, range truncation, absolute
URL construction, and fallback body error behavior.

#### `read_confluence_attachment`

**Purpose**: Read one attachment from a Confluence page, including supported
document conversion.

**Inputs And Trust Boundaries**: The model supplies page id and exactly one
attachment selector: id or filename. Confluence is an external trust boundary,
and the local attachment cache is a filesystem boundary.

**Algorithm**: Require the Confluence gateway and base URL secret. Fetch page
metadata. Resolve the attachment by filename or id; input validation rejects
missing or duplicate selectors. Compute a cache key from base URL, page id, and
attachment id; sanitize the filename for cache storage. If cache metadata
matches attachment version, size, and title, reuse the cached bytes. Otherwise
download attachment bytes to memory, write them to the internal temp cache, and
write metadata. Load readable content from the cached file using the shared
readable-content algorithm, including document conversion and character-range
limits. Return page, attachment, URL, and bounded content metadata.

**Required Capabilities**: Network access, filesystem access, local-write
side-effect permission, and Confluence secrets. Feature-gated in the assistant
by `atlassian_tools_enabled`; as a local-write tool it is also subject to
write-side-effect policy and approval rules.

**Security-Relevant Controls**: Required scoped secrets, explicit selector
validation, sanitized cache filename, cache freshness metadata, temp-directory
internal cache, readable byte/character limits, conversion error handling,
timeout, source provenance, and local-write side-effect classification.

**Persistence, Cache, Logs, And Provenance**: Writes downloaded attachment bytes
and conversion metadata to the internal Confluence attachment cache under the
platform temp directory. Successful reads log the action, add a remote artifact
identity, and add source provenance.

**Failure Behavior**: Missing selectors, duplicate selectors, missing
attachments, failed downloads, invalid cache metadata, unsupported content, and
conversion failures become validation errors, normalized remote errors, or
structured read statuses depending on failure point.

**Residual Risks**: This tool stores remote attachment bytes in a local temp
cache. Attachments may be large or parser-hostile, though byte limits and
conversion controls constrain readable output. Operators should treat the cache
as sensitive and rely on filesystem policy and approval gates before enabling
the tool.

**Test Evidence**: Tests cover attachment selector validation, selection by id
or filename, cache reuse, download payload variants, invalid cache metadata,
filename sanitization helpers, conversion reuse, and missing attachment errors.

### Assistant Product Capabilities

The assistant app supports persistent chats, temporary chats, provider settings,
model discovery, response-mode selection, session permissions, approvals,
session-scoped credentials, workspace selection, context compaction, structured
final responses, workbench inspection, local users, admin settings, and admin
branding in the normal product surface.

The assistant also contains implemented default-hidden capabilities:
information protection (`information_protection_enabled`), local and bundled
skills (`skills_enabled`), native Ollama (`ollama_native_provider_enabled`),
native Ask Sage (`ask_sage_native_provider_enabled`), the `write_file` tool
(`write_file_tool_enabled`), Atlassian tools (`atlassian_tools_enabled`), GitLab
tools (`gitlab_tools_enabled`), and Deep Task mode (`deep_task_mode_enabled`).

This document intentionally does not describe button-by-button usage. The
architectural point is that assistant UI state is translated into runtime
assembly inputs, while tool execution remains mediated by lower layers.

### Bundled Skills

The package ships bundled example skills under `skills_api/bundled/`. Bundled
skills have the lowest skill-scope precedence. They are instructions and
supporting files, not hidden runtime capabilities. Assistant discovery and use
of bundled skills is nevertheless hidden by default behind `skills_enabled`, the
same gate used for project, user, and enterprise skills. Loading a skill does
not bypass tool policy, approvals, filesystem permissions, shell restrictions,
network restrictions, or credential rules.

## Dependency And Supply Chain Inventory

### Inventory Generation

`scripts/generate_dependency_inventory.py` reads `pyproject.toml`, curated
purpose text from `docs/dependency-purposes.toml`, and installed distribution
metadata from the active Python environment.

The supported outputs are:

```bash
python scripts/generate_dependency_inventory.py --format markdown
python scripts/generate_dependency_inventory.py --format cyclonedx-json
```

`make dependency-inventory` prints the direct-dependency Markdown table.
`make sbom` emits a full transitive CycloneDX JSON SBOM for the selected
environment. Generated SBOM files are not committed without a lockfile because
the transitive environment is not otherwise pinned.

The generator is tolerant by default. `--strict` fails when direct dependency
metadata or curated purpose text is missing. CycloneDX output requires installed
metadata because it describes an actual environment.

### Direct Dependency Table

The following table is generated from the current development environment and
curated purpose file. Runtime dependencies are package requirements for
installed behavior. Development dependencies support local engineering
workflows.

### Runtime Dependencies

| Package | Constraint | Installed | Purpose | Provenance | License |
| --- | --- | --- | --- | --- | --- |
| argon2-cffi | argon2-cffi>=23,<26 | 25.1.0 | Password hashing for the assistant app's local username/password authentication. | https://argon2-cffi.readthedocs.io/ | MIT |
| atlassian-python-api | atlassian-python-api>=4,<5 | 4.0.7 | Client library for bundled Jira, Confluence, and Bitbucket read integrations, which are default-hidden in the assistant behind atlassian_tools_enabled. | https://github.com/atlassian-api/atlassian-python-api | Apache Software License |
| cryptography | cryptography>=42,<47 | 46.0.7 | Authenticated encryption and key wrapping for assistant persistence. | https://github.com/pyca/cryptography/ | Apache-2.0 OR BSD-3-Clause |
| instructor | instructor>=1,<2 | 1.12.0 | Structured response parsing support for OpenAI-compatible provider interactions. | https://github.com/instructor-ai/instructor | MIT |
| markitdown | markitdown>=0.1.5,<0.2 | 0.1.5 | Document-to-Markdown conversion for readable filesystem file access. | https://github.com/microsoft/markitdown | MIT |
| mpxj | mpxj>=16,<17 | 16.1.0 | Microsoft Project file conversion support in the filesystem read pipeline. | https://github.com/joniles/mpxj | LGPL-2.0-or-later |
| nicegui | nicegui>=3,<4 | 3.11.1 | Browser UI runtime for the LLM Tools Assistant app. | https://github.com/zauberzeug/nicegui | MIT |
| ollama | ollama>=0.6,<0.7 | 0.6.2 | Native Ollama provider transport for local model endpoints; assistant exposure is default-hidden behind ollama_native_provider_enabled. | https://github.com/ollama/ollama-python | MIT |
| openai | openai>=1,<2 | 1.109.1 | OpenAI-compatible model transport and native OpenAI API client support. | https://github.com/openai/openai-python | Apache Software License |
| pathspec | pathspec>=0.12,<0.13 | 1.0.4 | Gitignore-style hidden and ignored path matching for workspace discovery. | https://github.com/cpburnz/python-pathspec | Mozilla Public License 2.0 (MPL 2.0) |
| pydantic | pydantic>=2,<3 | 2.13.2 | Typed contract boundary for tool, workflow, harness, provider, app, and persistence models. | https://github.com/pydantic/pydantic | MIT |
| python-gitlab | python-gitlab>=4,<6 | 5.6.0 | Client library for bundled GitLab read integrations, which are default-hidden in the assistant behind gitlab_tools_enabled. | https://github.com/python-gitlab/python-gitlab | GNU Lesser General Public License v3 (LGPLv3) |
| PyYAML | PyYAML>=6,<7 | 6.0.3 | YAML parsing for assistant configuration files and skill manifests. | https://github.com/yaml/pyyaml | MIT License |
| SQLAlchemy | SQLAlchemy>=2.0,<2.1 | 2.0.49 | Synchronous database access layer for SQLCipher-backed assistant persistence. | https://docs.sqlalchemy.org | MIT |
| sqlcipher3-wheels | sqlcipher3-wheels>=0.5.7,<0.6 | 0.5.7 | SQLCipher-backed SQLite driver for encrypted assistant persistence. | https://github.com/laggykiller/sqlcipher3 | zlib/libpng |

### Development Dependencies

| Package | Constraint | Installed | Purpose | Provenance | License |
| --- | --- | --- | --- | --- | --- |
| build | build>=1.2,<2 | 1.4.3 | Package build frontend used by the local packaging workflow. | https://github.com/pypa/build | MIT |
| mypy | mypy>=1.10,<2 | 1.20.1 | Strict static type checking for the Python package. | https://github.com/python/mypy | MIT |
| pytest | pytest>=8.3,<9 | 8.4.2 | Test runner for unit, integration, architecture, and example tests. | https://github.com/pytest-dev/pytest | MIT License |
| pytest-cov | pytest-cov>=5,<7 | 6.3.0 | Coverage reporting integration for pytest. | https://pytest-cov.readthedocs.io/ | MIT |
| ruff | ruff>=0.6,<1 | 0.15.11 | Formatter and linter for source and test code. | https://github.com/astral-sh/ruff | MIT |
| setuptools | setuptools>=82,<83 | 82.0.1 | Build backend and editable-install support for the src-layout package. | https://github.com/pypa/setuptools | MIT |
| vulture | vulture>=2.16,<3 | 2.16 | Dead-code review lead generation for reachability maintenance. | https://github.com/jendrikseipp/vulture | MIT License |
| wheel | wheel>=0.46,<0.47 | 0.46.3 | Wheel distribution build support. | https://github.com/pypa/wheel | MIT |

## Implementation Source Map

This map connects the architectural sections above to implementation and test
evidence. It is intentionally package and module oriented rather than
line-number oriented: exact class and function locations can move without
changing the design, but the owning module boundaries should remain stable.

| Design Area | Primary Implementation Modules | Test Evidence | Notes |
| --- | --- | --- | --- |
| Package shape and public surfaces | `src/llm_tools/__init__.py`, public `__init__.py` files under `tool_api`, `llm_adapters`, `llm_providers`, `tools`, `skills_api`, `workflow_api`, and `harness_api` | `tests/test_package_imports.py`, `tests/architecture/test_private_surface_hygiene.py` | Public package exports are the supported library surface. |
| Layering and dependency direction | All packages under `src/llm_tools/` | `tests/architecture/test_layering.py`, `tests/architecture/test_model_file_boundaries.py` | Architecture tests are the enforceable source for allowed import direction. |
| Tool contracts and typed execution substrate | `src/llm_tools/tool_api/models.py`, `tool.py`, `registry.py`, `policy.py`, `policy_models.py`, `execution.py`, `runtime.py`, `redaction.py`, `redaction_models.py`, `errors.py` | `tests/tool_api/test_models.py`, `test_tool.py`, `test_registry.py`, `test_policy.py`, `test_execution.py`, `test_runtime.py`, `test_redaction.py` | Covers tool definition, validation, policy, execution records, redaction, and runtime mediation. |
| Runtime mediation and no direct tool invocation | `src/llm_tools/tool_api/runtime.py`, `src/llm_tools/tool_api/execution.py`, `src/llm_tools/apps/assistant_execution.py`, `src/llm_tools/workflow_api/executor.py` | `tests/architecture/test_runtime_mediation.py`, `tests/architecture/test_no_direct_tool_invocation.py`, `tests/tools/test_runtime_integration.py` | These tests protect the central security invariant that model requests pass through `ToolRuntime`. |
| Model-output adapters | `src/llm_tools/llm_adapters/base.py`, `base_models.py`, `action_envelope.py`, `action_envelope_models.py`, `prompt_tools.py` | `tests/llm_adapters/test_base.py`, `test_action_envelope.py`, `test_prompt_tools.py` | Adapter code owns provider-output parsing, not provider transport or tool execution. |
| Provider transports and response modes | `src/llm_tools/llm_providers/openai_compatible.py`, `openai_compatible_models.py`, `ollama_native.py`, `ollama_native_models.py`, `ask_sage_native.py` | `tests/llm_providers/test_openai_compatible.py`, `test_openai_compatible_extra.py`, `test_ollama_native.py`, `test_ask_sage_native.py` | Native Ollama and native Ask Sage are implemented but default-hidden in the assistant behind provider feature gates. |
| Skills discovery, loading, resolution, and rendering | `src/llm_tools/skills_api/discovery.py`, `loading.py`, `resolution.py`, `rendering.py`, `models.py`, `errors.py`, bundled skills under `src/llm_tools/skills_api/bundled/` | `tests/skills_api/test_skills_api.py` | Assistant skill integration is default-hidden behind `skills_enabled`; the library API remains available. |
| Workflow turn execution | `src/llm_tools/workflow_api/executor.py`, `models.py`, `model_turn_protocol.py`, `staged_structured.py` | `tests/workflow_api/test_executor.py`, `test_executor_additional.py`, `test_model_turn_protocol.py`, `test_staged_structured.py` | Owns one-turn model interaction, parsed tool invocation execution, repair/fallback, and transient approvals. |
| Interactive chat state and inspection | `src/llm_tools/workflow_api/chat_models.py`, `chat_state.py`, `chat_session.py`, `chat_runner.py`, `chat_inspector.py`; app presentation helpers in `src/llm_tools/apps/chat_runtime.py`, `chat_presentation.py`, `chat_config.py`, `chat_config_models.py` | `tests/workflow_api/test_chat_session.py`, `test_chat_models_extra.py`, `tests/apps/test_chat_shared.py`, `tests/tools/test_chat.py`, `tests/tools/test_chat_helpers.py` | Covers provider-visible projection, transcript state, chat helpers, and UI-facing presentation boundaries. |
| Protection subsystem algorithms | `src/llm_tools/workflow_api/protection.py`, `protection_models.py`, `protection_controller.py`, `protection_provenance.py`, `protection_store.py`, `protection_store_models.py`; app assembly in `src/llm_tools/apps/protection_runtime.py`, `protection_runtime_models.py`; harness scrubbing in `src/llm_tools/harness_api/protection.py` | `tests/workflow_api/test_protection.py`, `tests/apps/test_protection_runtime.py`, `tests/apps/test_assistant_runtime_protection.py`, `tests/harness_api/test_protection_scrub.py`, `tests/scripts/test_backend_matrix.py` | Experimental and default-hidden behind `information_protection_enabled`; only assembled when runtime protection config also permits it. |
| Harness state, execution, replay, and resume | `src/llm_tools/harness_api/models.py`, `executor.py`, `executor_loop.py`, `executor_persistence.py`, `session.py`, `session_service.py`, `store.py`, `resume.py`, `replay.py`, `tasks.py`, `context.py`, `approval_context.py`, `verification.py` and companion `*_models.py` files | `tests/harness_api/test_harness_models.py`, `test_harness_executor.py`, `test_session_api.py`, `test_session_additional_coverage.py`, `test_store.py`, `test_resume.py`, `test_replay_golden.py`, `test_task_lifecycle.py`, `test_context.py`, `test_verification_models.py` | Deep Task uses this layer through the assistant when `deep_task_mode_enabled` is enabled. |
| Harness planning and deterministic task selection | `src/llm_tools/harness_api/planning.py`, `planning_models.py`, `defaults.py` | `tests/harness_api/test_planning.py` | Covers deterministic planning, selected task control, and related harness defaults. |
| Assistant runtime assembly | `src/llm_tools/apps/assistant_runtime.py`, `assistant_execution.py`, `assistant_tool_registry.py`, `assistant_tool_capabilities.py`, `assistant_tool_capabilities_models.py`, `assistant_prompts.py`, `assistant_research_provider.py`, `chat_runtime.py` | `tests/apps/test_assistant_runtime.py`, `tests/apps/test_assistant_config_prompts.py`, `tests/tools/test_registration.py` | The app composes lower layers and filters visible tools/providers by feature flags, policy, credentials, and workspace readiness. |
| Assistant UI, settings, feature gates, and launch | `src/llm_tools/apps/assistant_app/app.py`, `__main__.py`, `ui.py`, `controller.py`, `controller_core.py`, `models.py`, `features.py`, `project_defaults.py`, `provider_endpoints.py`, `paths.py`; config modules `assistant_config.py`, `assistant_config_models.py` | `tests/apps/test_assistant_app.py`, `tests/apps/test_launch.py`, `tests/apps/test_assistant_config_prompts.py`, `tests/examples/test_assistant_config_examples.py` | Feature flags are modeled as administrator settings and default to disabled for beta or security-sensitive capabilities. |
| Assistant authentication, credentials, encryption, and persistence | `src/llm_tools/apps/assistant_app/auth.py`, `crypto.py`, `store.py`, `store_sqlite.py`, `models.py`, `paths.py` | `tests/apps/test_assistant_app_store.py`, `tests/apps/test_assistant_app.py` | Covers local auth, SQLCipher-backed persistence, user-key wrapping, in-memory credential handling, and app-owned data models. |
| Harness CLI product entrypoint | `src/llm_tools/apps/harness_cli.py` | `tests/apps/test_harness_cli.py`, `tests/examples/test_examples_smoke.py` | CLI composes harness/workflow/provider primitives without assistant UI state. |
| Filesystem tools and readable-content pipeline | `src/llm_tools/tools/filesystem/`, `src/llm_tools/tools/_path_utils.py` | `tests/tools/test_filesystem.py`, `test_path_utils.py`, `test_project_reads.py`, `test_text.py`, `test_chat_models_paths_extra.py` | `write_file` is implemented but default-hidden in the assistant behind `write_file_tool_enabled`; read/conversion caches are internal filesystem surfaces. |
| Git tools | `src/llm_tools/tools/git/` | `tests/tools/test_git.py` | Fixed non-interactive subprocess commands are mediated by runtime policy and workspace repository resolution. |
| GitLab tools | `src/llm_tools/tools/gitlab/` | `tests/tools/test_gitlab.py` | Implemented remote read integrations; default-hidden in the assistant behind `gitlab_tools_enabled` and dependent on scoped GitLab credentials. |
| Atlassian tools | `src/llm_tools/tools/atlassian/` | `tests/tools/test_atlassian.py` | Implemented Jira, Bitbucket, and Confluence integrations; default-hidden in the assistant behind `atlassian_tools_enabled` and dependent on scoped service credentials. |
| Built-in tool registration | `src/llm_tools/tools/filesystem/tools.py`, `src/llm_tools/tools/git/tools.py`, `src/llm_tools/tools/gitlab/tools.py`, `src/llm_tools/tools/atlassian/tools.py`, `src/llm_tools/apps/assistant_tool_registry.py` | `tests/tools/test_registration.py`, `tests/tools/test_runtime_integration.py` | Registration is separate from model exposure; app runtime filters visibility before tool specs reach the model. |
| Dependency inventory and SBOM generation | `scripts/generate_dependency_inventory.py`, `docs/dependency-purposes.toml`, `pyproject.toml`, `Makefile` | `tests/scripts/test_dependency_inventory.py` | `make dependency-inventory` emits the documentation table; `make sbom` emits a transitive CycloneDX JSON artifact for the active environment. |
| Backend behavior matrix and scripted evaluations | `scripts/` evaluation helpers where present | `tests/scripts/test_backend_matrix.py`, `test_gemini_agent_eval.py`, `test_ollama_skill_eval.py` | These tests exercise selected app/workflow behavior across provider-style scenarios and scripted examples. |

## Verification And Decision Records

### Test Strategy

The project uses:

- Ruff for formatting and linting.
- mypy in strict mode for package type checking.
- pytest for unit, integration, architecture, app, example, and script tests.
- pytest-cov plus a per-file coverage checker.
- Vulture as a dead-code review lead, not an automatic deletion oracle.

Architecture tests enforce dependency direction, runtime mediation, model-file
boundaries, public/private surface hygiene, and no direct tool invocation
outside the runtime path. Harness tests cover model integrity, replay, resume,
store behavior, approvals, planning, verification, and golden traces. App tests
cover assistant runtime assembly, persistence, launch, protection runtime, and
configuration examples.

### Decision Records

ADRs live in `docs/adr/`. They are useful rationale references for surprising
or hard-to-reverse decisions, including tool runtime mediation, canonical
harness state, fail-closed recovery, encrypted assistant persistence,
session-memory credentials, local auth by default, provider/adapter separation,
prompt-tool support, staged JSON tool use, workflow-level protection,
workspace-relative paths, public facades over split internals, default-deny
assistant permissions, deterministic harness planning, NiceGUI app runtime,
tested implementation as documentation truth, runtime-owned execution service
wiring, provider-neutral harness state, and explicit harness schema versioning.

This design document does not restate every ADR. When a future change revises a
hard-to-reverse architectural decision, update the relevant design section and
add or supersede the ADR as appropriate.

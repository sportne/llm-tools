# llm-tools Context

This context captures the domain language for `llm-tools`: typed tools, model
turns, workflow execution, durable harness sessions, and assistant product
entrypoints.

## Language

**Tool**:
A typed callable capability exposed to a model and executed through the runtime.
_Avoid_: function, plugin

**Skill**:
A discoverable instruction package that teaches an agent a repeatable workflow and may reference supporting files, scripts, examples, or existing **Tools**.
_Avoid_: tool, plugin, prompt snippet

**Skill Package**:
A directory-based **Skill** distribution containing a required `SKILL.md` manifest and optional supporting files.
_Avoid_: command file, prompt file, tool module

**Skill Metadata**:
The lightweight discoverability record for a **Skill**, including its name, description, source, scope, and validation status.
_Avoid_: loaded skill, full skill

**Loaded Skill**:
A **Skill** whose full instructions have been selected for use in the current agent context.
_Avoid_: discovered skill, installed skill

**Skill Scope**:
The authority level of a discovered **Skill**, used to resolve same-name skills across enterprise, user, project, and bundled sources.
_Avoid_: folder, location

**Skill Invocation**:
An explicit request to use a **Skill** by name for the current agent task.
_Avoid_: selection, discovery

**Skill Selection**:
A runtime or agent decision to use a **Skill** because its **Skill Metadata** is relevant to the current task.
_Avoid_: invocation, discovery

**Available Skills Context**:
A budgeted model-visible catalog of discovered **Skill Metadata** used to support **Skill Selection**.
_Avoid_: loaded skills, skill body

**Loaded Skill Context**:
The model-visible contribution for one **Loaded Skill**, including its name, source path, and full `SKILL.md` instructions.
_Avoid_: available skills list, metadata

**Skill Dependency**:
A declarative requirement that a **Skill** expects from the surrounding agent environment, such as a **Tool**, connector, or MCP server.
_Avoid_: installed tool, enabled app

**Skill Discovery Result**:
The outcome of scanning skill roots, containing valid **Skill Metadata** records plus path-specific validation errors.
_Avoid_: loaded skills, registry

**Skill Resolution**:
The process of choosing one effective **Skill** from discovered candidates by invocation name, path, scope, and ambiguity rules.
_Avoid_: discovery, selection

**Skill Enablement**:
A user- or administrator-controlled availability setting that can disable an otherwise valid and policy-allowed **Skill**.
_Avoid_: validation, policy permission

**Skill Usage Record**:
A durable metadata record that a **Skill** was invoked or selected for a turn, without persisting the full skill body by default.
_Avoid_: loaded skill context, transcript copy

**Tool Runtime**:
The policy-aware execution substrate that validates, mediates, invokes, and normalizes **Tool** calls.
_Avoid_: executor, runner

**Tool Policy**:
The session or runtime rules that determine whether a **Tool** may be exposed, executed, denied, or routed through approval.
_Avoid_: permission flags, UI settings

**Tool Exposure**:
The act of making a registered **Tool** visible and callable by the model for the current session.
_Avoid_: registration, installation, availability

**Approval Request**:
A workflow-level request for an operator decision before executing one policy-gated **Tool** invocation.
_Avoid_: permission prompt, confirmation dialog

**Execution Services**:
The mediated host capabilities, credentials, filesystem access, subprocess access, and remote gateways supplied to **Tools** by the **Tool Runtime**.
_Avoid_: dependencies, clients, service locator

**Workspace**:
The configured filesystem root that scopes model-visible local file operations for a session or tool invocation.
_Avoid_: project, cwd, repository when the selected root may differ

**Workspace-Relative Tool Path**:
A POSIX-style path used in model-visible filesystem tool arguments and outputs, resolved relative to the **Workspace**.
_Avoid_: absolute path, native path, Windows path

**Discovery**:
A broad listing or search operation that surfaces candidate files from a **Workspace**.
_Avoid_: direct read, explicit access

**Direct Access**:
A focused file operation against a caller-provided **Workspace-Relative Tool Path**.
_Avoid_: discovery, search result

**Hidden/Ignored Path**:
A dot-hidden or ignore-file-matched workspace path excluded from default **Discovery** but still eligible for policy-governed **Direct Access** when explicitly named.
_Avoid_: forbidden path, inaccessible path

**Model Turn**:
One model response cycle that may produce a final response or one or more **Tool** invocations.
_Avoid_: request, completion

**Model-Turn Protocol**:
The model-facing contract used to obtain and parse a **Model Turn**, including native structured output, staged schemas, and prompt-emitted tool calls.
_Avoid_: response mode, strategy

**Model-Turn Event**:
A user- or operator-visible redacted progress record emitted while a **Model-Turn Protocol** is producing a **Model Turn**.
_Avoid_: UI event, chat event

**Protection**:
Workflow-level handling of sensitive information before, during, and after **Model Turns**.
_Avoid_: UI moderation, content filter

**Sensitivity Category**:
A configured label describing a kind of protected information that **Protection** decisions can detect and enforce.
_Avoid_: tag, policy rule

**Protection Decision**:
A workflow decision to allow, constrain, challenge, sanitize, or block protected prompt or response content.
_Avoid_: moderation result, classifier output

**Source Provenance**:
A structured reference to source material that was visible to tool execution or **Protection** checks.
_Avoid_: citation, artifact

**Workflow Turn**:
One parsed **Model Turn** executed through the workflow layer against the **Tool Runtime**.
_Avoid_: chat turn

**Harness Session**:
A durable, resumable record of multi-turn work built from persisted **Workflow Turn** results.
_Avoid_: job, run

**Harness State**:
The canonical durable truth for one **Harness Session**, including session metadata, **Harness Tasks**, **Harness Turns**, verification evidence, and pending approvals.
_Avoid_: trace, summary, replay artifact

**Harness Turn**:
One persisted orchestration step in a **Harness Session** that records selected **Harness Tasks**, the embedded **Workflow Turn** result, approval audit metadata, verification status, no-progress signals, and the next **Turn Decision**.
_Avoid_: workflow turn, model turn, chat turn

**Harness Task**:
A durable unit of work tracked by the harness task graph, with lifecycle status, origin, dependencies, verification expectations, artifacts, and retry state.
_Avoid_: todo, plan item, task record when speaking outside code

**Turn Decision**:
The harness-level decision after a **Harness Turn** completes, indicating whether to continue, select tasks, or stop with a reason.
_Avoid_: model decision, tool result

**Pending Approval**:
A durable pause in a **Harness Session** waiting for an operator decision on one policy-gated tool invocation.
_Avoid_: approval dialog, permission prompt

**Resume Disposition**:
The resume-time classification of a persisted **Harness State**, such as runnable, waiting for approval, interrupted, terminal, incompatible, or corrupt.
_Avoid_: resume status, session status

**Verification Expectation**:
A task-level statement of what evidence or condition should be checked for a **Harness Task**.
_Avoid_: acceptance criterion when referring to durable harness state

**Verification Evidence**:
A persisted record supporting the verification outcome for a **Harness Task**.
_Avoid_: artifact, log

**No-Progress Signal**:
A structured indication that a **Harness Session** is stalled, repeating, or otherwise failing to advance.
_Avoid_: failure, timeout

**Assistant Chat**:
The interactive product flow that keeps conversational state while executing **Workflow Turn** results.
_Avoid_: chat UI, nicegui chat

**Assistant User Guide**:
The end-user documentation for operating the assistant app.
_Avoid_: system design, architecture document

**Temporary Chat**:
An Assistant chat session that uses normal runtime behavior but is not written to durable Assistant persistence or shown in recent sessions.
_Avoid_: incognito chat, private mode

**Local/Solo Assistant Use**:
The documented Assistant product workflow where one user runs one Assistant app process for their own browser session and workspace.
_Avoid_: hosted mode, multi-user deployment

**Local Owner Account**:
The first local Assistant account that can sign in, unlock per-user data, and change administrator-gated local settings for solo use.
_Avoid_: hosted administrator, team admin

**Experimental Feature**:
An Assistant capability that may exist in code or behind administrator settings but is not yet part of the stable documented product workflow.
_Avoid_: supported feature, roadmap item

**Assistant Workbench**:
The assistant product surface for inspecting runtime, execution, approval, protection, provider, and deep-task details.
_Avoid_: canvas, artifact editor, artifact workbench

**Assistant Help**:
The in-app help surface that exposes the Assistant user guide from the product UI.
_Avoid_: external docs only, support portal

**Deep Task**:
The assistant product flow that runs a user request through a durable **Harness Session**.
_Avoid_: research session

**Assistant Runtime Assembly**:
The app-layer construction of provider, **Tool** exposure, policy, prompts, protection, and execution objects for **Assistant Chat** or **Deep Task**.
_Avoid_: bootstrap, setup

**Final Answer Details**:
Model-produced supplemental metadata shown with an assistant answer, such as citations, confidence, uncertainty, missing information, and follow-up suggestions.
_Avoid_: verification result, proof, audit record

**Project Defaults Module**:
A repository-authored Python module shipped with a project or app variant to provide startup defaults for assistant app behavior without making those values persisted runtime configuration.
_Avoid_: app defaults, runtime config, operator hook

**Selected Model**:
The provider-specific model identifier chosen for a **Model Turn** after discovery or explicit user entry.
_Avoid_: default model, initial model

**Provider Connection**:
The concrete model-service connection context used for discovery or a **Model Turn**, including API shape, endpoint, and credential scope.
_Avoid_: provider when referring only to a preset name

**Provider Protocol**:
The model-service API shape used by a **Provider Connection**, such as an OpenAI-compatible API.
_Avoid_: provider preset, vendor

**Provider Connection Preset**:
A saved, built-in, or project-supplied non-secret template that can populate **Provider Protocol**, **Provider Connection**, response mode, and optionally **Selected Model** by stable preset id.
_Avoid_: provider, provider protocol

**Provider Auth Scheme**:
The non-secret credential shape required by a **Provider Connection**, such as no credential, bearer token, or `x-access-tokens` header.
_Avoid_: bearer-token requirement

**System Design Document**:
The single current-system technical design document that describes implemented architecture, cybersecurity-relevant controls, algorithmic behavior, subsystem boundaries, contracts, dependencies, LLM interaction design, and built-in capabilities.
_Avoid_: target architecture, aspirational design, stale design notes

**Dependency Inventory**:
A reproducible project artifact that lists package dependencies, provenance, license metadata, and their design-level purpose.
_Avoid_: hand-written SBOM dump, dependency notes

**Runtime Dependency**:
A package required for installed `llm-tools` behavior, including library APIs, built-in integrations, assistant app runtime, persistence, provider transport, or CLI entrypoints.
_Avoid_: dev tool, test dependency

**Development Dependency**:
A package required for local development, testing, linting, type checking, dead-code review, coverage, or packaging workflows.
_Avoid_: runtime dependency, shipped capability

**Software Bill of Materials**:
A generated full transitive CycloneDX JSON dependency artifact for the project environment.
_Avoid_: curated dependency table, design dependency summary

**Feature-Gated Capability**:
An implemented system behavior that may be disabled, hidden, or restricted by runtime configuration, administrator settings, or feature flags.
_Avoid_: backlog item, planned feature

## Relationships

- A **Model-Turn Protocol** produces one parsed **Model Turn**.
- A **Skill** may guide how an agent uses zero or more **Tools**.
- A **Skill Package** contains exactly one canonical **Skill** manifest.
- **Skill Metadata** may be discovered without creating a **Loaded Skill**.
- A **Loaded Skill** is created from **Skill Metadata** when the user invokes it or the agent selects it as relevant.
- Every discovered **Skill** has exactly one **Skill Scope**.
- Same-name **Skills** resolve by **Skill Scope** precedence: enterprise, user, project, then bundled.
- **Skill Invocation** and **Skill Selection** can both produce a **Loaded Skill**.
- **Available Skills Context** is derived from **Skill Metadata**.
- **Loaded Skill Context** is derived from a **Loaded Skill**.
- A **Skill** may declare zero or more **Skill Dependencies**.
- Skill scanning produces a **Skill Discovery Result**.
- A **Skill Discovery Result** may contain valid **Skill Metadata** and validation errors from different **Skill Packages**.
- **Skill Resolution** chooses from a **Skill Discovery Result** when a plain name or explicit path must identify one **Skill**.
- **Skill Enablement** gates whether discovered **Skills** are eligible for **Skill Selection** or **Skill Invocation**.
- **Workflow Turns** and **Harness Sessions** may contain **Skill Usage Records**.
- **Skills** are discovered, parsed, and validated by a reusable skills API before workflow, harness, or app layers consume them.
- **Tool Policy** gates **Tool Exposure** and **Tool** execution.
- **Tool Exposure** may include fewer **Tools** than are registered in the runtime.
- A **Tool Policy** may produce an **Approval Request** for one **Tool** invocation.
- An **Approval Request** may become a **Pending Approval** when persisted by a **Harness Session**.
- **Execution Services** are supplied by the **Tool Runtime** to mediated **Tool** execution.
- **Workspace-Relative Tool Paths** are resolved through the **Workspace** before host filesystem access.
- **Discovery** may exclude **Hidden/Ignored Paths** by default.
- **Direct Access** may target a **Hidden/Ignored Path** when the path is explicitly named and policy allows it.
- A **Model-Turn Protocol** may emit zero or more **Model-Turn Events** before it produces a parsed **Model Turn**.
- **Protection** may evaluate prompt content before a **Model Turn**.
- **Protection** may evaluate final response content before it is retained or shown.
- A **Sensitivity Category** may influence one or more **Protection Decisions**.
- A **Protection Decision** may constrain, challenge, sanitize, or block a **Model Turn** or final response.
- **Source Provenance** may be visible to **Protection** checks and tool execution.
- A **Workflow Turn** executes one parsed **Model Turn** through the **Tool Runtime**.
- A **Project Defaults Module** may provide startup defaults for **Assistant Chat**, **Deep Task**, and **Provider Connection Presets**.
- An **Assistant Chat** contains one or more **Workflow Turn** results.
- A **Temporary Chat** is an **Assistant Chat** without durable Assistant persistence.
- **Local/Solo Assistant Use** normally has exactly one **Local Owner Account**.
- **Assistant Help** should make the user guide reachable from within the Assistant app.
- The **Assistant Workbench** may inspect **Assistant Chat** and **Deep Task** runtime details.
- A **Deep Task** owns exactly one active **Harness Session** at a time.
- An **Experimental Feature** may be hidden by default or visible with limitations, but it should be documented separately from stable **Local/Solo Assistant Use**.
- A **Harness Session** may persist zero or more **Workflow Turn** results.
- A **Harness Session** has exactly one canonical **Harness State** at any persisted snapshot.
- **Harness State** contains exactly one root user-requested **Harness Task**.
- **Harness State** contains zero or more **Harness Turns** with contiguous turn indices.
- A **Harness Turn** may embed exactly one **Workflow Turn** result.
- A **Harness Turn** may contain one **Pending Approval** audit record.
- **Harness State** may contain at most one active **Pending Approval**.
- A **Pending Approval** references one tool invocation in one parsed **Model Turn**.
- A **Harness Task** may depend on other **Harness Tasks**.
- A **Harness Task** may have zero or more **Verification Expectations**.
- **Verification Evidence** may support the verification outcome for exactly one **Harness Task**.
- A **Resume Disposition** is derived from a persisted **Harness State** before execution continues.
- A **No-Progress Signal** may cause a **Turn Decision** to stop a **Harness Session**.
- **Assistant Runtime Assembly** prepares the app-layer objects used by **Assistant Chat** and **Deep Task**.
- **Final Answer Details** may be tool-grounded when tools were used, but they are not the same as **Verification Evidence**.
- A **Model Turn** requires exactly one **Selected Model**.
- An **Assistant Chat** or **Deep Task** may exist with no **Selected Model** until its first **Model Turn**.
- Model discovery results belong to exactly one **Provider Connection**.
- A **Provider Connection** uses exactly one **Provider Protocol**.
- A **Provider Connection** uses exactly one **Provider Auth Scheme**.
- A **Provider Connection Preset** can populate fields for one **Provider Connection**.

## Example Dialogue

> **Dev:** "Should prompt-tool fallback live in the **Assistant Chat**?"
> **Domain expert:** "No. Prompt-tool fallback is part of the **Model-Turn Protocol**, because **Deep Task** needs the same contract before a **Workflow Turn** can execute."

## Flagged Ambiguities

- "research session" and **Deep Task** have both been used for the assistant's durable harness-backed flow; prefer **Deep Task** for product behavior and **Harness Session** for durable state.
- **Harness State** is the canonical durable truth; traces, summaries, replay results, and stored artifacts are derived observability views and must not redefine session truth.
- **Harness Turn** and **Workflow Turn** are related but distinct: a **Harness Turn** is durable orchestration state and may embed the **Workflow Turn** result produced by one-turn execution.
- **Harness Task** is the domain term for durable harness work; use `TaskRecord` only when referring to the implementation model.
- A **Pending Approval** is a durable execution pause, not merely a UI prompt; denial, expiration, cancel, and interrupted recovery fail closed.
- **Protection** is workflow-level behavior shared by **Assistant Chat** and **Deep Task**, not an app-only moderation feature.
- **Protection Decision** is the canonical term for enforced prompt or response handling; classifier assessments are inputs, not the final decision.
- **Source Provenance** is not the same as a user-facing citation; it records source visibility for runtime and protection decisions.
- A registered **Tool** is not necessarily exposed to the model; **Tool Exposure** depends on session policy, credentials, features, and runtime readiness.
- **Tool Policy** is broader than UI permission toggles; it is enforced by the runtime and workflow path.
- An **Approval Request** is transient workflow state unless a durable harness pause turns it into a **Pending Approval**.
- **Execution Services** are runtime-mediated capabilities, not objects that **Tools** should construct for themselves.
- **Workspace-Relative Tool Paths** are the model-facing contract; native absolute paths belong at app settings and runtime boundaries.
- **Discovery** visibility is not the same as access permission; a **Hidden/Ignored Path** can be hidden from broad search while still readable through explicit, policy-governed **Direct Access**.
- **Assistant Workbench** is inspector-first; it is not yet a Canvas-style artifact editing surface.
- The current **Assistant Workbench** product story should not reserve or imply Canvas/artifact editing because that work is not on the current roadmap.
- **Final Answer Details** are produced by the model and should not be described as app guarantees; scalar confidence is especially easy to overread.
- **Temporary Chat** means non-durable in the Assistant database; it must not be described as provider-private, browser-private, or tool-private.
- Older docs and startup flags mention hosted Assistant operation, but the current documented product workflow is **Local/Solo Assistant Use**; hosted/server mode is early plumbing, not a supported user workflow.
- "response mode" has been used for both transport configuration and the **Model-Turn Protocol**; use **Model-Turn Protocol** when discussing the model-facing parse contract.
- **Model-Turn Event** payloads are redacted at the **Model-Turn Protocol** seam; raw provider messages and responses should not be emitted to callers.
- The app must not invent a **Selected Model** before model discovery, but explicit user or config-provided model identifiers are valid without discovery.
- Model discovery may populate model choices, but it does not create a **Selected Model** without an explicit user or config choice.
- Configured and CLI-provided model identifiers count as explicit **Selected Model** choices, even before discovery has run.
- Provider transport objects are created for a **Model Turn** and still require a **Selected Model**; the no-selection state belongs to app/runtime assembly.
- Attempts to start a **Model Turn** without a **Selected Model** should be blocked before provider creation with a transient app notice, not a persisted transcript entry.
- A blocked no-model attempt should not append the user's prompt to the persisted transcript.
- Provider and endpoint defaults may exist without implying a **Selected Model**.
- Recent model history may suggest choices, but it does not create a **Selected Model** for a new session.
- Persisted session model identifiers are treated as explicit **Selected Model** choices, including values that match former built-in defaults.
- Packaged app defaults should not include a **Selected Model**, while deployment configuration may provide one explicitly.
- Model discovery may run without a **Selected Model**; provider preflight for an executable **Model Turn** requires one.
- Discovery results should be refreshed or invalidated when the **Provider Connection** changes.
- The app UX may group **Provider Protocol**, endpoint, credential, and **Selected Model** controls under the label "Provider"; internally those concepts stay distinct.
- Ollama, Gemini, Ask Sage, and similar endpoints may initially be reached through an OpenAI-compatible **Provider Protocol**, with native **Provider Protocols** added later.
- Native Ollama, Gemini, and Ask Sage **Provider Protocols** mean those services' native API shapes, not vendor-branded presets over the OpenAI API shape.
- Native Gemini **Provider Protocol** work is deferred until provider-native capabilities such as grounding with Google Search are in scope.
- Native Ollama **Provider Protocol** work should land before native Ask Sage **Provider Protocol** work, with Ask Sage following immediately after.
- **Provider Auth Scheme** refactor should land as a prep slice before native Ollama **Provider Protocol** work.
- Native **Provider Protocol** transports should first satisfy the existing **Model-Turn Protocol** provider surface before introducing a lower-level provider abstraction.
- Native Ollama **Provider Protocol** should initially support JSON-schema structured output, prompt-tool text, and Ollama's native tool-calling API shape.
- For native Ollama **Provider Protocol**, `auto` response mode should try native tools, then JSON-schema structured output, then prompt tools.
- Native provider tool-call responses may map to multiple canonical **Tool** invocations in one **Model Turn**.
- Native provider responses that contain both tool calls and substantive final-answer text are ambiguous and should be rejected or repaired before becoming a parsed **Model Turn**.
- Native Ollama **Provider Protocol** uses the `ollama` Python package rather than direct HTTP because its transitive dependencies already overlap the app runtime.
- Native Ollama tool schemas should be passed as plain mappings generated from `llm-tools` schemas; Ollama package Pydantic types should stay at the client boundary.
- Ask Sage native persona, dataset, reference-limit, and similar controls are provider-native request settings, not **Provider Connection** identity.
- OpenAI-compatible transport should not infer behavior from vendor-specific provider families; endpoint capabilities and auth requirements should be explicit connection or protocol settings.
- The **Provider Auth Scheme** belongs to the **Provider Connection**.
- Structured-output and fallback behavior should be explicit **Model-Turn Protocol** capability configuration, not inferred from vendor identity.
- The initial **Provider Protocol** selector should expose one option, "OpenAI API"; native Gemini, Ollama, and Ask Sage protocols can be added later.
- **Provider Connection Presets** may populate provider fields, but tokens are not persisted to disk as part of a preset.
- The **System Design Document** describes current implemented behavior and references ADRs for decision rationale.
- The canonical **System Design Document** lives at `docs/system-design.md`.
- A **Dependency Inventory** supports the **System Design Document** rather than replacing its narrative design sections.
- The **System Design Document** documents public architectural contracts rather than every implementation field or function signature.
- The **System Design Document** includes **Feature-Gated Capabilities** because they are implemented system behavior.
- **Feature-Gated Capabilities** are documented in the owning subsystem section, not in a backlog section.
- The **System Design Document** includes an **Implementation Source Map** appendix that links design areas to primary modules and tests; it is package/module traceability, not line-level source commentary.
- The **System Design Document** uses a consolidated top-level structure: purpose and truth model, system architecture, subsystem designs, cross-cutting designs, built-in capabilities, dependency and supply-chain inventory, and verification and decision records.
- The **System Design Document** contains a full security design section intended to supersede `docs/security.md` later, excluding active hardening backlog and dated review-log content.
- A **Software Bill of Materials** is generated as a full transitive artifact, while the **System Design Document** includes a curated Markdown dependency table.
- The **Software Bill of Materials** generator is committed, but generated full transitive SBOM files are produced on demand rather than committed without a lockfile.
- Curated direct-dependency purpose text lives in `docs/dependency-purposes.toml` and is consumed by the **Dependency Inventory** generator.
- The **Dependency Inventory** distinguishes **Runtime Dependencies** from **Development Dependencies**.
- The **System Design Document** describes individual built-in **Tools** at architectural and security depth, not full input/output schema-reference depth.
- The **System Design Document** is written for a wider technical audience including maintainers, contributors, security reviewers, auditors, and advanced integrators, not end users.
- The **System Design Document** is simultaneously an architecture document, cybersecurity design document, and algorithm description document.
- Built-in **Tool** and **Protection** sections in the **System Design Document** use a consistent assurance template: purpose, inputs and trust boundaries, algorithm, required capabilities, security-relevant controls, persistence/cache/logs/provenance, failure behavior, residual risks, and test evidence.
- **Protection** is a distinctive first-class subsystem in the **System Design Document**, not merely a short security-control note.
- The **Protection** section in the **System Design Document** is organized by algorithms: corpus loading, prompt assessment and decision, prompt challenge and feedback, response review and sanitization/blocking, source provenance and single-source allowance, persistence/caches/scrubbing, assistant integration, controls, risks, and test evidence.
- The **Assistant App** section in the **System Design Document** covers app runtime assembly, chat turn flow, Deep Task integration, workbench/inspector behavior, admin settings, hosted mode, and feature-gate enforcement.
- The **LLM Interaction Design** section in the **System Design Document** describes model-turn protocol selection and the native, staged structured JSON, prompt-tool, protection, fallback, repair, and event-redaction paths as algorithms.
- The **Persistence Design** section in the **System Design Document** covers assistant table responsibilities, encrypted field envelopes, key hierarchy, save/load algorithms, harness persistence, caches, and corruption behavior.
- The **Provider** section in the **System Design Document** covers provider connection identity, support matrix, preflight algorithm, response-mode support, trust boundaries, and feature-gated native protocols.
- The **Skills** section in the **System Design Document** covers bounded discovery, deterministic resolution, assistant prompt-context construction, explicit invocation, and the trust model that skills influence prompts without granting capabilities.
- The **Harness** section in the **System Design Document** covers canonical state, session lifecycle, checkpoint/commit behavior, resume classifications, durable approvals, retry/stop behavior, replay/summaries, verification, and Assistant Deep Task wrapping.
- The **Tool API** and **Workflow API** sections in the **System Design Document** cover tool abstractions, policy evaluation, runtime execution, scoped services, action-envelope parsing, model-facing schema preparation, parsed-response execution, approval semantics, and normalized workflow observability.
- The **Security Design** section in the **System Design Document** is a threat-model-oriented section with assets, trust boundaries, attacker assumptions, control matrix, deployment modes, residual risks, and non-goals.
- The **System Design Document** covers Assistant app architecture, while the **Assistant User Guide** covers end-user operating instructions.
- The **System Design Document** may reference ADRs for rationale but should not depend on ADRs as its primary structure.
- The **Dependency Inventory** generator is tolerant by default and strict on request.
- The **Dependency Inventory** generator has Makefile convenience targets but is not part of the required CI gate without a lockfile.
- **Provider Connection Presets** should have stable ids separate from display labels so later user- or administrator-managed overrides can target them.
- The initial **Provider Protocol** should be named `openai_api` internally and "OpenAI API" in the app.
- OpenAI API protocol endpoints should be entered and stored as actual API base URLs, including path prefixes such as `/v1` when required by the endpoint.
- Native **Provider Protocol** endpoints should be entered and stored as protocol-specific API base URLs that append native paths rather than pretending to use OpenAI API path prefixes.
- Packaged app defaults should not include a concrete **Provider Connection** endpoint; presets or deployment configuration may provide one explicitly.
- **Provider Connection Presets** should start from the **Project Defaults Module** catalog; the packaged module contains the generic starting catalog.
- The assistant app should load exactly one conventional in-package **Project Defaults Module** rather than selecting one through CLI startup arguments.
- A **Project Defaults Module** should expose one typed defaults object covering assistant startup configuration, **Provider Connection Presets**, provider Base URL help text, and initial administrator settings.
- A **Project Defaults Module** should not own process/runtime deployment settings such as database paths, encryption key paths, host, port, TLS, auth mode, or browser launch behavior.
- The assistant app **Project Defaults Module** is `llm_tools.apps.assistant_app.project_defaults` and should export `PROJECT_DEFAULTS`.
- Assistant startup configuration precedence is **Project Defaults Module** baseline, then optional YAML config overlay, then CLI overrides.
- YAML startup configuration should deeply overlay mapping fields from the **Project Defaults Module**, while list fields replace rather than concatenate.
- Initial administrator settings from a **Project Defaults Module** apply only when no persisted administrator settings exist; persisted administrator settings win afterward.
- Loading initial administrator settings should use **Project Defaults Module** values as an absent-row fallback without writing them to persistence until an administrator saves settings.
- Changing to a different **Provider Connection** should clear the **Selected Model** unless the connection identity is unchanged.
- **Provider Connection** identity for model/discovery invalidation is defined by **Provider Protocol**, normalized API base URL, **Provider Auth Scheme**, and credential identity.
- Applying provider settings with a **Selected Model** that is not available for the current **Provider Connection** should show a transient app notice.
- Applying provider settings should block an unavailable **Selected Model** only when discovery succeeds with a non-empty model list; discovery failure should allow explicit model entry with a transient warning.
- Model controls should support manual **Selected Model** entry even when discovery populates known options.
- Applying provider settings should run discovery only when needed to validate an existing **Selected Model**.
- Discovered model lists may be cached in memory for a single login/app session across chats, scoped to **Provider Connection** identity, but should not be persisted long term.
- In-memory model discovery caches should invalidate on **Provider Connection** identity changes and support manual refresh plus time-based expiry.
- Model discovery cache identity should include the in-memory provider credential version.
- Failed discovery refreshes may keep the last successful in-memory model list for the same cache identity while showing a transient warning.
- Stale cached model lists may suggest choices but should not be used to block Apply validation when a fresh discovery attempt fails.
- Response mode is not part of model discovery cache identity.
- Provider preflight results should not be cached as executable readiness.
- Editing provider settings should not mutate the active **Provider Connection** or **Selected Model** until Apply succeeds.
- A **Selected Model** explicitly chosen or entered in the same successful Apply operation survives **Provider Connection** changes.
- The configuration summary below the composer should visibly represent incomplete Provider configuration such as missing endpoint, missing **Selected Model**, or missing required credential.
- Missing Provider items in the composer summary should open the relevant Provider configuration control when practical.
- **Model-Turn Protocol** capability selection should remain directly editable in the Provider configuration area and configuration summary because enterprise endpoints may support different capabilities over time.
- The app should label **Model-Turn Protocol** capability selection as "Response mode", not "Provider mode".
- Pre-alpha config and storage may make breaking renames such as replacing `response_mode_strategy` with response-mode terminology without compatibility migration.
- The breaking refactor should rename low-level provider-mode symbols to response-mode terminology as part of the same change.
- The response-mode enum should be named `ResponseModeStrategy`.
- Response-mode values should remain `auto`, `tools`, `json`, and `prompt_tools`.
- `auto` response mode should use a stable generic fallback order, not vendor-specific inference.
- For the OpenAI API **Provider Protocol**, `auto` response mode should try `tools`, then `json`, then `prompt_tools`.
- **Provider Connection Presets** may populate a visible, editable recommended response mode.
- **Provider Connection Presets** may include a **Selected Model** when authored for a specific deployment; generic built-in presets should avoid model assumptions.
- Selecting a **Provider Connection Preset** should copy its **Selected Model**, when present, into the editable model field before Apply.
- The repo should provide one packaged **Project Defaults Module** for generic use that can be replaced by deployment-specific branches or build variants.
- A **Project Defaults Module** replaces the packaged starting catalog without preventing later user- or administrator-managed preset additions or overrides.
- User- or administrator-managed **Provider Connection Preset** persistence is deferred until a concrete workflow requires it.
- Applying a **Provider Connection Preset** copies fields into the **Provider Connection**; execution uses resolved fields, not preset lookup.
- Credentials for a **Provider Connection** are application-memory-only secrets, not persisted configuration, and should remain in browser UI only long enough to submit them to the backend.
- In-memory provider credentials are scoped to **Provider Connection** identity and should be cleared when that identity changes unless a new credential is submitted in the same Apply operation.
- Credential identity may initially be represented by the in-memory provider secret version rather than a user-facing credential name.
- The first provider-configuration refactor should not add user-facing provider credential slots, API-key labels, or persisted preset metadata unless a concrete use case requires them.
- **Provider Connection** config should contain endpoint and **Provider Auth Scheme**; **Provider Protocol**, **Selected Model**, and response mode remain sibling settings.
- A saved **Provider Connection** may omit an endpoint, but discovery or **Model Turn** actions should block with a transient notice until an API base URL is provided.
- Apply may save an incomplete Provider configuration; action handlers enforce readiness for discovery and **Model Turns**.
- Provider transport creation should receive execution-ready resolved fields, not unresolved app config plus overrides.
- Model discovery may remain a lightweight protocol-specific helper, but it should consume the same resolved **Provider Connection** and credential fields as provider transport creation.
- Assistant app credentials are entered and held in memory; lower-level library or CLI provider calls may still accept explicit or conventional environment-variable credentials.
- Saved sessions may persist non-secret **Provider Connection** fields and **Selected Model** while requiring credentials to be re-entered after restart or logout.
- Discovery for a credential-required **Provider Connection** should be disabled with a transient notice when no in-memory credential is available.
- A **Model Turn** for a credential-required **Provider Connection** with no in-memory credential should be blocked before provider creation with a transient notice and no transcript mutation.
- Missing tool credentials block tool exposure rather than the whole **Model Turn**, and the UI should visibly indicate selected tools that are blocked.
- "Requires bearer token" was the first provider credential shape; prefer **Provider Auth Scheme** now that native Ask Sage requires a non-bearer `x-access-tokens` header.
- The canonical **Skill Package** format is the portable `SKILL.md` directory shape with required `name` and `description` frontmatter plus optional supporting files such as `scripts/`, `references/`, `examples/`, and `assets/`.
- The skills API parses and validates **Skill Packages**, but does not execute referenced scripts during parsing or validation.
- **Skill Metadata** is safe to include in broad discovery context; **Loaded Skill** instructions are included only after explicit invocation or relevance selection.
- **Skill Scope** is domain authority, not a filesystem path; apps may map scopes to different directories.
- Public **Skill Scope** values should be `enterprise`, `user`, `project`, and `bundled`.
- **Skill Scope** precedence is `enterprise`, then `user`, then `project`, then `bundled`; user skills outrank project skills for plain-name resolution.
- Plugin-provided **Skills** may use namespaced names later, but the first skill model does not require a plugin system.
- The skills API supports deterministic **Skill Invocation** by name; heuristic or semantic **Skill Selection** belongs to workflow, harness, or app consumers until a selector API is explicitly introduced.
- Bundled **Skills** are in scope primarily as examples of skill structure and use, not as a hidden way to add runtime capabilities.
- Bundled **Skills** have the lowest **Skill Scope** precedence and should be opt-in or disable-able by consumers.
- The skills API produces structured **Available Skills Context** and **Loaded Skill Context**; provider message role placement belongs to workflow, harness, or app consumers.
- The default app/workflow integration should mirror Codex's two-stage context shape: **Available Skills Context** in developer/system-like instructions and **Loaded Skill Context** as tagged contextual task input.
- **Available Skills Context** should be represented as structured data and render by default in a Codex-style `## Skills` block with skill metadata, paths or aliases, and progressive-disclosure usage instructions.
- **Available Skills Context** rendering should support caller-provided budgets and return report or warning metadata when descriptions are truncated or skills are omitted.
- **Loaded Skill Context** should be represented as structured data and render by default to a Codex-compatible `<skill>` envelope with `<name>`, `<path>`, and the full original `SKILL.md` contents.
- Supporting files referenced by a **Skill Package** remain references until a consumer explicitly loads them.
- The skills API owns safe **Skill Package** discovery, parsing, and path resolution, including resolving support-file paths relative to the skill directory.
- The skills API must not execute **Skill Package** scripts or bypass **Tool Runtime**, app policy, or harness approval flows.
- Script execution, shell approval, filesystem permissions, network access, and credential use remain governed by existing runtime and policy layers.
- **Skill Dependencies** are declarative metadata; dependency installation, enablement, prompting, and blocking decisions belong to workflow, harness, or app consumers.
- The first skills API should require only `name` and `description` in `SKILL.md`; optional dependency metadata may be added without becoming part of the required portable manifest.
- "Codex-compatible skills" means the portable `SKILL.md` package shape for the first implementation, not immediate support for Codex's optional `agents/openai.yaml` sidecar metadata.
- Unknown `SKILL.md` frontmatter fields should not make a **Skill Package** invalid unless they break required fields.
- Codex's `agents/openai.yaml` sidecar metadata is the preferred first compatibility target if optional sidecar support is added later.
- Skill scanning is tolerant: one malformed **Skill Package** should produce a path-specific validation error without preventing other valid **Skills** from loading.
- A missing or invalid `name` or `description` makes that **Skill Package** invalid.
- The skills API may normalize metadata whitespace and require single-line metadata, but should not silently invent canonical names from invalid explicit names.
- Skill names should be trimmed, non-empty, single-line, length-limited, and restricted to letters, digits, `_`, `-`, `.`, and `:`.
- Skill names should not be silently slugified; invalid names should produce validation errors.
- Skill descriptions should be trimmed, non-empty, single-line, and length-limited because they are the primary discovery and selection metadata.
- Plain-name **Skill Resolution** should be exact by default, with optional case-insensitive matching only when unambiguous.
- Full Markdown instruction bodies are not semantically validated by the skills API.
- **Skill Discovery Result** includes all valid **Skills**, including same-name skills from different scopes or paths.
- Plain-name **Skill Resolution** applies **Skill Scope** precedence only when one effective winner is needed.
- Same-name **Skills** at the same **Skill Scope** make plain-name **Skill Invocation** ambiguous unless an explicit path is provided.
- Explicit-path **Skill Invocation** may target any enabled discovered **Skill** regardless of name ambiguity.
- **Skill Enablement** is distinct from policy: a policy-allowed **Skill** may still be disabled by user or administrator choice.
- Automatic **Skill Selection** must respect **Skill Enablement** and must not use a disabled **Skill** just because policy would otherwise allow it.
- The skills API enforces caller-supplied **Skill Enablement** during context rendering and **Skill Resolution**, but does not persist enablement choices.
- Apps, harness consumers, or embedding products own persistence of **Skill Enablement** choices.
- The first implementation slice should establish the reusable skills API, deterministic resolution, context rendering, and example bundled **Skills** without app UI, auto-selection, sidecar metadata, plugin integration, or dependency prompting.
- The second implementation slice should add **Skill Enablement** management and assistant app UI for viewing, enabling, disabling, and invoking **Skills**.
- A **Loaded Skill** is turn-scoped by default and should not keep contributing instructions to later turns unless invoked, selected, or explicitly pinned again.
- Persisting or pinning **Loaded Skills** across a **Harness Session** is a future explicit orchestration feature, not the base skills behavior.
- **Skill Usage Records** should include metadata such as skill name, **Skill Scope**, source path or stable id, invocation type, and optionally a content hash.
- Full `SKILL.md` bodies should not be persisted into transcripts or durable sessions by default.
- Deterministic skill replay may later snapshot skill bodies or warn on content-hash mismatch, but that is explicit replay behavior.
- The assistant app should use "Skills" as the user-facing label for **Skills**.
- The assistant app should support `$skill-name` or picker-based **Skill Invocation** rather than treating **Skills** as slash commands.
- Skill management UI should expose name, description, **Skill Scope** or source, enablement state, path, and validation errors.
- Project or repo **Skills** may be discovered automatically, but they do not gain trust or execution rights beyond normal instruction loading.
- Loading a project **Skill** must not bypass **Tool Runtime**, app policy, shell approval, credential, or filesystem restrictions.
- Future automatic **Skill Selection** should keep **Skill Scope** visible and avoid treating project **Skills** as inherently more trusted than user or administrator choices.
- The skills API is local-filesystem-only for the first implementation, and remote skill catalogs are not assumed to be required later.
- Skill installation, update, marketplace discovery, and remote provenance are outside the first skills API boundary.
- Skill scanning should recursively discover files named exactly `SKILL.md` under configured local roots, with bounded depth and directory-count limits.
- Skill scanning should ignore hidden directories by default, dedupe by canonical path, and avoid following symlinks unless a caller explicitly opts in.
- The first skills API may be synchronous because local skill discovery and loading are filesystem-bound, but its model should not preclude later async APIs.
- Async skill APIs are expected to become useful for subagent orchestration or background refresh, so sync-first implementations should avoid global mutable state and blocking-only abstractions that cannot be mirrored asynchronously.
- Public skills API data structures should use Pydantic models in `skills_api/*_models.py` or `skills_api/models.py`, matching existing public typed model conventions.
- Skills API behavior should live in focused modules for discovery, resolution, loading, and rendering, with YAML parsing using the existing `PyYAML` dependency.
- **Assistant Runtime Assembly** composes session-specific skill roots, **Skill Enablement**, available-skill context, and loaded-skill context for **Assistant Chat** and **Deep Task**.
- The skills API owns reusable skill mechanics; **Assistant Runtime Assembly** owns app-session skill composition.
- Assistant app inspector/debug UI should expose turn-visible **Available Skills Context**, **Loaded Skill Context**, **Skill Usage Records**, and skill validation or enablement warnings.
- Supporting files should not be shown in inspector/debug UI unless the user explicitly opens or loads them.
- The new **System Design Document** and **Assistant User Guide** are intended to replace the old `docs/design/*` and `docs/security.md` documents later, while `docs/CONTEXT.md` and `docs/adr/*` remain supporting records.
- Old `docs/design/*` and `docs/security.md` artifacts remain until a dedicated cleanup removes the superseded documents.
- The first **System Design Document** pass includes `docs/system-design.md`, `docs/dependency-purposes.toml`, a dependency inventory generator, and supporting tests where practical.
- The **System Design Document** is hand-authored, with generated dependency tables used as supporting content rather than generated system narrative.
- The first **System Design Document** draft should be dense and section-complete, while avoiding source-code reference depth.
- The **System Design Document** should not contain the active backlog or general known-gaps list.

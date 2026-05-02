# llm-tools Context

This context captures the domain language for `llm-tools`: typed tools, model
turns, workflow execution, durable harness sessions, and assistant product
entrypoints.

## Language

**Tool**:
A typed callable capability exposed to a model and executed through the runtime.
_Avoid_: function, plugin

**Tool Runtime**:
The policy-aware execution substrate that validates, mediates, invokes, and normalizes **Tool** calls.
_Avoid_: executor, runner

**Model Turn**:
One model response cycle that may produce a final response or one or more **Tool** invocations.
_Avoid_: request, completion

**Model-Turn Protocol**:
The model-facing contract used to obtain and parse a **Model Turn**, including native structured output, staged schemas, and prompt-emitted tool calls.
_Avoid_: response mode, strategy

**Model-Turn Event**:
A user- or operator-visible redacted progress record emitted while a **Model-Turn Protocol** is producing a **Model Turn**.
_Avoid_: UI event, chat event

**Workflow Turn**:
One parsed **Model Turn** executed through the workflow layer against the **Tool Runtime**.
_Avoid_: chat turn

**Harness Session**:
A durable, resumable record of multi-turn work built from persisted **Workflow Turn** results.
_Avoid_: job, run

**Assistant Chat**:
The interactive product flow that keeps conversational state while executing **Workflow Turn** results.
_Avoid_: chat UI, nicegui chat

**Deep Task**:
The assistant product flow that runs a user request through a durable **Harness Session**.
_Avoid_: research session

**Assistant Runtime Assembly**:
The app-layer construction of provider, **Tool** exposure, policy, prompts, protection, and execution objects for **Assistant Chat** or **Deep Task**.
_Avoid_: bootstrap, setup

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
A saved or built-in non-secret template for a **Provider Connection** that can populate protocol, endpoint, auth policy, and capability settings.
_Avoid_: provider, provider protocol

## Relationships

- A **Model-Turn Protocol** produces one parsed **Model Turn**.
- A **Model-Turn Protocol** may emit zero or more **Model-Turn Events** before it produces a parsed **Model Turn**.
- A **Workflow Turn** executes one parsed **Model Turn** through the **Tool Runtime**.
- An **Assistant Chat** contains one or more **Workflow Turn** results.
- A **Deep Task** owns exactly one active **Harness Session** at a time.
- A **Harness Session** persists one or more **Workflow Turn** results.
- **Assistant Runtime Assembly** prepares the app-layer objects used by **Assistant Chat** and **Deep Task**.
- A **Model Turn** requires exactly one **Selected Model**.
- An **Assistant Chat** or **Deep Task** may exist with no **Selected Model** until its first **Model Turn**.
- Model discovery results belong to exactly one **Provider Connection**.
- A **Provider Connection** uses exactly one **Provider Protocol**.
- A **Provider Connection Preset** can populate fields for one **Provider Connection**.

## Example Dialogue

> **Dev:** "Should prompt-tool fallback live in the **Assistant Chat**?"
> **Domain expert:** "No. Prompt-tool fallback is part of the **Model-Turn Protocol**, because **Deep Task** needs the same contract before a **Workflow Turn** can execute."

## Flagged Ambiguities

- "research session" and **Deep Task** have both been used for the assistant's durable harness-backed flow; prefer **Deep Task** for product behavior and **Harness Session** for durable state.
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
- OpenAI-compatible transport should not infer behavior from vendor-specific provider families; endpoint capabilities and auth requirements should be explicit connection or protocol settings.
- Whether bearer-token authentication is required belongs to the **Provider Connection**.
- Structured-output and fallback behavior should be explicit **Model-Turn Protocol** capability configuration, not inferred from vendor identity.
- The initial **Provider Protocol** selector should expose one option, "OpenAI API"; native Gemini, Ollama, and Ask Sage protocols can be added later.
- **Provider Connection Presets** may populate provider fields, but tokens are not persisted to disk as part of a preset.
- The initial **Provider Protocol** should be named `openai_api` internally and "OpenAI API" in the app.
- OpenAI API protocol endpoints should be entered and stored as actual API base URLs, including path prefixes such as `/v1` when required by the endpoint.
- Packaged app defaults should not include a concrete **Provider Connection** endpoint; presets or deployment configuration may provide one explicitly.
- Built-in **Provider Connection Presets** should live in one easy-to-edit repo file and may include local Ollama's OpenAI API URL.
- Changing to a different **Provider Connection** should clear the **Selected Model** unless the connection identity is unchanged.
- **Provider Connection** identity for model/discovery invalidation is defined by **Provider Protocol**, normalized API base URL, bearer-token requirement, and credential identity.
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
- The repo should provide one built-in preset catalog and a local/customer Python extension hook for deployment-specific presets.
- Local/customer **Provider Connection Presets** may add or override built-in presets by stable preset id, with local definitions winning on id collision.
- Applying a **Provider Connection Preset** copies fields into the **Provider Connection**; execution uses resolved fields, not preset lookup.
- Credentials for a **Provider Connection** are application-memory-only secrets, not persisted configuration, and should remain in browser UI only long enough to submit them to the backend.
- In-memory provider credentials are scoped to **Provider Connection** identity and should be cleared when that identity changes unless a new credential is submitted in the same Apply operation.
- Credential identity may initially be represented by the in-memory provider secret version rather than a user-facing credential name.
- The first provider-configuration refactor should not add user-facing provider credential slots, API-key labels, or persisted preset metadata unless a concrete use case requires them.
- The first **Provider Connection** config should contain only endpoint and bearer-token requirement; **Provider Protocol**, **Selected Model**, and response mode remain sibling settings.
- A saved **Provider Connection** may omit an endpoint, but discovery or **Model Turn** actions should block with a transient notice until an API base URL is provided.
- Apply may save an incomplete Provider configuration; action handlers enforce readiness for discovery and **Model Turns**.
- Provider transport creation should receive execution-ready resolved fields, not unresolved app config plus overrides.
- Model discovery may remain a lightweight protocol-specific helper, but it should consume the same resolved **Provider Connection** and credential fields as provider transport creation.
- Assistant app credentials are entered and held in memory; lower-level library or CLI provider calls may still accept explicit or conventional environment-variable credentials.
- Saved sessions may persist non-secret **Provider Connection** fields and **Selected Model** while requiring credentials to be re-entered after restart or logout.
- Discovery for a bearer-token-required **Provider Connection** should be disabled with a transient notice when no in-memory credential is available.
- A **Model Turn** for a bearer-token-required **Provider Connection** with no in-memory credential should be blocked before provider creation with a transient notice and no transcript mutation.
- Missing tool credentials block tool exposure rather than the whole **Model Turn**, and the UI should visibly indicate selected tools that are blocked.

# Split provider configuration into protocol, connection, presets, and selected model

The assistant's current provider settings blur API shape, endpoint, credentials, model selection, and response-mode behavior under a single provider preset, which caused packaged defaults such as local Ollama and `gemma4:26b` to be treated as implicit runtime choices. We will refactor provider configuration into explicit **Provider Protocol**, **Provider Connection**, **Provider Connection Preset**, **Selected Model**, and response-mode concepts while the app may still present those controls under a single "Provider" area.

The packaged app will not invent a concrete endpoint or **Selected Model**. A
model becomes selected only through explicit user choice, explicit deployment or
assistant configuration, or persisted session state. Built-in presets and
packaged examples may suggest or populate fields, but generic packaged defaults
must not silently route prompts to a model just because it was convenient for a
local example. The initial **Provider Protocol** is `openai_api` ("OpenAI API"),
endpoints are stored as actual API base URLs, and OpenAI-compatible behavior
will not branch on vendor-specific provider families such as Ollama. Native
provider behavior must be selected through explicit **Provider Protocol** and
capability settings, not inferred from endpoint URLs, preset names, or vendor
branding. Presets may populate non-secret connection fields and visible
recommended response-mode or deployment-specific model choices, but execution
uses the resolved copied fields rather than a preset lookup.
Provider transport creation receives execution-ready resolved protocol,
endpoint, auth, credential, selected-model, and response-mode fields instead of
unresolved app configuration plus ad hoc overrides.

Credentials are application-memory-only secrets. They are not stored in config, browser storage, SQLite, or presets, and they are scoped to **Provider Connection** identity. Missing model or missing required provider credentials blocks model-turn submission before provider creation with a transient UI notice and no transcript mutation. The response-mode refactor is a pre-alpha breaking change: `response_mode_strategy` and related low-level symbols should move to `ResponseModeStrategy` terminology without compatibility migration.

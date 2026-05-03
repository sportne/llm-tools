# Generalize provider auth from bearer-token requirement to auth schemes

Native provider protocols need different credential header shapes: OpenAI-style endpoints use bearer tokens, local Ollama usually uses no provider credential, and Ask Sage native uses an `x-access-tokens` header. Replace the narrower provider-connection boolean for "requires bearer token" with a **Provider Auth Scheme** so connection identity, credential prompts, model discovery, and provider construction can vary by protocol without vendor-specific special cases.


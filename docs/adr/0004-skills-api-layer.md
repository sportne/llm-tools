# Add a Reusable Skills API Layer

`llm-tools` will treat **Skills** as reusable agent instruction packages, not executable **Tools** or app-only prompt snippets. Skill discovery, parsing, and validation should live in a new `skills_api` library surface that depends only downward, while `workflow_api`, `harness_api`, and `apps` consume resolved skills without owning the canonical format. This keeps skill support available to future agent-framework layers without pushing app-specific behavior into the core tool runtime or durable harness.

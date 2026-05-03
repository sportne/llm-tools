# ADR 0016: Bundle Enterprise Read Integrations

## Status

Accepted

## Context

`llm-tools` includes typed built-in tools for GitLab and Atlassian surfaces such
as Jira, Confluence, Bitbucket, merge requests, repository files, and code
search. These integrations add runtime dependencies and expand the security
review surface beyond local filesystem and Git tools, so future maintainers
could reasonably ask why they are not external plugins.

## Decision

Bundle common enterprise read integrations inside `llm_tools.tools` as
supported but secondary tool families.

The project intentionally favors a more inclusive supported distribution and
expects fewer extension points to be required for common assistant deployments
than many comparable tool or agent frameworks. Bundled remote integrations still
use typed tool contracts, explicit side-effect and credential metadata,
runtime-mediated execution, policy checks, redaction, timeouts, output limits,
and assistant exposure controls. These integrations are initially read-focused
and should prefer bounded, inspectable outputs over broad write-capable remote
automation.

## Consequences

Enterprise assistant use cases can rely on consistent built-in behavior for
local files, Git, GitLab, and Atlassian reads instead of reimplementing those
surfaces as ad hoc app glue or third-party plugins.

The cost is a larger dependency set and a broader hardening burden in the core
repository. These integrations should remain modular within `tools`, and the
lowest layers must not learn vendor-specific behavior just because the
integrations are bundled.

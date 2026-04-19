# Security Reviews

## 2026-04-19: GitLab and Atlassian Built-In Tool Families

### Scope

- Reviewed code paths:
  - `src/llm_tools/tools/gitlab/`
  - `src/llm_tools/tools/atlassian/`
  - shared execution, secret, and policy surfaces in `src/llm_tools/tool_api/`
  - relevant tests in `tests/tools/`, plus policy and contract coverage relevant to these tool families
- Primary focus areas:
  - credential handling and secret scoping
  - request scoping and remote trust boundaries
  - pagination and data-exposure risks
  - unsafe assumptions about remote content
  - network error handling and retry behavior
  - approval and side-effect expectations
  - tool spec vs actual capability flags / required secrets / side effects
  - missing negative tests

### Tests

- Inspected:
  - `tests/tools/test_gitlab.py`
  - `tests/tools/test_atlassian.py`
  - `tests/tools/test_runtime_integration.py`
  - `tests/tool_api/test_policy.py`
  - `tests/architecture/test_tool_contracts.py`
- Attempted targeted execution:
  - `python3 -m pytest -q tests/tools/test_gitlab.py tests/tools/test_atlassian.py tests/tools/test_runtime_integration.py`
- Execution result:
  - blocked in the current environment because `pytest` is unavailable: `No module named pytest`

### Findings

- `High`: `read_confluence_content` writes remote attachment bytes and metadata into a temp cache but declares only `EXTERNAL_READ` and does not set `writes_internal_workspace_cache=True`.
  - Impact: policy that denies internal cache writes will not stop this path, and approval semantics can understate the actual side effects.
  - References:
    - `src/llm_tools/tools/atlassian/tools.py`
    - `src/llm_tools/tool_api/policy.py`
- `Medium`: pagination and output bounding are inconsistent across the tool families.
  - Impact: Jira, Bitbucket, and Confluence searches accept effectively unbounded positive limits, and MR/PR readers materialize full commit and change collections without local truncation metadata, increasing data exposure and availability risk on large remotes.
  - References:
    - `src/llm_tools/tools/atlassian/tools.py`
    - `src/llm_tools/tools/gitlab/tools.py`
- `Medium`: `read_jira_issue` exposes the full Jira `fields` map through `raw_fields` instead of an allowlisted subset.
  - Impact: the effective data exposure is broader than the tool description suggests and can include arbitrary custom fields or sensitive instance-specific metadata.
  - References:
    - `src/llm_tools/tools/atlassian/tools.py`
    - `tests/tools/test_atlassian.py`
- `Medium`: network resilience is weak because the remote tool specs do not set per-tool timeouts and the gateway builders rely on client-library defaults.
  - Impact: slow or stalled remotes can hang invocations longer than intended, and transient upstream failures are surfaced as generic execution failures without explicit retry policy.
  - References:
    - `src/llm_tools/tool_api/execution.py`
    - `src/llm_tools/tool_api/runtime.py`
    - `src/llm_tools/tools/gitlab/tools.py`
    - `src/llm_tools/tools/atlassian/tools.py`

### Recommended Remediation

- Split Confluence page reads from attachment reads, or mark attachment caching with `writes_internal_workspace_cache=True` so policy and approval behavior match the actual side effects.
- Add hard upper bounds and explicit truncation metadata for Jira, Bitbucket, and Confluence searches and for GitLab and Bitbucket MR/PR commit and change collections.
- Replace default `raw_fields` exposure in Jira reads with an allowlisted issue view, and make broader field access explicit and opt-in.
- Define per-tool network timeouts, decide which upstream failures should be retryable, and add negative tests for timeout, oversized result sets, and cache-write policy denial.

### Residual Risk

- Secret scoping in the runtime is generally sound because tools only receive the secrets listed in `required_secrets`, and matching provider gateways are built from that scoped view.
- Residual risk remains at the remote-boundary level: once a provider token is granted, there is no finer local allowlist for hosts, projects, repositories, spaces, or issue scopes.
- Remote HTML, excerpts, issue fields, and similar payloads are mostly treated as untrusted pass-through data rather than normalized content, so downstream consumers still need to treat these outputs as untrusted remote material.

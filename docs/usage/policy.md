# Policy

`ToolPolicy` controls whether a tool invocation is allowed before execution.

By default, the policy is conservative:

- `SideEffectClass.NONE` is allowed
- `SideEffectClass.LOCAL_READ` is allowed
- more powerful side effects are denied unless you opt in

## Common Controls

`ToolPolicy` can restrict execution by:

- tool name allowlists and denylists
- tag allowlists and denylists
- allowed side-effect classes
- approval-required side-effect classes
- network, filesystem, and subprocess capability flags
- required secrets present in `ToolContext.env`

## Example

```python
from llm_tools.tool_api import SideEffectClass, ToolPolicy

policy = ToolPolicy(
    allowed_side_effects={
        SideEffectClass.NONE,
        SideEffectClass.LOCAL_READ,
        SideEffectClass.LOCAL_WRITE,
    },
    denied_tools={"delete_everything"},
)
```

## Approval-Required Cases

If a tool's side effects are listed in `require_approval_for`, policy returns a
denial-shaped `PolicyDecision` with `requires_approval=True`.

That keeps approval signaling explicit without introducing approval workflow
machinery into the base runtime.

## Secrets and Capabilities

`ToolPolicy` checks required secrets against `ToolContext.env`.

Examples:

- Jira tools require `JIRA_BASE_URL`, `JIRA_USERNAME`, and `JIRA_API_TOKEN`
- GitLab tools require `GITLAB_BASE_URL` and `GITLAB_API_TOKEN`
- tools that set `requires_network=True` are denied if `allow_network=False`
- tools that set `requires_filesystem=True` are denied if
  `allow_filesystem=False`

## Runtime Interaction

Policy itself does not raise runtime-specific errors. `ToolRuntime` uses the
returned `PolicyDecision` and normalizes denied executions into `ToolResult`
with `error.code == "policy_denied"`.


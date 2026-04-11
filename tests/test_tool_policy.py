"""Tests for tool policy evaluation."""

from __future__ import annotations

from pydantic import BaseModel

from llm_tools.tool_api import SideEffectClass, Tool, ToolContext, ToolPolicy, ToolSpec


class PolicyInput(BaseModel):
    value: str


class PolicyOutput(BaseModel):
    value: str


class NoOpTool(Tool[PolicyInput, PolicyOutput]):
    spec = ToolSpec(
        name="noop",
        description="Do nothing.",
        tags=["utility"],
        side_effects=SideEffectClass.NONE,
    )
    input_model = PolicyInput
    output_model = PolicyOutput

    def invoke(self, context: ToolContext, args: PolicyInput) -> PolicyOutput:
        return PolicyOutput(value=f"{context.invocation_id}:{args.value}")


class LocalReadTool(Tool[PolicyInput, PolicyOutput]):
    spec = ToolSpec(
        name="read_file",
        description="Read a file.",
        tags=["filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
    )
    input_model = PolicyInput
    output_model = PolicyOutput

    def invoke(self, context: ToolContext, args: PolicyInput) -> PolicyOutput:
        return PolicyOutput(value=args.value)


class LocalWriteTool(Tool[PolicyInput, PolicyOutput]):
    spec = ToolSpec(
        name="write_file",
        description="Write a file.",
        tags=["filesystem", "write"],
        side_effects=SideEffectClass.LOCAL_WRITE,
        requires_filesystem=True,
    )
    input_model = PolicyInput
    output_model = PolicyOutput

    def invoke(self, context: ToolContext, args: PolicyInput) -> PolicyOutput:
        return PolicyOutput(value=args.value)


class ExternalReadTool(Tool[PolicyInput, PolicyOutput]):
    spec = ToolSpec(
        name="fetch_url",
        description="Fetch a URL.",
        tags=["http", "read"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
    )
    input_model = PolicyInput
    output_model = PolicyOutput

    def invoke(self, context: ToolContext, args: PolicyInput) -> PolicyOutput:
        return PolicyOutput(value=args.value)


class ProcessTool(Tool[PolicyInput, PolicyOutput]):
    spec = ToolSpec(
        name="run_process",
        description="Run a subprocess.",
        tags=["process"],
        side_effects=SideEffectClass.NONE,
        requires_subprocess=True,
    )
    input_model = PolicyInput
    output_model = PolicyOutput

    def invoke(self, context: ToolContext, args: PolicyInput) -> PolicyOutput:
        return PolicyOutput(value=args.value)


class SecretTool(Tool[PolicyInput, PolicyOutput]):
    spec = ToolSpec(
        name="secret_tool",
        description="Needs a secret.",
        tags=["secret"],
        side_effects=SideEffectClass.NONE,
        required_secrets=["API_KEY"],
    )
    input_model = PolicyInput
    output_model = PolicyOutput

    def invoke(self, context: ToolContext, args: PolicyInput) -> PolicyOutput:
        return PolicyOutput(value=args.value)


def _context(**env: str) -> ToolContext:
    return ToolContext(invocation_id="inv-1", env=env)


def test_default_policy_allows_none_side_effects() -> None:
    decision = ToolPolicy().evaluate(NoOpTool(), _context())

    assert decision.allowed is True
    assert decision.requires_approval is False
    assert decision.reason == "allowed"


def test_default_policy_allows_local_read_tools() -> None:
    decision = ToolPolicy().evaluate(LocalReadTool(), _context())

    assert decision.allowed is True
    assert decision.reason == "allowed"


def test_default_policy_denies_local_write_tools() -> None:
    decision = ToolPolicy().evaluate(LocalWriteTool(), _context())

    assert decision.allowed is False
    assert decision.requires_approval is False
    assert decision.reason == "side effect not allowed"


def test_default_policy_denies_external_read_tools() -> None:
    decision = ToolPolicy().evaluate(ExternalReadTool(), _context())

    assert decision.allowed is False
    assert decision.reason == "side effect not allowed"


def test_denied_tool_names_block_execution() -> None:
    decision = ToolPolicy(denied_tools={"noop"}).evaluate(NoOpTool(), _context())

    assert decision.allowed is False
    assert decision.reason == "tool name denied"


def test_allowed_tools_acts_as_an_allowlist() -> None:
    decision = ToolPolicy(allowed_tools={"read_file"}).evaluate(NoOpTool(), _context())

    assert decision.allowed is False
    assert decision.reason == "tool name not allowed"


def test_denied_tags_block_execution() -> None:
    decision = ToolPolicy(denied_tags={"filesystem"}).evaluate(
        LocalReadTool(), _context()
    )

    assert decision.allowed is False
    assert decision.reason == "tool tag denied"
    assert decision.metadata["matched_tags"] == ["filesystem"]


def test_allowed_tags_require_any_overlap() -> None:
    denied = ToolPolicy(allowed_tags={"http", "process"}).evaluate(
        LocalReadTool(), _context()
    )
    allowed = ToolPolicy(allowed_tags={"http", "process"}).evaluate(
        ProcessTool(), _context()
    )

    assert denied.allowed is False
    assert denied.reason == "tool tags not allowed"
    assert allowed.allowed is True


def test_allowed_side_effects_gate_exact_side_effect_values() -> None:
    decision = ToolPolicy(
        allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_WRITE}
    ).evaluate(LocalReadTool(), _context())

    assert decision.allowed is False
    assert decision.reason == "side effect not allowed"


def test_network_restrictions_deny_tools_requiring_network() -> None:
    decision = ToolPolicy(
        allowed_side_effects={
            SideEffectClass.NONE,
            SideEffectClass.LOCAL_READ,
            SideEffectClass.EXTERNAL_READ,
        },
        allow_network=False,
    ).evaluate(ExternalReadTool(), _context())

    assert decision.allowed is False
    assert decision.reason == "network access denied"
    assert decision.metadata["blocked_capability"] == "network"


def test_filesystem_restrictions_deny_tools_requiring_filesystem() -> None:
    decision = ToolPolicy(allow_filesystem=False).evaluate(LocalReadTool(), _context())

    assert decision.allowed is False
    assert decision.reason == "filesystem access denied"
    assert decision.metadata["blocked_capability"] == "filesystem"


def test_subprocess_restrictions_deny_tools_requiring_subprocess() -> None:
    decision = ToolPolicy(allow_subprocess=False).evaluate(ProcessTool(), _context())

    assert decision.allowed is False
    assert decision.reason == "subprocess access denied"
    assert decision.metadata["blocked_capability"] == "subprocess"


def test_require_approval_returns_denied_decision_with_flag() -> None:
    decision = ToolPolicy(
        allowed_side_effects={
            SideEffectClass.NONE,
            SideEffectClass.LOCAL_READ,
            SideEffectClass.LOCAL_WRITE,
        },
        require_approval_for={SideEffectClass.LOCAL_WRITE},
    ).evaluate(LocalWriteTool(), _context())

    assert decision.allowed is False
    assert decision.requires_approval is True
    assert decision.reason == "approval required"


def test_missing_required_secrets_deny_execution() -> None:
    decision = ToolPolicy().evaluate(SecretTool(), _context())

    assert decision.allowed is False
    assert decision.reason == "required secrets missing"
    assert decision.metadata["missing_secrets"] == ["API_KEY"]


def test_present_required_secrets_allow_execution_if_nothing_else_blocks() -> None:
    decision = ToolPolicy().evaluate(SecretTool(), _context(API_KEY="secret"))

    assert decision.allowed is True
    assert decision.reason == "allowed"

"""Runtime helpers for the assistant-focused app surfaces."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_tools.apps.assistant_config import AssistantConfig
from llm_tools.apps.assistant_execution import (
    build_assistant_context,
    build_assistant_policy,
)
from llm_tools.apps.assistant_prompts import (
    build_assistant_system_prompt,
    build_research_system_prompt,
)
from llm_tools.apps.assistant_research_provider import AssistantHarnessTurnProvider
from llm_tools.apps.assistant_tool_capabilities import (
    AssistantToolApprovalGate,
    AssistantToolCapability,
    AssistantToolCapabilityReason,
    AssistantToolCapabilityReasonCode,
    AssistantToolGroupCapabilitySummary,
    AssistantToolStatus,
    assistant_tool_group,
    build_tool_capabilities,
    build_tool_group_capability_summaries,
)
from llm_tools.apps.assistant_tool_registry import (
    build_assistant_available_tool_specs,
    build_assistant_executor,
    build_assistant_registry,
    resolve_assistant_default_enabled_tools,
)
from llm_tools.apps.chat_config import ChatLLMConfig
from llm_tools.apps.chat_runtime import create_provider
from llm_tools.apps.protection_runtime import (
    build_protection_controller,
    build_protection_environment,
)
from llm_tools.harness_api import (
    DefaultHarnessContextBuilder,
    HarnessSessionService,
    HarnessStateStore,
)
from llm_tools.harness_api.context import TurnContextBundle
from llm_tools.llm_providers import ResponseModeStrategy
from llm_tools.skills_api import (
    AvailableSkillsContext,
    LoadedSkillContext,
    SkillDiscoveryResult,
    SkillEnablement,
    SkillInvocation,
    SkillInvocationType,
    SkillMetadata,
    SkillRoot,
    SkillScope,
    SkillUsageRecord,
    build_skill_usage_record,
    bundled_skill_root,
    discover_skills,
    enabled_skills,
    is_skill_enabled,
    load_skill_context,
    render_available_skills_context,
    render_loaded_skill_context,
    resolve_skill,
)
from llm_tools.tool_api import ToolPolicy, ToolRegistry, ToolSpec
from llm_tools.workflow_api import (
    ChatSessionState,
    ChatSessionTurnRunner,
    ModelTurnProvider,
    ProtectionConfig,
    ProtectionController,
    WorkflowExecutor,
    inspect_protection_corpus,
    run_interactive_chat_session_turn,
)

if TYPE_CHECKING:
    from llm_tools.apps.assistant_app.models import (
        NiceGUIAdminSettings,
        NiceGUIRuntimeConfig,
    )

AssistantProviderFactory = Callable[["NiceGUIRuntimeConfig"], ModelTurnProvider]
AssistantProtectionControllerFactory = Callable[
    ...,
    ProtectionController | None,
]
_SKILL_INVOCATION_RE = re.compile(r"(?<!\w)\$([A-Za-z0-9_.:-]+)")


class NiceGUIHarnessContextBuilder:
    """Harness context builder that uses NiceGUI session-scoped tool values."""

    def __init__(self, *, env_overrides: dict[str, str] | None = None) -> None:
        self._delegate = DefaultHarnessContextBuilder()
        self._env_overrides = dict(env_overrides or {})

    def build(
        self,
        *,
        state: Any,
        selected_task_ids: Sequence[str],
        turn_index: int,
        workspace: str | None = None,
    ) -> TurnContextBundle:
        bundle = self._delegate.build(
            state=state,
            selected_task_ids=selected_task_ids,
            turn_index=turn_index,
            workspace=workspace,
        )
        metadata = dict(bundle.tool_context.metadata)
        metadata["assistant_mode"] = "assistant_app_deep_task"
        return bundle.model_copy(
            update={
                "tool_context": bundle.tool_context.model_copy(
                    update={
                        "env": dict(self._env_overrides),
                        "metadata": metadata,
                    }
                )
            },
            deep=True,
        )


@dataclass(frozen=True, slots=True)
class AssistantRuntimeBundle:
    """App-layer runtime assembly shared by Assistant Chat and Deep Task."""

    session_id: str
    runtime: NiceGUIRuntimeConfig
    effective_config: AssistantConfig
    provider: ModelTurnProvider
    tool_specs: dict[str, ToolSpec]
    enabled_tool_names: set[str]
    exposed_tool_names: set[str]
    policy: ToolPolicy
    registry: ToolRegistry
    workflow_executor: WorkflowExecutor
    env_overrides: dict[str, str]
    root: Path | None
    chat_system_prompt: str
    deep_task_system_prompt: str
    skill_discovery: SkillDiscoveryResult
    available_skills_context: AvailableSkillsContext | None
    loaded_skill_contexts: tuple[LoadedSkillContext, ...]
    skill_usage_records: tuple[SkillUsageRecord, ...]
    chat_protection_controller: ProtectionController | None
    deep_task_protection_controller: ProtectionController | None

    def build_chat_runner(
        self,
        *,
        session_state: ChatSessionState,
        user_message: str,
    ) -> ChatSessionTurnRunner:
        """Build the Assistant Chat runner for this assembled runtime."""
        return run_interactive_chat_session_turn(
            user_message=user_message,
            session_state=session_state,
            executor=self.workflow_executor,
            provider=self.provider,
            system_prompt=self.chat_system_prompt,
            base_context=build_assistant_context(
                root_path=self.root,
                config=self.effective_config,
                app_name=f"assistant-app-{self.session_id}",
                env_overrides=self.env_overrides,
                include_process_env=False,
            ),
            session_config=self.effective_config.session,
            tool_limits=self.effective_config.tool_limits,
            redaction_config=self.effective_config.policy.redaction,
            temperature=self.effective_config.llm.temperature,
            protection_controller=self.chat_protection_controller,
            enabled_tool_names=self.exposed_tool_names,
        )

    def build_harness_service(
        self,
        *,
        store: HarnessStateStore,
    ) -> HarnessSessionService:
        """Build the Deep Task harness service for this assembled runtime."""
        harness_provider = AssistantHarnessTurnProvider(
            provider=self.provider,  # type: ignore[arg-type]
            temperature=self.effective_config.llm.temperature,
            system_prompt=self.deep_task_system_prompt,
            protection_controller=self.deep_task_protection_controller,
        )
        return HarnessSessionService(
            store=store,
            workflow_executor=self.workflow_executor,
            provider=harness_provider,
            context_builder=NiceGUIHarnessContextBuilder(
                env_overrides=self.env_overrides
            ),
            approval_context_env=self.env_overrides,
            workspace=str(self.root) if self.root is not None else None,
        )


def build_assistant_runtime_bundle(
    *,
    config: AssistantConfig,
    runtime: NiceGUIRuntimeConfig,
    admin_settings: NiceGUIAdminSettings,
    session_id: str,
    provider_factory: AssistantProviderFactory | None = None,
    provider_api_key: str | None = None,
    env_overrides: dict[str, str] | None = None,
    protection_controller_factory: AssistantProtectionControllerFactory | None = None,
    information_protection_enabled: bool = False,
    chat_has_pending_protection_prompt: bool = False,
    skill_invocation_text: str | None = None,
) -> AssistantRuntimeBundle:
    """Build the app-layer runtime bundle for one assistant session."""
    effective_config = _effective_assistant_config(config=config, runtime=runtime)
    provider = _create_bundle_provider(
        effective_config=effective_config,
        runtime=runtime,
        provider_factory=provider_factory,
        provider_api_key=provider_api_key,
    )
    all_tool_specs = build_assistant_available_tool_specs()
    tool_specs = _filter_assistant_tool_specs_for_features(
        all_tool_specs, admin_settings
    )
    root = Path(runtime.root_path) if runtime.root_path is not None else None
    enabled_tool_names = _visible_enabled_tool_names(
        runtime.enabled_tools,
        all_tool_specs,
        admin_settings,
    )
    effective_env_overrides = dict(env_overrides or {})
    exposed_tool_names = _exposed_tool_names_for_runtime(
        tool_specs=tool_specs,
        runtime=runtime,
        root=root,
        env=effective_env_overrides,
    )
    policy = build_assistant_policy(
        enabled_tools=enabled_tool_names,
        tool_specs=tool_specs,
        require_approval_for=set(runtime.require_approval_for),
        allow_network=runtime.allow_network,
        allow_filesystem=runtime.allow_filesystem and root is not None,
        allow_subprocess=runtime.allow_subprocess and root is not None,
        redaction_config=effective_config.policy.redaction,
    )
    registry, workflow_executor = build_assistant_executor(policy=policy)
    skill_context = build_assistant_skill_context(
        runtime=runtime,
        invocation_text=skill_invocation_text,
    )
    chat_system_prompt = _prompt_with_skill_context(
        build_assistant_system_prompt(
            tool_registry=registry,
            tool_limits=effective_config.tool_limits,
            enabled_tool_names=exposed_tool_names,
            workspace_enabled=root is not None,
            staged_schema_protocol=_is_staged_schema_protocol(provider),
            interaction_protocol=_interaction_protocol(provider),
        ),
        available_context=skill_context.available_skills_context,
        loaded_contexts=skill_context.loaded_skill_contexts,
    )
    deep_task_system_prompt = _prompt_with_skill_context(
        build_research_system_prompt(
            tool_registry=registry,
            tool_limits=effective_config.tool_limits,
            enabled_tool_names=exposed_tool_names,
            workspace_enabled=root is not None,
            staged_schema_protocol=_is_staged_schema_protocol(provider),
        ),
        available_context=skill_context.available_skills_context,
        loaded_contexts=skill_context.loaded_skill_contexts,
    )
    chat_protection_controller = _build_bundle_protection_controller(
        app_name="assistant_app",
        runtime=runtime,
        provider=provider,
        exposed_tool_names=exposed_tool_names,
        root=root,
        enabled=(
            information_protection_enabled
            and (
                _nicegui_protection_is_ready(runtime.protection)
                or chat_has_pending_protection_prompt
            )
        ),
        force_config_enabled=chat_has_pending_protection_prompt,
        protection_controller_factory=protection_controller_factory,
    )
    deep_task_protection_controller = _build_bundle_protection_controller(
        app_name="assistant_app_deep_task",
        runtime=runtime,
        provider=provider,
        exposed_tool_names=exposed_tool_names,
        root=root,
        enabled=(
            information_protection_enabled
            and _nicegui_protection_is_ready(runtime.protection)
        ),
        force_config_enabled=False,
        protection_controller_factory=protection_controller_factory,
    )
    return AssistantRuntimeBundle(
        session_id=session_id,
        runtime=runtime,
        effective_config=effective_config,
        provider=provider,
        tool_specs=tool_specs,
        enabled_tool_names=enabled_tool_names,
        exposed_tool_names=exposed_tool_names,
        policy=policy,
        registry=registry,
        workflow_executor=workflow_executor,
        env_overrides=effective_env_overrides,
        root=root,
        chat_system_prompt=chat_system_prompt,
        deep_task_system_prompt=deep_task_system_prompt,
        skill_discovery=skill_context.discovery,
        available_skills_context=skill_context.available_skills_context,
        loaded_skill_contexts=skill_context.loaded_skill_contexts,
        skill_usage_records=skill_context.usage_records,
        chat_protection_controller=chat_protection_controller,
        deep_task_protection_controller=deep_task_protection_controller,
    )


@dataclass(frozen=True, slots=True)
class AssistantSkillContext:
    """Resolved skill context for one assistant runtime turn."""

    discovery: SkillDiscoveryResult
    available_skills_context: AvailableSkillsContext | None
    loaded_skill_contexts: tuple[LoadedSkillContext, ...]
    usage_records: tuple[SkillUsageRecord, ...]


def assistant_skill_roots(runtime: Any) -> tuple[SkillRoot, ...]:
    """Return local skill roots for one assistant runtime."""
    roots: list[SkillRoot] = []
    root_path = getattr(runtime, "root_path", None)
    if root_path:
        roots.append(SkillRoot(path=Path(root_path), scope=SkillScope.PROJECT))
    roots.append(bundled_skill_root())
    return tuple(roots)


def assistant_skill_enablement(runtime: Any) -> SkillEnablement:
    """Return caller-supplied skill enablement for one assistant runtime."""
    return SkillEnablement(
        disabled_names=tuple(getattr(runtime, "disabled_skill_names", ()) or ()),
        disabled_paths=tuple(
            Path(path) for path in getattr(runtime, "disabled_skill_paths", ()) or ()
        ),
    )


def discover_assistant_skills(runtime: Any) -> SkillDiscoveryResult:
    """Discover local skills visible to one assistant runtime."""
    return discover_skills(assistant_skill_roots(runtime))


def assistant_enabled_skills(runtime: Any) -> tuple[SkillMetadata, ...]:
    """Return enabled local skills visible to one assistant runtime."""
    discovery = discover_assistant_skills(runtime)
    return enabled_skills(discovery, assistant_skill_enablement(runtime))


def assistant_skill_is_enabled(runtime: Any, skill: SkillMetadata) -> bool:
    """Return whether one discovered assistant skill is enabled."""
    return is_skill_enabled(skill, assistant_skill_enablement(runtime))


def build_assistant_skill_context(
    *,
    runtime: Any,
    invocation_text: str | None = None,
) -> AssistantSkillContext:
    """Build available and loaded skill context for one assistant turn."""
    discovery = discover_assistant_skills(runtime)
    enablement = assistant_skill_enablement(runtime)
    available_context = render_available_skills_context(
        discovery,
        enablement=enablement,
    )
    loaded_contexts: list[LoadedSkillContext] = []
    usage_records: list[SkillUsageRecord] = []
    for skill_name in _skill_names_from_text(invocation_text or ""):
        resolved = resolve_skill(
            discovery,
            SkillInvocation(name=skill_name),
            enablement,
        )
        loaded = load_skill_context(resolved.skill)
        loaded_contexts.append(loaded)
        usage_records.append(
            build_skill_usage_record(
                resolved.skill,
                invocation_type=SkillInvocationType.EXPLICIT,
                contents=loaded.contents,
            )
        )
    return AssistantSkillContext(
        discovery=discovery,
        available_skills_context=available_context,
        loaded_skill_contexts=tuple(loaded_contexts),
        usage_records=tuple(usage_records),
    )


def _skill_names_from_text(text: str) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for match in _SKILL_INVOCATION_RE.finditer(text):
        name = match.group(1)
        if name not in seen:
            names.append(name)
            seen.add(name)
    return tuple(names)


def _prompt_with_skill_context(
    prompt: str,
    *,
    available_context: AvailableSkillsContext | None,
    loaded_contexts: Sequence[LoadedSkillContext],
) -> str:
    sections = [prompt]
    if available_context is not None:
        sections.append(available_context.rendered_text.strip())
        if available_context.warning_message:
            sections.append(f"Skills warning: {available_context.warning_message}")
    if loaded_contexts:
        sections.append(
            "\n\n".join(
                render_loaded_skill_context(context) for context in loaded_contexts
            )
        )
    return "\n\n".join(section for section in sections if section.strip())


def _create_bundle_provider(
    *,
    effective_config: AssistantConfig,
    runtime: NiceGUIRuntimeConfig,
    provider_factory: AssistantProviderFactory | None,
    provider_api_key: str | None,
) -> ModelTurnProvider:
    if provider_factory is not None:
        return provider_factory(runtime)
    if runtime.selected_model is None:
        raise ValueError("Choose a model before running a model turn.")
    if runtime.provider_connection.requires_bearer_token and not provider_api_key:
        raise ValueError("Enter provider credentials before running a model turn.")
    return create_provider(
        provider_protocol=runtime.provider_protocol,
        provider_connection=runtime.provider_connection,
        api_key=provider_api_key,
        selected_model=runtime.selected_model,
        response_mode_strategy=runtime.response_mode_strategy,
        timeout_seconds=effective_config.llm.timeout_seconds,
        allow_env_api_key=False,
    )


def _assistant_tool_is_feature_visible(
    spec: ToolSpec, admin_settings: NiceGUIAdminSettings
) -> bool:
    if spec.name == "write_file":
        return admin_settings.write_file_tool_enabled
    tags = set(spec.tags)
    if "gitlab" in tags:
        return admin_settings.gitlab_tools_enabled
    if tags.intersection({"jira", "confluence", "bitbucket"}):
        return admin_settings.atlassian_tools_enabled
    return True


def _filter_assistant_tool_specs_for_features(
    tool_specs: dict[str, ToolSpec], admin_settings: NiceGUIAdminSettings
) -> dict[str, ToolSpec]:
    return {
        name: spec
        for name, spec in tool_specs.items()
        if _assistant_tool_is_feature_visible(spec, admin_settings)
    }


def _visible_enabled_tool_names(
    enabled_tools: Sequence[str],
    tool_specs: dict[str, ToolSpec],
    admin_settings: NiceGUIAdminSettings,
) -> set[str]:
    visible_specs = _filter_assistant_tool_specs_for_features(
        tool_specs, admin_settings
    )
    return set(enabled_tools).intersection(visible_specs)


def _build_bundle_protection_controller(
    *,
    app_name: str,
    runtime: NiceGUIRuntimeConfig,
    provider: ModelTurnProvider,
    exposed_tool_names: set[str],
    root: Path | None,
    enabled: bool,
    force_config_enabled: bool,
    protection_controller_factory: AssistantProtectionControllerFactory | None,
) -> ProtectionController | None:
    if not enabled:
        return None
    protection_config = (
        runtime.protection.model_copy(update={"enabled": True})
        if force_config_enabled and not runtime.protection.enabled
        else runtime.protection
    )
    protection_environment = build_protection_environment(
        app_name=app_name,
        model_name=runtime.selected_model or "",
        workspace=runtime.root_path,
        enabled_tools=sorted(exposed_tool_names),
        allow_network=runtime.allow_network,
        allow_filesystem=runtime.allow_filesystem and root is not None,
        allow_subprocess=runtime.allow_subprocess and root is not None,
    )
    protection_environment["allowed_sensitivity_labels"] = list(
        runtime.protection.allowed_sensitivity_labels
    )
    protection_environment["sensitivity_categories"] = [
        category.model_dump(mode="json")
        for category in runtime.protection.sensitivity_categories
    ]
    factory = protection_controller_factory or build_protection_controller
    return factory(
        config=protection_config,
        provider=provider,
        environment=protection_environment,
    )


def _effective_assistant_config(
    *,
    config: AssistantConfig,
    runtime: NiceGUIRuntimeConfig,
) -> AssistantConfig:
    workspace_default_root = runtime.default_workspace_root
    if workspace_default_root is None:
        workspace_default_root = config.workspace.default_root
    return config.model_copy(
        deep=True,
        update={
            "llm": config.llm.model_copy(
                update={
                    "provider_protocol": runtime.provider_protocol,
                    "provider_connection": runtime.provider_connection.model_copy(
                        deep=True
                    ),
                    "response_mode_strategy": runtime.response_mode_strategy,
                    "selected_model": runtime.selected_model,
                    "temperature": runtime.temperature,
                    "timeout_seconds": runtime.timeout_seconds,
                }
            ),
            "session": runtime.session_config.model_copy(deep=True),
            "tool_limits": runtime.tool_limits.model_copy(deep=True),
            "policy": config.policy.model_copy(
                update={
                    "enabled_tools": list(runtime.enabled_tools),
                    "require_approval_for": set(runtime.require_approval_for),
                }
            ),
            "workspace": config.workspace.model_copy(
                update={"default_root": workspace_default_root}
            ),
            "ui": config.ui.model_copy(
                update={
                    "show_token_usage": runtime.show_token_usage,
                    "show_footer_help": runtime.show_footer_help,
                    "inspector_open_by_default": runtime.inspector_open,
                }
            ),
            "protection": runtime.protection.model_copy(deep=True),
            "research": config.research.model_copy(
                update=runtime.research.model_dump(mode="python", exclude_unset=True)
            ),
        },
    )


def _is_staged_schema_protocol(provider: ModelTurnProvider) -> bool:
    preference = getattr(provider, "uses_staged_schema_protocol", None)
    return bool(callable(preference) and preference())


def _interaction_protocol(provider: ModelTurnProvider) -> str:
    prompt_tool_preference = getattr(provider, "uses_prompt_tool_protocol", None)
    if callable(prompt_tool_preference) and bool(prompt_tool_preference()):
        return "prompt_tools"
    if _is_staged_schema_protocol(provider):
        return "staged_json"
    return "native_tools"


def _nicegui_protection_is_ready(config: ProtectionConfig) -> bool:
    """Return whether NiceGUI should engage workflow protection for a turn."""
    if (
        not config.enabled
        or not config.allowed_sensitivity_labels
        or not config.document_paths
    ):
        return False
    report = inspect_protection_corpus(config)
    return report.usable_document_count > 0


def _exposed_tool_names_for_runtime(
    *,
    tool_specs: dict[str, ToolSpec],
    runtime: NiceGUIRuntimeConfig,
    root: Path | None,
    env: dict[str, str],
) -> set[str]:
    capability_groups = build_tool_capabilities(
        tool_specs=tool_specs,
        enabled_tools=set(runtime.enabled_tools),
        root_path=runtime.root_path,
        env=env,
        allow_network=runtime.allow_network,
        allow_filesystem=runtime.allow_filesystem and root is not None,
        allow_subprocess=runtime.allow_subprocess and root is not None,
        require_approval_for=set(runtime.require_approval_for),
    )
    return {
        tool.tool_name
        for group in capability_groups.values()
        for tool in group
        if tool.exposed_to_model
    }


def build_live_harness_provider(
    *,
    config: AssistantConfig,
    provider_config: ChatLLMConfig,
    selected_model: str,
    api_key: str | None,
    response_mode_strategy: ResponseModeStrategy,
    tool_registry: ToolRegistry,
    enabled_tool_names: set[str],
    workspace_enabled: bool,
    workspace: str | None,
    allow_network: bool,
    allow_filesystem: bool,
    allow_subprocess: bool,
) -> AssistantHarnessTurnProvider:
    """Create the live provider wrapper used by research sessions."""
    provider = create_provider(
        provider_protocol=provider_config.provider_protocol,
        provider_connection=provider_config.provider_connection,
        api_key=api_key,
        selected_model=selected_model,
        response_mode_strategy=response_mode_strategy,
        timeout_seconds=provider_config.timeout_seconds,
    )
    protection_controller = build_protection_controller(
        config=config.protection,
        provider=provider,
        environment=build_protection_environment(
            app_name="assistant_app_research",
            model_name=selected_model,
            workspace=workspace,
            enabled_tools=enabled_tool_names,
            allow_network=allow_network,
            allow_filesystem=allow_filesystem,
            allow_subprocess=allow_subprocess,
        ),
    )
    return AssistantHarnessTurnProvider(
        provider=provider,
        temperature=config.llm.temperature,
        system_prompt=build_research_system_prompt(
            tool_registry=tool_registry,
            tool_limits=config.tool_limits,
            enabled_tool_names=enabled_tool_names,
            workspace_enabled=workspace_enabled,
        ),
        protection_controller=protection_controller,
    )


__all__ = [
    "AssistantProviderFactory",
    "AssistantRuntimeBundle",
    "AssistantSkillContext",
    "AssistantHarnessTurnProvider",
    "AssistantToolApprovalGate",
    "AssistantToolCapability",
    "AssistantToolCapabilityReason",
    "AssistantToolCapabilityReasonCode",
    "AssistantToolGroupCapabilitySummary",
    "AssistantToolStatus",
    "NiceGUIHarnessContextBuilder",
    "assistant_enabled_skills",
    "assistant_skill_enablement",
    "assistant_skill_is_enabled",
    "assistant_skill_roots",
    "assistant_tool_group",
    "build_assistant_skill_context",
    "build_assistant_available_tool_specs",
    "build_assistant_context",
    "build_assistant_executor",
    "build_assistant_policy",
    "build_assistant_registry",
    "build_assistant_runtime_bundle",
    "build_live_harness_provider",
    "build_protection_controller",
    "build_protection_environment",
    "build_research_system_prompt",
    "build_tool_capabilities",
    "build_tool_group_capability_summaries",
    "create_provider",
    "discover_assistant_skills",
    "resolve_assistant_default_enabled_tools",
    "_effective_assistant_config",
    "_exposed_tool_names_for_runtime",
    "_interaction_protocol",
    "_is_staged_schema_protocol",
    "_nicegui_protection_is_ready",
]

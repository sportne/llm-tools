"""Controller layer for the assistant app."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from llm_tools.apps import assistant_runtime as _assistant_runtime
from llm_tools.apps.assistant_app.auth import LocalAuthProvider
from llm_tools.apps.assistant_app.features import (
    filter_assistant_tool_specs_for_features,
)
from llm_tools.apps.assistant_app.models import (
    AssistantBranding,
    NiceGUIAdminSettings,
    NiceGUIHostedConfig,
    NiceGUIInspectorEntry,
    NiceGUIPreferences,
    NiceGUIRuntimeConfig,
    NiceGUISessionRecord,
    NiceGUITranscriptEntry,
    NiceGUIUser,
    NiceGUIWorkbenchItem,
)
from llm_tools.apps.assistant_app.store import (
    SQLiteNiceGUIChatStore,
    remember_default_db_path,
)
from llm_tools.apps.assistant_config import AssistantConfig
from llm_tools.apps.assistant_runtime import (
    AssistantRuntimeBundle,
    assistant_skill_is_enabled,
    build_assistant_available_tool_specs,
    build_assistant_runtime_bundle,
    discover_assistant_skills,
    resolve_assistant_default_enabled_tools,
)
from llm_tools.harness_api import (
    ApprovalResolution,
    BudgetPolicy,
    HarnessSessionCreateRequest,
    HarnessSessionInspection,
    HarnessSessionInspectRequest,
    HarnessSessionResumeRequest,
    HarnessSessionRunRequest,
    HarnessSessionService,
    HarnessSessionStopRequest,
    ResumeDisposition,
    resume_session,
)
from llm_tools.skills_api import SkillError, SkillMetadata
from llm_tools.tool_api import ToolSpec
from llm_tools.workflow_api import (
    ChatSessionState,
    ChatSessionTurnRunner,
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowApprovalState,
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
    ModelTurnProvider,
    ProtectionPendingPrompt,
)

TurnEventKind = Literal[
    "status",
    "approval_requested",
    "approval_resolved",
    "inspector",
    "result",
    "harness_result",
    "error",
    "complete",
]
INSPECTOR_WORKBENCH_LABELS = {
    "provider_messages": "User To LLM",
    "provider_response": "From LLM",
    "parsed_response": "Parsed LLM Response",
    "tool_execution": "Tool To LLM",
}
_effective_assistant_config = _assistant_runtime._effective_assistant_config
_exposed_tool_names_for_runtime = _assistant_runtime._exposed_tool_names_for_runtime
_interaction_protocol = _assistant_runtime._interaction_protocol
_is_staged_schema_protocol = _assistant_runtime._is_staged_schema_protocol
_nicegui_protection_is_ready = _assistant_runtime._nicegui_protection_is_ready
build_protection_controller = _assistant_runtime.build_protection_controller


@dataclass
class NiceGUITurnState:
    """Transient turn state owned by the UI controller."""

    busy: bool = False
    status_text: str = ""
    status_history: list[str] = field(default_factory=list)
    pending_approval: ChatWorkflowApprovalState | None = None
    active_turn_number: int = 0
    active_turn_started_at: str | None = None
    last_workbench_entry_at: str | None = None
    pending_harness_session_id: str | None = None
    queued_follow_up_prompt: str | None = None
    cancelling: bool = False


@dataclass
class NiceGUIQueuedEvent:
    """Workflow event serialized across the background turn boundary."""

    kind: TurnEventKind
    payload: object
    turn_number: int
    session_id: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class NiceGUIActiveTurnHandle:
    """Background runner handle for one active turn."""

    session_id: str
    mode: Literal["chat", "deep_task"]
    event_queue: queue.Queue[NiceGUIQueuedEvent]
    thread: threading.Thread
    turn_number: int
    runner: ChatSessionTurnRunner | None = None
    harness_service: HarnessSessionService | None = None
    harness_session_id: str | None = None
    include_harness_replay: bool = False


ProviderFactory = Callable[[NiceGUIRuntimeConfig], ModelTurnProvider]
PROVIDER_API_KEY_FIELD = "__provider_api_key__"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _seconds_between_iso(
    started_at: str | None, finished_at: str | None
) -> float | None:
    if not started_at or not finished_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
        finished = datetime.fromisoformat(finished_at)
    except ValueError:
        return None
    return max((finished - started).total_seconds(), 0.0)


def _workbench_inspector_title(*, turn_number: int, round_index: int, kind: str) -> str:
    """Return a compact, directional title for an inspector payload."""
    direction = INSPECTOR_WORKBENCH_LABELS.get(kind, kind.replace("_", " ").title())
    return f"T{turn_number}R{round_index}: {direction}"


def _remember_status(turn_state: NiceGUITurnState, status: str) -> None:
    if not status:
        return
    if not turn_state.status_history or turn_state.status_history[-1] != status:
        turn_state.status_history.append(status)
    turn_state.status_history = turn_state.status_history[-8:]


def _protection_feedback_message(
    *,
    analysis_is_correct: bool,
    expected_sensitivity_label: str | None = None,
    rationale: str | None = None,
) -> str:
    """Serialize protection feedback for the shared chat runner."""
    lines = [f"analysis_is_correct: {str(analysis_is_correct).lower()}"]
    if expected_sensitivity_label:
        lines.append(f"expected_sensitivity_label: {expected_sensitivity_label}")
    if rationale:
        lines.append(f"rationale: {rationale}")
    return "\n".join(lines)


def _deep_task_summary_text(summary: Any) -> str:
    """Return the assistant transcript summary for one deep task session."""
    lines = [
        f"Deep task session: {summary.session_id}",
        f"Stop reason: {summary.stop_reason.value if summary.stop_reason else 'running'}",
        f"Turns: {summary.total_turns}",
        ("Completed tasks: " + (", ".join(summary.completed_task_ids) or "none")),
        f"Active tasks: {', '.join(summary.active_task_ids) or 'none'}",
    ]
    if summary.pending_approval_ids:
        lines.append("Pending approvals: " + ", ".join(summary.pending_approval_ids))
    if summary.latest_decision_summary:
        lines.append(f"Latest decision: {summary.latest_decision_summary}")
    return "\n".join(lines)


def default_runtime_config(
    config: AssistantConfig,
    *,
    root_path: Path | None,
) -> NiceGUIRuntimeConfig:
    """Build the default runtime for a new NiceGUI chat session."""
    effective_root = root_path or (
        Path(config.workspace.default_root).expanduser()
        if config.workspace.default_root
        else None
    )
    return NiceGUIRuntimeConfig(
        provider_protocol=config.llm.provider_protocol,
        provider_connection=config.llm.provider_connection.model_copy(deep=True),
        response_mode_strategy=config.llm.response_mode_strategy,
        selected_model=config.llm.selected_model,
        temperature=config.llm.temperature,
        timeout_seconds=config.llm.timeout_seconds,
        root_path=str(effective_root) if effective_root is not None else None,
        default_workspace_root=config.workspace.default_root,
        enabled_tools=sorted(resolve_assistant_default_enabled_tools(config)),
        require_approval_for=set(config.policy.require_approval_for),
        allow_network=True,
        allow_filesystem=effective_root is not None,
        allow_subprocess=effective_root is not None,
        inspector_open=config.ui.inspector_open_by_default,
        show_token_usage=config.ui.show_token_usage,
        show_footer_help=config.ui.show_footer_help,
        session_config=config.session.model_copy(deep=True),
        tool_limits=config.tool_limits.model_copy(deep=True),
        research=config.research.model_copy(deep=True),
        protection=config.protection.model_copy(deep=True),
    )


class NiceGUIChatController:
    """Stateful controller for the NiceGUI app shell."""

    def __init__(
        self,
        *,
        store: SQLiteNiceGUIChatStore,
        config: AssistantConfig,
        root_path: Path | None = None,
        provider_factory: ProviderFactory | None = None,
        hosted_config: NiceGUIHostedConfig | None = None,
        current_user: NiceGUIUser | None = None,
        auth_provider: LocalAuthProvider | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.root_path = root_path
        self.provider_factory = provider_factory
        self.hosted_config = hosted_config or NiceGUIHostedConfig()
        self.current_user = current_user
        self.auth_provider = auth_provider
        self.owner_user_id = current_user.user_id if current_user is not None else None
        self.preferences: NiceGUIPreferences = store.load_preferences(
            owner_user_id=self.owner_user_id
        )
        self.admin_settings: NiceGUIAdminSettings = store.load_admin_settings()
        self.sessions: dict[str, NiceGUISessionRecord] = {}
        self.session_order: list[str] = []
        self.turn_states: dict[str, NiceGUITurnState] = {}
        self.active_turns: dict[str, NiceGUIActiveTurnHandle] = {}
        self.session_secrets: dict[str, dict[str, str]] = {}
        self._lock = threading.RLock()
        self._load_initial_state()

    @property
    def active_session_id(self) -> str:
        """Return the active session id, creating a session if needed."""
        if self.preferences.active_session_id in self.sessions:
            return str(self.preferences.active_session_id)
        record = self.create_session(temporary=False)
        return record.summary.session_id

    @property
    def active_record(self) -> NiceGUISessionRecord:
        """Return the active session record."""
        return self.sessions[self.active_session_id]

    @property
    def active_turn_state(self) -> NiceGUITurnState:
        """Return the active session's turn state."""
        return self.turn_state_for(self.active_session_id)

    def list_session_summaries(self, *, query: str | None = None) -> list[Any]:
        """Return summaries for the chat rail."""
        summaries = [
            self.sessions[session_id].summary
            for session_id in self.session_order
            if not self.sessions[session_id].summary.temporary
        ]
        cleaned = (query or "").strip().lower()
        if not cleaned:
            return summaries
        return [
            summary
            for summary in summaries
            if cleaned in summary.title.lower()
            or (
                summary.selected_model is not None
                and cleaned in summary.selected_model.lower()
            )
            or cleaned in (summary.root_path or "").lower()
        ]

    def create_session(self, *, temporary: bool = False) -> NiceGUISessionRecord:
        """Create and activate a new session."""
        runtime = default_runtime_config(self.config, root_path=self.root_path)
        self._filter_hidden_runtime_tools(runtime)
        record = self.store.create_session(
            runtime, temporary=temporary, owner_user_id=self.owner_user_id
        )
        with self._lock:
            self.sessions[record.summary.session_id] = record
            self.turn_states[record.summary.session_id] = NiceGUITurnState()
            self.session_secrets.setdefault(record.summary.session_id, {})
            if not record.summary.temporary:
                self.session_order.insert(0, record.summary.session_id)
                self.preferences.active_session_id = record.summary.session_id
                self.store.save_preferences(
                    self.preferences, owner_user_id=self.owner_user_id
                )
            else:
                self.preferences.active_session_id = record.summary.session_id
            return record

    def select_session(self, session_id: str) -> bool:
        """Activate a known session."""
        if session_id not in self.sessions:
            loaded = self.store.load_session(
                session_id, owner_user_id=self.owner_user_id
            )
            if loaded is None:
                return False
            self.sessions[session_id] = loaded
            self.turn_states[session_id] = NiceGUITurnState()
            self.session_secrets.setdefault(session_id, {})
        self.preferences.active_session_id = session_id
        if not self.sessions[session_id].summary.temporary:
            self.store.save_preferences(
                self.preferences, owner_user_id=self.owner_user_id
            )
        return True

    def delete_session(self, session_id: str) -> None:
        """Delete a durable session."""
        record = self.sessions.get(session_id)
        if record is not None and record.summary.temporary:
            self.sessions.pop(session_id, None)
            self.turn_states.pop(session_id, None)
            self.session_secrets.pop(session_id, None)
        else:
            self.store.delete_session(session_id)
            self.sessions.pop(session_id, None)
            self.turn_states.pop(session_id, None)
            self.session_secrets.pop(session_id, None)
            if session_id in self.session_order:
                self.session_order.remove(session_id)
        if self.preferences.active_session_id == session_id:
            replacement = self.session_order[0] if self.session_order else None
            if replacement is None:
                replacement = self.create_session(temporary=False).summary.session_id
            self.preferences.active_session_id = replacement
            self.store.save_preferences(
                self.preferences, owner_user_id=self.owner_user_id
            )

    def rename_session(self, session_id: str, title: str) -> None:
        """Rename a session."""
        record = self.sessions[session_id]
        cleaned = title.strip() or "New chat"
        record.summary.title = cleaned
        record.summary.updated_at = _now_iso()
        self._persist_record(record)

    def turn_state_for(self, session_id: str) -> NiceGUITurnState:
        """Return transient turn state for a session."""
        return self.turn_states.setdefault(session_id, NiceGUITurnState())

    def set_session_secret(
        self, name: str, value: str, *, session_id: str | None = None
    ) -> None:
        """Set one in-memory secret for the selected session."""
        cleaned_name = name.strip()
        cleaned_value = value.strip()
        if not cleaned_name:
            return
        effective_session_id = session_id or self.active_session_id
        secrets = self.session_secrets.setdefault(effective_session_id, {})
        if cleaned_value:
            secrets[cleaned_name] = cleaned_value

    def clear_session_secret(self, name: str, *, session_id: str | None = None) -> None:
        """Clear one in-memory secret for the selected session."""
        effective_session_id = session_id or self.active_session_id
        self.session_secrets.setdefault(effective_session_id, {}).pop(name, None)

    def has_session_secret(self, name: str, *, session_id: str | None = None) -> bool:
        """Return whether the selected session has an in-memory secret."""
        effective_session_id = session_id or self.active_session_id
        return bool(self.session_secrets.get(effective_session_id, {}).get(name))

    def session_tool_env(self, *, session_id: str | None = None) -> dict[str, str]:
        """Return session-scoped tool credentials without provider-only secrets."""
        effective_session_id = session_id or self.active_session_id
        return {
            name: value
            for name, value in self.session_secrets.get(
                effective_session_id, {}
            ).items()
            if name != PROVIDER_API_KEY_FIELD
        }

    def runtime_tool_urls(
        self, runtime: NiceGUIRuntimeConfig | None = None
    ) -> dict[str, str]:
        """Return persisted non-secret tool URLs for one runtime."""
        effective_runtime = runtime or self.active_record.runtime
        return dict(effective_runtime.tool_urls)

    def tool_env_overrides(
        self,
        *,
        runtime: NiceGUIRuntimeConfig | None = None,
        session_id: str | None = None,
    ) -> dict[str, str]:
        """Return session runtime values and credentials passed to tools."""
        env = self.runtime_tool_urls(runtime)
        env.update(self.session_tool_env(session_id=session_id))
        return env

    def effective_tool_env(
        self,
        *,
        runtime: NiceGUIRuntimeConfig | None = None,
        session_id: str | None = None,
    ) -> dict[str, str]:
        """Return NiceGUI session-scoped tool values without ambient env fallback."""
        env: dict[str, str] = {}
        env.update(self.tool_env_overrides(runtime=runtime, session_id=session_id))
        return env

    def provider_api_key(self, *, session_id: str | None = None) -> str | None:
        """Return the selected session's in-memory provider API key."""
        effective_session_id = session_id or self.active_session_id
        return self.session_secrets.get(effective_session_id, {}).get(
            PROVIDER_API_KEY_FIELD
        )

    def interaction_mode_locked(self, *, session_id: str | None = None) -> bool:
        """Return whether the session has sent its first visible user message."""
        effective_session_id = session_id or self.active_session_id
        record = self.sessions[effective_session_id]
        return any(
            entry.role == "user" and entry.show_in_transcript
            for entry in record.transcript
        )

    def set_interaction_mode(self, mode: Literal["chat", "deep_task"]) -> bool:
        """Set the active session mode when the session has no user messages."""
        if mode == "deep_task" and not self.deep_task_mode_enabled():
            return False
        if self.interaction_mode_locked():
            return False
        record = self.active_record
        record.runtime.interaction_mode = mode
        self._persist_record(record)
        return True

    def deep_task_mode_enabled(self) -> bool:
        """Return whether the administrator has enabled deep task mode."""
        return self.admin_settings.deep_task_mode_enabled

    def information_protection_enabled(self) -> bool:
        """Return whether the administrator has enabled Information Protection."""
        return self.admin_settings.information_protection_enabled

    def set_deep_task_mode_enabled(self, enabled: bool) -> None:
        """Persist the administrator-controlled deep task feature flag."""
        self.set_beta_feature_flags(deep_task_mode_enabled=enabled)

    def set_beta_feature_flags(
        self,
        *,
        deep_task_mode_enabled: bool | None = None,
        information_protection_enabled: bool | None = None,
        write_file_tool_enabled: bool | None = None,
        atlassian_tools_enabled: bool | None = None,
        gitlab_tools_enabled: bool | None = None,
    ) -> None:
        """Persist administrator-controlled beta feature flags."""
        update: dict[str, bool] = {}
        if deep_task_mode_enabled is not None:
            update["deep_task_mode_enabled"] = bool(deep_task_mode_enabled)
        if information_protection_enabled is not None:
            update["information_protection_enabled"] = bool(
                information_protection_enabled
            )
        if write_file_tool_enabled is not None:
            update["write_file_tool_enabled"] = bool(write_file_tool_enabled)
        if atlassian_tools_enabled is not None:
            update["atlassian_tools_enabled"] = bool(atlassian_tools_enabled)
        if gitlab_tools_enabled is not None:
            update["gitlab_tools_enabled"] = bool(gitlab_tools_enabled)
        if not update:
            return
        self.admin_settings = self.admin_settings.model_copy(update=update)
        self.store.save_admin_settings(self.admin_settings)
        self._coerce_sessions_for_feature_flags()

    def visible_tool_specs(self) -> dict[str, ToolSpec]:
        """Return Assistant tool specs visible under current admin feature flags."""
        return filter_assistant_tool_specs_for_features(
            build_assistant_available_tool_specs(), self.admin_settings
        )

    def visible_skills(self) -> tuple[SkillMetadata, ...]:
        """Return discovered skills visible to the active session."""
        return discover_assistant_skills(self.active_record.runtime).skills

    def visible_skill_errors(self) -> tuple[SkillError, ...]:
        """Return discovered skill validation errors for the active session."""
        return discover_assistant_skills(self.active_record.runtime).errors

    def skill_enabled(self, skill: SkillMetadata) -> bool:
        """Return whether one discovered skill is enabled for the active session."""
        return assistant_skill_is_enabled(self.active_record.runtime, skill)

    def set_skill_enabled(self, skill: SkillMetadata, enabled: bool) -> None:
        """Persist per-session enablement for one skill."""
        runtime = self.active_record.runtime
        disabled_paths = {
            str(Path(path).expanduser().resolve(strict=False))
            for path in runtime.disabled_skill_paths
        }
        skill_path = str(Path(skill.path).expanduser().resolve(strict=False))
        if enabled:
            disabled_paths.discard(skill_path)
        else:
            disabled_paths.add(skill_path)
        runtime.disabled_skill_paths = sorted(disabled_paths)
        self.save_active_session()

    def _coerce_sessions_for_feature_flags(self) -> None:
        """Drop now-hidden beta features from unlocked/current session runtime state."""
        visible_specs = self.visible_tool_specs()
        changed_records: set[str] = set()
        for record in self.sessions.values():
            session_id = record.summary.session_id
            if (
                record.runtime.interaction_mode == "deep_task"
                and not self.deep_task_mode_enabled()
                and not self.interaction_mode_locked(session_id=session_id)
            ):
                record.runtime.interaction_mode = "chat"
                changed_records.add(session_id)
            visible_tools = sorted(
                set(record.runtime.enabled_tools).intersection(visible_specs)
            )
            if visible_tools != record.runtime.enabled_tools:
                record.runtime.enabled_tools = visible_tools
                changed_records.add(session_id)
        for session_id in changed_records:
            self._persist_record(self.sessions[session_id])

    def set_branding(self, branding: AssistantBranding) -> None:
        """Persist administrator-controlled app branding."""
        self.admin_settings = self.admin_settings.model_copy(
            update={"branding": branding}
        )
        self.store.save_admin_settings(self.admin_settings)

    def submit_prompt(
        self, prompt: str, *, show_in_transcript: bool = True
    ) -> str | None:
        """Start a turn for the active session. Returns an error string when blocked."""
        cleaned = prompt.strip()
        if not cleaned:
            return "Enter a message first."
        session_id = self.active_session_id
        turn_state = self.turn_state_for(session_id)
        if turn_state.busy:
            turn_state.queued_follow_up_prompt = cleaned
            return None
        record = self.sessions[session_id]
        provider_error = self._provider_readiness_error(
            runtime=record.runtime,
            session_id=session_id,
        )
        if provider_error is not None:
            return provider_error
        user_entry = NiceGUITranscriptEntry(
            role="user",
            text=cleaned,
            show_in_transcript=show_in_transcript,
            created_at=_now_iso(),
        )
        record.transcript.append(user_entry)
        if show_in_transcript and record.summary.title == "New chat":
            record.summary.title = _title_from_prompt(cleaned)
        if (
            record.runtime.interaction_mode == "deep_task"
            and not self.deep_task_mode_enabled()
        ):
            record.runtime.interaction_mode = "chat"
        self._persist_record(record)
        if (
            record.runtime.interaction_mode == "deep_task"
            and self.deep_task_mode_enabled()
        ):
            return self._submit_deep_task_prompt(
                session_id=session_id,
                record=record,
                prompt=cleaned,
            )
        try:
            runner = self._build_runner(
                session_id=session_id,
                runtime=record.runtime,
                session_state=record.workflow_session_state,
                user_message=cleaned,
                turn_number=turn_state.active_turn_number + 1,
            )
        except Exception as exc:
            record.transcript.append(
                NiceGUITranscriptEntry(
                    role="error", text=str(exc), created_at=_now_iso()
                )
            )
            self._persist_record(record)
            return str(exc)

        event_queue: queue.Queue[NiceGUIQueuedEvent] = queue.Queue()
        turn_state.busy = True
        turn_state.cancelling = False
        turn_state.status_text = "thinking"
        turn_state.status_history = ["thinking"]
        turn_state.active_turn_started_at = _now_iso()
        turn_state.last_workbench_entry_at = turn_state.active_turn_started_at
        turn_state.pending_approval = None
        turn_state.active_turn_number += 1
        handle = NiceGUIActiveTurnHandle(
            session_id=session_id,
            mode="chat",
            event_queue=event_queue,
            thread=threading.Thread(target=lambda: None),
            turn_number=turn_state.active_turn_number,
            runner=runner,
        )
        handle.thread = threading.Thread(
            target=_worker_run_turn,
            args=(handle,),
            name=f"llm-tools-assistant-turn-{session_id}",
            daemon=True,
        )
        self.active_turns[session_id] = handle
        handle.thread.start()
        return None

    def _submit_deep_task_prompt(
        self,
        *,
        session_id: str,
        record: NiceGUISessionRecord,
        prompt: str,
    ) -> str | None:
        """Start a durable harness session for one deep task prompt."""
        turn_state = self.turn_state_for(session_id)
        try:
            service = self._build_harness_service(
                session_id=session_id,
                record=record,
                user_message=prompt,
                turn_number=turn_state.active_turn_number + 1,
            )
            budget_policy = BudgetPolicy(
                max_turns=record.runtime.research.default_max_turns,
                max_tool_invocations=record.runtime.research.default_max_tool_invocations,
                max_elapsed_seconds=record.runtime.research.default_max_elapsed_seconds,
            )
            created = service.create_session(
                HarnessSessionCreateRequest(
                    title=_title_from_prompt(prompt),
                    intent=prompt,
                    budget_policy=budget_policy,
                )
            )
        except Exception as exc:
            record.transcript.append(
                NiceGUITranscriptEntry(
                    role="error", text=str(exc), created_at=_now_iso()
                )
            )
            self._persist_record(record)
            return str(exc)

        record.transcript.append(
            NiceGUITranscriptEntry(
                role="system",
                text=f"Started deep task `{created.session_id}`.",
                created_at=_now_iso(),
            )
        )
        self._persist_record(record)
        event_queue: queue.Queue[NiceGUIQueuedEvent] = queue.Queue()
        turn_state.busy = True
        turn_state.cancelling = False
        turn_state.status_text = "running deep task"
        turn_state.status_history = ["running deep task"]
        turn_state.active_turn_started_at = _now_iso()
        turn_state.last_workbench_entry_at = turn_state.active_turn_started_at
        turn_state.pending_approval = None
        turn_state.active_turn_number += 1
        handle = NiceGUIActiveTurnHandle(
            session_id=session_id,
            mode="deep_task",
            event_queue=event_queue,
            thread=threading.Thread(target=lambda: None),
            turn_number=turn_state.active_turn_number,
            harness_service=service,
            harness_session_id=created.session_id,
            include_harness_replay=record.runtime.research.include_replay_by_default,
        )
        handle.thread = threading.Thread(
            target=_worker_run_harness,
            args=(handle,),
            name=f"llm-tools-assistant-deep-task-{session_id}",
            daemon=True,
        )
        self.active_turns[session_id] = handle
        handle.thread.start()
        return None

    def _resume_deep_task_approval(
        self,
        *,
        session_id: str,
        harness_session_id: str,
        approved: bool,
    ) -> bool:
        """Resume a deep task after a durable harness approval decision."""
        if session_id in self.active_turns:
            return False
        record = self.sessions[session_id]
        turn_state = self.turn_state_for(session_id)
        try:
            service = self._build_harness_service(session_id=session_id, record=record)
        except Exception as exc:
            record.transcript.append(
                NiceGUITranscriptEntry(
                    role="error", text=str(exc), created_at=_now_iso()
                )
            )
            self._persist_record(record)
            return False
        turn_state.busy = True
        turn_state.cancelling = False
        turn_state.status_text = "resuming deep task"
        _remember_status(turn_state, "resuming deep task")
        turn_state.active_turn_started_at = _now_iso()
        turn_state.last_workbench_entry_at = turn_state.active_turn_started_at
        turn_state.active_turn_number += 1
        event_queue: queue.Queue[NiceGUIQueuedEvent] = queue.Queue()
        handle = NiceGUIActiveTurnHandle(
            session_id=session_id,
            mode="deep_task",
            event_queue=event_queue,
            thread=threading.Thread(target=lambda: None),
            turn_number=turn_state.active_turn_number,
            harness_service=service,
            harness_session_id=harness_session_id,
            include_harness_replay=record.runtime.research.include_replay_by_default,
        )
        handle.thread = threading.Thread(
            target=_worker_resume_harness,
            args=(
                handle,
                ApprovalResolution.APPROVE if approved else ApprovalResolution.DENY,
            ),
            name=f"llm-tools-assistant-deep-task-resume-{session_id}",
            daemon=True,
        )
        self.active_turns[session_id] = handle
        handle.thread.start()
        return True

    def drain_events(self) -> list[NiceGUIQueuedEvent]:
        """Drain queued workflow events and update controller state."""
        applied: list[NiceGUIQueuedEvent] = []
        for session_id, handle in list(self.active_turns.items()):
            while True:
                try:
                    event = handle.event_queue.get_nowait()
                except queue.Empty:
                    break
                applied.append(event)
                try:
                    pending_prompt = self._apply_queued_event(event)
                    if pending_prompt is not None and event.kind != "complete":
                        self.turn_state_for(
                            session_id
                        ).queued_follow_up_prompt = pending_prompt
                except Exception as exc:
                    self._apply_turn_error(
                        session_id=session_id, error_message=str(exc)
                    )
                if event.kind == "complete":
                    self.active_turns.pop(session_id, None)
                    completed_turn_state = self.turn_state_for(session_id)
                    pending = completed_turn_state.queued_follow_up_prompt
                    if pending and completed_turn_state.pending_approval is None:
                        completed_turn_state.queued_follow_up_prompt = None
                        if self.preferences.active_session_id == session_id:
                            self.submit_prompt(pending)
        return applied

    def resolve_approval(self, *, approved: bool) -> bool:
        """Resolve the active session's pending approval."""
        session_id = self.active_session_id
        handle = self.active_turns.get(session_id)
        turn_state = self.turn_state_for(session_id)
        if handle is None:
            if turn_state.pending_harness_session_id is None:
                return False
            return self._resume_deep_task_approval(
                session_id=session_id,
                harness_session_id=turn_state.pending_harness_session_id,
                approved=approved,
            )
        if handle.mode == "deep_task":
            if handle.harness_session_id is None:
                return False
            return self._resume_deep_task_approval(
                session_id=session_id,
                harness_session_id=handle.harness_session_id,
                approved=approved,
            )
        turn_state = self.turn_state_for(session_id)
        if handle.runner is None:
            return False
        resolved = handle.runner.resolve_pending_approval(approved)
        if resolved:
            return resolved
        return False

    def cancel_active_turn(self) -> bool:
        """Request cancellation for the active turn."""
        session_id = self.active_session_id
        handle = self.active_turns.get(session_id)
        if handle is None:
            return False
        turn_state = self.turn_state_for(session_id)
        turn_state.cancelling = True
        turn_state.status_text = "stopping"
        if handle.mode == "chat" and handle.runner is not None:
            handle.runner.cancel()
        elif handle.mode == "deep_task" and handle.harness_service is not None:
            try:
                handle.harness_service.stop_session(
                    HarnessSessionStopRequest(
                        session_id=str(handle.harness_session_id or "")
                    )
                )
            except Exception:
                return False
        return True

    def pending_protection_prompt(self) -> ProtectionPendingPrompt | None:
        """Return the active session's unresolved protection challenge."""
        return self.active_record.workflow_session_state.pending_protection_prompt

    def submit_protection_accept(self) -> str | None:
        """Accept the active protection ruling through the shared feedback path."""
        if self.pending_protection_prompt() is None:
            return "No protection challenge is pending."
        return self.submit_prompt(
            _protection_feedback_message(analysis_is_correct=True),
            show_in_transcript=False,
        )

    def submit_protection_overrule(
        self,
        *,
        expected_sensitivity_label: str,
        rationale: str,
    ) -> str | None:
        """Record a protection correction and requeue the original request."""
        pending = self.pending_protection_prompt()
        if pending is None:
            return "No protection challenge is pending."
        cleaned_label = expected_sensitivity_label.strip()
        cleaned_rationale = rationale.strip()
        if not cleaned_label:
            return "Expected category is required."
        if not cleaned_rationale:
            return "Explanation is required."
        turn_state = self.active_turn_state
        if turn_state.busy:
            return "Wait for the current turn to finish."
        error = self.submit_prompt(
            _protection_feedback_message(
                analysis_is_correct=False,
                expected_sensitivity_label=cleaned_label,
                rationale=cleaned_rationale,
            ),
            show_in_transcript=False,
        )
        if error:
            turn_state.queued_follow_up_prompt = None
            return error
        turn_state.queued_follow_up_prompt = pending.original_user_message
        return None

    def save_active_session(self) -> None:
        """Persist the active record and preferences."""
        self._filter_hidden_runtime_tools(self.active_record.runtime)
        self._persist_record(self.active_record)
        self.save_preferences()

    def save_preferences(self) -> None:
        """Persist active user preferences."""
        self.store.save_preferences(self.preferences, owner_user_id=self.owner_user_id)

    def switch_database(self, db_path: Path | str) -> None:
        """Switch persistence to a new SQLite database and copy active sessions."""
        if self.current_user is not None:
            raise RuntimeError("Hosted sessions cannot switch SQLite databases.")
        new_store = SQLiteNiceGUIChatStore(
            db_path,
            db_key_file=self.store.db_key_file,
            user_key_file=self.store.user_key_file,
        )
        new_store.initialize()
        for session_id in self.session_order:
            record = self.sessions.get(session_id)
            if record is not None and not record.summary.temporary:
                new_store.save_session(record)
        if (
            self.preferences.active_session_id is not None
            and self.preferences.active_session_id
            not in [session_id for session_id in self.session_order]
        ):
            self.preferences.active_session_id = (
                self.session_order[0] if self.session_order else None
            )
        new_store.save_preferences(self.preferences, owner_user_id=self.owner_user_id)
        self.store = new_store
        remember_default_db_path(new_store.db_path)

    def _load_initial_state(self) -> None:
        self.store.initialize()
        summaries = self.store.list_sessions(
            limit=100, owner_user_id=self.owner_user_id
        )
        self.session_order = [summary.session_id for summary in summaries]
        for summary in summaries:
            loaded = self.store.load_session(
                summary.session_id, owner_user_id=self.owner_user_id
            )
            if loaded is not None:
                self.sessions[summary.session_id] = loaded
                self.turn_states[summary.session_id] = NiceGUITurnState()
                self.session_secrets.setdefault(summary.session_id, {})
                self._filter_hidden_runtime_tools(loaded.runtime)
                self._restore_pending_harness_approval(loaded)
        active_id = self.preferences.active_session_id
        if active_id not in self.sessions:
            active_id = self.session_order[0] if self.session_order else None
        if active_id is None:
            active_id = self.create_session(temporary=False).summary.session_id
        self.preferences.active_session_id = active_id
        self.store.save_preferences(self.preferences, owner_user_id=self.owner_user_id)

    def _restore_pending_harness_approval(self, record: NiceGUISessionRecord) -> None:
        """Restore a durable deep-task approval prompt into transient UI state."""
        turn_state = self.turn_state_for(record.summary.session_id)
        harness_store = self.store.harness_store(
            chat_session_id=record.summary.session_id,
            owner_user_id=record.summary.owner_user_id,
        )
        for snapshot in harness_store.list_sessions():
            resumed = resume_session(snapshot)
            if resumed.disposition is not ResumeDisposition.WAITING_FOR_APPROVAL:
                continue
            if resumed.pending_approval is None:
                continue
            approval_request = resumed.pending_approval.approval_request
            turn_state.pending_approval = ChatWorkflowApprovalState(
                approval_request=approval_request,
                tool_name=approval_request.tool_name,
                redacted_arguments=dict(approval_request.request.arguments),
                policy_reason=approval_request.policy_reason,
                policy_metadata=dict(approval_request.policy_metadata),
            )
            turn_state.pending_harness_session_id = snapshot.session_id
            turn_state.status_text = "approval required"
            turn_state.status_history = ["approval required"]
            return

    def _persist_record(self, record: NiceGUISessionRecord) -> None:
        self._filter_hidden_runtime_tools(record.runtime)
        record.summary.root_path = record.runtime.root_path
        record.summary.owner_user_id = self.owner_user_id
        record.summary.provider_protocol = record.runtime.provider_protocol
        record.summary.selected_model = record.runtime.selected_model
        record.summary.message_count = len(
            [entry for entry in record.transcript if entry.show_in_transcript]
        )
        record.summary.updated_at = _now_iso()
        if record.summary.temporary:
            return
        self.store.save_session(record)
        if record.summary.session_id not in self.session_order:
            self.session_order.insert(0, record.summary.session_id)

    def _filter_hidden_runtime_tools(self, runtime: NiceGUIRuntimeConfig) -> None:
        """Remove tool selections hidden by current admin feature flags."""
        visible_specs = self.visible_tool_specs()
        runtime.enabled_tools = sorted(
            set(runtime.enabled_tools).intersection(visible_specs)
        )

    def _build_runner(
        self,
        *,
        session_id: str,
        runtime: NiceGUIRuntimeConfig,
        session_state: ChatSessionState,
        user_message: str,
        turn_number: int,
    ) -> ChatSessionTurnRunner:
        bundle = build_assistant_runtime_bundle(
            config=self.config,
            runtime=runtime,
            admin_settings=self.admin_settings,
            session_id=session_id,
            provider_factory=self.provider_factory,
            provider_api_key=self.provider_api_key(session_id=session_id),
            env_overrides=self.tool_env_overrides(
                runtime=runtime,
                session_id=session_id,
            ),
            protection_controller_factory=build_protection_controller,
            information_protection_enabled=self.information_protection_enabled(),
            chat_has_pending_protection_prompt=(
                session_state.pending_protection_prompt is not None
            ),
            skill_invocation_text=user_message,
        )
        self._record_skill_context(
            record=self.sessions[session_id],
            bundle=bundle,
            turn_number=turn_number,
        )
        return bundle.build_chat_runner(
            session_state=session_state,
            user_message=user_message,
        )

    def _provider_readiness_error(
        self,
        *,
        runtime: NiceGUIRuntimeConfig,
        session_id: str,
    ) -> str | None:
        """Return a transient provider setup error before a model turn starts."""
        if not runtime.provider_connection.api_base_url:
            return "Enter an API base URL before running a model turn."
        if runtime.selected_model is None:
            return "Choose a model before running a model turn."
        if (
            runtime.provider_connection.requires_bearer_token
            and not self.provider_api_key(session_id=session_id)
        ):
            return "Enter provider credentials before running a model turn."
        return None

    def _build_harness_service(
        self,
        *,
        session_id: str,
        record: NiceGUISessionRecord,
        user_message: str | None = None,
        turn_number: int | None = None,
    ) -> HarnessSessionService:
        """Build the shared harness service for a NiceGUI deep task."""
        bundle = build_assistant_runtime_bundle(
            config=self.config,
            runtime=record.runtime,
            admin_settings=self.admin_settings,
            session_id=session_id,
            provider_factory=self.provider_factory,
            provider_api_key=self.provider_api_key(session_id=session_id),
            env_overrides=self.tool_env_overrides(
                runtime=record.runtime,
                session_id=session_id,
            ),
            protection_controller_factory=build_protection_controller,
            information_protection_enabled=self.information_protection_enabled(),
            skill_invocation_text=user_message,
        )
        if turn_number is not None:
            self._record_skill_context(
                record=record,
                bundle=bundle,
                turn_number=turn_number,
            )
        return bundle.build_harness_service(
            store=self.store.harness_store(
                chat_session_id=session_id,
                owner_user_id=record.summary.owner_user_id,
            )
        )

    def _record_skill_context(
        self,
        *,
        record: NiceGUISessionRecord,
        bundle: AssistantRuntimeBundle,
        turn_number: int,
    ) -> None:
        """Persist skill visibility and usage metadata without loaded skill bodies."""
        if (
            bundle.available_skills_context is None
            and not bundle.loaded_skill_contexts
            and not bundle.skill_discovery.errors
        ):
            return
        payload = {
            "available_skills_context": (
                bundle.available_skills_context.model_dump(mode="json")
                if bundle.available_skills_context is not None
                else None
            ),
            "loaded_skills": [
                {
                    "name": context.name,
                    "path": str(context.path),
                    "content_hash": usage.content_hash,
                }
                for context, usage in zip(
                    bundle.loaded_skill_contexts,
                    bundle.skill_usage_records,
                    strict=True,
                )
            ],
            "usage_records": [
                usage.model_dump(mode="json") for usage in bundle.skill_usage_records
            ],
            "validation_errors": [
                error.model_dump(mode="json") for error in bundle.skill_discovery.errors
            ],
        }
        now = _now_iso()
        record.workbench_items.append(
            NiceGUIWorkbenchItem(
                item_id=f"workbench-{uuid4().hex}",
                kind="inspector",
                title=f"T{turn_number}: Skills",
                payload=payload,
                version=1,
                active=True,
                created_at=now,
                updated_at=now,
            )
        )

    def _apply_queued_event(self, event: NiceGUIQueuedEvent) -> str | None:
        if event.session_id not in self.sessions:
            return None
        record = self.sessions[event.session_id]
        turn_state = self.turn_state_for(event.session_id)
        if event.kind == "status":
            if turn_state.cancelling:
                return None
            status_event = ChatWorkflowStatusEvent.model_validate(event.payload)
            turn_state.status_text = status_event.status
            _remember_status(turn_state, status_event.status)
            return None
        if event.kind == "approval_requested":
            approval_event = ChatWorkflowApprovalEvent.model_validate(event.payload)
            turn_state.pending_approval = approval_event.approval
            turn_state.status_text = "approval required"
            _remember_status(turn_state, "approval required")
            record.transcript.append(
                NiceGUITranscriptEntry(
                    role="system",
                    text=_approval_request_copy(approval_event.approval),
                    created_at=_now_iso(),
                )
            )
            self._persist_record(record)
            return None
        if event.kind == "approval_resolved":
            approval_resolved_event = ChatWorkflowApprovalResolvedEvent.model_validate(
                event.payload
            )
            turn_state.pending_approval = None
            record.transcript.append(
                NiceGUITranscriptEntry(
                    role="system",
                    text=_approval_resolution_copy(approval_resolved_event.resolution),
                    created_at=_now_iso(),
                )
            )
            turn_state.status_text = {
                "approved": "resuming turn",
                "denied": "continuing without approval",
                "timed_out": "approval timed out",
                "cancelled": "",
            }[approval_resolved_event.resolution]
            _remember_status(turn_state, turn_state.status_text)
            self._persist_record(record)
            return None
        if event.kind == "inspector":
            inspector_event = ChatWorkflowInspectorEvent.model_validate(event.payload)
            started_at = (
                turn_state.last_workbench_entry_at or turn_state.active_turn_started_at
            )
            finished_at = event.created_at
            duration_seconds = _seconds_between_iso(started_at, finished_at)
            turn_state.last_workbench_entry_at = finished_at
            label = _workbench_inspector_title(
                turn_number=event.turn_number,
                round_index=inspector_event.round_index,
                kind=inspector_event.kind,
            )
            entry = NiceGUIInspectorEntry(
                label=label,
                payload=inspector_event.payload,
                created_at=_now_iso(),
            )
            if inspector_event.kind == "provider_messages":
                record.inspector_state.provider_messages.append(entry)
            elif inspector_event.kind == "tool_execution":
                record.inspector_state.tool_executions.append(entry)
            else:
                record.inspector_state.parsed_responses.append(entry)
            record.workbench_items.append(
                NiceGUIWorkbenchItem(
                    item_id=f"workbench-{uuid4().hex}",
                    kind="inspector",
                    title=label,
                    payload=inspector_event.payload,
                    version=1,
                    active=True,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=duration_seconds,
                    created_at=_now_iso(),
                    updated_at=_now_iso(),
                )
            )
            self._persist_record(record)
            return None
        if event.kind == "result":
            result_event = ChatWorkflowResultEvent.model_validate(event.payload)
            return self._apply_turn_result(
                session_id=event.session_id,
                event=result_event,
            )
        if event.kind == "harness_result":
            inspection = HarnessSessionInspection.model_validate(event.payload)
            return self._apply_harness_result(
                session_id=event.session_id,
                turn_number=event.turn_number,
                inspection=inspection,
                event_created_at=event.created_at,
            )
        if event.kind == "error":
            return self._apply_turn_error(
                session_id=event.session_id,
                error_message=str(event.payload),
            )
        if event.kind == "complete":
            if turn_state.busy and turn_state.cancelling:
                pending = turn_state.queued_follow_up_prompt
                self._reset_turn_state(turn_state)
                record.transcript.append(
                    NiceGUITranscriptEntry(
                        role="system",
                        text="Stopped the active turn.",
                        created_at=_now_iso(),
                    )
                )
                self._persist_record(record)
                return pending
            return None
        raise ValueError(f"Unsupported queued event kind: {event.kind}")

    def _apply_turn_result(
        self,
        *,
        session_id: str,
        event: ChatWorkflowResultEvent,
    ) -> str | None:
        record = self.sessions[session_id]
        turn_state = self.turn_state_for(session_id)
        result = ChatWorkflowTurnResult.model_validate(event.result)
        record.workflow_session_state = (
            result.session_state or record.workflow_session_state
        )
        record.token_usage = result.token_usage
        turn_state.pending_approval = None
        if result.context_warning:
            record.transcript.append(
                NiceGUITranscriptEntry(
                    role="system",
                    text=result.context_warning,
                    created_at=_now_iso(),
                )
            )
        if result.status == "needs_continuation" and result.continuation_reason:
            record.transcript.append(
                NiceGUITranscriptEntry(
                    role="system",
                    text=result.continuation_reason,
                    created_at=_now_iso(),
                )
            )
        if result.final_response is not None:
            record.transcript.append(
                NiceGUITranscriptEntry(
                    role="assistant",
                    text=result.final_response.answer,
                    final_response=result.final_response,
                    created_at=_now_iso(),
                )
            )
            record.confidence = result.final_response.confidence
        elif result.status == "interrupted":
            interrupted = next(
                (
                    message
                    for message in reversed(result.new_messages)
                    if message.role == "assistant"
                    and message.completion_state == "interrupted"
                ),
                None,
            )
            record.transcript.append(
                NiceGUITranscriptEntry(
                    role="assistant" if interrupted is not None else "system",
                    text=(
                        interrupted.content
                        if interrupted is not None
                        else result.interruption_reason or "Interrupted."
                    ),
                    assistant_completion_state="interrupted",
                    created_at=_now_iso(),
                )
            )
        pending = turn_state.queued_follow_up_prompt
        self._reset_turn_state(turn_state)
        self._persist_record(record)
        return pending

    def _apply_harness_result(
        self,
        *,
        session_id: str,
        turn_number: int,
        inspection: HarnessSessionInspection,
        event_created_at: str,
    ) -> str | None:
        record = self.sessions[session_id]
        turn_state = self.turn_state_for(session_id)
        pending_approval = inspection.resumed.pending_approval
        turn_state.pending_approval = None
        turn_state.pending_harness_session_id = None
        if pending_approval is not None:
            approval_request = pending_approval.approval_request
            turn_state.pending_approval = ChatWorkflowApprovalState(
                approval_request=approval_request,
                tool_name=approval_request.tool_name,
                redacted_arguments=dict(approval_request.request.arguments),
                policy_reason=approval_request.policy_reason,
                policy_metadata=dict(approval_request.policy_metadata),
            )
            turn_state.pending_harness_session_id = inspection.summary.session_id
            turn_state.status_text = "approval required"
            _remember_status(turn_state, "approval required")
            record.transcript.append(
                NiceGUITranscriptEntry(
                    role="system",
                    text=_approval_request_copy(turn_state.pending_approval),
                    created_at=_now_iso(),
                )
            )
        else:
            record.transcript.append(
                NiceGUITranscriptEntry(
                    role="assistant",
                    text=_deep_task_summary_text(inspection.summary),
                    created_at=_now_iso(),
                )
            )
        started_at = (
            turn_state.last_workbench_entry_at or turn_state.active_turn_started_at
        )
        duration_seconds = _seconds_between_iso(started_at, event_created_at)
        record.workbench_items.append(
            NiceGUIWorkbenchItem(
                item_id=f"workbench-{uuid4().hex}",
                kind="result",
                title=f"T{turn_number}: Deep Task",
                payload=inspection.model_dump(mode="json"),
                version=1,
                active=True,
                started_at=started_at,
                finished_at=event_created_at,
                duration_seconds=duration_seconds,
                created_at=_now_iso(),
                updated_at=_now_iso(),
            )
        )
        pending = turn_state.queued_follow_up_prompt
        keep_approval = turn_state.pending_approval
        keep_harness_session_id = turn_state.pending_harness_session_id
        self._reset_turn_state(turn_state)
        turn_state.pending_approval = keep_approval
        turn_state.pending_harness_session_id = keep_harness_session_id
        if keep_approval is not None:
            turn_state.status_text = "approval required"
            turn_state.status_history = ["approval required"]
        self._persist_record(record)
        return pending

    def _apply_turn_error(
        self,
        *,
        session_id: str,
        error_message: str,
    ) -> str | None:
        record = self.sessions[session_id]
        turn_state = self.turn_state_for(session_id)
        record.transcript.append(
            NiceGUITranscriptEntry(
                role="error",
                text=_user_facing_turn_error_message(error_message),
                created_at=_now_iso(),
            )
        )
        pending = turn_state.queued_follow_up_prompt
        self._reset_turn_state(turn_state)
        self._persist_record(record)
        return pending

    def _reset_turn_state(self, turn_state: NiceGUITurnState) -> None:
        pending = turn_state.queued_follow_up_prompt
        turn_state.busy = False
        turn_state.status_text = ""
        turn_state.status_history = []
        turn_state.active_turn_started_at = None
        turn_state.last_workbench_entry_at = None
        turn_state.pending_approval = None
        turn_state.pending_harness_session_id = None
        turn_state.queued_follow_up_prompt = pending
        turn_state.cancelling = False


def _serialize_workflow_event(
    event: object,
    *,
    turn_number: int,
    session_id: str,
) -> NiceGUIQueuedEvent:
    if isinstance(event, ChatWorkflowStatusEvent):
        return NiceGUIQueuedEvent(
            kind="status",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowApprovalEvent):
        return NiceGUIQueuedEvent(
            kind="approval_requested",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowApprovalResolvedEvent):
        return NiceGUIQueuedEvent(
            kind="approval_resolved",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowInspectorEvent):
        return NiceGUIQueuedEvent(
            kind="inspector",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    if isinstance(event, ChatWorkflowResultEvent):
        return NiceGUIQueuedEvent(
            kind="result",
            payload=event.model_dump(mode="json"),
            turn_number=turn_number,
            session_id=session_id,
        )
    raise TypeError(f"Unsupported workflow event type: {type(event)!r}")


def _worker_run_turn(handle: NiceGUIActiveTurnHandle) -> None:
    try:
        if handle.runner is None:
            raise RuntimeError("Chat turn handle is missing a runner.")
        for event in handle.runner:
            handle.event_queue.put(
                _serialize_workflow_event(
                    event,
                    turn_number=handle.turn_number,
                    session_id=handle.session_id,
                )
            )
    except Exception as exc:
        handle.event_queue.put(
            NiceGUIQueuedEvent(
                kind="error",
                payload=str(exc),
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )
    finally:
        handle.event_queue.put(
            NiceGUIQueuedEvent(
                kind="complete",
                payload=None,
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )


def _worker_run_harness(handle: NiceGUIActiveTurnHandle) -> None:
    try:
        if handle.harness_service is None or handle.harness_session_id is None:
            raise RuntimeError("Deep task handle is missing a harness session.")
        handle.harness_service.run_session(
            HarnessSessionRunRequest(session_id=handle.harness_session_id)
        )
        inspection = handle.harness_service.inspect_session(
            HarnessSessionInspectRequest(
                session_id=handle.harness_session_id,
                include_replay=handle.include_harness_replay,
            )
        )
        handle.event_queue.put(
            NiceGUIQueuedEvent(
                kind="harness_result",
                payload=inspection.model_dump(mode="json"),
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )
    except Exception as exc:
        handle.event_queue.put(
            NiceGUIQueuedEvent(
                kind="error",
                payload=str(exc),
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )
    finally:
        handle.event_queue.put(
            NiceGUIQueuedEvent(
                kind="complete",
                payload=None,
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )


def _worker_resume_harness(
    handle: NiceGUIActiveTurnHandle, approval_resolution: ApprovalResolution
) -> None:
    try:
        if handle.harness_service is None or handle.harness_session_id is None:
            raise RuntimeError("Deep task handle is missing a harness session.")
        handle.harness_service.resume_session(
            HarnessSessionResumeRequest(
                session_id=handle.harness_session_id,
                approval_resolution=approval_resolution,
            )
        )
        inspection = handle.harness_service.inspect_session(
            HarnessSessionInspectRequest(
                session_id=handle.harness_session_id,
                include_replay=handle.include_harness_replay,
            )
        )
        handle.event_queue.put(
            NiceGUIQueuedEvent(
                kind="harness_result",
                payload=inspection.model_dump(mode="json"),
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )
    except Exception as exc:
        handle.event_queue.put(
            NiceGUIQueuedEvent(
                kind="error",
                payload=str(exc),
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )
    finally:
        handle.event_queue.put(
            NiceGUIQueuedEvent(
                kind="complete",
                payload=None,
                turn_number=handle.turn_number,
                session_id=handle.session_id,
            )
        )


def _approval_request_copy(approval: ChatWorkflowApprovalState) -> str:
    return (
        f"Approval required for `{approval.tool_name}`. "
        f"Reason: {approval.policy_reason}"
    )


def _approval_resolution_copy(resolution: str) -> str:
    return f"Tool approval {resolution}."


def _user_facing_turn_error_message(error_message: str) -> str:
    if "All response mode attempts failed." in error_message:
        return (
            "Provider compatibility error. The endpoint did not return a usable "
            f"structured response in any fallback mode. {error_message}"
        )
    return error_message


def _title_from_prompt(prompt: str) -> str:
    first_line = prompt.strip().splitlines()[0] if prompt.strip() else "New chat"
    return first_line[:60] + ("..." if len(first_line) > 60 else "")


__all__ = [
    "NiceGUIActiveTurnHandle",
    "NiceGUIChatController",
    "NiceGUIQueuedEvent",
    "NiceGUITurnState",
    "default_runtime_config",
]

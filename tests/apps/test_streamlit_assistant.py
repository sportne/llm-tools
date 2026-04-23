"""Tests for the Streamlit assistant app layer."""

from __future__ import annotations

import importlib
import queue
import runpy
import sys
import time
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from tests.apps._imports import import_streamlit_assistant_modules

from llm_tools.apps.assistant_config import (
    AssistantResearchConfig,
    AssistantWorkspaceConfig,
    StreamlitAssistantConfig,
    load_streamlit_assistant_config,
)
from llm_tools.apps.assistant_prompts import (
    build_assistant_system_prompt,
    build_research_system_prompt,
)
from llm_tools.apps.assistant_runtime import (
    AssistantHarnessTurnProvider,
    build_assistant_available_tool_specs,
    build_assistant_context,
    build_assistant_executor,
    build_assistant_policy,
    build_assistant_registry,
    build_tool_capabilities,
    resolve_assistant_default_enabled_tools,
)
from llm_tools.apps.chat_runtime import build_chat_executor
from llm_tools.harness_api import (
    ApprovalResolution,
    BudgetPolicy,
    HarnessSessionCreateRequest,
    HarnessSessionInspectRequest,
    HarnessSessionRunRequest,
    HarnessSessionService,
    HarnessStopReason,
    HarnessTurn,
    InMemoryHarnessStateStore,
    ResumeDisposition,
    ScriptedParsedResponseProvider,
)
from llm_tools.harness_api.models import (
    TaskLifecycleStatus,
    TurnDecision,
    TurnDecisionAction,
)
from llm_tools.harness_api.tasks import complete_task, start_task
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolInvocationRequest
from llm_tools.workflow_api import (
    ApprovalRequest,
    ChatFinalResponse,
    ChatMessage,
    ChatSessionState,
    ChatTokenUsage,
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowApprovalState,
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
    DefaultEnvironmentComparator,
    PromptProtectionDecision,
    ProtectionAction,
    ProtectionAssessment,
    ProtectionConfig,
    ProtectionController,
    ProtectionFeedbackStore,
    ResponseProtectionDecision,
    WorkflowInvocationStatus,
    load_protection_corpus,
)

_MODULES = import_streamlit_assistant_modules()
build_parser = _MODULES.app.build_parser
process_streamlit_assistant_turn = _MODULES.app.process_streamlit_assistant_turn
_resolve_assistant_config = _MODULES.app._resolve_assistant_config
AssistantResearchSessionController = _MODULES.app.AssistantResearchSessionController


class _FakeProvider:
    def __init__(self, responses: list[ParsedModelResponse]) -> None:
        self._responses = list(responses)

    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        return self._responses.pop(0)


class _RecordingProtectionController:
    def __init__(
        self,
        *,
        prompt_decision: PromptProtectionDecision,
        response_decision: ResponseProtectionDecision,
    ) -> None:
        self.prompt_decision = prompt_decision
        self.response_decision = response_decision
        self.prompt_calls: list[dict[str, object]] = []
        self.response_calls: list[dict[str, object]] = []

    def assess_prompt(self, **kwargs) -> PromptProtectionDecision:
        self.prompt_calls.append(dict(kwargs))
        return self.prompt_decision

    def review_response(self, **kwargs) -> ResponseProtectionDecision:
        self.response_calls.append(dict(kwargs))
        return self.response_decision


class _UnexpectedProvider:
    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("provider should not be called")


class _DeterministicProtectionClassifier:
    def __init__(self, *, document_id: str) -> None:
        self._document_id = document_id

    def assess_prompt(self, **kwargs) -> ProtectionAssessment:
        messages = list(kwargs["messages"])
        latest_message = str(messages[-1].get("content", "")).lower()
        if "secret plan" in latest_message or "proprietary playbook" in latest_message:
            return ProtectionAssessment(
                sensitivity_label="restricted",
                reasoning="Prompt requests proprietary planning material.",
                confidence=0.99,
                referenced_document_ids=[self._document_id],
                recommended_action=ProtectionAction.CHALLENGE,
            )
        return ProtectionAssessment(
            sensitivity_label="public",
            reasoning="Prompt is safe to process.",
            confidence=0.85,
        )

    def assess_response(self, **kwargs) -> ProtectionAssessment:
        response_text = str(kwargs["response_payload"]).lower()
        if "top secret token" in response_text:
            return ProtectionAssessment(
                sensitivity_label="restricted",
                reasoning="Candidate answer contains proprietary material.",
                confidence=0.99,
                referenced_document_ids=[self._document_id],
                recommended_action=ProtectionAction.SANITIZE,
                sanitized_text="Safe replacement",
            )
        return ProtectionAssessment(
            sensitivity_label="public",
            reasoning="Candidate answer is safe.",
            confidence=0.9,
        )


class _FakeBlock:
    def __init__(self, streamlit: _FakeStreamlit) -> None:
        self._streamlit = streamlit

    def __enter__(self) -> _FakeBlock:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def button(self, label: str, **kwargs: object) -> bool:
        return self._streamlit.button(label, **kwargs)

    def text_input(self, label: str, **kwargs: object) -> str:
        return self._streamlit.text_input(label, **kwargs)

    def text_area(self, label: str, *args: object, **kwargs: object) -> str:
        return self._streamlit.text_area(label, *args, **kwargs)

    def checkbox(self, label: str, **kwargs: object) -> bool:
        return self._streamlit.checkbox(label, **kwargs)

    def selectbox(self, label: str, **kwargs: object) -> object:
        return self._streamlit.selectbox(label, **kwargs)

    def number_input(self, label: str, **kwargs: object) -> object:
        return self._streamlit.number_input(label, **kwargs)

    def columns(self, spec: int | list[int]) -> list[_FakeBlock]:
        return self._streamlit.columns(spec)

    def expander(self, label: str, expanded: bool = False) -> _FakeBlock:
        return self._streamlit.expander(label, expanded=expanded)

    def markdown(self, text: str, **kwargs: object) -> None:
        self._streamlit.markdown(text, **kwargs)

    def caption(self, text: str) -> None:
        self._streamlit.caption(text)

    def code(self, text: str) -> None:
        self._streamlit.code(text)

    def warning(self, text: str) -> None:
        self._streamlit.warning(text)

    def error(self, text: str) -> None:
        self._streamlit.error(text)


class _FakeStreamlit:
    def __init__(
        self,
        *,
        button_values: dict[str, bool] | None = None,
        text_input_values: dict[str, str] | None = None,
        checkbox_values: dict[str, bool] | None = None,
        selectbox_values: dict[str, object] | None = None,
        number_input_values: dict[str, int | float] | None = None,
    ) -> None:
        self.session_state: dict[str, object] = {}
        self.button_values = button_values or {}
        self.text_input_values = text_input_values or {}
        self.checkbox_values = checkbox_values or {}
        self.selectbox_values = selectbox_values or {}
        self.number_input_values = number_input_values or {}
        self.sidebar = _FakeBlock(self)
        self.components = SimpleNamespace(v1=SimpleNamespace(html=self._component_html))
        self.page_config_kwargs: list[dict[str, object]] = []
        self.markdown_messages: list[str] = []
        self.component_html_calls: list[tuple[str, int | None, int | None]] = []
        self.caption_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.error_messages: list[str] = []
        self.button_labels: list[str] = []
        self.selectbox_labels: list[str] = []
        self.text_area_values: dict[str, str] = {}
        self.chat_roles: list[str] = []
        self.rerun_called = False

    def set_page_config(self, **kwargs: object) -> None:
        self.page_config_kwargs.append(kwargs)

    def markdown(self, text: str, unsafe_allow_html: bool = False) -> None:
        del unsafe_allow_html
        self.markdown_messages.append(text)

    def _component_html(
        self,
        html: str,
        *,
        height: int | None = None,
        width: int | None = None,
        scrolling: bool = False,
    ) -> None:
        del scrolling
        self.component_html_calls.append((html, height, width))

    def caption(self, text: str) -> None:
        self.caption_messages.append(text)

    def warning(self, text: str) -> None:
        self.warning_messages.append(text)

    def error(self, text: str) -> None:
        self.error_messages.append(text)

    def code(self, text: str) -> None:
        self.markdown_messages.append(text)

    def button(
        self,
        label: str,
        *,
        key: str | None = None,
        use_container_width: bool = False,
        disabled: bool = False,
    ) -> bool:
        del use_container_width
        self.button_labels.append(label)
        if disabled:
            return False
        token = key or label
        return self.button_values.get(token, self.button_values.get(label, False))

    def text_input(
        self,
        label: str,
        *,
        value: str = "",
        key: str | None = None,
        disabled: bool = False,
        placeholder: str | None = None,
        type: str = "default",
    ) -> str:
        del disabled, placeholder, type
        token = key or label
        return self.text_input_values.get(
            token, self.text_input_values.get(label, value)
        )

    def text_area(
        self,
        label: str,
        value: str = "",
        *,
        key: str | None = None,
        height: int | None = None,
        placeholder: str | None = None,
        label_visibility: str | None = None,
    ) -> str:
        del height, placeholder, label_visibility
        token = key or label
        resolved = self.text_input_values.get(
            token, self.text_input_values.get(label, value)
        )
        self.text_area_values[token] = str(resolved)
        self.session_state[token] = str(resolved)
        return str(resolved)

    def checkbox(
        self,
        label: str,
        *,
        value: bool = False,
        key: str | None = None,
        disabled: bool = False,
    ) -> bool:
        if disabled:
            return value
        token = key or label
        return self.checkbox_values.get(token, self.checkbox_values.get(label, value))

    def toggle(
        self,
        label: str,
        *,
        value: bool = False,
        key: str | None = None,
        disabled: bool = False,
    ) -> bool:
        return self.checkbox(
            label,
            value=value,
            key=key,
            disabled=disabled,
        )

    def selectbox(
        self,
        label: str,
        *,
        options: list[object],
        index: int = 0,
        key: str | None = None,
        disabled: bool = False,
        format_func=None,
    ) -> object:
        del format_func
        self.selectbox_labels.append(label)
        if disabled:
            return options[index]
        token = key or label
        return self.selectbox_values.get(
            token, self.selectbox_values.get(label, options[index])
        )

    def number_input(
        self,
        label: str,
        *,
        value: int | float = 0,
        key: str | None = None,
        min_value: int | float | None = None,
        max_value: int | float | None = None,
        step: int | float | None = None,
    ) -> int | float:
        del min_value, max_value, step
        token = key or label
        return self.number_input_values.get(
            token, self.number_input_values.get(label, value)
        )

    def columns(self, spec: int | list[int]) -> list[_FakeBlock]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_FakeBlock(self) for _ in range(count)]

    def expander(self, label: str, expanded: bool = False) -> _FakeBlock:
        del expanded
        self.button_labels.append(label)
        return _FakeBlock(self)

    def chat_message(self, role: str) -> _FakeBlock:
        self.chat_roles.append(role)
        return _FakeBlock(self)

    def rerun(self) -> None:
        self.rerun_called = True


class _DeadThread:
    def is_alive(self) -> bool:
        return False


class _FakeRunnerHandle:
    def __init__(self) -> None:
        self.cancelled = False
        self.approvals: list[bool] = []

    def cancel(self) -> None:
        self.cancelled = True

    def resolve_pending_approval(self, approved: bool) -> bool:
        self.approvals.append(approved)
        return True


class _FakeIterableRunner:
    def __init__(self, events: list[object]) -> None:
        self._events = list(events)
        self.approvals: list[bool] = []
        self.cancelled = False

    def __iter__(self):
        return iter(self._events)

    def cancel(self) -> None:
        self.cancelled = True

    def resolve_pending_approval(self, approved: bool) -> bool:
        self.approvals.append(approved)
        return True


class _RejectingRunnerHandle(_FakeRunnerHandle):
    def resolve_pending_approval(self, approved: bool) -> bool:
        self.approvals.append(approved)
        return False


class _FakeThread:
    def __init__(self, target=None, args=(), daemon: bool | None = None) -> None:
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started


def _make_runtime(*, root_path: str | None = None):
    runtime = _MODULES.models.StreamlitRuntimeConfig(root_path=root_path)
    if root_path is not None:
        runtime.enabled_tools = ["read_file"]
        runtime.allow_filesystem = True
    return runtime


def _make_record(*, session_id: str = "session-1", root_path: str | None = None):
    runtime = _make_runtime(root_path=root_path)
    return _MODULES.app._new_session_record(session_id, runtime)


def _make_app_state(*, session_id: str = "session-1", root_path: str | None = None):
    record = _make_record(session_id=session_id, root_path=root_path)
    return _MODULES.app.AssistantWorkspaceState(
        sessions={session_id: record},
        session_order=[session_id],
        active_session_id=session_id,
        preferences=_MODULES.models.StreamlitPreferences(theme_mode="dark"),
        turn_states={session_id: _MODULES.app.AssistantTurnState()},
    )


class _FakeModel(SimpleNamespace):
    def model_dump(self, mode: str = "json") -> dict[str, object]:
        del mode

        def _convert(value: object) -> object:
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            if isinstance(value, list):
                return [_convert(item) for item in value]
            if isinstance(value, dict):
                return {key: _convert(item) for key, item in value.items()}
            if isinstance(value, SimpleNamespace):
                return {key: _convert(item) for key, item in vars(value).items()}
            return value

        return {key: _convert(item) for key, item in vars(self).items()}


def _fake_research_inspection(
    *,
    session_id: str = "research-1",
    stop_reason: str | None = None,
    total_turns: int = 3,
    pending_approval_ids: list[str] | None = None,
    resumed_disposition=None,
    latest_decision_summary: str | None = "waiting",
    replay_limitations: list[str] | None = None,
) -> _FakeModel:
    approval_ids = (
        ["approval-1"] if pending_approval_ids is None else pending_approval_ids
    )
    disposition = (
        resumed_disposition or _MODULES.app.ResumeDisposition.WAITING_FOR_APPROVAL
    )
    approval_request = None
    pending_approval = None
    if approval_ids:
        approval_request = _FakeModel(
            approval_id=approval_ids[0],
            tool_name="read_file",
            expires_at="2026-04-19T10:30:00Z",
            request=_FakeModel(tool_name="read_file", arguments={"path": "note.txt"}),
        )
        pending_approval = _FakeModel(approval_request=approval_request)
    invocation = _FakeModel(
        invocation_index=1,
        tool_name="read_file",
        status=SimpleNamespace(value="approval_requested"),
        approval_id=approval_ids[0] if approval_ids else None,
        error_code=None,
        policy_snapshot=_FakeModel(
            reason="approval required",
            requires_approval=bool(approval_ids),
        ),
        logs=["checked workspace"],
        artifacts=["artifact-1"],
    )
    turn = _FakeModel(
        turn_index=1,
        selected_task_ids=["task-1"],
        planner_selected_task_ids=["task-1"],
        replanning_triggers=["new evidence"],
        workflow_outcome_statuses=[SimpleNamespace(value="approval_requested")],
        verification_status_by_task_id={"task-1": "not_run"},
        pending_approval_id=approval_ids[0] if approval_ids else None,
        no_progress_signals=["stalled"],
        decision_action=SimpleNamespace(value="stop"),
        decision_summary="Need approval before reading the file.",
        invocation_traces=[invocation],
    )
    return _FakeModel(
        summary=_FakeModel(
            session_id=session_id,
            stop_reason=None
            if stop_reason is None
            else SimpleNamespace(value=stop_reason),
            total_turns=total_turns,
            completed_task_ids=["task-1"] if stop_reason else [],
            active_task_ids=[] if stop_reason else ["task-1"],
            pending_approval_ids=approval_ids,
            verification_status_counts={"not_run": 1},
            latest_decision_summary=latest_decision_summary,
        ),
        resumed=_FakeModel(
            disposition=disposition,
            pending_approval=pending_approval,
            issues=[],
        ),
        snapshot=_FakeModel(
            revision="7",
            saved_at="2026-04-19T10:00:00Z",
            state=_FakeModel(
                session=_FakeModel(root_task_id="task-1"),
                tasks=[_FakeModel(task_id="task-1", title="Investigate regression")],
            ),
            artifacts=_FakeModel(
                trace=_FakeModel(
                    final_stop_reason=None
                    if stop_reason is None
                    else SimpleNamespace(value=stop_reason),
                    turns=[turn],
                )
            ),
        ),
        replay=_FakeModel(
            steps=[
                _FakeModel(
                    turn_index=1,
                    selected_task_ids=["task-1"],
                    decision_action=SimpleNamespace(value="stop"),
                    decision_stop_reason=None
                    if stop_reason is None
                    else SimpleNamespace(value=stop_reason),
                    decision_summary="Need approval before continuing.",
                )
            ],
            limitations=replay_limitations or ["Replay omitted tool payload bodies."],
        ),
    )


def test_streamlit_assistant_package_imports_without_loading_streamlit() -> None:
    module = importlib.import_module("llm_tools.apps.streamlit_assistant")
    main_module = importlib.import_module("llm_tools.apps.streamlit_assistant.__main__")

    assert hasattr(module, "main")
    assert hasattr(module, "run_streamlit_assistant_app")
    assert hasattr(main_module, "main")


def test_console_script_target_is_declared_in_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["llm-tools-streamlit-assistant"]
        == "llm_tools.apps.streamlit_assistant:main"
    )


def test_module_main_helpers_dispatch_to_package_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_assistant")
    main_module = importlib.import_module("llm_tools.apps.streamlit_assistant.__main__")

    monkeypatch.setattr(package, "main", lambda: 9)

    assert main_module._main() == 9
    assert main_module.main() == 9


def test_module_entrypoint_raises_system_exit_with_main_return_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_assistant")
    called: list[str] = []
    monkeypatch.setattr(package, "main", lambda: called.append("main") or 0)

    sys.modules.pop("llm_tools.apps.streamlit_assistant.__main__", None)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(
            "llm_tools.apps.streamlit_assistant.__main__", run_name="__main__"
        )

    assert exc.value.code == 0
    assert called == ["main"]


def test_resolve_assistant_config_uses_new_model(tmp_path: Path) -> None:
    config_path = tmp_path / "assistant.yml"
    config_path.write_text(
        """
llm:
  provider: ollama
  model_name: demo-model
workspace:
  default_root: .
research:
  default_max_turns: 9
""".strip(),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["--config", str(config_path), "--model", "override-model"]
    )
    config = _resolve_assistant_config(args)

    assert isinstance(config, StreamlitAssistantConfig)
    assert config.llm.model_name == "override-model"
    assert config.workspace.default_root == "."
    assert config.research.default_max_turns == 9


def test_assistant_runtime_exposes_full_registry_and_defaults_to_no_enabled_tools() -> (
    None
):
    tool_specs = build_assistant_available_tool_specs()
    config = StreamlitAssistantConfig()

    assert "write_file" in tool_specs
    assert "search_gitlab_code" in tool_specs
    assert "search_jira" in tool_specs
    assert resolve_assistant_default_enabled_tools(config) == set()


def test_build_tool_capabilities_reports_blocked_states() -> None:
    tool_specs = build_assistant_available_tool_specs()
    capabilities = build_tool_capabilities(
        tool_specs=tool_specs,
        enabled_tools={"read_file", "search_gitlab_code", "write_file"},
        root_path=None,
        env={},
        allow_network=False,
        allow_filesystem=False,
        allow_subprocess=False,
        require_approval_for={SideEffectClass.LOCAL_WRITE},
    )

    all_items = {
        item.tool_name: item for items in capabilities.values() for item in items
    }
    assert all_items["read_file"].status == "missing_workspace"
    assert all_items["search_gitlab_code"].status == "missing_credentials"
    assert all_items["write_file"].approval_required is True


def test_process_streamlit_assistant_turn_supports_direct_answers() -> None:
    outcome = process_streamlit_assistant_turn(
        root_path=None,
        config=StreamlitAssistantConfig(),
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    final_response={
                        "answer": "Plain answer",
                        "citations": [],
                        "confidence": 0.8,
                        "uncertainty": [],
                        "missing_information": [],
                        "follow_up_suggestions": [],
                    }
                )
            ]
        ),
        session_state=ChatSessionState(),
        user_message="hello",
    )

    assert outcome.session_state.turns
    assert outcome.transcript_entries[-1].final_response is not None
    assert outcome.transcript_entries[-1].final_response.answer == "Plain answer"


def test_process_streamlit_assistant_turn_brokered_tool_provenance_reaches_protection_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret_text = "TOP SECRET TOKEN"  # noqa: S105
    (tmp_path / "secret.txt").write_text(secret_text, encoding="utf-8")
    protection_controller = _RecordingProtectionController(
        prompt_decision=PromptProtectionDecision(action=ProtectionAction.ALLOW),
        response_decision=ResponseProtectionDecision(
            action=ProtectionAction.SANITIZE,
            sanitized_payload={
                "answer": "Safe replacement",
                "citations": [],
                "confidence": 0.4,
                "uncertainty": [],
                "missing_information": [],
                "follow_up_suggestions": [],
            },
        ),
    )
    monkeypatch.setattr(
        _MODULES.app,
        "build_protection_controller",
        lambda **kwargs: protection_controller,
    )

    outcome = process_streamlit_assistant_turn(
        root_path=tmp_path,
        config=StreamlitAssistantConfig(),
        runtime_config=_MODULES.models.StreamlitRuntimeConfig(
            root_path=str(tmp_path),
            enabled_tools=["read_file"],
            allow_filesystem=True,
        ),
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "read_file", "arguments": {"path": "secret.txt"}}
                    ]
                ),
                ParsedModelResponse(
                    final_response={
                        "answer": secret_text,
                        "citations": [],
                        "confidence": 0.9,
                        "uncertainty": [],
                        "missing_information": [],
                        "follow_up_suggestions": [],
                    }
                ),
            ]
        ),
        session_state=ChatSessionState(),
        user_message="Read the secret file.",
    )

    turn = outcome.session_state.turns[-1]
    assert turn.tool_results[0].metadata["execution_record"]["tool_name"] == "read_file"
    assert (
        protection_controller.response_calls[0]["provenance"]
        .sources[0]
        .metadata["path"]
        == "secret.txt"
    )
    assert outcome.transcript_entries[-1].final_response is not None
    assert outcome.transcript_entries[-1].final_response.answer == "Safe replacement"
    assert secret_text not in outcome.transcript_entries[-1].text


def test_process_streamlit_assistant_turn_protection_demo_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readme_path = tmp_path / "README.md"
    readme_path.write_text("Workspace read target.\n", encoding="utf-8")
    guidance_path = tmp_path / "proprietary-guidance.md"
    guidance_path.write_text(
        "Secret plans and proprietary playbooks are restricted.\n",
        encoding="utf-8",
    )
    corrections_path = tmp_path / "corrections.json"
    protection_config = ProtectionConfig(
        enabled=True,
        document_paths=[str(guidance_path)],
        corrections_path=str(corrections_path),
    )

    def _build_demo_controller(**kwargs) -> ProtectionController:
        config = kwargs["config"]
        feedback_store = ProtectionFeedbackStore(config.corrections_path)
        corpus = load_protection_corpus(config, feedback_store=feedback_store)
        return ProtectionController(
            config=config,
            classifier=_DeterministicProtectionClassifier(
                document_id=guidance_path.name
            ),
            environment=dict(kwargs["environment"]),
            comparator=DefaultEnvironmentComparator(),
            corpus=corpus,
            feedback_store=feedback_store,
        )

    monkeypatch.setattr(
        _MODULES.app,
        "build_protection_controller",
        _build_demo_controller,
    )

    runtime_config = _MODULES.models.StreamlitRuntimeConfig(
        root_path=str(tmp_path),
        enabled_tools=["read_file"],
        allow_filesystem=True,
        protection=protection_config,
    )

    challenge_outcome = process_streamlit_assistant_turn(
        root_path=tmp_path,
        config=StreamlitAssistantConfig(),
        runtime_config=runtime_config,
        provider=_UnexpectedProvider(),
        session_state=ChatSessionState(),
        user_message="Tell me the secret plan from the proprietary playbook.",
    )

    assert (
        "Potential sensitivity issue" in challenge_outcome.transcript_entries[-1].text
    )
    assert challenge_outcome.session_state.pending_protection_prompt is not None

    feedback_outcome = process_streamlit_assistant_turn(
        root_path=tmp_path,
        config=StreamlitAssistantConfig(),
        runtime_config=runtime_config,
        provider=_UnexpectedProvider(),
        session_state=challenge_outcome.session_state,
        user_message=(
            "analysis_is_correct: false\n"
            "expected_sensitivity_label: public\n"
            "rationale: This request was already approved for external-safe sharing."
        ),
    )

    assert feedback_outcome.session_state.pending_protection_prompt is None
    assert "Recorded your correction" in feedback_outcome.transcript_entries[-1].text
    feedback_entries = ProtectionFeedbackStore(corrections_path).load_entries()
    assert feedback_entries[-1].expected_sensitivity_label == "public"

    sanitize_outcome = process_streamlit_assistant_turn(
        root_path=tmp_path,
        config=StreamlitAssistantConfig(),
        runtime_config=runtime_config,
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "read_file", "arguments": {"path": "README.md"}}
                    ]
                ),
                ParsedModelResponse(
                    final_response={
                        "answer": "TOP SECRET TOKEN",
                        "citations": [],
                        "confidence": 0.9,
                        "uncertainty": [],
                        "missing_information": [],
                        "follow_up_suggestions": [],
                    }
                ),
            ]
        ),
        session_state=feedback_outcome.session_state,
        user_message="Use a local workspace tool to inspect README.md, then answer safely.",
    )

    assert sanitize_outcome.transcript_entries[-1].final_response is not None
    assert (
        sanitize_outcome.transcript_entries[-1].final_response.answer
        == "Safe replacement"
    )
    assert "TOP SECRET TOKEN" not in sanitize_outcome.transcript_entries[-1].text
    tool_result = sanitize_outcome.session_state.turns[-1].tool_results[0]
    assert tool_result.metadata["execution_record"]["tool_name"] == "read_file"
    assert tool_result.source_provenance[0].metadata["path"] == "README.md"


def test_research_controller_can_launch_list_and_stop_sessions() -> None:
    store = InMemoryHarnessStateStore()

    def service_factory() -> HarnessSessionService:
        _, workflow_executor = build_assistant_executor()
        return HarnessSessionService(
            store=store,
            workflow_executor=workflow_executor,
            provider=ScriptedParsedResponseProvider(
                [ParsedModelResponse(final_response="research complete")]
            ),
            workspace=".",
        )

    controller = AssistantResearchSessionController(
        service_factory=service_factory,
        budget_policy=BudgetPolicy(max_turns=2),
        include_replay_by_default=False,
        list_limit=5,
    )

    inspection = controller.launch(prompt="Investigate the workspace")
    listed = controller.list_recent()
    stopped = controller.stop(inspection.snapshot.session_id)

    assert listed.sessions
    assert listed.sessions[0].snapshot.session_id == inspection.snapshot.session_id
    assert "Research session:" in controller.summary_text(inspection)
    assert stopped.snapshot.session_id == inspection.snapshot.session_id


def test_package_main_and_runner_dispatch_to_app_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_assistant")
    called: list[str] = []

    monkeypatch.setattr(
        "llm_tools.apps.streamlit_assistant.app.run_streamlit_assistant_app",
        lambda **kwargs: called.append(f"run:{kwargs['root_path']}"),
    )
    monkeypatch.setattr(
        "llm_tools.apps.streamlit_assistant.app.main",
        lambda argv=None: called.append(f"main:{argv}") or 0,
    )

    package.run_streamlit_assistant_app(
        root_path=None, config=StreamlitAssistantConfig()
    )
    assert package.main() == 0
    assert called == ["run:None", "main:None"]


def test_assistant_prompts_cover_normal_and_research_modes() -> None:
    registry = build_assistant_registry()
    enabled = {"read_file", "search_jira"}

    assistant_prompt = build_assistant_system_prompt(
        tool_registry=registry,
        tool_limits=StreamlitAssistantConfig().tool_limits,
        enabled_tool_names=enabled,
        workspace_enabled=False,
    )
    research_prompt = build_research_system_prompt(
        tool_registry=registry,
        tool_limits=StreamlitAssistantConfig().tool_limits,
        enabled_tool_names=enabled,
        workspace_enabled=True,
    )
    staged_assistant_prompt = build_assistant_system_prompt(
        tool_registry=registry,
        tool_limits=StreamlitAssistantConfig().tool_limits,
        enabled_tool_names=enabled,
        workspace_enabled=True,
        staged_schema_protocol=True,
    )
    staged_research_prompt = build_research_system_prompt(
        tool_registry=registry,
        tool_limits=StreamlitAssistantConfig().tool_limits,
        enabled_tool_names=enabled,
        workspace_enabled=True,
        staged_schema_protocol=True,
    )

    assert "general-purpose assistant" in assistant_prompt
    assert "No workspace root is configured" in assistant_prompt
    assert "search_jira" in assistant_prompt
    assert "inspect relevant local files or git data" in assistant_prompt
    assert "durable research assistant" in research_prompt
    assert "A workspace root is configured" in research_prompt
    assert "Structured interaction protocol:" in staged_assistant_prompt
    assert (
        "Do not invent tool arguments until the client sends the selected tool schema."
        in staged_assistant_prompt
    )
    assert "Required action format:" not in staged_assistant_prompt
    assert "Structured interaction protocol:" in staged_research_prompt
    assert "Final response fields:" not in staged_assistant_prompt


def test_assistant_config_validators_normalize_blank_values() -> None:
    assert AssistantWorkspaceConfig(default_root=None).default_root is None
    assert AssistantWorkspaceConfig(default_root="   ").default_root is None
    assert (
        AssistantWorkspaceConfig(default_root=" ./workspace ").default_root
        == "./workspace"
    )
    assert AssistantResearchConfig(store_dir=None).store_dir is None
    assert AssistantResearchConfig(store_dir="   ").store_dir is None
    assert AssistantResearchConfig(store_dir=" .assistant ").store_dir == ".assistant"


def test_load_streamlit_assistant_config_handles_missing_nonfile_and_empty_yaml(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.yml"
    empty_path = tmp_path / "empty.yml"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration file not found"):
        load_streamlit_assistant_config(missing_path)

    with pytest.raises(ValueError, match="Configuration path is not a file"):
        load_streamlit_assistant_config(tmp_path)

    config = load_streamlit_assistant_config(empty_path)

    assert isinstance(config, StreamlitAssistantConfig)


def test_load_streamlit_assistant_config_rejects_non_mapping_and_invalid_yaml(
    tmp_path: Path,
) -> None:
    scalar_path = tmp_path / "scalar.yml"
    scalar_path.write_text("- item\n", encoding="utf-8")
    invalid_yaml_path = tmp_path / "invalid.yml"
    invalid_yaml_path.write_text("llm: [oops\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected mapping at root of YAML file"):
        load_streamlit_assistant_config(scalar_path)

    with pytest.raises(ValueError, match="Invalid YAML at"):
        load_streamlit_assistant_config(invalid_yaml_path)


def test_load_streamlit_assistant_config_wraps_nested_validation_errors(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid-research.yml"
    config_path.write_text(
        "research:\n  default_max_turns: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid assistant config at"):
        load_streamlit_assistant_config(config_path)


def test_assistant_config_validation_rejects_invalid_shapes(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yml"
    config_path.write_text("session: nope", encoding="utf-8")

    with pytest.raises(ValueError):
        _resolve_assistant_config(
            build_parser().parse_args(["--config", str(config_path)])
        )


def test_build_assistant_context_merges_env_overrides_and_default_read_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEEP_ME", "from-env")
    config = StreamlitAssistantConfig()
    config.tool_limits.max_read_file_chars = None
    context = build_assistant_context(
        root_path=None,
        config=config,
        app_name="assistant-test",
        env_overrides={
            "KEEP_ME": " from-override ",
            "DROP_ME": "   ",
            "ADD_ME": " present ",
        },
    )

    assert context.workspace is None
    assert context.env["KEEP_ME"] == "from-override"
    assert context.env["ADD_ME"] == "present"
    assert "DROP_ME" not in context.env
    assert context.metadata["tool_limits"]["max_read_file_chars"] == (
        config.session.max_context_tokens * 4
    )


def test_assistant_runtime_builds_policy_and_context() -> None:
    tool_specs = build_assistant_available_tool_specs()
    policy = build_assistant_policy(
        enabled_tools={"read_file", "write_file"},
        tool_specs=tool_specs,
        require_approval_for={SideEffectClass.LOCAL_WRITE},
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
        redaction_config=StreamlitAssistantConfig().policy.redaction,
    )
    context = build_assistant_context(
        root_path=Path(".").resolve(),
        config=StreamlitAssistantConfig(),
        app_name="assistant-test",
    )

    assert policy.allowed_tools == {"read_file", "write_file"}
    assert policy.allow_filesystem is True
    assert context.workspace
    assert context.metadata["assistant_mode"] == "streamlit_assistant"
    assert "tool_limits" in context.metadata


class _RecordingProvider:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None

    def run(self, **kwargs: object) -> ParsedModelResponse:
        self.messages = kwargs["messages"]  # type: ignore[index]
        return ParsedModelResponse(final_response="done")

    async def run_async(self, **kwargs: object) -> ParsedModelResponse:
        self.messages = kwargs["messages"]  # type: ignore[index]
        return ParsedModelResponse(final_response="done")


class _RecordingStagedProvider:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def prefers_simplified_json_schema_contract(self) -> bool:
        return False

    def uses_staged_schema_protocol(self) -> bool:
        return True

    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("staged provider should use run_structured()")

    async def run_async(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("staged provider should use run_structured_async()")

    def run_structured(self, **kwargs: object) -> object:
        messages = kwargs["messages"]  # type: ignore[index]
        assert isinstance(messages, list)
        self.calls.append([dict(message) for message in messages])
        return self._responses.pop(0)

    async def run_structured_async(self, **kwargs: object) -> object:
        return self.run_structured(**kwargs)


def test_assistant_harness_turn_provider_builds_research_messages() -> None:
    provider = _RecordingProvider()
    harness_provider = AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )
    response = harness_provider.run(
        state=object(),
        selected_task_ids=["task-1"],
        context=ToolContext(
            invocation_id="turn-1",
            metadata={"harness_turn_context": {"turn_index": 1}},
        ),
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=build_assistant_executor()[1].prepare_model_interaction(
            ActionEnvelopeAdapter(),
            context=ToolContext(invocation_id="demo"),
            final_response_model=str,
        ),
    )

    assert response.final_response == "done"
    assert provider.messages is not None
    assert provider.messages[0]["content"] == "research-system"
    assert '"task-1"' in provider.messages[1]["content"]


def test_assistant_harness_turn_provider_repairs_invalid_staged_final_response() -> (
    None
):
    provider = _RecordingStagedProvider(
        [
            {"mode": "finalize"},
            {"mode": "finalize", "final_response": {"summary": "bad-shape"}},
            {"mode": "finalize", "final_response": "done"},
        ]
    )
    harness_provider = AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )
    response = harness_provider.run(
        state=object(),
        selected_task_ids=["task-1"],
        context=ToolContext(
            invocation_id="turn-1",
            metadata={"harness_turn_context": {"turn_index": 1}},
        ),
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=build_assistant_executor()[1].prepare_model_interaction(
            ActionEnvelopeAdapter(),
            context=ToolContext(invocation_id="demo"),
            final_response_model=str,
        ),
    )

    assert response.final_response == "done"
    assert len(provider.calls) == 3
    repair_message = provider.calls[-1][-1]["content"]
    assert "The previous final_response response was invalid." in repair_message
    assert "Validation summary:" in repair_message
    assert '"summary": "bad-shape"' in repair_message


def test_streamlit_assistant_helper_paths_and_preferences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))
    config = StreamlitAssistantConfig(
        workspace=AssistantWorkspaceConfig(default_root=str(tmp_path))
    )

    args = build_parser().parse_args([])
    assert _MODULES.app._resolve_root_argument(args, config) == tmp_path.resolve()
    assert _MODULES.app._storage_root() == (tmp_path / "state").resolve()
    assert _MODULES.app._sessions_dir() == (tmp_path / "state" / "sessions").resolve()
    assert (
        _MODULES.app._preferences_path()
        == (tmp_path / "state" / "preferences.json").resolve()
    )
    assert _MODULES.app._index_path() == (tmp_path / "state" / "index.json").resolve()
    assert (
        _MODULES.app._session_path("demo")
        == (tmp_path / "state" / "sessions" / "demo.json").resolve()
    )
    assert (
        _MODULES.app._research_store_dir(config)
        == (tmp_path / "state" / "research").resolve()
    )
    assert _MODULES.app._dedupe_preserve([" a ", "", "a", "b "]) == ["a", "b"]
    assert _MODULES.app._title_from_prompt("word " * 20).endswith("...")

    runtime = _MODULES.app._default_runtime_config(config, root_path=tmp_path)
    assert runtime.root_path == str(tmp_path)
    assert runtime.allow_network is True
    assert runtime.allow_filesystem is True
    assert runtime.allow_subprocess is True
    preferences = _MODULES.models.StreamlitPreferences(theme_mode="dark")
    _MODULES.app._remember_runtime_preferences(preferences, runtime)
    assert preferences.recent_roots[0] == str(tmp_path)
    assert preferences.recent_models[runtime.provider.value][0] == runtime.model_name

    record = _MODULES.app._new_session_record("session-1", runtime)
    record.transcript.extend(
        [
            _MODULES.models.StreamlitTranscriptEntry(role="user", text="hello"),
            _MODULES.models.StreamlitTranscriptEntry(role="assistant", text="hi"),
            _MODULES.models.StreamlitTranscriptEntry(
                role="system", text="hidden", show_in_transcript=False
            ),
        ]
    )
    _MODULES.app._sync_summary_fields(record)
    assert record.summary.message_count == 2
    _MODULES.app._touch_record(record)
    assert (
        _MODULES.app._visible_transcript_entries(record.transcript)
        == record.transcript[:2]
    )


def test_streamlit_assistant_provider_and_workspace_settings_keep_permissions_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(
        text_input_values={"Workspace root": str(tmp_path)},
    )
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    config = StreamlitAssistantConfig(
        workspace=AssistantWorkspaceConfig(default_root=str(tmp_path))
    )
    runtime = _MODULES.app._default_runtime_config(config, root_path=None)

    llm_config = _MODULES.app._render_sidebar_provider_connection_controls(
        runtime,
        config=config,
        session_id="session-1",
    )
    _MODULES.app._render_sidebar_model_selection_controls(
        runtime,
        llm_config=llm_config,
        session_id="session-1",
    )
    _MODULES.app._render_sidebar_remote_credentials_controls()
    _MODULES.app._render_sidebar_workspace_controls(runtime)
    _MODULES.app._render_sidebar_permission_controls(runtime)

    assert runtime.root_path == str(tmp_path.resolve())
    assert runtime.allow_network is True
    assert runtime.allow_filesystem is True
    assert runtime.allow_subprocess is True
    assert "Validate provider connection" in fake_st.button_labels
    assert any(
        "Validate the current provider settings before you choose a model" in message
        for message in fake_st.caption_messages
    )
    assert any(
        "Validate the provider connection first." in message
        for message in fake_st.caption_messages
    )
    assert any(
        "Workspace selected. Local tools are scoped now" in message
        for message in fake_st.caption_messages
    )
    assert any(
        "Network, filesystem, and subprocess access are on by default." in message
        for message in fake_st.caption_messages
    )


def test_streamlit_assistant_pick_local_path_rejects_invalid_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)

    class _FakeTkRoot:
        def withdraw(self) -> None:
            return None

        def attributes(self, *_args: object) -> None:
            return None

        def destroy(self) -> None:
            return None

    fake_tk = ModuleType("tkinter")
    fake_tk.Tk = lambda: _FakeTkRoot()
    fake_dialog = ModuleType("filedialog")
    fake_dialog.askopenfilename = lambda: "/work/example.txt"
    fake_tk.filedialog = fake_dialog
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)

    picked = _MODULES.app._pick_local_path(
        directory=False,
        allowed_suffixes={".json"},
    )

    assert picked is None
    assert any(
        "Selected file must use one of these extensions" in message
        for message in fake_st.warning_messages
    )


def test_streamlit_assistant_path_inputs_use_picker_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(
        button_values={
            "browse-workspace-root": True,
            "browse-corrections": True,
        }
    )
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    picks = iter([[str(tmp_path)], [str(tmp_path / "corrections.json")]])
    monkeypatch.setattr(_MODULES.app, "_pick_local_path", lambda **kwargs: next(picks))

    workspace = _MODULES.app._render_directory_path_input(
        label="Workspace root",
        value=None,
        placeholder="dir",
        browse_key="browse-workspace-root",
    )
    corrections = _MODULES.app._render_file_path_input(
        label="Corrections",
        value=None,
        placeholder="file",
        browse_key="browse-corrections",
        allowed_suffixes={".json"},
    )

    assert workspace == str(tmp_path)
    assert corrections == str(tmp_path / "corrections.json")


def test_streamlit_assistant_source_and_advanced_controls_update_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(
        checkbox_values={
            "source-group:GitLab": True,
            "Enable research tasks": True,
            "Show token usage": False,
            "Show footer help": False,
            "Show inspector details": True,
        },
        text_input_values={
            "remote-url:GITLAB_BASE_URL": "https://gitlab.internal",
            "remote-secret:GITLAB_API_TOKEN": "gitlab-token",
            "Default workspace root in exported YAML": str(tmp_path),
            "Research store directory": str(tmp_path / "research-store"),
            "Protection corrections path": str(tmp_path / "corrections.yaml"),
            "assistant-protection-paths": str(tmp_path),
        },
        number_input_values={
            "Temperature": 0.3,
            "Provider timeout seconds": 45.0,
            "Max context tokens": 32000,
            "Max tool round trips": 5,
            "Max tool calls per round": 3,
            "Max total tool calls per turn": 9,
            "Max entries per call": 111,
            "Max recursive depth": 7,
            "Max files scanned": 1234,
            "Max search matches": 15,
            "Max read lines": 77,
            "Max read input bytes": 2048,
            "Max file size characters": 4096,
            "Max read file characters (0 uses no override)": 512,
            "Max tool result characters": 9000,
            "Recent research sessions to show": 4,
            "Default research max turns": 8,
            "Default research max tool invocations (0 uses no limit)": 16,
            "Default research max elapsed seconds (0 uses no limit)": 120,
        },
    )
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    (tmp_path / "policy.txt").write_text("private", encoding="utf-8")
    runtime = _MODULES.models.StreamlitRuntimeConfig(
        protection=_MODULES.app.ProtectionConfig(enabled=True),
    )

    _MODULES.app._render_sidebar_source_controls(runtime)
    _MODULES.app._render_sidebar_advanced_controls(runtime)

    assert "search_gitlab_code" in runtime.enabled_tools
    assert (
        _MODULES.app._get_session_env_value("GITLAB_BASE_URL")
        == "https://gitlab.internal"
    )
    assert _MODULES.app._get_secret_value("GITLAB_API_TOKEN") == "gitlab-token"
    assert runtime.temperature == 0.3
    assert runtime.timeout_seconds == 45.0
    assert runtime.session_config.max_context_tokens == 32000
    assert runtime.tool_limits.max_read_file_chars == 512
    assert runtime.research.store_dir == str(tmp_path / "research-store")
    assert runtime.default_workspace_root == str(tmp_path)
    assert runtime.show_token_usage is False
    assert runtime.show_footer_help is False
    assert runtime.inspector_open is True
    assert any(
        "Current source readiness" in message for message in fake_st.markdown_messages
    )


def test_streamlit_assistant_sidebar_tool_controls_show_guided_readiness_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    runtime = _MODULES.models.StreamlitRuntimeConfig(
        enabled_tools=["read_file", "search_jira"],
    )

    _MODULES.app._render_sidebar_tool_controls(runtime)

    assert any(
        "Current source readiness" in message for message in fake_st.markdown_messages
    )
    assert any(
        "Enabled sources: Jira: 1 enabled | needs credentials; Local Files: 1 enabled | needs workspace"
        in message
        for message in fake_st.caption_messages
    )
    assert any(
        "Next: choose a workspace root in the Workspace section for local file and git sources."
        in message
        for message in fake_st.caption_messages
    )
    assert any(
        "Next: add the required URLs and credentials in the Sources section for the enabled remote sources."
        in message
        for message in fake_st.caption_messages
    )


def test_streamlit_assistant_session_secret_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")

    assert _MODULES.app._get_secret_value("OPENAI_API_KEY") == "env-secret"

    _MODULES.app._set_secret_value("OPENAI_API_KEY", "session-secret")

    assert _MODULES.app._get_secret_value("OPENAI_API_KEY") == "session-secret"


def test_streamlit_assistant_execution_blocker_uses_session_only_remote_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(
        _MODULES.app,
        "_ensure_model_connection_ready",
        lambda **kwargs: _MODULES.app.ProviderPreflightResult(
            ok=True,
            connection_succeeded=True,
            model_accepted=True,
            selected_mode_supported=True,
            model_listing_supported=True,
            available_models=["demo-model"],
            resolved_mode=None,
            actionable_message="ready",
        ),
    )
    runtime = _MODULES.models.StreamlitRuntimeConfig(
        enabled_tools=["search_jira"],
        allow_network=True,
    )

    blocker = _MODULES.app._execution_blocker(
        session_id="session-1",
        config=StreamlitAssistantConfig(),
        runtime=runtime,
        require_tool_readiness=True,
    )

    assert blocker == (
        "Research is not ready yet. Add the required URLs and credentials for the enabled remote "
        "sources in the Sources section."
    )

    _MODULES.app._set_session_env_value("JIRA_BASE_URL", "https://jira.internal")
    _MODULES.app._set_secret_value("JIRA_API_TOKEN", "jira-token")

    assert (
        _MODULES.app._execution_blocker(
            session_id="session-1",
            config=StreamlitAssistantConfig(),
            runtime=runtime,
            require_tool_readiness=True,
        )
        is None
    )


def test_streamlit_assistant_protection_controls_accept_directory_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(
        checkbox_values={"Enable proprietary protection": True},
        text_input_values={
            "assistant-protection-paths": str(tmp_path),
            "Protection corrections path": str(tmp_path / "corrections.yaml"),
        },
    )
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    (tmp_path / "policy.txt").write_text("keep private", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    runtime = _MODULES.models.StreamlitRuntimeConfig()

    _MODULES.app._render_sidebar_protection_controls(runtime)

    assert runtime.protection.enabled is True
    assert runtime.protection.document_paths == [str(tmp_path)]
    assert runtime.protection.corrections_path == str(tmp_path / "corrections.yaml")
    assert any(
        "Loaded protection documents: 1" in message
        for message in fake_st.caption_messages
    )
    assert any(
        "Skipped unsupported file type for protection corpus." in message
        for message in fake_st.caption_messages
    )


def test_streamlit_assistant_config_export_and_state_persistence_omit_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_path = tmp_path / "exports" / "assistant.yaml"
    fake_st = _FakeStreamlit(
        button_values={
            "generate-config:session-1": True,
            "save-config:session-1": True,
        },
        text_input_values={
            "save-config-path:session-1": str(export_path),
        },
    )
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))
    config = StreamlitAssistantConfig()
    runtime = _MODULES.models.StreamlitRuntimeConfig(
        provider="custom_openai_compatible",
        provider_mode_strategy="json",
        model_name="internal-model",
        api_base_url="http://llm.internal/v1",
        root_path=str(tmp_path),
        enabled_tools=["search_jira"],
        protection=_MODULES.app.ProtectionConfig(
            enabled=True,
            document_paths=[str(tmp_path)],
            corrections_path=str(tmp_path / "corrections.yaml"),
        ),
    )
    _MODULES.app._set_secret_value("OPENAI_API_KEY", "model-secret")
    _MODULES.app._set_secret_value("JIRA_API_TOKEN", "jira-secret")
    _MODULES.app._set_session_env_value("JIRA_BASE_URL", "https://jira.internal")

    _MODULES.app._render_sidebar_config_export(
        config=config,
        runtime=runtime,
        session_id="session-1",
    )

    exported = export_path.read_text(encoding="utf-8")
    assert "provider_mode_strategy: json" in exported
    assert "document_paths:" in exported
    assert str(tmp_path) in exported
    assert "model-secret" not in exported
    assert "jira-secret" not in exported

    round_tripped = load_streamlit_assistant_config(export_path)
    assert round_tripped.llm.provider_mode_strategy.value == "json"
    assert round_tripped.workspace.default_root is None
    assert round_tripped.protection.document_paths == [str(tmp_path)]
    assert round_tripped.protection.corrections_path == str(
        tmp_path / "corrections.yaml"
    )

    app_state = _make_app_state()
    _MODULES.app._save_workspace_state(app_state)
    persisted_payloads = [
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "state").rglob("*.json")
    ]
    assert persisted_payloads
    assert all("model-secret" not in payload for payload in persisted_payloads)
    assert all("jira-secret" not in payload for payload in persisted_payloads)


def test_streamlit_assistant_tool_capability_caption_uses_explicit_blockers() -> None:
    tool_specs = _MODULES.app._all_tool_specs()
    capabilities = build_tool_capabilities(
        tool_specs=tool_specs,
        enabled_tools={"read_file", "search_jira", "run_git_status"},
        root_path=None,
        env={},
        allow_network=False,
        allow_filesystem=False,
        allow_subprocess=False,
        require_approval_for={SideEffectClass.LOCAL_READ},
    )
    items = {item.tool_name: item for group in capabilities.values() for item in group}

    assert _MODULES.app._tool_capability_caption(items["read_file"]) == (
        "Not ready yet. Choose a workspace root in the Workspace section. "
        "This tool pauses for approval before it runs."
    )
    assert _MODULES.app._tool_capability_caption(items["search_jira"]) == (
        "Not ready yet. Add credentials: JIRA_API_TOKEN, JIRA_BASE_URL. "
        "Turn on network access in the Advanced section."
    )
    assert _MODULES.app._tool_capability_caption(items["run_git_status"]) == (
        "Not ready yet. Choose a workspace root in the Workspace section. "
        "This tool pauses for approval before it runs."
    )


def test_streamlit_assistant_copy_helpers_cover_status_sources_and_session_meta() -> (
    None
):
    runtime = _make_runtime(root_path=None)
    runtime.enabled_tools = []
    assert _MODULES.app._assistant_status_copy("thinking") == (
        "Assistant is drafting a response."
    )
    assert _MODULES.app._assistant_status_copy("gathering evidence") == (
        "Assistant is gathering workspace evidence before answering."
    )
    assert _MODULES.app._source_readiness_tokens(runtime) == ["sources: chat only"]

    runtime = _make_runtime(root_path=None)
    runtime.enabled_tools = ["read_file", "write_file"]
    runtime.allow_filesystem = False
    runtime.require_approval_for = {SideEffectClass.LOCAL_WRITE}
    tokens = _MODULES.app._source_readiness_tokens(runtime)
    assert "Local Files: needs workspace" in tokens

    record = _make_record(root_path=None)
    record.summary.message_count = 3
    turn_state = _MODULES.app.AssistantTurnState(
        busy=True, queued_follow_up_prompt="follow up"
    )
    meta = _MODULES.app._session_meta_copy(
        record,
        turn_state=turn_state,
        draft="draft text",
        is_active=True,
    )
    assert meta == "current | 3 msgs | working | follow-up queued | draft saved"


def test_streamlit_assistant_recent_status_history_copy_trims_to_recent_steps() -> None:
    turn_state = _MODULES.app.AssistantTurnState(
        busy=True,
        status_history=[
            "thinking",
            "gathering evidence",
            "searching text",
            "reading file",
            "drafting answer",
        ],
    )

    assert _MODULES.app._recent_status_history_copy(turn_state) == (
        "Recent steps: Assistant is gathering workspace evidence before answering"
        " -> Assistant is searching the workspace"
        " -> Assistant is reading relevant files"
        " -> Drafting answer"
    )


def test_streamlit_assistant_defaults_to_dark_theme_for_new_workspace_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))

    app_state = _MODULES.app._load_workspace_state(
        root_path=None,
        config=StreamlitAssistantConfig(),
    )

    assert app_state.preferences.theme_mode == "dark"


def test_streamlit_assistant_migrates_legacy_light_preference_to_dark_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "state"
    monkeypatch.setenv(_MODULES.app._STORAGE_ENV_VAR, str(storage_root))
    storage_root.mkdir(parents=True)
    (storage_root / "preferences.json").write_text(
        _MODULES.models.StreamlitPreferences(theme_mode="light").model_dump_json(
            indent=2
        )
    )

    app_state = _MODULES.app._load_workspace_state(
        root_path=None,
        config=StreamlitAssistantConfig(),
    )

    assert app_state.preferences.theme_mode == "dark"
    assert app_state.preferences.appearance_mode_explicit is False


def test_streamlit_assistant_theme_sync_applies_widget_choice_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state["assistant-theme-mode"] = False
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    preferences = _MODULES.models.StreamlitPreferences(theme_mode="dark")

    _MODULES.app._sync_theme_preference_from_widget_state(preferences)

    assert preferences.theme_mode == "light"
    assert preferences.appearance_mode_explicit is True


def test_streamlit_assistant_theme_sync_initializes_widget_state_without_forcing_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    preferences = _MODULES.models.StreamlitPreferences(theme_mode="dark")

    _MODULES.app._sync_theme_preference_from_widget_state(preferences)

    assert fake_st.session_state["assistant-theme-mode"] is True
    assert preferences.appearance_mode_explicit is False


def test_streamlit_assistant_theme_sync_accepts_legacy_string_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state["assistant-theme-mode"] = "light"
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    preferences = _MODULES.models.StreamlitPreferences(theme_mode="dark")

    _MODULES.app._sync_theme_preference_from_widget_state(preferences)

    assert preferences.theme_mode == "light"
    assert preferences.appearance_mode_explicit is True


def test_streamlit_assistant_appearance_controls_render_non_editable_theme_combo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit(checkbox_values={"assistant-theme-mode": False})
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    preferences = _MODULES.models.StreamlitPreferences(theme_mode="dark")
    fake_st.session_state["assistant-theme-mode"] = "dark"

    _MODULES.app._render_sidebar_appearance_controls(preferences)

    assert fake_st.checkbox_values["assistant-theme-mode"] is False
    assert preferences.theme_mode == "dark"


def test_streamlit_assistant_page_config_uses_speech_bubble_icon() -> None:
    page_config = _MODULES.app._page_config()

    assert page_config["page_title"] == "llm-tools assistant"
    assert page_config["page_icon"] == "💬"


def test_streamlit_assistant_render_theme_supports_dark_and_light(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)

    _MODULES.app._render_theme(_MODULES.models.StreamlitPreferences(theme_mode="dark"))
    dark_css = fake_st.markdown_messages[-1]
    assert "--assistant-bg: #060b14;" in dark_css
    assert "--assistant-accent: #5e93ff;" in dark_css
    assert ".assistant-chip--ready" in dark_css
    assert ".assistant-empty-state" in dark_css
    assert ".assistant-summary-panel" in dark_css
    assert '[data-testid="stSidebarResizer"]::before' in dark_css
    assert 'header[data-testid="stHeader"] button' in dark_css
    assert (
        '[data-testid="stSidebar"] > div:first-child > div:first-child button'
        in dark_css
    )
    assert ".stAppToolbar" in dark_css
    assert '[data-testid="stAppViewContainer"] .main .block-container' in dark_css

    _MODULES.app._render_theme(_MODULES.models.StreamlitPreferences(theme_mode="light"))
    light_css = fake_st.markdown_messages[-1]
    assert "--assistant-bg: #edf3fb;" in light_css
    assert "--assistant-accent: #245dff;" in light_css
    assert light_css != dark_css


def test_streamlit_assistant_injects_connection_error_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)

    _MODULES.app._render_connection_error_override()

    assert fake_st.component_html_calls
    html, height, width = fake_st.component_html_calls[-1]
    assert "window.parent?.document" in html
    assert '[data-testid="stConnectionStatus"]' in html
    assert "Connection to llm-tools assistant lost" in html
    assert "Reconnect to the llm-tools assistant" in html
    assert "MutationObserver" in html
    assert "streamlit run +yourscript[.]py" in html
    assert height == 0
    assert width == 0


def test_streamlit_assistant_provider_type_choice_maps_runtime_defaults() -> None:
    runtime = _MODULES.models.StreamlitRuntimeConfig(
        provider="ollama",
        provider_mode_strategy="auto",
        api_base_url="http://127.0.0.1:11434/v1",
    )

    _MODULES.app._apply_provider_type_choice(runtime, choice="OpenAI Compatible")
    assert runtime.provider.value == "custom_openai_compatible"
    assert runtime.provider_mode_strategy.value == "json"
    assert runtime.api_base_url is None

    _MODULES.app._apply_provider_type_choice(runtime, choice="Ollama")
    assert runtime.provider.value == "ollama"
    assert runtime.provider_mode_strategy.value == "auto"
    assert runtime.api_base_url == "http://127.0.0.1:11434/v1"


def test_streamlit_assistant_provider_connection_controls_allow_mode_selection_for_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit(
        text_input_values={
            "Provider Base URL": "http://127.0.0.1:11434/v1",
        },
        selectbox_values={
            "Provider Type": "Ollama",
            "Provider mode": "json",
        },
    )
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)

    runtime = _MODULES.models.StreamlitRuntimeConfig(
        provider="ollama",
        provider_mode_strategy="auto",
        api_base_url="http://127.0.0.1:11434/v1",
    )
    _MODULES.app._render_sidebar_provider_connection_controls(
        runtime,
        config=StreamlitAssistantConfig(),
        session_id="session-1",
    )

    assert runtime.provider.value == "ollama"
    assert runtime.provider_mode_strategy.value == "json"
    assert "Provider mode" in fake_st.selectbox_labels


def test_streamlit_assistant_provider_connection_status_copy_covers_variants() -> None:
    ok_with_models = _MODULES.app.ProviderPreflightResult(
        ok=True,
        connection_succeeded=True,
        model_accepted=True,
        selected_mode_supported=True,
        model_listing_supported=True,
        available_models=["a", "b"],
        resolved_mode=None,
        actionable_message="ready",
    )
    assert _MODULES.app._provider_connection_status_copy(ok_with_models) == (
        "Provider connection is ready. Retrieved 2 model(s) for this session.",
        False,
    )

    ok_without_models = _MODULES.app.ProviderPreflightResult(
        ok=True,
        connection_succeeded=True,
        model_accepted=True,
        selected_mode_supported=True,
        model_listing_supported=False,
        available_models=[],
        resolved_mode=None,
        actionable_message="ready",
    )
    assert _MODULES.app._provider_connection_status_copy(ok_without_models) == (
        "Provider connection is ready, but this endpoint did not expose a model list.",
        True,
    )

    failed = _MODULES.app.ProviderPreflightResult(
        ok=False,
        connection_succeeded=False,
        model_accepted=False,
        selected_mode_supported=False,
        model_listing_supported=False,
        available_models=[],
        resolved_mode=None,
        actionable_message="broken",
    )
    assert _MODULES.app._provider_connection_status_copy(failed) == ("broken", True)


def test_streamlit_assistant_provider_and_model_sections_use_validated_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit(
        button_values={"validate-connection:session-1": True},
        text_input_values={
            "Provider Base URL": "http://llm.internal/v1",
            "OPENAI_API_KEY": "secret",
        },
        selectbox_values={
            "Provider Type": "OpenAI Compatible",
            "Provider mode": "json",
            "Model": "model-b",
        },
    )
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(
        _MODULES.app,
        "_validate_model_connection",
        lambda **kwargs: _MODULES.app.ProviderPreflightResult(
            ok=True,
            connection_succeeded=True,
            model_accepted=True,
            selected_mode_supported=True,
            model_listing_supported=True,
            available_models=["model-a", "model-b"],
            resolved_mode=None,
            actionable_message="ready",
        ),
    )
    runtime = _MODULES.models.StreamlitRuntimeConfig()
    config = StreamlitAssistantConfig()

    llm_config = _MODULES.app._render_sidebar_provider_connection_controls(
        runtime,
        config=config,
        session_id="session-1",
    )
    _MODULES.app._store_connection_report(
        "session-1",
        signature=_MODULES.app._provider_signature(
            runtime=runtime,
            api_key=_MODULES.app._current_api_key(llm_config),
        ),
        report=_MODULES.app.ProviderPreflightResult(
            ok=True,
            connection_succeeded=True,
            model_accepted=True,
            selected_mode_supported=True,
            model_listing_supported=True,
            available_models=["model-a", "model-b"],
            resolved_mode=None,
            actionable_message="ready",
        ),
    )
    _MODULES.app._render_sidebar_model_selection_controls(
        runtime,
        llm_config=llm_config,
        session_id="session-1",
    )

    assert runtime.provider.value == "custom_openai_compatible"
    assert runtime.provider_mode_strategy.value == "json"
    assert runtime.api_base_url == "http://llm.internal/v1"
    assert runtime.model_name == "model-b"
    assert "Validate provider connection" in fake_st.button_labels


def test_streamlit_assistant_research_sidebar_helpers_cover_edge_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)

    assert _MODULES.app._selected_research_session_id() is None
    _MODULES.app._select_research_session("research-1")
    assert _MODULES.app._selected_research_session_id() == "research-1"

    summary = SimpleNamespace(
        session_id="research-1",
        total_turns=2,
        active_task_ids=["task-1"],
        pending_approval_ids=[],
        stop_reason=None,
    )
    resumed = SimpleNamespace(
        disposition=_MODULES.app.ResumeDisposition.TERMINAL,
        issues=[SimpleNamespace(message="needs review")],
    )
    view = _MODULES.app._build_research_session_view(
        summary,
        resumed=resumed,
        summarized=True,
    )
    assert view.state_label == "stopped"
    assert "needs attention" in view.state_detail
    assert view.issue_messages == ["needs review"]

    assert _MODULES.app._payload_for_display({"ok": True}) == {"ok": True}
    inspection = SimpleNamespace(
        snapshot=SimpleNamespace(
            state=SimpleNamespace(
                session=SimpleNamespace(root_task_id="task-1"),
                tasks=[SimpleNamespace(task_id="task-1", title="Root Task")],
            )
        )
    )
    assert _MODULES.app._research_session_title(inspection) == "Root Task"


def test_streamlit_assistant_chip_classifies_readiness_states() -> None:
    assert (
        _MODULES.app._chip_class_for_token("workspace ready") == "assistant-chip--ready"
    )
    assert (
        _MODULES.app._chip_class_for_token("network blocked")
        == "assistant-chip--blocked"
    )
    assert (
        _MODULES.app._chip_class_for_token("approval required")
        == "assistant-chip--approval"
    )
    assert (
        _MODULES.app._chip_class_for_token("missing credentials")
        == "assistant-chip--warning"
    )
    assert (
        _MODULES.app._chip_class_for_token("needs attention")
        == "assistant-chip--warning"
    )
    assert _MODULES.app._chip_class_for_token("local files") == ""


def test_streamlit_assistant_sidebar_sections_are_collapsible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)

    app_state = _make_app_state(root_path=str(tmp_path))
    _MODULES.app._render_sidebar(
        app_state,
        config=StreamlitAssistantConfig(),
        root_path=tmp_path,
    )

    for label in (
        "Appearance",
        "Assistant sessions",
        "1. Connect to provider",
        "2. Model selection",
        "3. Workspace",
        "4. Sources",
        "5. Advanced",
        "6. Save configuration",
        "Research tasks",
    ):
        assert label in fake_st.button_labels


def test_streamlit_assistant_render_helpers_show_updated_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(
        button_values={"Queue follow-up": True},
        text_input_values={"composer:session-1": "next question"},
    )
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)

    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    turn_state = _MODULES.app._turn_state_for(app_state, session_id)
    turn_state.busy = True
    turn_state.status_text = "thinking"
    turn_state.queued_follow_up_prompt = "already queued"
    app_state.drafts[session_id] = "draft text"

    created: list[str] = []
    monkeypatch.setattr(
        _MODULES.app,
        "_submit_streamlit_prompt",
        lambda **kwargs: created.append(kwargs["prompt"]) or True,
    )

    _MODULES.app._render_sidebar_session_controls(
        app_state,
        config=StreamlitAssistantConfig(),
        root_path=tmp_path,
        runtime=app_state.sessions[session_id].runtime,
    )
    _MODULES.app._render_summary_chips(app_state.sessions[session_id])
    _MODULES.app._render_empty_state(app_state.sessions[session_id])
    _MODULES.app._render_status_and_composer(
        app_state,
        session_id=session_id,
        config=StreamlitAssistantConfig(),
    )

    assert "New session from current setup" in fake_st.button_labels
    assert "Delete" in fake_st.button_labels
    assert "Queue follow-up" in fake_st.button_labels
    assert any(
        "New sessions reuse the current model, permissions, and enabled sources."
        in item
        for item in fake_st.caption_messages
    )
    assert any(
        "current | 0 msgs | working | follow-up queued | draft saved | workspace on"
        in item
        for item in fake_st.caption_messages
    )
    assert any(
        "Assistant is drafting a response." in item for item in fake_st.caption_messages
    )
    assert any(
        "Queued follow-up: already queued" in item for item in fake_st.caption_messages
    )
    assert any(
        "Start with a normal question" in item for item in fake_st.markdown_messages
    )
    assert any(
        "This assistant can answer directly without tools." in item
        for item in fake_st.markdown_messages
    )
    assert any("Current Session" in item for item in fake_st.markdown_messages)
    assert any("Local Files: ready" in item for item in fake_st.markdown_messages)
    assert created == ["next question"]
    assert app_state.drafts[session_id] == ""


def test_streamlit_assistant_sidebar_session_controls_can_switch_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(button_values={"session:session-2": True})
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)

    app_state = _make_app_state(root_path=str(tmp_path))
    other = _make_record(session_id="session-2", root_path=str(tmp_path))
    other.summary.title = "Second session"
    app_state.sessions["session-2"] = other
    app_state.session_order.append("session-2")
    app_state.turn_states["session-2"] = _MODULES.app.AssistantTurnState()

    switched = _MODULES.app._render_sidebar_session_controls(
        app_state,
        config=StreamlitAssistantConfig(),
        root_path=tmp_path,
        runtime=app_state.sessions[app_state.active_session_id].runtime,
    )

    assert switched is False
    assert app_state.active_session_id == "session-2"


def test_streamlit_assistant_persists_multiple_sessions_and_deletes_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))

    state = _MODULES.app._load_workspace_state(
        root_path=tmp_path,
        config=StreamlitAssistantConfig(),
    )
    original_id = state.active_session_id
    _MODULES.app._create_session(
        state, template_runtime=state.sessions[original_id].runtime
    )

    reloaded = _MODULES.app._load_workspace_state(
        root_path=tmp_path,
        config=StreamlitAssistantConfig(),
    )
    assert len(reloaded.session_order) == 2

    session_to_delete = reloaded.session_order[0]
    _MODULES.app._delete_session(
        reloaded,
        session_id=session_to_delete,
        config=StreamlitAssistantConfig(),
        root_path=tmp_path,
    )
    assert session_to_delete not in reloaded.session_order
    assert not _MODULES.app._session_path(session_to_delete).exists()


def test_streamlit_assistant_skips_corrupt_persisted_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "state"
    monkeypatch.setenv(_MODULES.app._STORAGE_ENV_VAR, str(storage_root))
    (storage_root / "sessions").mkdir(parents=True)
    (storage_root / "index.json").write_text(
        _MODULES.models.StreamlitSessionIndex(
            active_session_id="broken",
            session_order=["broken"],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (storage_root / "sessions" / "broken.json").write_text(
        "{not-json", encoding="utf-8"
    )

    state = _MODULES.app._load_workspace_state(
        root_path=None,
        config=StreamlitAssistantConfig(),
    )

    assert state.session_order
    assert any(
        "Skipped unreadable assistant session broken" in notice
        for notice in state.startup_notices
    )


def test_streamlit_assistant_event_reducers_and_drain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))
    app_state = _make_app_state(root_path=str(tmp_path))
    approval = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-1",
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {"path": "a.txt"}},
            tool_name="read_file",
            tool_version="0.1.0",
            policy_reason="approval required",
            policy_metadata={},
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:05:00Z",
        ),
        tool_name="read_file",
        redacted_arguments={"path": "a.txt"},
        policy_reason="approval required",
        policy_metadata={},
    )
    queued_events = [
        _MODULES.app._serialize_workflow_event(
            ChatWorkflowStatusEvent(status="thinking"),
            turn_number=1,
            session_id="session-1",
        ),
        _MODULES.app._serialize_workflow_event(
            ChatWorkflowApprovalEvent(approval=approval),
            turn_number=1,
            session_id="session-1",
        ),
        _MODULES.app._serialize_workflow_event(
            ChatWorkflowApprovalResolvedEvent(approval=approval, resolution="approved"),
            turn_number=1,
            session_id="session-1",
        ),
        _MODULES.app._serialize_workflow_event(
            ChatWorkflowInspectorEvent(
                round_index=1, kind="provider_messages", payload=[{"role": "user"}]
            ),
            turn_number=1,
            session_id="session-1",
        ),
        _MODULES.app._serialize_workflow_event(
            ChatWorkflowResultEvent(
                result=ChatWorkflowTurnResult(
                    status="completed",
                    new_messages=[],
                    context_warning="warn",
                    final_response=ChatFinalResponse(answer="done", confidence=0.5),
                    token_usage=ChatTokenUsage(
                        session_tokens=22, active_context_tokens=10
                    ),
                    session_state=ChatSessionState(),
                )
            ),
            turn_number=1,
            session_id="session-1",
        ),
        _MODULES.app.AssistantQueuedEvent(
            kind="error",
            payload="boom",
            turn_number=1,
            session_id="session-1",
        ),
    ]
    queue_obj: queue.Queue[object] = queue.Queue()
    for event in queued_events:
        queue_obj.put(event)
    fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _MODULES.app.AssistantActiveTurnHandle(
            session_id="session-1",
            runner=_FakeRunnerHandle(),
            event_queue=queue_obj,
            thread=_DeadThread(),
            turn_number=1,
        )
    )

    pending_prompt = _MODULES.app._drain_active_turn_events(app_state)
    record = app_state.sessions["session-1"]
    assert pending_prompt is None
    assert any(entry.text == "warn" for entry in record.transcript)
    assert any(
        entry.text == "Approved the requested tool run." for entry in record.transcript
    )
    assert any(entry.text == "boom" for entry in record.transcript)
    assert record.inspector_state.provider_messages
    assert record.token_usage is not None
    assert fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] is None

    interrupted_event = ChatWorkflowResultEvent(
        result=ChatWorkflowTurnResult(
            status="interrupted",
            new_messages=[
                ChatMessage(
                    role="assistant",
                    content="partial answer",
                    completion_state="interrupted",
                )
            ],
            interruption_reason="stopped",
        )
    )
    _MODULES.app._apply_turn_result(
        app_state, session_id="session-1", event=interrupted_event
    )
    assert record.transcript[-1].assistant_completion_state == "interrupted"

    interrupted_no_message = ChatWorkflowResultEvent(
        result=ChatWorkflowTurnResult(
            status="interrupted",
            new_messages=[],
            interruption_reason="stopped hard",
        )
    )
    _MODULES.app._apply_turn_result(
        app_state, session_id="session-1", event=interrupted_no_message
    )
    assert record.transcript[-1].text == "stopped hard"

    with pytest.raises(TypeError):
        _MODULES.app._serialize_workflow_event(
            object(), turn_number=1, session_id="session-1"
        )


def test_streamlit_assistant_drain_returns_originating_session_for_queued_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)

    app_state = _make_app_state(root_path=str(tmp_path))
    other = _make_record(session_id="session-2", root_path=str(tmp_path))
    app_state.sessions["session-2"] = other
    app_state.session_order.append("session-2")
    app_state.turn_states["session-2"] = _MODULES.app.AssistantTurnState()
    app_state.active_session_id = "session-2"

    queue_obj: queue.Queue[object] = queue.Queue()
    queue_obj.put(
        _MODULES.app.AssistantQueuedEvent(
            kind="complete",
            payload=None,
            turn_number=1,
            session_id="session-1",
        )
    )
    handle = _MODULES.app.AssistantActiveTurnHandle(
        session_id="session-1",
        runner=_FakeRunnerHandle(),
        event_queue=queue_obj,
        thread=_DeadThread(),
        turn_number=1,
    )
    fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] = handle
    turn_state = _MODULES.app._turn_state_for(app_state, "session-1")
    turn_state.busy = True
    turn_state.cancelling = True
    turn_state.queued_follow_up_prompt = "resume work"

    pending_prompt = _MODULES.app._drain_active_turn_events(app_state)

    assert pending_prompt == ("session-1", "resume work")
    assert fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] is None


def test_streamlit_assistant_complete_event_finishes_cancelled_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)

    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    turn_state = _MODULES.app._turn_state_for(app_state, session_id)
    turn_state.busy = True
    turn_state.cancelling = True
    turn_state.status_text = "stopping"
    turn_state.queued_follow_up_prompt = "next prompt"

    ignored = _MODULES.app._apply_queued_event(
        app_state,
        _MODULES.app.AssistantQueuedEvent(
            kind="status",
            payload=ChatWorkflowStatusEvent(status="thinking").model_dump(mode="json"),
            turn_number=1,
            session_id=session_id,
        ),
    )
    assert ignored is None
    assert turn_state.status_text == "stopping"

    pending_prompt = _MODULES.app._apply_queued_event(
        app_state,
        _MODULES.app.AssistantQueuedEvent(
            kind="complete",
            payload=None,
            turn_number=1,
            session_id=session_id,
        ),
    )
    assert pending_prompt == "next prompt"
    assert turn_state.busy is False
    assert turn_state.cancelling is False
    assert (
        app_state.sessions[session_id].transcript[-1].text == "Stopped the active turn."
    )


def test_streamlit_assistant_submit_prompt_and_approval_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)
    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id

    started: list[str] = []
    monkeypatch.setattr(
        _MODULES.app, "_create_provider_for_runtime", lambda *args, **kwargs: "provider"
    )
    monkeypatch.setattr(
        _MODULES.app,
        "_start_streamlit_turn",
        lambda **kwargs: started.append(kwargs["user_message"]),
    )

    turn_state = _MODULES.app._turn_state_for(app_state, session_id)
    assert (
        _MODULES.app._submit_streamlit_prompt(
            app_state=app_state,
            session_id=session_id,
            config=StreamlitAssistantConfig(),
            prompt="   ",
        )
        is False
    )
    turn_state.busy = True
    assert (
        _MODULES.app._submit_streamlit_prompt(
            app_state=app_state,
            session_id=session_id,
            config=StreamlitAssistantConfig(),
            prompt="pending",
        )
        is True
    )
    assert turn_state.queued_follow_up_prompt == "pending"
    assert (
        _MODULES.app._submit_streamlit_prompt(
            app_state=app_state,
            session_id=session_id,
            config=StreamlitAssistantConfig(),
            prompt="replacement",
        )
        is True
    )
    assert turn_state.queued_follow_up_prompt == "replacement"
    turn_state.busy = False
    assert (
        _MODULES.app._submit_streamlit_prompt(
            app_state=app_state,
            session_id=session_id,
            config=StreamlitAssistantConfig(),
            prompt="hello",
        )
        is True
    )
    assert started == ["hello"]

    runner = _FakeRunnerHandle()
    approval = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-1",
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {"path": "a.txt"}},
            tool_name="read_file",
            tool_version="0.1.0",
            policy_reason="approval required",
            policy_metadata={},
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:05:00Z",
        ),
        tool_name="read_file",
        redacted_arguments={"path": "a.txt"},
        policy_reason="approval required",
        policy_metadata={},
    )
    fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _MODULES.app.AssistantActiveTurnHandle(
            session_id=session_id,
            runner=runner,
            event_queue=queue.Queue(),
            thread=_DeadThread(),
            turn_number=1,
        )
    )
    turn_state.pending_approval = approval

    _MODULES.app._resolve_active_approval(
        app_state, session_id=session_id, approved=True
    )
    assert runner.approvals == [True]
    assert turn_state.approval_decision_in_flight is True


def test_streamlit_assistant_render_status_and_composer_shows_approval_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)

    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    turn_state = _MODULES.app._turn_state_for(app_state, session_id)
    turn_state.status_text = "approval required"
    turn_state.pending_approval = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-3",
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {"path": "a.txt"}},
            tool_name="read_file",
            tool_version="0.1.0",
            policy_reason="approval required",
            policy_metadata={},
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:05:00Z",
        ),
        tool_name="read_file",
        redacted_arguments={"path": "a.txt"},
        policy_reason="approval required",
        policy_metadata={},
    )

    _MODULES.app._render_status_and_composer(
        app_state,
        session_id=session_id,
        config=StreamlitAssistantConfig(),
    )

    assert "Allow tool" in fake_st.button_labels
    assert "Skip tool" in fake_st.button_labels
    assert any(
        "Approval is needed to continue." in item for item in fake_st.caption_messages
    )
    assert any(
        "Approval needed before using read_file. approval required." in item
        for item in fake_st.markdown_messages
    )


def test_streamlit_assistant_render_status_and_composer_shows_recent_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)

    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    turn_state = _MODULES.app._turn_state_for(app_state, session_id)
    turn_state.busy = True
    turn_state.status_text = "searching text"
    turn_state.status_history = [
        "thinking",
        "gathering evidence",
        "searching text",
    ]

    _MODULES.app._render_status_and_composer(
        app_state,
        session_id=session_id,
        config=StreamlitAssistantConfig(),
    )

    assert any(
        "Assistant is searching the workspace." in item
        for item in fake_st.caption_messages
    )
    assert any(
        "Recent steps: Assistant is drafting a response"
        " -> Assistant is gathering workspace evidence before answering"
        " -> Assistant is searching the workspace" in item
        for item in fake_st.caption_messages
    )


def test_streamlit_assistant_research_controls_explain_transition_and_seed_from_draft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)

    class _FakeController:
        def list_recent(self) -> object:
            return SimpleNamespace(sessions=[])

    monkeypatch.setattr(
        _MODULES.app, "_build_research_controller", lambda **kwargs: _FakeController()
    )
    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    app_state.drafts[session_id] = (
        "Investigate why approvals keep pausing this workflow"
    )
    active = app_state.sessions[session_id]

    _MODULES.app._render_sidebar_research_controls(
        app_state,
        config=StreamlitAssistantConfig(),
        runtime=active.runtime,
        active=active,
    )

    assert any(
        "Stay in chat for quick back-and-forth." in item
        for item in fake_st.caption_messages
    )
    assert any(
        "Your current chat draft is loaded below" in item
        for item in fake_st.caption_messages
    )
    assert fake_st.text_area_values["assistant-research-prompt"] == (
        "Investigate why approvals keep pausing this workflow"
    )


def test_streamlit_assistant_research_controls_resume_with_explicit_approval_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(button_values={"research-approve:research-1": True})
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    inspection = _fake_research_inspection(
        session_id="research-1",
        pending_approval_ids=["approval-1"],
    )

    class _FakeController:
        def __init__(self) -> None:
            self.resolutions: list[ApprovalResolution | None] = []

        def launch(self, *, prompt: str) -> object:
            raise AssertionError(prompt)

        def list_recent(self) -> object:
            return SimpleNamespace(
                sessions=[
                    SimpleNamespace(
                        summary=inspection.summary,
                        snapshot=inspection.snapshot,
                    )
                ]
            )

        def inspect(self, session_id: str) -> object:
            assert session_id == "research-1"
            return inspection

        def resume(
            self,
            session_id: str,
            *,
            approval_resolution: ApprovalResolution | None = None,
        ) -> object:
            assert session_id == "research-1"
            self.resolutions.append(approval_resolution)
            return inspection

        def stop(self, session_id: str) -> object:
            assert session_id == "research-1"
            return inspection

    controller = _FakeController()
    monkeypatch.setattr(
        _MODULES.app, "_build_research_controller", lambda **kwargs: controller
    )
    monkeypatch.setattr(
        _MODULES.app, "resume_session", lambda snapshot: inspection.resumed
    )
    monkeypatch.setattr(_MODULES.app, "_execution_blocker", lambda **kwargs: None)
    appended: list[object] = []
    monkeypatch.setattr(
        _MODULES.app,
        "_append_research_summary",
        lambda active, inspection, app_state: appended.append(inspection),
    )
    app_state = _make_app_state(root_path=str(tmp_path))
    active = app_state.sessions[app_state.active_session_id]

    _MODULES.app._render_sidebar_research_controls(
        app_state,
        config=StreamlitAssistantConfig(),
        runtime=active.runtime,
        active=active,
    )

    assert controller.resolutions == [ApprovalResolution.APPROVE]
    assert appended and appended[0].summary.session_id == "research-1"
    assert any("awaiting approval" in item for item in fake_st.caption_messages)


def test_streamlit_assistant_research_controls_show_states_and_select_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(button_values={"research-view:resumable-1": True})
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    running = _fake_research_inspection(
        session_id="running-1",
        pending_approval_ids=[],
        resumed_disposition=_MODULES.app.ResumeDisposition.RUNNABLE,
        total_turns=0,
        latest_decision_summary=None,
    )
    resumable = _fake_research_inspection(
        session_id="resumable-1",
        pending_approval_ids=[],
        resumed_disposition=_MODULES.app.ResumeDisposition.RUNNABLE,
        total_turns=2,
    )
    stopped = _fake_research_inspection(
        session_id="stopped-1",
        stop_reason="completed",
        pending_approval_ids=[],
        resumed_disposition=_MODULES.app.ResumeDisposition.TERMINAL,
    )

    class _FakeController:
        def launch(self, *, prompt: str) -> object:
            raise AssertionError(prompt)

        def list_recent(self) -> object:
            return SimpleNamespace(
                sessions=[
                    SimpleNamespace(summary=running.summary, snapshot=running.snapshot),
                    SimpleNamespace(
                        summary=resumable.summary, snapshot=resumable.snapshot
                    ),
                    SimpleNamespace(summary=stopped.summary, snapshot=stopped.snapshot),
                ]
            )

    monkeypatch.setattr(
        _MODULES.app,
        "_build_research_controller",
        lambda **kwargs: _FakeController(),
    )
    resumptions = {
        id(running.snapshot): running.resumed,
        id(resumable.snapshot): resumable.resumed,
        id(stopped.snapshot): stopped.resumed,
    }
    monkeypatch.setattr(
        _MODULES.app,
        "resume_session",
        lambda snapshot: resumptions[id(snapshot)],
    )
    app_state = _make_app_state(root_path=str(tmp_path))
    active = app_state.sessions[app_state.active_session_id]
    active.transcript.append(
        _MODULES.models.StreamlitTranscriptEntry(
            role="system",
            text="Research session: stopped-1\nStop reason: completed",
        )
    )

    _MODULES.app._render_sidebar_research_controls(
        app_state,
        config=StreamlitAssistantConfig(),
        runtime=active.runtime,
        active=active,
    )

    assert any("running | turns=0" in item for item in fake_st.caption_messages)
    assert any("resumable | turns=2" in item for item in fake_st.caption_messages)
    assert any("stopped | turns=3" in item for item in fake_st.caption_messages)
    assert any("summarized" in item for item in fake_st.caption_messages)
    assert (
        fake_st.session_state[_MODULES.app._SELECTED_RESEARCH_SESSION_SLOT]
        == "resumable-1"
    )


def test_streamlit_assistant_render_research_session_details_shows_replay_and_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state[_MODULES.app._SELECTED_RESEARCH_SESSION_SLOT] = "research-1"
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    inspection = _fake_research_inspection(session_id="research-1")

    class _FakeController:
        def inspect(self, session_id: str) -> object:
            assert session_id == "research-1"
            return inspection

    monkeypatch.setattr(
        _MODULES.app, "_build_research_controller", lambda **kwargs: _FakeController()
    )
    app_state = _make_app_state(root_path=str(tmp_path))
    active = app_state.sessions[app_state.active_session_id]

    _MODULES.app._render_research_session_details(
        app_state,
        config=StreamlitAssistantConfig(),
        runtime=active.runtime,
        active=active,
    )

    assert "Research session details" in fake_st.button_labels
    assert "Turn 1" in fake_st.button_labels
    assert "Raw inspection payload" in fake_st.button_labels
    assert any("**Replay**" in item for item in fake_st.markdown_messages)
    assert any("**Trace**" in item for item in fake_st.markdown_messages)
    assert any("Limitations:" in item for item in fake_st.caption_messages)
    assert any("approval=approval-1" in item for item in fake_st.caption_messages)


def test_streamlit_assistant_research_detail_does_not_render_purged_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sensitive_text = "TOP SECRET TOKEN"
    safe_message = "Safe replacement"
    fake_st = _FakeStreamlit()
    fake_st.session_state[_MODULES.app._SELECTED_RESEARCH_SESSION_SLOT] = (
        "research-purge"
    )
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)

    class _PurgingDriver:
        def select_task_ids(self, *, state):
            del state
            return ["task-1"]

        def build_context(self, *, state, selected_task_ids, turn_index):
            del state, selected_task_ids
            return ToolContext(
                invocation_id=f"turn-{turn_index}",
                metadata={
                    "protection_review": {
                        "purge_requested": True,
                        "safe_message": safe_message,
                    }
                },
            )

        def run_turn(self, *, state, selected_task_ids, context):
            del state, selected_task_ids, context
            return ParsedModelResponse(final_response=sensitive_text)

        async def run_turn_async(self, *, state, selected_task_ids, context):
            return self.run_turn(
                state=state,
                selected_task_ids=selected_task_ids,
                context=context,
            )

    store = InMemoryHarnessStateStore()
    _, workflow_executor = build_chat_executor()
    service = HarnessSessionService(
        store=store,
        workflow_executor=workflow_executor,
        driver=_PurgingDriver(),
        workspace=str(tmp_path),
    )
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Purged research",
            intent="Do protected work.",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="research-purge",
        )
    )
    service.run_session(HarnessSessionRunRequest(session_id=created.session_id))
    inspection = service.inspect_session(
        HarnessSessionInspectRequest(
            session_id=created.session_id,
            include_replay=True,
        )
    )

    class _FakeController:
        def inspect(self, session_id: str) -> object:
            assert session_id == "research-purge"
            return inspection

    monkeypatch.setattr(
        _MODULES.app, "_build_research_controller", lambda **kwargs: _FakeController()
    )
    app_state = _make_app_state(root_path=str(tmp_path))
    active = app_state.sessions[app_state.active_session_id]

    _MODULES.app._render_research_session_details(
        app_state,
        config=StreamlitAssistantConfig(),
        runtime=active.runtime,
        active=active,
    )

    rendered = "\n".join(
        [
            *fake_st.markdown_messages,
            *fake_st.caption_messages,
            *fake_st.warning_messages,
            *fake_st.error_messages,
        ]
    )
    assert sensitive_text not in rendered
    assert safe_message in rendered


def test_streamlit_assistant_research_detail_brokered_execution_preserves_provenance_and_purge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret_text = "TOP SECRET TOKEN"  # noqa: S105
    safe_message = "Safe replacement"
    (tmp_path / "secret.txt").write_text(secret_text, encoding="utf-8")
    protection_controller = _RecordingProtectionController(
        prompt_decision=PromptProtectionDecision(action=ProtectionAction.ALLOW),
        response_decision=ResponseProtectionDecision(
            action=ProtectionAction.SANITIZE,
            sanitized_payload=safe_message,
            should_purge=True,
        ),
    )

    class _ContinueUntilFinalResponseApplier:
        def apply_turn(self, *, state, turn: HarnessTurn):
            workflow_result = turn.workflow_result
            assert workflow_result is not None
            updated_state = state
            for task_id in turn.selected_task_ids:
                task = next(
                    task for task in updated_state.tasks if task.task_id == task_id
                )
                if task.status is TaskLifecycleStatus.PENDING:
                    updated_state = start_task(
                        updated_state,
                        task_id=task_id,
                        started_at=turn.started_at,
                    )

            if workflow_result.parsed_response.final_response is None:
                return updated_state, TurnDecision(
                    action=TurnDecisionAction.CONTINUE,
                    selected_task_ids=list(turn.selected_task_ids),
                    summary="Continue researching.",
                )

            for task_id in turn.selected_task_ids:
                task = next(
                    task for task in updated_state.tasks if task.task_id == task_id
                )
                if task.status is TaskLifecycleStatus.IN_PROGRESS:
                    updated_state = complete_task(
                        updated_state,
                        task_id=task_id,
                        finished_at=turn.started_at,
                    )
            return updated_state, TurnDecision(
                action=TurnDecisionAction.STOP,
                selected_task_ids=list(turn.selected_task_ids),
                stop_reason=HarnessStopReason.COMPLETED,
                summary="Research complete.",
            )

    store = InMemoryHarnessStateStore()
    _, workflow_executor = build_chat_executor()
    service = HarnessSessionService(
        store=store,
        workflow_executor=workflow_executor,
        provider=AssistantHarnessTurnProvider(
            provider=_FakeProvider(
                [
                    ParsedModelResponse(
                        invocations=[
                            {
                                "tool_name": "read_file",
                                "arguments": {"path": "secret.txt"},
                            }
                        ]
                    ),
                    ParsedModelResponse(final_response=secret_text),
                ]
            ),
            temperature=0.1,
            system_prompt="research-system",
            protection_controller=protection_controller,
        ),
        applier=_ContinueUntilFinalResponseApplier(),
        workspace=str(tmp_path),
    )
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Brokered purge",
            intent="Read and summarize protected research.",
            budget_policy=BudgetPolicy(max_turns=4),
            session_id="research-brokered-purge",
        )
    )
    service.run_session(HarnessSessionRunRequest(session_id=created.session_id))
    inspection = service.inspect_session(
        HarnessSessionInspectRequest(
            session_id=created.session_id,
            include_replay=True,
        )
    )

    assert (
        protection_controller.response_calls[0]["provenance"]
        .sources[0]
        .metadata["path"]
        == "secret.txt"
    )

    fake_st = _FakeStreamlit()
    fake_st.session_state[_MODULES.app._SELECTED_RESEARCH_SESSION_SLOT] = (
        created.session_id
    )
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)

    class _FakeController:
        def inspect(self, session_id: str) -> object:
            assert session_id == created.session_id
            return inspection

    monkeypatch.setattr(
        _MODULES.app, "_build_research_controller", lambda **kwargs: _FakeController()
    )
    app_state = _make_app_state(root_path=str(tmp_path))
    active = app_state.sessions[app_state.active_session_id]

    _MODULES.app._render_research_session_details(
        app_state,
        config=StreamlitAssistantConfig(),
        runtime=active.runtime,
        active=active,
    )

    rendered = "\n".join(
        [
            *fake_st.markdown_messages,
            *fake_st.caption_messages,
            *fake_st.warning_messages,
            *fake_st.error_messages,
        ]
    )
    assert inspection.summary.stop_reason is HarnessStopReason.COMPLETED
    assert safe_message in rendered
    assert secret_text not in rendered


def test_streamlit_assistant_research_detail_resolves_approval_and_appends_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(button_values={"research-detail-deny:research-1": True})
    fake_st.session_state[_MODULES.app._SELECTED_RESEARCH_SESSION_SLOT] = "research-1"
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    initial = _fake_research_inspection(session_id="research-1")
    updated = _fake_research_inspection(
        session_id="research-1",
        stop_reason="approval_denied",
        pending_approval_ids=[],
        resumed_disposition=_MODULES.app.ResumeDisposition.TERMINAL,
    )

    class _FakeController:
        def __init__(self) -> None:
            self.resolutions: list[ApprovalResolution | None] = []

        def inspect(self, session_id: str) -> object:
            assert session_id == "research-1"
            return initial

        def resume(
            self,
            session_id: str,
            *,
            approval_resolution: ApprovalResolution | None = None,
        ) -> object:
            assert session_id == "research-1"
            self.resolutions.append(approval_resolution)
            return updated

        def stop(self, session_id: str) -> object:
            raise AssertionError(session_id)

    controller = _FakeController()
    monkeypatch.setattr(
        _MODULES.app, "_build_research_controller", lambda **kwargs: controller
    )
    monkeypatch.setattr(_MODULES.app, "_execution_blocker", lambda **kwargs: None)
    appended: list[object] = []
    monkeypatch.setattr(
        _MODULES.app,
        "_append_research_summary",
        lambda active, inspection, app_state: appended.append(inspection),
    )
    app_state = _make_app_state(root_path=str(tmp_path))
    active = app_state.sessions[app_state.active_session_id]

    _MODULES.app._render_research_session_details(
        app_state,
        config=StreamlitAssistantConfig(),
        runtime=active.runtime,
        active=active,
    )

    assert controller.resolutions == [ApprovalResolution.DENY]
    assert appended and appended[0].summary.stop_reason.value == "approval_denied"


def test_streamlit_assistant_research_detail_shows_resume_for_resumable_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state[_MODULES.app._SELECTED_RESEARCH_SESSION_SLOT] = "research-2"
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    inspection = _fake_research_inspection(
        session_id="research-2",
        pending_approval_ids=[],
        resumed_disposition=_MODULES.app.ResumeDisposition.RUNNABLE,
    )

    class _FakeController:
        def inspect(self, session_id: str) -> object:
            assert session_id == "research-2"
            return inspection

    monkeypatch.setattr(
        _MODULES.app, "_build_research_controller", lambda **kwargs: _FakeController()
    )
    app_state = _make_app_state(root_path=str(tmp_path))
    active = app_state.sessions[app_state.active_session_id]

    _MODULES.app._render_research_session_details(
        app_state,
        config=StreamlitAssistantConfig(),
        runtime=active.runtime,
        active=active,
    )

    assert "Resume" in fake_st.button_labels
    assert "Approve" not in fake_st.button_labels
    assert "Deny" not in fake_st.button_labels


def test_streamlit_assistant_research_detail_warns_when_inspection_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state[_MODULES.app._SELECTED_RESEARCH_SESSION_SLOT] = "missing-1"
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)

    class _FakeController:
        def inspect(self, session_id: str) -> object:
            raise RuntimeError(f"missing session: {session_id}")

    monkeypatch.setattr(
        _MODULES.app, "_build_research_controller", lambda **kwargs: _FakeController()
    )
    app_state = _make_app_state(root_path=str(tmp_path))
    active = app_state.sessions[app_state.active_session_id]

    _MODULES.app._render_research_session_details(
        app_state,
        config=StreamlitAssistantConfig(),
        runtime=active.runtime,
        active=active,
    )

    assert any(
        "Research inspection failed for missing-1" in item
        for item in fake_st.warning_messages
    )


def test_streamlit_assistant_build_research_controller_filters_blocked_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(root_path=str(tmp_path))
    runtime.enabled_tools = ["read_file", "search_gitlab_code"]
    runtime.allow_network = False
    runtime.allow_filesystem = True
    captured: list[set[str]] = []

    monkeypatch.setattr(_MODULES.app, "_current_api_key", lambda llm_config: None)
    monkeypatch.setattr(
        _MODULES.app,
        "build_assistant_executor",
        lambda policy: ("registry", "workflow"),
    )
    monkeypatch.setattr(
        _MODULES.app,
        "build_live_harness_provider",
        lambda **kwargs: (
            captured.append(set(kwargs["enabled_tool_names"])) or "provider"
        ),
    )

    class _FakeHarnessService:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def list_sessions(self, request: object) -> object:
            del request
            return SimpleNamespace(sessions=[])

    monkeypatch.setattr(_MODULES.app, "HarnessSessionService", _FakeHarnessService)

    controller = _MODULES.app._build_research_controller(
        config=StreamlitAssistantConfig(
            research=AssistantResearchConfig(store_dir=str(tmp_path / "research"))
        ),
        runtime=runtime,
    )
    controller.list_recent()

    assert captured == [{"read_file"}]


def test_streamlit_assistant_launch_and_script_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str] = []
    fake_cli = SimpleNamespace(main=lambda: captured.extend(sys.argv) or 0)
    monkeypatch.setitem(sys.modules, "streamlit.web.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "streamlit.web", SimpleNamespace(cli=fake_cli))
    monkeypatch.setitem(
        sys.modules, "streamlit", SimpleNamespace(web=SimpleNamespace(cli=fake_cli))
    )

    assert _MODULES.app._launch_streamlit_app(["--config", "assistant.yaml"]) == 0
    assert captured[:3] == [
        "streamlit",
        "run",
        str(Path(_MODULES.app.__file__).resolve()),
    ]

    config_path = tmp_path / "assistant.yaml"
    config_path.write_text(
        """
llm:
  provider: ollama
""".lstrip(),
        encoding="utf-8",
    )
    called: list[tuple[Path | None, str]] = []
    monkeypatch.setattr(
        _MODULES.app,
        "run_streamlit_assistant_app",
        lambda *, root_path, config: called.append(
            (root_path, config.llm.provider.value)
        ),
    )
    _MODULES.app._run_streamlit_script([str(tmp_path), "--config", str(config_path)])
    assert called == [(tmp_path.resolve(), "ollama")]


def test_launch_research_task_appends_transition_note_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)
    inspection = _fake_research_inspection(
        session_id="research-42",
        pending_approval_ids=[],
        resumed_disposition=_MODULES.app.ResumeDisposition.RUNNABLE,
    )

    class _FakeController:
        def launch(self, *, prompt: str) -> object:
            assert prompt == "Investigate the failing integration path"
            return inspection

    monkeypatch.setattr(
        _MODULES.app,
        "_run_research_action",
        lambda session_id, *, action, failure_prefix: action(),
    )
    app_state = _make_app_state(root_path=str(tmp_path))
    active = app_state.sessions[app_state.active_session_id]

    _MODULES.app._launch_research_task(
        "Investigate the failing integration path",
        controller=_FakeController(),
        active=active,
        app_state=app_state,
    )

    assert active.transcript[-2].text == (
        "Started research task from this chat.\n"
        "Prompt: Investigate the failing integration path\n"
        "Research session: research-42"
    )
    assert active.transcript[-1].text.startswith("Research session: research-42")
    assert fake_st.session_state["assistant-research-prompt"] == ""


def test_assistant_research_controller_inspect_resume_stop_and_summary() -> None:
    calls: list[tuple[str, object]] = []

    class _FakeService:
        def inspect_session(self, request: object) -> object:
            calls.append(("inspect", request))
            return inspection

        def resume_session(self, request: object) -> object:
            calls.append(("resume", request))
            return None

        def stop_session(self, request: object) -> object:
            calls.append(("stop", request))
            return inspection

        def list_sessions(self, request: object) -> object:
            calls.append(("list", request))
            return SimpleNamespace(sessions=[])

    inspection = SimpleNamespace(
        summary=SimpleNamespace(
            session_id="research-1",
            stop_reason=SimpleNamespace(value="awaiting_approval"),
            total_turns=4,
            completed_task_ids=["task-1"],
            active_task_ids=[],
            pending_approval_ids=["approval-1"],
            latest_decision_summary="waiting",
        )
    )
    controller = AssistantResearchSessionController(
        service_factory=_FakeService,
        budget_policy=BudgetPolicy(max_turns=2),
        include_replay_by_default=True,
        list_limit=3,
    )

    assert controller.inspect("research-1") is inspection
    assert (
        controller.resume("research-1", approval_resolution=ApprovalResolution.DENY)
        is inspection
    )
    assert controller.stop("research-1") is inspection
    assert controller.list_recent().sessions == []
    summary = controller.summary_text(inspection)
    assert "Pending approvals: approval-1" in summary
    assert "Latest decision: waiting" in summary
    assert [name for name, _ in calls] == [
        "inspect",
        "resume",
        "inspect",
        "stop",
        "list",
    ]


def test_resolve_assistant_config_applies_cli_overrides_for_all_runtime_limits(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "assistant.yml"
    config_path.write_text(
        """
llm:
  provider: ollama
  model_name: base
""".lstrip(),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "--temperature",
            "0.3",
            "--api-base-url",
            "https://example.test/v1",
            "--max-context-tokens",
            "123",
            "--max-tool-round-trips",
            "4",
            "--max-tool-calls-per-round",
            "5",
            "--max-total-tool-calls-per-turn",
            "6",
            "--max-entries-per-call",
            "7",
            "--max-recursive-depth",
            "8",
            "--max-search-matches",
            "9",
            "--max-read-lines",
            "10",
            "--max-file-size-characters",
            "11",
            "--max-tool-result-chars",
            "12",
        ]
    )

    resolved = _resolve_assistant_config(args)

    assert resolved.llm.provider.value == "openai"
    assert resolved.llm.model_name == "gpt-test"
    assert resolved.llm.temperature == 0.3
    assert resolved.llm.api_base_url == "https://example.test/v1"
    assert resolved.session.max_context_tokens == 123
    assert resolved.session.max_tool_round_trips == 4
    assert resolved.session.max_tool_calls_per_round == 5
    assert resolved.session.max_total_tool_calls_per_turn == 6
    assert resolved.tool_limits.max_entries_per_call == 7
    assert resolved.tool_limits.max_recursive_depth == 8
    assert resolved.tool_limits.max_search_matches == 9
    assert resolved.tool_limits.max_read_lines == 10
    assert resolved.tool_limits.max_file_size_characters == 11
    assert resolved.tool_limits.max_tool_result_chars == 12


def test_streamlit_assistant_fallback_storage_provider_helpers_and_missing_session_notice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(_MODULES.app._STORAGE_ENV_VAR, raising=False)
    monkeypatch.setattr(_MODULES.app.Path, "home", lambda: tmp_path)
    assert (
        _MODULES.app._storage_root()
        == (tmp_path / ".llm-tools" / "assistant" / "streamlit").resolve()
    )

    runtime = _make_runtime(root_path=str(tmp_path))
    runtime.api_base_url = None
    preferences = _MODULES.models.StreamlitPreferences(theme_mode="dark")
    _MODULES.app._remember_runtime_preferences(preferences, runtime)
    assert runtime.provider.value in preferences.recent_models
    assert runtime.provider.value not in preferences.recent_base_urls

    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(
        _MODULES.app,
        "create_provider",
        lambda *args, **kwargs: recorded.append(dict(kwargs)) or "provider",
    )
    assert (
        _MODULES.app._create_provider_for_runtime(
            StreamlitAssistantConfig().llm,
            _make_runtime(root_path=None),
            api_key=None,
            model_name="demo-model",
        )
        == "provider"
    )
    assert recorded[-1]["model_name"] == "demo-model"
    assert "OPENAI_API_KEY" in _MODULES.app._missing_api_key_text(
        SimpleNamespace(api_key_env_var=None)
    )

    storage_root = tmp_path / "state"
    monkeypatch.setenv(_MODULES.app._STORAGE_ENV_VAR, str(storage_root))
    (storage_root / "sessions").mkdir(parents=True)
    (storage_root / "index.json").write_text(
        _MODULES.models.StreamlitSessionIndex(
            active_session_id="missing",
            session_order=["missing"],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    loaded = _MODULES.app._load_workspace_state(
        root_path=None,
        config=StreamlitAssistantConfig(),
    )
    assert any(
        "Skipped missing assistant session missing." in notice
        for notice in loaded.startup_notices
    )


def test_streamlit_assistant_load_save_delete_edge_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))

    storage_root = tmp_path / "state"
    storage_root.mkdir(parents=True, exist_ok=True)
    (storage_root / "preferences.json").write_text("{bad-json", encoding="utf-8")
    (storage_root / "index.json").write_text("{bad-json", encoding="utf-8")
    state = _MODULES.app._load_workspace_state(
        root_path=None,
        config=StreamlitAssistantConfig(),
    )
    assert any(
        "Unable to load preferences" in notice for notice in state.startup_notices
    )
    assert any(
        "Unable to load session index" in notice for notice in state.startup_notices
    )

    stale_path = _MODULES.app._session_path("stale")
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("{}", encoding="utf-8")
    state.session_order.append("ghost")
    _MODULES.app._save_workspace_state(state)
    assert not stale_path.exists()

    handle = _MODULES.app.AssistantActiveTurnHandle(
        session_id=state.active_session_id,
        runner=_FakeRunnerHandle(),
        event_queue=queue.Queue(),
        thread=_DeadThread(),
        turn_number=1,
    )
    fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] = handle
    _MODULES.app._delete_session(
        state,
        session_id=state.active_session_id,
        config=StreamlitAssistantConfig(),
        root_path=tmp_path,
    )
    assert handle.runner.cancelled is True
    assert fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] is None
    assert state.session_order

    monkeypatch.setattr(
        _MODULES.app,
        "_streamlit_module",
        lambda: (_ for _ in ()).throw(ModuleNotFoundError()),
    )
    single = _make_app_state(root_path=str(tmp_path))
    _MODULES.app._delete_session(
        single,
        session_id=single.active_session_id,
        config=StreamlitAssistantConfig(),
        root_path=tmp_path,
    )
    assert single.session_order
    assert single.active_session_id in single.sessions


def test_streamlit_assistant_start_turn_renames_legacy_default_title(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)
    fake_runner = _FakeIterableRunner([])
    monkeypatch.setattr(
        _MODULES.app, "_build_assistant_runner", lambda **kwargs: fake_runner
    )
    monkeypatch.setattr(_MODULES.app.threading, "Thread", _FakeThread)

    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    app_state.sessions[session_id].summary.title = "New assistant chat"

    _MODULES.app._start_streamlit_turn(
        app_state=app_state,
        session_id=session_id,
        config=StreamlitAssistantConfig(),
        provider="provider",
        user_message="legacy title prompt",
    )

    assert app_state.sessions[session_id].summary.title == "legacy title prompt"


def test_streamlit_assistant_start_cancel_and_resolve_guard_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)
    fake_runner = _FakeIterableRunner([])
    monkeypatch.setattr(
        _MODULES.app, "_build_assistant_runner", lambda **kwargs: fake_runner
    )
    monkeypatch.setattr(_MODULES.app.threading, "Thread", _FakeThread)

    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    _MODULES.app._start_streamlit_turn(
        app_state=app_state,
        session_id=session_id,
        config=StreamlitAssistantConfig(),
        provider="provider",
        user_message="hello world",
    )
    turn_state = _MODULES.app._turn_state_for(app_state, session_id)
    assert turn_state.busy is True
    assert app_state.sessions[session_id].transcript[-1].text == "hello world"
    assert (
        fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT].thread.started
        is True
    )
    assert _MODULES.app._active_session(app_state) is app_state.sessions[session_id]

    _MODULES.app._cancel_active_turn(app_state, session_id=session_id)
    assert fake_runner.cancelled is True

    fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] = None
    turn_state.queued_follow_up_prompt = "draft"
    _MODULES.app._cancel_active_turn(app_state, session_id=session_id)
    assert turn_state.busy is False
    assert (
        app_state.sessions[session_id].transcript[-1].text == "Stopped the active turn."
    )

    _MODULES.app._resolve_active_approval(
        app_state, session_id=session_id, approved=True
    )
    mismatch = _FakeRunnerHandle()
    fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _MODULES.app.AssistantActiveTurnHandle(
            session_id="other",
            runner=mismatch,
            event_queue=queue.Queue(),
            thread=_DeadThread(),
            turn_number=1,
        )
    )
    _MODULES.app._resolve_active_approval(
        app_state, session_id=session_id, approved=True
    )
    assert mismatch.approvals == []

    rejecting = _RejectingRunnerHandle()
    approval = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-2",
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {"path": "a.txt"}},
            tool_name="read_file",
            tool_version="0.1.0",
            policy_reason="approval required",
            policy_metadata={},
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:05:00Z",
        ),
        tool_name="read_file",
        redacted_arguments={"path": "a.txt"},
        policy_reason="approval required",
        policy_metadata={},
    )
    turn_state.pending_approval = approval
    fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _MODULES.app.AssistantActiveTurnHandle(
            session_id=session_id,
            runner=rejecting,
            event_queue=queue.Queue(),
            thread=_DeadThread(),
            turn_number=2,
        )
    )
    turn_state.approval_decision_in_flight = False
    _MODULES.app._resolve_active_approval(
        app_state, session_id=session_id, approved=False
    )
    assert rejecting.approvals == [False]
    assert turn_state.approval_decision_in_flight is False


def test_streamlit_assistant_process_turn_handles_approval_and_continuation() -> None:
    approval = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-1",
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {"path": "a.txt"}},
            tool_name="read_file",
            tool_version="0.1.0",
            policy_reason="approval required",
            policy_metadata={},
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:05:00Z",
        ),
        tool_name="read_file",
        redacted_arguments={"path": "a.txt"},
        policy_reason="approval required",
        policy_metadata={},
    )
    runner = _FakeIterableRunner(
        [
            ChatWorkflowApprovalEvent(approval=approval),
            ChatWorkflowApprovalResolvedEvent(approval=approval, resolution="approved"),
            ChatWorkflowResultEvent(
                result=ChatWorkflowTurnResult(
                    status="needs_continuation",
                    new_messages=[],
                    context_warning="warn",
                    continuation_reason="continue",
                    session_state=ChatSessionState(),
                )
            ),
        ]
    )
    original = _MODULES.app._build_assistant_runner
    _MODULES.app._build_assistant_runner = lambda **kwargs: runner
    try:
        outcome = process_streamlit_assistant_turn(
            root_path=None,
            config=StreamlitAssistantConfig(),
            provider=_FakeProvider([]),
            session_state=ChatSessionState(),
            user_message="hello",
            approval_resolver=lambda approval: True,
        )
    finally:
        _MODULES.app._build_assistant_runner = original

    assert runner.approvals == [True]
    assert [entry.text for entry in outcome.transcript_entries] == [
        "Approval needed before using read_file. approval required.",
        "Approved the requested tool run.",
        "warn",
        "continue",
    ]


def test_streamlit_assistant_apply_turn_error_wraps_provider_compatibility_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)
    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id

    _MODULES.app._apply_turn_error(
        app_state,
        session_id=session_id,
        error_message=(
            "All provider mode attempts failed. Overall failure type: schema/parse-related. "
            "Tried modes: tools: schema/parse-related (RuntimeError: schema validation failed)."
        ),
    )

    assert (
        app_state.sessions[session_id]
        .transcript[-1]
        .text.startswith("Provider compatibility error.")
    )
    assert "Tried modes: tools: schema/parse-related" in (
        app_state.sessions[session_id].transcript[-1].text
    )


def test_streamlit_assistant_apply_turn_result_and_queue_error_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)
    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id

    continuation_event = ChatWorkflowResultEvent(
        result=ChatWorkflowTurnResult(
            status="needs_continuation",
            new_messages=[],
            continuation_reason="continue next",
            session_state=ChatSessionState(),
        )
    )
    _MODULES.app._turn_state_for(
        app_state, session_id
    ).queued_follow_up_prompt = "next prompt"
    returned = _MODULES.app._apply_turn_result(
        app_state, session_id=session_id, event=continuation_event
    )
    assert returned == "next prompt"
    assert app_state.sessions[session_id].transcript[-1].text == "continue next"

    _MODULES.app._turn_state_for(
        app_state, session_id
    ).queued_follow_up_prompt = "retry prompt"
    queued_prompt = _MODULES.app._apply_queued_event(
        app_state,
        _MODULES.app.AssistantQueuedEvent(
            kind="error",
            payload="boom again",
            turn_number=2,
            session_id=session_id,
        ),
    )
    assert queued_prompt == "retry prompt"

    with pytest.raises(ValueError, match="Unsupported queued event kind"):
        _MODULES.app._apply_queued_event(
            app_state,
            _MODULES.app.AssistantQueuedEvent(
                kind="mystery",
                payload=object(),
                turn_number=1,
                session_id=session_id,
            ),
        )

    assert _MODULES.app._drain_active_turn_events(app_state) is None


def test_streamlit_assistant_drain_active_turn_events_recovers_orphaned_busy_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)
    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    turn_state = _MODULES.app._turn_state_for(app_state, session_id)
    turn_state.busy = True
    turn_state.status_text = "drafting answer"
    fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _MODULES.app.AssistantActiveTurnHandle(
            session_id=session_id,
            runner=_FakeRunnerHandle(),
            event_queue=queue.Queue(),
            thread=_DeadThread(),
            turn_number=1,
        )
    )

    pending = _MODULES.app._drain_active_turn_events(app_state)

    assert pending is None
    assert fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] is None
    assert turn_state.busy is False
    assert (
        app_state.sessions[session_id].transcript[-1].text
        == "Assistant turn ended before a final response was applied."
    )


def test_streamlit_assistant_drain_active_turn_events_surfaces_queue_apply_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)
    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    turn_state = _MODULES.app._turn_state_for(app_state, session_id)
    turn_state.busy = True
    event_queue: queue.Queue[_MODULES.app.AssistantQueuedEvent] = queue.Queue()
    event_queue.put(
        _MODULES.app.AssistantQueuedEvent(
            kind="mystery",
            payload=object(),
            turn_number=1,
            session_id=session_id,
        )
    )
    fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _MODULES.app.AssistantActiveTurnHandle(
            session_id=session_id,
            runner=_FakeRunnerHandle(),
            event_queue=event_queue,
            thread=_DeadThread(),
            turn_number=1,
        )
    )

    pending = _MODULES.app._drain_active_turn_events(app_state)

    assert pending is None
    assert fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] is None
    assert turn_state.busy is False
    assert "Failed to apply assistant turn event." in (
        app_state.sessions[session_id].transcript[-1].text
    )


def test_streamlit_assistant_drain_active_turn_events_accepts_legacy_handle_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_MODULES.app, "_save_workspace_state", lambda app_state: None)
    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    event_queue: queue.Queue[_MODULES.app.AssistantQueuedEvent] = queue.Queue()
    event_queue.put(
        _MODULES.app.AssistantQueuedEvent(
            kind="result",
            payload=ChatWorkflowResultEvent(
                result=ChatWorkflowTurnResult(
                    status="completed",
                    new_messages=[],
                    final_response=ChatFinalResponse(answer="done"),
                    session_state=ChatSessionState(),
                )
            ).model_dump(mode="json"),
            turn_number=1,
            session_id=session_id,
        )
    )
    event_queue.put(
        _MODULES.app.AssistantQueuedEvent(
            kind="complete",
            payload=None,
            turn_number=1,
            session_id=session_id,
        )
    )
    turn_state = _MODULES.app._turn_state_for(app_state, session_id)
    turn_state.busy = True
    fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] = SimpleNamespace(
        session_id=session_id,
        runner=_FakeRunnerHandle(),
        event_queue=event_queue,
        thread=_DeadThread(),
        turn_number=1,
    )

    pending = _MODULES.app._drain_active_turn_events(app_state)

    assert pending is None
    assert fake_st.session_state[_MODULES.app._ACTIVE_TURN_STATE_SLOT] is None
    assert app_state.sessions[session_id].transcript[-1].text == "done"
    assert turn_state.busy is False


def _example_config_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "assistant_configs" / name


def _final_response_payload(answer: str) -> dict[str, object]:
    return {
        "answer": answer,
        "citations": [],
        "confidence": 0.8,
        "uncertainty": [],
        "missing_information": [],
        "follow_up_suggestions": [],
    }


def _drain_turn_until_idle(
    app_state: object,
    *,
    fake_st: _FakeStreamlit,
    max_attempts: int = 100,
) -> None:
    for _ in range(max_attempts):
        _MODULES.app._drain_active_turn_events(app_state)
        handle = fake_st.session_state.get(_MODULES.app._ACTIVE_TURN_STATE_SLOT)
        if handle is None:
            return
        time.sleep(0.01)
    raise AssertionError("assistant turn did not become idle")


@pytest.mark.filterwarnings(
    "ignore:coroutine 'BaseSubprocessTransport.__del__':RuntimeWarning"
)
def test_streamlit_assistant_product_journey_persists_direct_chat_and_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_streamlit_assistant_config(
        _example_config_path("local-only-chat.yaml")
    )
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "assistant-state"))

    app_state = _MODULES.app._load_workspace_state(root_path=None, config=config)
    session_id = app_state.active_session_id
    _MODULES.app._render_empty_state(app_state.sessions[session_id])

    monkeypatch.setattr(
        _MODULES.app,
        "_create_provider_for_runtime",
        lambda *args, **kwargs: _FakeProvider(
            [
                ParsedModelResponse(
                    final_response=_final_response_payload("Direct answer")
                )
            ]
        ),
    )

    assert any(
        "Start with a normal question" in item for item in fake_st.markdown_messages
    )
    assert (
        _MODULES.app._submit_streamlit_prompt(
            app_state=app_state,
            session_id=session_id,
            config=config,
            prompt="How does this assistant behave with chat only?",
        )
        is True
    )
    _drain_turn_until_idle(app_state, fake_st=fake_st)

    record = app_state.sessions[session_id]
    assert [entry.role for entry in record.transcript] == ["user", "assistant"]
    assert record.transcript[-1].final_response is not None
    assert record.transcript[-1].final_response.answer == "Direct answer"
    assert _MODULES.app._session_path(session_id).exists()

    reloaded = _MODULES.app._load_workspace_state(root_path=None, config=config)
    reloaded_record = reloaded.sessions[session_id]
    assert reloaded.active_session_id == session_id
    assert reloaded_record.transcript[-1].final_response is not None
    assert reloaded_record.transcript[-1].final_response.answer == "Direct answer"


@pytest.mark.filterwarnings(
    "ignore:coroutine 'BaseSubprocessTransport.__del__':RuntimeWarning"
)
def test_streamlit_assistant_product_journey_supports_workspace_tool_turn_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_streamlit_assistant_config(
        _example_config_path("local-only-chat.yaml")
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "plan.txt").write_text(
        "Ship the evaluation workflow.", encoding="utf-8"
    )
    fake_st = _FakeStreamlit(
        text_input_values={"Workspace root": str(workspace)},
        checkbox_values={"Filesystem access": True},
    )
    monkeypatch.setattr(_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "assistant-state"))

    app_state = _MODULES.app._load_workspace_state(root_path=None, config=config)
    session_id = app_state.active_session_id
    _MODULES.app._render_sidebar(app_state, config=config, root_path=None)

    record = app_state.sessions[session_id]
    assert record.runtime.root_path == str(workspace.resolve())
    assert record.runtime.allow_filesystem is True

    monkeypatch.setattr(
        _MODULES.app,
        "_create_provider_for_runtime",
        lambda *args, **kwargs: _FakeProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        ToolInvocationRequest(
                            tool_name="read_file",
                            arguments={"path": "plan.txt"},
                        )
                    ]
                ),
                ParsedModelResponse(
                    final_response=_final_response_payload("Read the workspace plan.")
                ),
            ]
        ),
    )

    assert (
        _MODULES.app._submit_streamlit_prompt(
            app_state=app_state,
            session_id=session_id,
            config=config,
            prompt="Read plan.txt and summarize it.",
        )
        is True
    )
    _drain_turn_until_idle(app_state, fake_st=fake_st)

    assert record.transcript[-1].final_response is not None
    assert record.transcript[-1].final_response.answer == "Read the workspace plan."

    reloaded = _MODULES.app._load_workspace_state(root_path=None, config=config)
    reloaded_record = reloaded.sessions[session_id]
    assert reloaded_record.runtime.root_path == str(workspace.resolve())
    assert reloaded_record.runtime.allow_filesystem is True
    assert reloaded_record.transcript[-1].final_response is not None
    assert reloaded_record.transcript[-1].final_response.answer == (
        "Read the workspace plan."
    )


@pytest.mark.filterwarnings(
    "ignore:coroutine 'BaseSubprocessTransport.__del__':RuntimeWarning"
)
def test_streamlit_assistant_product_journey_supports_research_approval_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_streamlit_assistant_config(
        _example_config_path("harness-research-chat.yaml")
    )
    config = config.model_copy(
        update={
            "research": config.research.model_copy(
                update={"store_dir": str(tmp_path / "research-store")}
            )
        }
    )
    (tmp_path / "research-note.txt").write_text(
        "Research details for the harness flow.",
        encoding="utf-8",
    )
    runtime = _MODULES.models.StreamlitRuntimeConfig(
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        api_base_url=config.llm.api_base_url,
        root_path=str(tmp_path),
        enabled_tools=["read_file"],
        allow_filesystem=True,
        require_approval_for={SideEffectClass.LOCAL_READ},
    )
    provider = ScriptedParsedResponseProvider(
        [
            ParsedModelResponse(
                invocations=[
                    ToolInvocationRequest(
                        tool_name="read_file",
                        arguments={"path": "research-note.txt"},
                    )
                ]
            ),
            ParsedModelResponse(final_response="Research summary complete"),
        ]
    )

    monkeypatch.setattr(_MODULES.app, "_current_api_key", lambda llm_config: None)
    monkeypatch.setattr(
        _MODULES.app,
        "build_live_harness_provider",
        lambda **kwargs: provider,
    )

    controller = _MODULES.app._build_research_controller(
        config=config,
        runtime=runtime,
    )
    waiting = controller.launch(prompt="Read research-note.txt then summarize it.")

    assert waiting.resumed.disposition is ResumeDisposition.WAITING_FOR_APPROVAL
    assert waiting.summary.pending_approval_ids

    approved = controller.resume(
        waiting.snapshot.session_id,
        approval_resolution=ApprovalResolution.APPROVE,
    )

    assert approved.summary.stop_reason is HarnessStopReason.COMPLETED
    assert approved.summary.pending_approval_ids == []
    assert approved.snapshot.state.session.stop_reason is HarnessStopReason.COMPLETED


def test_research_controller_approval_resume_write_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = StreamlitAssistantConfig().model_copy(
        update={"research": AssistantResearchConfig(include_replay_by_default=True)}
    )
    runtime = _MODULES.models.StreamlitRuntimeConfig(
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        api_base_url=config.llm.api_base_url,
        root_path=str(tmp_path / "approve-workspace"),
        default_workspace_root=str(tmp_path / "approve-workspace"),
        enabled_tools=["list_directory", "write_file"],
        allow_filesystem=True,
        allow_subprocess=False,
        require_approval_for={SideEffectClass.LOCAL_WRITE},
    )
    deny_workspace = tmp_path / "deny-workspace"
    approve_workspace = tmp_path / "approve-workspace"

    def _run_flow(
        *,
        workspace: Path,
        approval_resolution: ApprovalResolution,
    ) -> object:
        workspace.mkdir(parents=True, exist_ok=True)
        provider = ScriptedParsedResponseProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        ToolInvocationRequest(
                            tool_name="write_file",
                            arguments={
                                "path": "notes/approved.txt",
                                "content": "approved research output\n",
                                "create_parents": True,
                            },
                        ),
                    ]
                ),
                ParsedModelResponse(
                    final_response="Wrote the approved research note successfully."
                ),
            ]
        )
        monkeypatch.setattr(_MODULES.app, "_current_api_key", lambda llm_config: None)
        monkeypatch.setattr(
            _MODULES.app,
            "build_live_harness_provider",
            lambda **kwargs: provider,
        )
        controller = _MODULES.app._build_research_controller(
            config=config,
            runtime=runtime.model_copy(
                update={
                    "root_path": str(workspace),
                    "default_workspace_root": str(workspace),
                }
            ),
        )
        waiting = controller.launch(
            prompt=(
                "List the workspace, then write an approval-gated note and confirm "
                "completion."
            )
        )
        resumed = controller.resume(
            waiting.snapshot.session_id,
            approval_resolution=approval_resolution,
        )
        return waiting, resumed

    waiting_deny, denied = _run_flow(
        workspace=deny_workspace,
        approval_resolution=ApprovalResolution.DENY,
    )
    waiting_approve, approved = _run_flow(
        workspace=approve_workspace,
        approval_resolution=ApprovalResolution.APPROVE,
    )

    assert waiting_deny.summary.pending_approval_ids
    assert waiting_deny.resumed.disposition is ResumeDisposition.WAITING_FOR_APPROVAL
    assert not (deny_workspace / "notes" / "approved.txt").exists()
    assert denied.summary.stop_reason is HarnessStopReason.APPROVAL_DENIED
    assert denied.summary.pending_approval_ids == []

    approved_path = approve_workspace / "notes" / "approved.txt"
    assert waiting_approve.summary.pending_approval_ids
    assert approved.summary.stop_reason is HarnessStopReason.COMPLETED
    assert approved.summary.pending_approval_ids == []
    assert approved_path.read_text(encoding="utf-8") == "approved research output\n"
    trace = approved.snapshot.artifacts.trace
    assert trace is not None
    invocation_statuses = [
        invocation.status
        for turn in trace.turns
        for invocation in turn.invocation_traces
    ]
    tool_names = [
        invocation.tool_name
        for turn in trace.turns
        for invocation in turn.invocation_traces
    ]
    assert invocation_statuses == [
        WorkflowInvocationStatus.APPROVAL_REQUESTED,
        WorkflowInvocationStatus.EXECUTED,
    ]
    assert tool_names == ["write_file", "write_file"]
    assert approved.replay is not None
    assert approved.replay.steps[0].workflow_outcome_statuses == [
        WorkflowInvocationStatus.APPROVAL_REQUESTED,
        WorkflowInvocationStatus.EXECUTED,
    ]

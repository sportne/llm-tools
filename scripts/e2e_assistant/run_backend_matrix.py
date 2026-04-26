"""Run temporary backend E2E probes for the NiceGUI assistant."""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from collections import Counter
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from uuid import uuid4

import common

import llm_tools.apps.nicegui_chat.controller as nicegui_controller_module
from llm_tools.apps.nicegui_chat.controller import NiceGUIChatController
from llm_tools.apps.nicegui_chat.models import NiceGUITranscriptEntry
from llm_tools.apps.nicegui_chat.store import SQLiteNiceGUIChatStore
from llm_tools.apps.protection_runtime import build_protection_environment
from llm_tools.harness_api import (
    ApprovalResolution,
    HarnessSessionInspection,
    HarnessStopReason,
    ResumeDisposition,
    ScriptedParsedResponseProvider,
    resume_session,
)
from llm_tools.harness_api.replay import replay_session
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.llm_providers import ProviderModeStrategy
from llm_tools.tool_api import SideEffectClass
from llm_tools.workflow_api import (
    ChatSessionState,
    DefaultEnvironmentComparator,
    ProtectionAction,
    ProtectionAssessment,
    ProtectionConfig,
    ProtectionController,
    ProtectionFeedbackStore,
    load_protection_corpus,
)

ALL_SCENARIOS = [
    "chat_repo_lookup",
    "chat_multi_turn_back_and_forth",
    "chat_git_inspection",
    "chat_remote_request_blocked",
    "chat_proprietary_protection_demo",
    "research_approval_resume_write_flow",
    "research_repo_investigation",
    "research_followup_or_inspect",
]
PROMPT_TOOL_VARIANT_MODES = {
    "prompt_tools_split": "split",
    "prompt_tools_single_action": "single_action",
    "prompt_tools_category": "category",
}
RESEARCH_SESSION_TO_PROBE_SESSION: dict[str, str] = {}
CONTINUATION_REFUSAL_PROMPT = (
    "Do not continue with more tool use. Use only the local tool evidence already "
    "available in this chat session and answer now. Mention uncertainty if the "
    "evidence is incomplete."
)

SAFE_PROTECTION_REPLACEMENT = (
    "I can't provide the proprietary material directly. I can offer a safe summary "
    "without restricted details."
)


class _UnexpectedProvider:
    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("provider should not be called for this turn")


class _FakeProvider:
    def __init__(self, responses: list[ParsedModelResponse]) -> None:
        self._responses = list(responses)

    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        if not self._responses:
            raise AssertionError("provider ran more times than expected")
        return self._responses.pop(0)


class _DeterministicProtectionClassifier:
    def __init__(self, *, document_id: str) -> None:
        self._document_id = document_id

    def assess_prompt(self, **kwargs: object) -> ProtectionAssessment:
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
            confidence=0.88,
        )

    def assess_response(self, **kwargs: object) -> ProtectionAssessment:
        response_payload = kwargs["response_payload"]
        response_text = str(response_payload).lower()
        if (
            "top secret token" in response_text
            or "proprietary playbook" in response_text
        ):
            return ProtectionAssessment(
                sensitivity_label="restricted",
                reasoning="Candidate answer contains restricted proprietary material.",
                confidence=0.99,
                referenced_document_ids=[self._document_id],
                recommended_action=ProtectionAction.SANITIZE,
                sanitized_text=SAFE_PROTECTION_REPLACEMENT,
            )
        return ProtectionAssessment(
            sensitivity_label="public",
            reasoning="Candidate answer is safe.",
            confidence=0.9,
        )


def _run_with_timeout(*, timeout_seconds: float, fn: Any) -> Any:
    payload_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            payload_queue.put(("result", fn()))
        except Exception as exc:  # pragma: no cover - worker failure path
            payload_queue.put(("error", exc))

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"Scenario exceeded {timeout_seconds:.1f}s.")
    kind, payload = payload_queue.get_nowait()
    if kind == "error":
        raise payload
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the backend probe CLI."""
    parser = argparse.ArgumentParser(
        description="Temporary backend E2E probes for the NiceGUI assistant."
    )
    parser.add_argument("--workspace", type=Path, default=common.REPO_ROOT)
    parser.add_argument(
        "--provider",
        default="ollama",
        help="Assistant provider preset, such as ollama or custom_openai_compatible.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=common.DEFAULT_OLLAMA_BASE_URL,
    )
    parser.add_argument(
        "--api-base-url",
        help="OpenAI-compatible API base URL. Overrides --ollama-base-url.",
    )
    parser.add_argument(
        "--api-key-env-var",
        help="Environment variable used by custom OpenAI-compatible providers.",
    )
    parser.add_argument("--model", default=common.DEFAULT_MODEL)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Run one named scenario. May be repeated.",
    )
    parser.add_argument(
        "--scenarios",
        help="Comma-separated scenario names.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=common.DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--provider-mode",
        action="append",
        default=[],
        help="Run one provider mode. May be repeated.",
    )
    parser.add_argument(
        "--provider-modes",
        help=(
            "Comma-separated provider modes: tools,json,prompt_tools,"
            "prompt_tools_split,prompt_tools_single_action,prompt_tools_category. "
            "The prompt_tools mode uses the default single-action prompt protocol."
        ),
    )
    return parser


def _select_provider_mode_variants(
    *,
    mode_args: list[str],
    modes_csv: str | None,
) -> list[tuple[str, ProviderModeStrategy, str | None]]:
    raw_values: list[str] = []
    for raw in mode_args:
        value = raw.strip()
        if value:
            raw_values.append(value)
    if modes_csv:
        raw_values.extend(part.strip() for part in modes_csv.split(",") if part.strip())
    if not raw_values:
        raw_values = [mode.value for mode in common.DEFAULT_PROVIDER_MODES]

    selected: list[tuple[str, ProviderModeStrategy, str | None]] = []
    seen: set[str] = set()
    valid_values = [
        *(mode.value for mode in common.DEFAULT_PROVIDER_MODES),
        *PROMPT_TOOL_VARIANT_MODES,
    ]
    for raw in raw_values:
        if raw in PROMPT_TOOL_VARIANT_MODES:
            label = raw
            mode = ProviderModeStrategy.PROMPT_TOOLS
            strategy = PROMPT_TOOL_VARIANT_MODES[raw]
        else:
            try:
                mode = ProviderModeStrategy(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown provider mode '{raw}'. Expected one of: {', '.join(valid_values)}"
                ) from exc
            label = mode.value
            strategy = None
        if label in seen:
            continue
        selected.append((label, mode, strategy))
        seen.add(label)
    return selected


def _chat_prompt_for(name: str) -> str:
    prompts = {
        "chat_repo_lookup": (
            "Explain how the NiceGUI assistant supports durable research sessions. "
            "You must use one or more local workspace tools before answering, cite the "
            "most relevant local files, and keep the answer focused on app/runtime flow."
        ),
        "chat_multi_turn_back_and_forth": (
            "Use local workspace tools to find the two most relevant files for tracing "
            "normal chat and durable research in the NiceGUI assistant. Keep the "
            "answer short."
        ),
        "chat_git_inspection": (
            "Summarize the current workspace git status and recent commit context for "
            "this repository. Mention any obvious uncommitted work that stands out."
        ),
        "chat_remote_request_blocked": (
            "Find the latest Jira tickets, Confluence notes, and GitLab merge requests "
            "related to the NiceGUI assistant. If you cannot access those systems, "
            "say that clearly instead of guessing."
        ),
    }
    return prompts[name]


def _build_probe_controller(
    *,
    session_id: str,
    config: Any,
    runtime: Any,
    provider: Any | None,
) -> NiceGUIChatController:
    """Build a NiceGUI controller with isolated encrypted probe persistence."""
    store_dir = Path(gettempdir()) / "llm-tools-e2e-nicegui" / session_id
    store_dir.mkdir(parents=True, exist_ok=True)
    store = SQLiteNiceGUIChatStore(
        store_dir / "chat.sqlite3",
        db_key_file=store_dir / "db.key",
        user_key_file=store_dir / "user-kek.key",
    )
    store.initialize()

    def _factory(active_runtime: Any) -> Any:
        if provider is not None:
            return provider
        return common.create_provider_for_runtime(
            config,
            active_runtime,
            api_key=None,
            model_name=active_runtime.model_name,
        )

    controller = NiceGUIChatController(
        store=store,
        config=config,
        root_path=Path(runtime.root_path) if runtime.root_path else None,
        provider_factory=_factory,
    )
    record = controller.active_record
    record.runtime = runtime.model_copy(deep=True)
    controller.save_active_session()
    return controller


def _drain_controller(
    controller: NiceGUIChatController,
    *,
    timeout_seconds: float,
    auto_resolve_approvals: bool = True,
) -> None:
    """Drain a NiceGUI controller until its background turn is idle."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        controller.drain_events()
        if (
            auto_resolve_approvals
            and controller.active_turn_state.pending_approval is not None
        ):
            controller.resolve_approval(approved=False)
        if not controller.active_turns:
            controller.drain_events()
            return
        time.sleep(0.01)
    raise TimeoutError(
        f"NiceGUI controller did not finish within {timeout_seconds:.1f}s."
    )


def _serialized_transcript(
    entries: list[NiceGUITranscriptEntry],
) -> list[dict[str, Any]]:
    return [entry.model_dump(mode="json") for entry in entries]


def _tool_sequence_from_workbench(controller: NiceGUIChatController) -> list[str]:
    executions = [
        item.payload
        for item in controller.active_record.workbench_items
        if item.title.endswith("Tool To LLM") and isinstance(item.payload, dict)
    ]
    return common.tool_sequence_from_inspector(executions)


def _run_chat_turn(
    *,
    scenario_name: str,
    session_id: str,
    prompt: str,
    config: Any,
    runtime: Any,
    session_state: ChatSessionState,
    provider: Any | None = None,
    protection_controller_factory: Any | None = None,
) -> dict[str, Any]:
    turn_provider = provider
    if turn_provider is None:
        turn_provider = common.create_provider_for_runtime(config, runtime)
    original_build_protection_controller = (
        nicegui_controller_module.build_protection_controller
    )
    if protection_controller_factory is not None:
        nicegui_controller_module.build_protection_controller = (
            protection_controller_factory
        )
    result_payload: dict[str, Any] | None = None
    final_response_payload: dict[str, Any] | None = None
    token_usage_payload: dict[str, Any] | None = None
    session_state_payload: dict[str, Any] | None = None
    failure_payload: dict[str, Any] | None = None
    controller = _build_probe_controller(
        session_id=session_id,
        config=config,
        runtime=runtime,
        provider=turn_provider,
    )
    controller.active_record.workflow_session_state = session_state
    controller.save_active_session()

    try:
        error = controller.submit_prompt(prompt)
        if error is not None:
            raise RuntimeError(error)
        _drain_controller(controller, timeout_seconds=max(runtime.timeout_seconds, 5.0))
        record = controller.active_record
        session_state_payload = common.dump_model(record.workflow_session_state)
        token_usage_payload = common.dump_model(record.token_usage)
        last_turn = (
            record.workflow_session_state.turns[-1]
            if record.workflow_session_state.turns
            else None
        )
        if last_turn is not None:
            result_payload = common.dump_model(last_turn)
            final_response_payload = common.dump_model(last_turn.final_response)
    except Exception as exc:
        failure_payload = common.failure_payload(exc)
        failure_payload["stage"] = controller.active_turn_state.status_text or None
        controller.active_record.transcript.append(
            NiceGUITranscriptEntry(
                role="system",
                text=(
                    "Scenario execution raised an exception after partial chat "
                    f"artifacts were captured: {type(exc).__name__}: {exc}"
                ),
            )
        )
    finally:
        nicegui_controller_module.build_protection_controller = (
            original_build_protection_controller
        )

    record = controller.active_record
    provider_responses = [
        item.payload
        for item in record.workbench_items
        if item.title.endswith("From LLM") and isinstance(item.payload, dict)
    ]
    payload = {
        "prompt": prompt,
        "status_events": list(controller.active_turn_state.status_history),
        "approvals": [],
        "transcript_entries": _serialized_transcript(record.transcript),
        "inspector": {
            "provider_messages": [
                common.dump_model(entry.payload)
                for entry in record.inspector_state.provider_messages
            ],
            "provider_responses": [
                common.dump_model(payload) for payload in provider_responses
            ],
            "parsed_responses": [
                common.dump_model(entry.payload)
                for entry in record.inspector_state.parsed_responses
            ],
            "tool_executions": [
                common.dump_model(entry.payload)
                for entry in record.inspector_state.tool_executions
            ],
        },
        "tool_sequence": _tool_sequence_from_workbench(controller),
        "result": result_payload,
        "final_response": final_response_payload,
        "token_usage": token_usage_payload,
        "session_state": session_state_payload,
    }
    if failure_payload is not None:
        payload["failure"] = failure_payload
    return payload


def _serialize_chat_turn(
    *,
    scenario_name: str,
    prompt: str,
    config: Any,
    runtime: Any,
) -> dict[str, Any]:
    return _run_chat_turn(
        scenario_name=scenario_name,
        session_id=f"e2e-{scenario_name}-{uuid4().hex[:8]}",
        prompt=prompt,
        config=config,
        runtime=runtime,
        session_state=ChatSessionState(),
    )


def _run_chat_turn_with_continuation_refusal(
    *,
    scenario_name: str,
    prompt: str,
    config: Any,
    runtime: Any,
) -> dict[str, Any]:
    session_id = f"e2e-{scenario_name}-{uuid4().hex[:8]}"
    first_turn = _run_chat_turn(
        scenario_name=scenario_name,
        session_id=session_id,
        prompt=prompt,
        config=config,
        runtime=runtime,
        session_state=ChatSessionState(),
    )
    result_payload = first_turn.get("result") or {}
    if result_payload.get("status") != "needs_continuation":
        return first_turn
    session_state_payload = first_turn.get("session_state")
    if session_state_payload is None:
        return first_turn

    second_turn = _run_chat_turn(
        scenario_name=f"{scenario_name}-continuation-refusal",
        session_id=session_id,
        prompt=CONTINUATION_REFUSAL_PROMPT,
        config=config,
        runtime=runtime,
        session_state=ChatSessionState.model_validate(session_state_payload),
    )
    combined = dict(second_turn)
    combined["initial_turn"] = first_turn
    combined["continuation_refusal_prompt"] = CONTINUATION_REFUSAL_PROMPT
    combined["tool_sequence"] = [
        *first_turn.get("tool_sequence", []),
        *second_turn.get("tool_sequence", []),
    ]
    combined["transcript_entries"] = [
        *first_turn.get("transcript_entries", []),
        *second_turn.get("transcript_entries", []),
    ]
    combined["continuation_refusal_attempted"] = True
    return combined


def _evaluate_chat_repo_lookup(
    chat_turn: dict[str, Any],
) -> tuple[dict[str, bool], str]:
    checks = {
        "final_response_present": chat_turn["final_response"] is not None,
        "local_read_tool_used": any(
            tool_name in {"read_file", "search_text", "find_files", "list_directory"}
            for tool_name in chat_turn["tool_sequence"]
        ),
        "provider_messages_captured": bool(chat_turn["inspector"]["provider_messages"]),
        "parsed_responses_captured": bool(chat_turn["inspector"]["parsed_responses"]),
    }
    summary = (
        "Captured a final response plus local workspace reads."
        if all(checks.values())
        else "Repo lookup did not produce the expected local tool-grounded trace."
    )
    return checks, summary


def _evaluate_chat_git_inspection(
    chat_turn: dict[str, Any],
) -> tuple[dict[str, bool], str]:
    checks = {
        "final_response_present": chat_turn["final_response"] is not None,
        "git_tool_used": any(
            tool_name in {"run_git_status", "run_git_diff", "run_git_log"}
            for tool_name in chat_turn["tool_sequence"]
        ),
        "local_only_tools": common.only_local_tools(chat_turn["tool_sequence"]),
    }
    summary = (
        "Captured a git-grounded response with local-only tool use."
        if all(checks.values())
        else "Git inspection did not stay inside the expected local git path."
    )
    return checks, summary


def _evaluate_chat_multi_turn(
    scenario_result: dict[str, Any],
) -> tuple[dict[str, bool], str]:
    turns = scenario_result["turns"]
    final_turn = turns[-1]
    final_session_state = final_turn.get("session_state") or {}
    turn_records = final_session_state.get("turns") or []
    all_tool_names = [
        tool_name for turn in turns for tool_name in turn.get("tool_sequence", [])
    ]
    checks = {
        "all_turns_have_final_response": all(
            turn.get("final_response") is not None for turn in turns
        ),
        "session_retains_multiple_turns": len(turn_records) == len(turns),
        "at_least_one_local_tool_used": any(
            tool_name in {"read_file", "search_text", "find_files", "list_directory"}
            for tool_name in all_tool_names
        ),
        "later_turn_completed": bool(final_turn.get("final_response")),
        "inspector_captured_each_turn": all(
            turn["inspector"]["provider_messages"]
            and turn["inspector"]["parsed_responses"]
            for turn in turns
        ),
    }
    summary = (
        "Completed a multi-turn back-and-forth with retained session state."
        if all(checks.values())
        else "Multi-turn chat did not retain or complete the expected back-and-forth."
    )
    return checks, summary


def _run_chat_multi_turn_scenario(
    *,
    provider_mode: ProviderModeStrategy,
    provider_health: dict[str, Any],
    config: Any,
    runtime: Any,
) -> dict[str, Any]:
    prompts = [
        (
            "Use local workspace tools to find the two most relevant files for tracing "
            "normal chat and durable research in the NiceGUI assistant. Keep the "
            "answer short."
        ),
        (
            "Follow up on your last answer. Tell me the first thing I should look for "
            "in each of those files. Keep the answer brief."
        ),
    ]
    result: dict[str, Any] = {
        "name": "chat_multi_turn_back_and_forth",
        "kind": "chat",
        "provider_mode": provider_mode.value,
        "provider_health": provider_health,
        "prompt": prompts[0],
        "prompts": prompts,
    }
    if not provider_health.get("ok", False):
        result.update(
            {
                "status": common.SCENARIO_STATUS_INFRA,
                "summary": (
                    "Provider preflight failed, so live chat behavior was not executed."
                ),
            }
        )
        return result

    session_id = f"e2e-chat-multi-turn-{uuid4().hex[:8]}"
    session_state = ChatSessionState()
    turns: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        turn = _run_chat_turn(
            scenario_name=f"chat_multi_turn_back_and_forth-turn-{index}",
            session_id=session_id,
            prompt=prompt,
            config=config,
            runtime=runtime,
            session_state=session_state,
        )
        turns.append(turn)
        session_state_payload = turn.get("session_state")
        if session_state_payload is not None:
            session_state = ChatSessionState.model_validate(session_state_payload)

    result["turns"] = turns
    result["tool_sequence"] = [
        tool_name for turn in turns for tool_name in turn.get("tool_sequence", [])
    ]
    result["transcript_entries"] = [
        entry for turn in turns for entry in turn.get("transcript_entries", [])
    ]
    result["final_response"] = turns[-1].get("final_response")
    result["session_state"] = turns[-1].get("session_state")
    checks, summary = _evaluate_chat_multi_turn(result)
    result["checks"] = checks
    result["summary"] = summary
    result["status"] = (
        common.SCENARIO_STATUS_PASSED
        if all(checks.values())
        else common.SCENARIO_STATUS_FAILED
    )
    return result


def _evaluate_chat_remote_blocked(
    chat_turn: dict[str, Any],
) -> tuple[dict[str, bool], str]:
    answer_text = None
    if chat_turn["final_response"] is not None:
        answer_text = chat_turn["final_response"].get("answer")
    checks = {
        "final_response_present": chat_turn["final_response"] is not None,
        "local_only_tools": common.only_local_tools(chat_turn["tool_sequence"]),
        "remote_unavailable_explained": common.looks_like_remote_unavailable(
            answer_text
        ),
    }
    summary = (
        "Remote request stayed local and clearly reported missing access."
        if all(checks.values())
        else "Remote request handling did not clearly communicate the local-only limit."
    )
    return checks, summary


def _evaluate_protection_demo(
    scenario_result: dict[str, Any],
) -> tuple[dict[str, bool], str]:
    turns = scenario_result["turns"]
    challenge_turn, feedback_turn, sanitize_turn = turns
    feedback_entries = scenario_result.get("corrections_entries", [])
    provenance_paths = scenario_result.get("sanitize_provenance_paths", [])
    sanitize_answer = ((sanitize_turn.get("final_response") or {}).get("answer")) or ""
    checks = {
        "challenge_returned_final_response": challenge_turn.get("final_response")
        is not None,
        "challenge_created_pending_prompt": bool(
            (challenge_turn.get("session_state") or {}).get("pending_protection_prompt")
        ),
        "feedback_cleared_pending_prompt": (
            (
                (feedback_turn.get("session_state") or {}).get(
                    "pending_protection_prompt"
                )
            )
            is None
        ),
        "feedback_recorded_to_store": bool(feedback_entries)
        and feedback_entries[-1].get("expected_sensitivity_label") == "public",
        "sanitize_turn_used_local_tool": "read_file"
        in sanitize_turn.get("tool_sequence", []),
        "sanitize_turn_returned_safe_replacement": sanitize_answer
        == SAFE_PROTECTION_REPLACEMENT,
        "sanitize_turn_removed_sensitive_text": "TOP SECRET TOKEN"
        not in sanitize_answer,
        "sanitize_turn_preserved_provenance": "README.md" in provenance_paths,
    }
    summary = (
        "Protection demo challenged the sensitive prompt, recorded feedback, and "
        "sanitized a later assistant answer."
        if all(checks.values())
        else "Protection demo missed one or more challenge, feedback, or sanitize expectations."
    )
    return checks, summary


def _run_chat_protection_demo(
    *,
    provider_mode: ProviderModeStrategy,
    provider_health: dict[str, Any],
    config: Any,
    runtime: Any,
    mode_output_dir: Path,
) -> dict[str, Any]:
    protection_dir = (mode_output_dir / "protection-demo").resolve()
    protection_dir.mkdir(parents=True, exist_ok=True)
    corrections_path = protection_dir / "corrections.json"
    guidance_path = protection_dir / "proprietary-guidance.md"
    guidance_path.write_text(
        "\n".join(
            [
                "# Proprietary Guidance",
                "",
                "Treat secret plans, partner lists, and unreleased go-to-market details as restricted.",
                "If a user requests them directly, challenge first and require a confirmed label correction.",
                "If a candidate final answer still contains restricted material, sanitize the answer before display.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    document_id = guidance_path.name
    runtime_with_protection = runtime.model_copy(
        update={
            "protection": ProtectionConfig(
                enabled=True,
                document_paths=[str(guidance_path)],
                corrections_path=str(corrections_path),
            )
        }
    )

    def _build_demo_controller(**kwargs: object) -> ProtectionController:
        config_arg = kwargs["config"]
        environment = dict(kwargs["environment"])
        feedback_store = ProtectionFeedbackStore(config_arg.corrections_path)
        corpus = load_protection_corpus(config_arg, feedback_store=feedback_store)
        return ProtectionController(
            config=config_arg,
            classifier=_DeterministicProtectionClassifier(document_id=document_id),
            environment=environment,
            comparator=DefaultEnvironmentComparator(),
            corpus=corpus,
            feedback_store=feedback_store,
        )

    prompts = [
        "Tell me the secret plan from the proprietary playbook.",
        (
            "analysis_is_correct: false\n"
            "expected_sensitivity_label: public\n"
            "rationale: This example is already approved for external-safe discussion."
        ),
        "Use a local workspace tool to inspect README.md, then answer directly.",
    ]
    result: dict[str, Any] = {
        "name": "chat_proprietary_protection_demo",
        "kind": "chat",
        "provider_mode": provider_mode.value,
        "provider_health": provider_health,
        "prompt": prompts[0],
        "prompts": prompts,
        "protection_document_path": str(guidance_path),
        "corrections_path": str(corrections_path),
    }

    session_id = f"e2e-chat-protection-{uuid4().hex[:8]}"
    session_state = ChatSessionState()
    challenge_turn = _run_chat_turn(
        scenario_name="chat_proprietary_protection_demo-challenge",
        session_id=session_id,
        prompt=prompts[0],
        config=config,
        runtime=runtime_with_protection,
        session_state=session_state,
        provider=_UnexpectedProvider(),
        protection_controller_factory=_build_demo_controller,
    )
    challenge_session_state = ChatSessionState.model_validate(
        challenge_turn["session_state"]
    )
    feedback_turn = _run_chat_turn(
        scenario_name="chat_proprietary_protection_demo-feedback",
        session_id=session_id,
        prompt=prompts[1],
        config=config,
        runtime=runtime_with_protection,
        session_state=challenge_session_state,
        provider=_UnexpectedProvider(),
        protection_controller_factory=_build_demo_controller,
    )
    feedback_session_state = ChatSessionState.model_validate(
        feedback_turn["session_state"]
    )
    sanitize_turn = _run_chat_turn(
        scenario_name="chat_proprietary_protection_demo-sanitize",
        session_id=session_id,
        prompt=prompts[2],
        config=config,
        runtime=runtime_with_protection,
        session_state=feedback_session_state,
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "read_file", "arguments": {"path": "README.md"}}
                    ]
                ),
                ParsedModelResponse(
                    final_response={
                        "answer": (
                            "TOP SECRET TOKEN: proprietary playbook says to prioritize "
                            "restricted launch partners first."
                        ),
                        "citations": [],
                        "confidence": 0.92,
                        "uncertainty": [],
                        "missing_information": [],
                        "follow_up_suggestions": [],
                    }
                ),
            ]
        ),
        protection_controller_factory=_build_demo_controller,
    )
    correction_entries = []
    if corrections_path.exists():
        correction_entries = ProtectionFeedbackStore(corrections_path).load_entries()
    sanitize_provenance_paths = []
    sanitize_result = sanitize_turn.get("result") or {}
    for tool_result in sanitize_result.get("tool_results") or []:
        for source in tool_result.get("source_provenance") or []:
            path = source.get("metadata", {}).get("path")
            if isinstance(path, str):
                sanitize_provenance_paths.append(path)

    result["turns"] = [challenge_turn, feedback_turn, sanitize_turn]
    result["transcript_entries"] = [
        entry
        for turn in result["turns"]
        for entry in turn.get("transcript_entries", [])
    ]
    result["tool_sequence"] = [
        tool_name
        for turn in result["turns"]
        for tool_name in turn.get("tool_sequence", [])
    ]
    result["final_response"] = sanitize_turn.get("final_response")
    result["session_state"] = sanitize_turn.get("session_state")
    result["corrections_entries"] = [
        entry.model_dump(mode="json") for entry in correction_entries
    ]
    result["sanitize_provenance_paths"] = sanitize_provenance_paths
    result["protection_environment"] = build_protection_environment(
        app_name="nicegui_chat",
        model_name=runtime.model_name,
        workspace=runtime.root_path,
        enabled_tools=runtime.enabled_tools,
        allow_network=runtime.allow_network,
        allow_filesystem=runtime.allow_filesystem,
        allow_subprocess=runtime.allow_subprocess,
    )
    checks, summary = _evaluate_protection_demo(result)
    result["checks"] = checks
    result["summary"] = summary
    result["status"] = (
        common.SCENARIO_STATUS_PASSED
        if all(checks.values())
        else common.SCENARIO_STATUS_FAILED
    )
    return result


def _run_chat_scenario(
    *,
    name: str,
    provider_mode: ProviderModeStrategy,
    provider_health: dict[str, Any],
    config: Any,
    runtime: Any,
    mode_output_dir: Path,
) -> dict[str, Any]:
    if name == "chat_proprietary_protection_demo":
        return _run_chat_protection_demo(
            provider_mode=provider_mode,
            provider_health=provider_health,
            config=config,
            runtime=runtime,
            mode_output_dir=mode_output_dir,
        )

    prompt = _chat_prompt_for(name)
    result: dict[str, Any] = {
        "name": name,
        "kind": "chat",
        "provider_mode": provider_mode.value,
        "provider_health": provider_health,
        "prompt": prompt,
    }
    if not provider_health.get("ok", False):
        result.update(
            {
                "status": common.SCENARIO_STATUS_INFRA,
                "summary": (
                    "Provider preflight failed, so live chat behavior was not executed."
                ),
            }
        )
        return result

    if name == "chat_multi_turn_back_and_forth":
        return _run_chat_multi_turn_scenario(
            provider_mode=provider_mode,
            provider_health=provider_health,
            config=config,
            runtime=runtime,
        )

    chat_turn = _run_chat_turn_with_continuation_refusal(
        scenario_name=name,
        prompt=prompt,
        config=config,
        runtime=runtime,
    )
    if name == "chat_repo_lookup":
        checks, summary = _evaluate_chat_repo_lookup(chat_turn)
    elif name == "chat_git_inspection":
        checks, summary = _evaluate_chat_git_inspection(chat_turn)
    else:
        checks, summary = _evaluate_chat_remote_blocked(chat_turn)

    result.update(chat_turn)
    result["checks"] = checks
    if chat_turn.get("failure"):
        summary = "Scenario execution raised an exception after partial artifacts."
    result["summary"] = summary
    result["status"] = (
        common.SCENARIO_STATUS_PASSED
        if all(checks.values())
        else common.SCENARIO_STATUS_FAILED
    )
    return result


def _research_prompt() -> str:
    return (
        "Investigate how the NiceGUI assistant distinguishes normal chat from "
        "durable research sessions in this repository. Focus on the assistant app, "
        "runtime wiring, and the harness-backed research controller."
    )


def _run_research_launch(
    *,
    provider_mode: ProviderModeStrategy,
    provider_health: dict[str, Any],
    config: Any,
    runtime: Any,
) -> dict[str, Any]:
    prompt = _research_prompt()
    result: dict[str, Any] = {
        "name": "research_repo_investigation",
        "kind": "research",
        "provider_mode": provider_mode.value,
        "provider_health": provider_health,
        "prompt": prompt,
    }
    if not provider_health.get("ok", False):
        result.update(
            {
                "status": common.SCENARIO_STATUS_INFRA,
                "summary": (
                    "Provider preflight failed, so live research execution was skipped."
                ),
            }
        )
        return result

    deep_runtime = runtime.model_copy(update={"interaction_mode": "deep_task"})
    probe_session_id = f"e2e-research-launch-{uuid4().hex[:8]}"
    controller = _build_probe_controller(
        session_id=probe_session_id,
        config=config,
        runtime=deep_runtime,
        provider=None,
    )
    controller.set_deep_task_mode_enabled(True)
    error = controller.submit_prompt(prompt)
    if error is not None:
        raise RuntimeError(error)
    _drain_controller(controller, timeout_seconds=max(runtime.timeout_seconds, 5.0))
    inspection_payload = next(
        (
            item.payload
            for item in reversed(controller.active_record.workbench_items)
            if item.kind == "result" and isinstance(item.payload, dict)
        ),
        None,
    )
    if inspection_payload is None:
        raise RuntimeError("Deep task did not produce an inspection artifact.")
    inspection = HarnessSessionInspection.model_validate(inspection_payload)
    RESEARCH_SESSION_TO_PROBE_SESSION[inspection.snapshot.session_id] = probe_session_id
    checks = {
        "session_created": bool(inspection.snapshot.session_id),
        "summary_present": inspection.summary.total_turns >= 0,
        "replay_present": inspection.replay is not None,
        "stop_state_recorded": inspection.summary.stop_reason is not None,
    }
    result.update(
        {
            "inspection": inspection_payload,
            "summary_text": "\n".join(
                [
                    f"session: {inspection.summary.session_id}",
                    f"turns: {inspection.summary.total_turns}",
                    f"stop: {inspection.summary.stop_reason}",
                ]
            ),
            "session_id": inspection.snapshot.session_id,
            "checks": checks,
            "status": (
                common.SCENARIO_STATUS_PASSED
                if all(checks.values())
                else common.SCENARIO_STATUS_FAILED
            ),
            "summary": (
                "Research session launched, stopped durably, and produced replay data."
                if all(checks.values())
                else "Research session did not produce the expected durable artifacts."
            ),
        }
    )
    return result


def _run_research_approval_flow(
    *,
    config: Any,
    runtime: Any,
    workspace_root: Path,
    output_relpath: str,
    approval_resolution: ApprovalResolution,
) -> dict[str, Any]:
    workspace_root.mkdir(parents=True, exist_ok=True)
    output_path = workspace_root / output_relpath
    provider = ScriptedParsedResponseProvider(
        [
            ParsedModelResponse(
                invocations=[
                    {
                        "tool_name": "write_file",
                        "arguments": {
                            "path": output_relpath,
                            "content": "approved research output\n",
                            "create_parents": True,
                        },
                    },
                ]
            ),
            ParsedModelResponse(
                final_response="Wrote the approved research note successfully."
            ),
        ]
    )
    deep_runtime = runtime.model_copy(update={"interaction_mode": "deep_task"})
    controller = _build_probe_controller(
        session_id=f"e2e-research-approval-{uuid4().hex[:8]}",
        config=config,
        runtime=deep_runtime,
        provider=provider,
    )
    controller.set_deep_task_mode_enabled(True)
    error = controller.submit_prompt(
        "List the scratch workspace, then write an approval-gated note and "
        "confirm completion."
    )
    if error is not None:
        raise RuntimeError(error)
    _drain_controller(
        controller,
        timeout_seconds=max(runtime.timeout_seconds, 5.0),
        auto_resolve_approvals=False,
    )
    waiting_payload = next(
        (
            item.payload
            for item in reversed(controller.active_record.workbench_items)
            if item.kind == "result" and isinstance(item.payload, dict)
        ),
        None,
    )
    if waiting_payload is None:
        raise RuntimeError("Deep task did not produce waiting inspection.")
    controller.resolve_approval(
        approved=approval_resolution is ApprovalResolution.APPROVE
    )
    _drain_controller(controller, timeout_seconds=max(runtime.timeout_seconds, 5.0))
    resumed_payload = next(
        (
            item.payload
            for item in reversed(controller.active_record.workbench_items)
            if item.kind == "result" and isinstance(item.payload, dict)
        ),
        None,
    )
    if resumed_payload is None:
        raise RuntimeError("Deep task did not produce resumed inspection.")
    resumed = HarnessSessionInspection.model_validate(resumed_payload)
    trace = resumed.snapshot.artifacts.trace
    replay = resumed.replay
    tool_name_counts: Counter[str] = Counter()
    workflow_statuses: list[str] = []
    if trace is not None:
        for turn in trace.turns:
            workflow_statuses.extend(
                status.value for status in turn.workflow_outcome_statuses
            )
            for invocation in turn.invocation_traces:
                tool_name_counts[invocation.tool_name] += 1
    return {
        "workspace_root": str(workspace_root),
        "output_path": str(output_path),
        "waiting": waiting_payload,
        "resumed": resumed_payload,
        "file_exists": output_path.exists(),
        "file_content": (
            output_path.read_text(encoding="utf-8") if output_path.exists() else None
        ),
        "tool_name_counts": dict(tool_name_counts),
        "workflow_statuses": workflow_statuses,
        "replay_step_statuses": (
            [
                [status.value for status in step.workflow_outcome_statuses]
                for step in replay.steps
            ]
            if replay is not None
            else []
        ),
    }


def _evaluate_research_approval_resume(
    scenario_result: dict[str, Any],
) -> tuple[dict[str, bool], str]:
    deny_flow = scenario_result["deny_flow"]
    approve_flow = scenario_result["approve_flow"]
    waiting_summary = approve_flow["waiting"]["summary"]
    denied_summary = deny_flow["resumed"]["summary"]
    approved_summary = approve_flow["resumed"]["summary"]
    checks = {
        "launch_waits_for_approval": (
            approve_flow["waiting"]["resumed"]["disposition"]
            == ResumeDisposition.WAITING_FOR_APPROVAL.value
        ),
        "pending_approval_visible": bool(waiting_summary.get("pending_approval_ids")),
        "deny_stops_cleanly": denied_summary.get("stop_reason")
        == HarnessStopReason.APPROVAL_DENIED.value,
        "deny_writes_nothing": not deny_flow["file_exists"],
        "approve_completes": approved_summary.get("stop_reason")
        == HarnessStopReason.COMPLETED.value,
        "approve_clears_pending_approvals": approved_summary.get("pending_approval_ids")
        == [],
        "approve_writes_expected_file": approve_flow["file_content"]
        == "approved research output\n",
        "resume_records_single_write_request_and_execution": approve_flow[
            "workflow_statuses"
        ]
        == ["approval_requested", "executed"]
        and approve_flow["tool_name_counts"].get("write_file") == 2,
        "replay_captures_approval_and_execution": any(
            "approval_requested" in statuses and "executed" in statuses
            for statuses in approve_flow["replay_step_statuses"]
        ),
    }
    summary = (
        "Research approval flow waited, denied safely, then approved and resumed a single write."
        if all(checks.values())
        else "Research approval flow missed one or more durable wait, deny, or resume expectations."
    )
    return checks, summary


def _run_research_approval_resume_write_flow(
    *,
    provider_mode: ProviderModeStrategy,
    provider_health: dict[str, Any],
    config: Any,
    runtime: Any,
    mode_output_dir: Path,
) -> dict[str, Any]:
    deny_workspace = (mode_output_dir / "approval-deny-workspace").resolve()
    approve_workspace = (mode_output_dir / "approval-approve-workspace").resolve()
    result: dict[str, Any] = {
        "name": "research_approval_resume_write_flow",
        "kind": "research",
        "provider_mode": provider_mode.value,
        "provider_health": provider_health,
        "prompt": (
            "List the scratch workspace, then write an approval-gated note and "
            "confirm completion."
        ),
    }
    runtime_with_write = runtime.model_copy(
        update={
            "enabled_tools": ["list_directory", "write_file"],
            "allow_filesystem": True,
            "allow_subprocess": False,
            "require_approval_for": set(runtime.require_approval_for).union(
                {SideEffectClass.LOCAL_WRITE}
            ),
        }
    )
    deny_flow = _run_research_approval_flow(
        config=config,
        runtime=runtime_with_write.model_copy(
            update={
                "root_path": str(deny_workspace),
                "default_workspace_root": str(deny_workspace),
            }
        ),
        workspace_root=deny_workspace,
        output_relpath="notes/approved.txt",
        approval_resolution=ApprovalResolution.DENY,
    )
    approve_flow = _run_research_approval_flow(
        config=config,
        runtime=runtime_with_write.model_copy(
            update={
                "root_path": str(approve_workspace),
                "default_workspace_root": str(approve_workspace),
            }
        ),
        workspace_root=approve_workspace,
        output_relpath="notes/approved.txt",
        approval_resolution=ApprovalResolution.APPROVE,
    )
    result["deny_flow"] = deny_flow
    result["approve_flow"] = approve_flow
    result["session_id"] = approve_flow["resumed"]["snapshot"]["session_id"]
    result["inspection"] = approve_flow["resumed"]
    checks, summary = _evaluate_research_approval_resume(result)
    result["checks"] = checks
    result["summary"] = summary
    result["status"] = (
        common.SCENARIO_STATUS_PASSED
        if all(checks.values())
        else common.SCENARIO_STATUS_FAILED
    )
    return result


def _run_research_followup(
    *,
    provider_mode: ProviderModeStrategy,
    provider_health: dict[str, Any],
    config: Any,
    runtime: Any,
    known_session_id: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": "research_followup_or_inspect",
        "kind": "research",
        "provider_mode": provider_mode.value,
        "provider_health": provider_health,
        "prompt": "Inspect the durable state the UI would show for the active session.",
    }
    bootstrap_created = False
    session_id = known_session_id
    if session_id is None and not provider_health.get("ok", False):
        result.update(
            {
                "status": common.SCENARIO_STATUS_INFRA,
                "summary": (
                    "No durable research session was available to inspect because the "
                    "provider was unreachable."
                ),
            }
        )
        return result

    if session_id is not None:
        probe_session_id = RESEARCH_SESSION_TO_PROBE_SESSION.get(session_id)
        if probe_session_id is None:
            result.update(
                {
                    "status": common.SCENARIO_STATUS_FAILED,
                    "summary": (
                        "Known durable research session id was not available in the "
                        "NiceGUI-backed probe store."
                    ),
                }
            )
            return result
        controller = _build_probe_controller(
            session_id=probe_session_id,
            config=config,
            runtime=runtime.model_copy(update={"interaction_mode": "deep_task"}),
            provider=None,
        )
        harness_store = controller.store.harness_store(
            chat_session_id=controller.active_session_id,
            owner_user_id=controller.active_record.summary.owner_user_id,
        )
        snapshot = harness_store.load_session(session_id)
        if snapshot is None or snapshot.artifacts.summary is None:
            result.update(
                {
                    "status": common.SCENARIO_STATUS_FAILED,
                    "summary": (
                        "Known durable research session could not be loaded from "
                        "NiceGUI persistence."
                    ),
                }
            )
            return result
        inspection = HarnessSessionInspection(
            snapshot=snapshot,
            resumed=resume_session(snapshot),
            summary=snapshot.artifacts.summary,
            replay=replay_session(snapshot),
        )
        inspection_payload = common.dump_model(inspection)
    else:
        launch = _run_research_launch(
            provider_mode=provider_mode,
            provider_health=provider_health,
            config=config,
            runtime=runtime,
        )
        bootstrap_created = True
        session_id = str(launch.get("session_id") or "")
        inspection_payload = launch["inspection"]
    summary_payload = inspection_payload["summary"]
    checks = {
        "session_summary_present": bool(summary_payload.get("session_id")),
        "turn_count_present": isinstance(summary_payload.get("total_turns"), int),
        "pending_approvals_present": isinstance(
            summary_payload.get("pending_approval_ids"),
            list,
        ),
        "replay_metadata_present": inspection_payload.get("replay") is not None,
    }
    result.update(
        {
            "session_id": session_id,
            "bootstrap_created_session": bootstrap_created,
            "inspection": inspection_payload,
            "checks": checks,
            "status": (
                common.SCENARIO_STATUS_PASSED
                if all(checks.values())
                else common.SCENARIO_STATUS_FAILED
            ),
            "summary": (
                "Inspected a durable research session and captured the UI-facing state."
                if all(checks.values())
                else "Research inspection missed one or more expected durable fields."
            ),
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the backend probe matrix."""
    args = build_parser().parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    output_dir = common.resolve_output_dir(args.output_dir, kind="backend")
    selected = common.select_scenarios(
        ALL_SCENARIOS,
        scenario_args=args.scenario,
        scenarios_csv=args.scenarios,
    )
    selected_modes = _select_provider_mode_variants(
        mode_args=args.provider_mode,
        modes_csv=args.provider_modes,
    )

    results: list[dict[str, Any]] = []
    provider_health_reports: list[dict[str, Any]] = []
    per_mode_runs: list[dict[str, Any]] = []
    for provider_mode_label, provider_mode, prompt_tool_strategy in selected_modes:
        mode_output_dir = (output_dir / provider_mode_label).resolve()
        mode_output_dir.mkdir(parents=True, exist_ok=True)
        old_prompt_tool_strategy = os.environ.get("LLM_TOOLS_PROMPT_TOOL_STRATEGY")
        if prompt_tool_strategy is not None:
            os.environ["LLM_TOOLS_PROMPT_TOOL_STRATEGY"] = prompt_tool_strategy
        else:
            os.environ.pop("LLM_TOOLS_PROMPT_TOOL_STRATEGY", None)
        config = common.build_assistant_config(
            workspace=workspace,
            output_dir=mode_output_dir,
            ollama_base_url=args.ollama_base_url,
            model=args.model,
            provider_mode=provider_mode,
            timeout_seconds=args.timeout_seconds,
            provider=args.provider,
            api_base_url=args.api_base_url,
            api_key_env_var=args.api_key_env_var,
        )
        runtime = common.build_runtime_config(config, workspace=workspace)
        provider_health = common.build_provider_health(config, runtime)
        provider_health["provider_mode"] = provider_mode_label
        provider_health["provider_mode_strategy"] = provider_mode.value
        if prompt_tool_strategy is not None:
            provider_health["prompt_tool_strategy"] = prompt_tool_strategy
        provider_health_reports.append(provider_health)

        mode_results: list[dict[str, Any]] = []
        known_session_id: str | None = None
        try:
            for name in selected:
                try:
                    if name.startswith("chat_"):
                        result = _run_with_timeout(
                            timeout_seconds=args.timeout_seconds,
                            fn=partial(
                                _run_chat_scenario,
                                name=name,
                                provider_mode=provider_mode,
                                provider_health=provider_health,
                                config=config,
                                runtime=runtime,
                                mode_output_dir=mode_output_dir,
                            ),
                        )
                    elif name == "research_repo_investigation":
                        result = _run_with_timeout(
                            timeout_seconds=args.timeout_seconds,
                            fn=partial(
                                _run_research_launch,
                                provider_mode=provider_mode,
                                provider_health=provider_health,
                                config=config,
                                runtime=runtime,
                            ),
                        )
                        if result.get("status") == common.SCENARIO_STATUS_PASSED:
                            known_session_id = str(result.get("session_id"))
                    elif name == "research_approval_resume_write_flow":
                        result = _run_with_timeout(
                            timeout_seconds=args.timeout_seconds,
                            fn=partial(
                                _run_research_approval_resume_write_flow,
                                provider_mode=provider_mode,
                                provider_health=provider_health,
                                config=config,
                                runtime=runtime,
                                mode_output_dir=mode_output_dir,
                            ),
                        )
                    else:
                        result = _run_with_timeout(
                            timeout_seconds=args.timeout_seconds,
                            fn=partial(
                                _run_research_followup,
                                provider_mode=provider_mode,
                                provider_health=provider_health,
                                config=config,
                                runtime=runtime,
                                known_session_id=known_session_id,
                            ),
                        )
                except Exception as exc:  # pragma: no cover - probe failure path
                    result = {
                        "name": name,
                        "kind": "chat" if name.startswith("chat_") else "research",
                        "provider_mode": provider_mode_label,
                        "provider_health": provider_health,
                        "status": common.SCENARIO_STATUS_FAILED,
                        "summary": "Scenario execution raised an exception.",
                        "failure": common.failure_payload(exc),
                    }
                result["provider_mode"] = provider_mode_label
                mode_results.append(result)
                results.append(result)
                common.write_json(mode_output_dir / f"{name}.json", result)
        finally:
            if old_prompt_tool_strategy is None:
                os.environ.pop("LLM_TOOLS_PROMPT_TOOL_STRATEGY", None)
            else:
                os.environ["LLM_TOOLS_PROMPT_TOOL_STRATEGY"] = old_prompt_tool_strategy

        mode_payload = {
            "run_kind": "backend_matrix",
            "workspace": str(workspace),
            "output_dir": str(mode_output_dir),
            "selected_scenarios": selected,
            "provider_mode": provider_mode_label,
            "provider_mode_strategy": provider_mode.value,
            "prompt_tool_strategy": prompt_tool_strategy,
            "provider_health": provider_health,
            "results": mode_results,
        }
        per_mode_runs.append(mode_payload)
        common.write_json(mode_output_dir / "results.json", mode_payload)
        common.write_text(
            mode_output_dir / "summary.md",
            common.results_markdown(
                title=f"Backend Assistant E2E Probe ({provider_mode_label})",
                results=mode_results,
                provider_health=provider_health,
            ),
        )

    run_payload = {
        "run_kind": "backend_matrix",
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "selected_scenarios": selected,
        "selected_provider_modes": [label for label, _, _ in selected_modes],
        "provider_health": provider_health_reports,
        "mode_runs": per_mode_runs,
        "results": results,
    }
    common.write_json(output_dir / "results.json", run_payload)
    common.write_text(
        output_dir / "summary.md",
        common.results_markdown(
            title="Backend Assistant E2E Probe",
            results=results,
            provider_health=provider_health_reports,
        ),
    )
    print(f"Backend probe artifacts written to {output_dir}")
    return common.final_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())

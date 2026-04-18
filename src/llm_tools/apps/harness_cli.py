"""Minimal CLI for persisted harness sessions."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_tools.apps.chat_runtime import build_chat_executor
from llm_tools.harness_api import (
    ApprovalResolution,
    BudgetPolicy,
    FileHarnessStateStore,
    HarnessSessionCreateRequest,
    HarnessSessionInspection,
    HarnessSessionInspectRequest,
    HarnessSessionListRequest,
    HarnessSessionResumeRequest,
    HarnessSessionRunRequest,
    HarnessSessionService,
    HarnessSessionStopRequest,
    ScriptedParsedResponseProvider,
)
from llm_tools.llm_adapters import ParsedModelResponse

DEFAULT_STORE_DIR = ".llm-tools-harness"


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal persisted harness CLI parser."""
    parser = argparse.ArgumentParser(prog="llm-tools-harness")
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Create a persisted harness session.")
    start.add_argument("--title", required=True)
    start.add_argument("--intent", required=True)
    start.add_argument("--session-id")
    start.add_argument("--root-task-id", default="task-1")
    start.add_argument("--max-turns", type=int, default=4)
    start.add_argument("--max-tool-invocations", type=int)
    start.add_argument("--max-elapsed-seconds", type=int)
    start.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run", help="Run a persisted harness session.")
    run.add_argument("session_id")
    run.add_argument("--workspace", default=".")
    run.add_argument("--script")
    run.add_argument("--json", action="store_true")

    resume = subparsers.add_parser("resume", help="Resume a persisted harness session.")
    resume.add_argument("session_id")
    resume.add_argument("--workspace", default=".")
    resume.add_argument("--script")
    resolution = resume.add_mutually_exclusive_group()
    resolution.add_argument("--approve", action="store_true")
    resolution.add_argument("--deny", action="store_true")
    resolution.add_argument("--expire", action="store_true")
    resolution.add_argument("--cancel", action="store_true")
    resume.add_argument("--json", action="store_true")

    inspect_cmd = subparsers.add_parser(
        "inspect", help="Inspect one persisted session."
    )
    inspect_cmd.add_argument("session_id")
    inspect_cmd.add_argument("--replay", action="store_true")
    inspect_cmd.add_argument("--json", action="store_true")

    list_cmd = subparsers.add_parser("list", help="List recent persisted sessions.")
    list_cmd.add_argument("--limit", type=int)
    list_cmd.add_argument("--replay", action="store_true")
    list_cmd.add_argument("--json", action="store_true")

    stop = subparsers.add_parser("stop", help="Stop a persisted harness session.")
    stop.add_argument("session_id")
    stop.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the minimal persisted harness CLI."""
    args = build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    store = FileHarnessStateStore(Path(args.store_dir))

    if args.command == "start":
        service = _build_service(store=store, workspace=".", script_path=None)
        snapshot = service.create_session(
            HarnessSessionCreateRequest(
                title=args.title,
                intent=args.intent,
                budget_policy=BudgetPolicy(
                    max_turns=args.max_turns,
                    max_tool_invocations=args.max_tool_invocations,
                    max_elapsed_seconds=args.max_elapsed_seconds,
                ),
                session_id=args.session_id,
                root_task_id=args.root_task_id,
            )
        )
        return _emit(snapshot.model_dump(mode="json"), as_json=args.json)

    if args.command == "run":
        service = _build_service(
            store=store,
            workspace=args.workspace,
            script_path=args.script,
        )
        run_result = service.run_session(
            HarnessSessionRunRequest(session_id=args.session_id)
        )
        inspection = service.inspect_session(
            HarnessSessionInspectRequest(session_id=run_result.snapshot.session_id)
        )
        return _emit_inspection(inspection, as_json=args.json)

    if args.command == "resume":
        service = _build_service(
            store=store,
            workspace=args.workspace,
            script_path=args.script,
        )
        resume_result = service.resume_session(
            HarnessSessionResumeRequest(
                session_id=args.session_id,
                approval_resolution=_approval_resolution_from_args(args),
            )
        )
        inspection = service.inspect_session(
            HarnessSessionInspectRequest(session_id=resume_result.snapshot.session_id)
        )
        return _emit_inspection(inspection, as_json=args.json)

    if args.command == "inspect":
        service = _build_service(store=store, workspace=".", script_path=None)
        inspection = service.inspect_session(
            HarnessSessionInspectRequest(
                session_id=args.session_id,
                include_replay=args.replay,
            )
        )
        return _emit_inspection(inspection, as_json=args.json)

    if args.command == "list":
        service = _build_service(store=store, workspace=".", script_path=None)
        list_result = service.list_sessions(
            HarnessSessionListRequest(
                limit=args.limit,
                include_replay=args.replay,
            )
        )
        if args.json:
            return _emit(list_result.model_dump(mode="json"), as_json=True)
        for item in list_result.sessions:
            summary = item.summary
            print(
                f"{item.snapshot.session_id} rev={item.snapshot.revision} "
                f"saved_at={item.snapshot.saved_at} "
                f"stop_reason={summary.stop_reason or 'running'} "
                f"turns={summary.total_turns} "
                f"pending_approvals={len(summary.pending_approval_ids)}"
            )
        return 0

    service = _build_service(store=store, workspace=".", script_path=None)
    inspection = service.stop_session(
        HarnessSessionStopRequest(session_id=args.session_id)
    )
    return _emit_inspection(inspection, as_json=args.json)


def _build_service(
    *,
    store: FileHarnessStateStore,
    workspace: str,
    script_path: str | None,
) -> HarnessSessionService:
    _, workflow_executor = build_chat_executor()
    provider = ScriptedParsedResponseProvider(_load_script(script_path))
    return HarnessSessionService(
        store=store,
        workflow_executor=workflow_executor,
        provider=provider,
        workspace=workspace,
    )


def _load_script(path: str | None) -> list[ParsedModelResponse]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Scripted response files must contain a JSON list.")
    return [ParsedModelResponse.model_validate(item) for item in payload]


def _approval_resolution_from_args(
    args: argparse.Namespace,
) -> ApprovalResolution | None:
    if args.approve:
        return ApprovalResolution.APPROVE
    if args.deny:
        return ApprovalResolution.DENY
    if args.expire:
        return ApprovalResolution.EXPIRE
    if args.cancel:
        return ApprovalResolution.CANCEL
    return None


def _emit_inspection(inspection: HarnessSessionInspection, *, as_json: bool) -> int:
    if as_json:
        return _emit(inspection.model_dump(mode="json"), as_json=True)
    snapshot = inspection.snapshot
    summary = inspection.summary
    print(f"session_id: {snapshot.session_id}")
    print(f"revision: {snapshot.revision}")
    print(f"saved_at: {snapshot.saved_at}")
    print(f"stop_reason: {summary.stop_reason or 'running'}")
    print(f"turns: {summary.total_turns}")
    print(f"completed_tasks: {', '.join(summary.completed_task_ids) or '-'}")
    print(f"active_tasks: {', '.join(summary.active_task_ids) or '-'}")
    print(f"pending_approvals: {', '.join(summary.pending_approval_ids) or '-'}")
    if inspection.replay is not None:
        print(f"replay_steps: {len(inspection.replay.steps)}")
    return 0


def _emit(payload: Any, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

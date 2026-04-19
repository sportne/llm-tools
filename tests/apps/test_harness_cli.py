"""CLI tests for the minimal persisted harness interface."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from llm_tools.apps.harness_cli import (
    _approval_resolution_from_args,
    _default_store_dir,
    _load_script,
    build_parser,
    main,
)


def test_harness_cli_start_list_and_inspect_json(tmp_path: Path, capsys) -> None:
    store_dir = tmp_path / "store"

    assert (
        main(
            [
                "--store-dir",
                str(store_dir),
                "start",
                "--title",
                "CLI task",
                "--intent",
                "Inspect the session.",
                "--session-id",
                "cli-session",
                "--json",
            ]
        )
        == 0
    )
    start_payload = json.loads(capsys.readouterr().out)
    assert start_payload["session_id"] == "cli-session"

    assert main(["--store-dir", str(store_dir), "list", "--json"]) == 0
    list_payload = json.loads(capsys.readouterr().out)
    assert list_payload["sessions"][0]["snapshot"]["session_id"] == "cli-session"

    assert (
        main(
            [
                "--store-dir",
                str(store_dir),
                "inspect",
                "cli-session",
                "--json",
            ]
        )
        == 0
    )
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["snapshot"]["session_id"] == "cli-session"


def test_harness_cli_run_scripted_session(tmp_path: Path, capsys) -> None:
    store_dir = tmp_path / "store"
    script_path = tmp_path / "script.json"
    script_path.write_text(
        json.dumps([{"final_response": "done", "invocations": []}]),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--store-dir",
                str(store_dir),
                "start",
                "--title",
                "CLI run",
                "--intent",
                "Run the session.",
                "--session-id",
                "cli-run-session",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "--store-dir",
                str(store_dir),
                "run",
                "cli-run-session",
                "--workspace",
                str(tmp_path),
                "--script",
                str(script_path),
                "--json",
            ]
        )
        == 0
    )
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["summary"]["stop_reason"] == "completed"


def test_harness_cli_resume_inspect_list_and_stop_human_output(
    tmp_path: Path,
    capsys,
) -> None:
    store_dir = tmp_path / "store"
    script_path = tmp_path / "script.json"
    script_path.write_text(
        json.dumps([{"final_response": "done", "invocations": []}]),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--store-dir",
                str(store_dir),
                "start",
                "--title",
                "CLI resume",
                "--intent",
                "Resume the session.",
                "--session-id",
                "cli-resume-session",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "--store-dir",
                str(store_dir),
                "resume",
                "cli-resume-session",
                "--workspace",
                str(tmp_path),
                "--script",
                str(script_path),
            ]
        )
        == 0
    )
    resume_output = capsys.readouterr().out
    assert "session_id: cli-resume-session" in resume_output

    assert (
        main(
            [
                "--store-dir",
                str(store_dir),
                "inspect",
                "cli-resume-session",
                "--replay",
            ]
        )
        == 0
    )
    inspect_output = capsys.readouterr().out
    assert "replay_steps:" in inspect_output

    assert main(["--store-dir", str(store_dir), "list"]) == 0
    list_output = capsys.readouterr().out
    assert "cli-resume-session" in list_output

    assert main(["--store-dir", str(store_dir), "stop", "cli-resume-session"]) == 0
    stop_output = capsys.readouterr().out
    assert "stop_reason: HarnessStopReason.CANCELED" in stop_output


def test_harness_cli_load_script_and_resolution_helpers(tmp_path: Path) -> None:
    script_path = tmp_path / "script.json"
    script_path.write_text(
        json.dumps([{"final_response": "done", "invocations": []}]),
        encoding="utf-8",
    )

    assert len(_load_script(str(script_path))) == 1
    assert _load_script(None) == []

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps({"final_response": "nope"}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        _load_script(str(invalid_path))

    args = type(
        "Args",
        (),
        {"approve": True, "deny": False, "expire": False, "cancel": False},
    )()
    assert _approval_resolution_from_args(args).value == "approve"

    args = type(
        "Args",
        (),
        {"approve": False, "deny": True, "expire": False, "cancel": False},
    )()
    assert _approval_resolution_from_args(args).value == "deny"

    args = type(
        "Args",
        (),
        {"approve": False, "deny": False, "expire": True, "cancel": False},
    )()
    assert _approval_resolution_from_args(args).value == "expire"

    args = type(
        "Args",
        (),
        {"approve": False, "deny": False, "expire": False, "cancel": True},
    )()
    assert _approval_resolution_from_args(args).value == "cancel"

    args = type(
        "Args",
        (),
        {"approve": False, "deny": False, "expire": False, "cancel": False},
    )()
    assert _approval_resolution_from_args(args) is None


def test_harness_cli_default_store_dir_is_user_scoped(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    args = build_parser().parse_args(["list"])

    assert Path(args.store_dir) == Path(_default_store_dir())
    assert Path(args.store_dir) == (tmp_path / ".llm-tools" / "harness").resolve()


def test_harness_cli_module_entrypoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llm_tools.apps.harness_cli",
            "start",
            "--title",
            "Entrypoint",
            "--intent",
            "Run as a module.",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("llm_tools.apps.harness_cli", run_name="__main__")

    assert exc_info.value.code == 0

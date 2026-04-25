"""Run a temporary UI smoke probe for the Streamlit assistant."""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import common

from llm_tools.llm_providers import ProviderModeStrategy

ALL_SCENARIOS = ["browser_smoke"]
CHAT_SMOKE_MARKER = "CHAT-SMOKE:"
CHAT_SMOKE_PROMPT = (
    "Use local workspace tools to find the two most relevant files for tracing "
    "normal chat and durable research in this assistant. "
    f"Begin your answer with {CHAT_SMOKE_MARKER}"
)
CHAT_FOLLOWUP_MARKER = "CHAT-FOLLOWUP:"
CHAT_FOLLOWUP_PROMPT = (
    "Follow up on your last answer. Tell me the first thing I should look for in "
    "each file and "
    f"begin your answer with {CHAT_FOLLOWUP_MARKER}"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the browser smoke CLI."""
    parser = argparse.ArgumentParser(
        description="Temporary UI smoke probe for the Streamlit assistant."
    )
    parser.add_argument("--workspace", type=Path, default=common.REPO_ROOT)
    parser.add_argument(
        "--ollama-base-url",
        default=common.DEFAULT_OLLAMA_BASE_URL,
    )
    parser.add_argument("--model", default=common.DEFAULT_MODEL)
    parser.add_argument("--output-dir")
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--scenarios")
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
        help="Comma-separated provider modes: tools,json,prompt_tools.",
    )
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _choose_port(port: int | None) -> int:
    if port is not None:
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _playwright_available() -> bool:
    try:
        return importlib.util.find_spec("playwright.sync_api") is not None
    except ModuleNotFoundError:
        return False


def _ui_snapshot_from_apptest(app: Any) -> dict[str, Any]:
    return {
        "captions": [item.value for item in app.caption],
        "warnings": [item.value for item in app.warning],
        "errors": [item.value for item in app.error],
        "buttons": [item.label for item in app.button],
        "text_areas": [
            {"label": item.label, "value": item.value} for item in app.text_area
        ],
        "chat_message_count": len(app.chat_message),
        "markdown": [item.value for item in app.markdown[:25]],
    }


def _visible_text_from_snapshot(snapshot: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("captions", "warnings", "errors", "markdown"):
        values = snapshot.get(key, [])
        if not isinstance(values, list):
            continue
        parts.extend(str(value) for value in values if value)
    return common.visible_text_block(parts)


def _status_updates_visible(visible_text: str) -> bool:
    lowered = visible_text.lower()
    return "assistant is " in lowered or "recent steps:" in lowered


def _snapshot_indicates_chat_response(snapshot: dict[str, Any]) -> bool:
    markdown = snapshot.get("markdown", [])
    return isinstance(markdown, list) and any(
        CHAT_SMOKE_MARKER in str(item) or CHAT_FOLLOWUP_MARKER in str(item)
        for item in markdown
    )


def _wait_for_apptest_chat_activity(
    app: Any,
    *,
    marker: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str, bool, bool]:
    status_updates_observed = False
    chat_response_visible = False
    latest_snapshot = _ui_snapshot_from_snapshot_target(app)
    latest_visible_text = _visible_text_from_snapshot(latest_snapshot)
    max_runs = max(1, min(12, int(timeout_seconds * 2)))
    for _ in range(max_runs):
        app.run(timeout=timeout_seconds)
        latest_snapshot = _ui_snapshot_from_snapshot_target(app)
        latest_visible_text = _visible_text_from_snapshot(latest_snapshot)
        status_updates_observed = status_updates_observed or _status_updates_visible(
            latest_visible_text
        )
        chat_response_visible = chat_response_visible or any(
            marker in str(item) for item in latest_snapshot.get("markdown", [])
        )
        if chat_response_visible:
            break
    return (
        latest_snapshot,
        latest_visible_text,
        status_updates_observed,
        chat_response_visible,
    )


def _ui_snapshot_from_snapshot_target(app: Any) -> dict[str, Any]:
    return _ui_snapshot_from_apptest(app)


def _apptest_send_chat_prompt(
    app: Any,
    *,
    prompt: str,
    marker: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str, bool, bool]:
    for text_area in app.text_area:
        if text_area.label == "Message the assistant":
            text_area.set_value(prompt).run(timeout=timeout_seconds)
            break
    for button in app.button:
        if button.label == "Send":
            button.click().run(timeout=timeout_seconds)
            break
    return _wait_for_apptest_chat_activity(
        app,
        marker=marker,
        timeout_seconds=timeout_seconds,
    )


def _apptest_wait_for_turn_idle(app: Any, *, timeout_seconds: float) -> bool:
    max_runs = max(1, min(12, int(timeout_seconds * 2)))
    for _ in range(max_runs):
        app.run(timeout=timeout_seconds)
        labels = [button.label for button in app.button]
        if "Queue follow-up" not in labels and "Send" in labels:
            return True
    return False


def _build_apptest_script(
    *,
    workspace: Path,
    config_path: Path,
    state_dir: Path,
) -> str:
    return f"""
from pathlib import Path
import os
import sys

sys.path.insert(0, {str(common.SRC_ROOT)!r})
os.environ["LLM_TOOLS_STREAMLIT_ASSISTANT_STATE_DIR"] = {str(state_dir)!r}

from llm_tools.apps.assistant_config import load_streamlit_assistant_config
from llm_tools.apps.streamlit_assistant import run_streamlit_assistant_app

workspace = Path({str(workspace)!r})
config = load_streamlit_assistant_config(Path({str(config_path)!r}))
run_streamlit_assistant_app(root_path=workspace, config=config)
""".strip()


def _run_apptest_smoke(
    *,
    provider_mode: ProviderModeStrategy,
    workspace: Path,
    config_path: Path,
    state_dir: Path,
    provider_health: dict[str, Any],
    timeout_seconds: float,
    output_dir: Path,
) -> dict[str, Any]:
    from streamlit.testing.v1 import AppTest

    script = _build_apptest_script(
        workspace=workspace,
        config_path=config_path,
        state_dir=state_dir,
    )
    app = AppTest.from_string(script, default_timeout=timeout_seconds)
    app.run(timeout=timeout_seconds)

    checks = {
        "ui_rendered": True,
        "workspace_visible": any(
            "Workspace:" in text for text in [item.value for item in app.caption]
        ),
        "research_visible": any(
            item.label == "Create a research task" for item in app.text_area
        ),
        "chat_composer_visible": any(
            item.label == "Message the assistant" for item in app.text_area
        ),
    }

    validated_provider = False
    if provider_health.get("ok", False):
        for button in app.button:
            if button.label == "Validate provider connection":
                button.click().run(timeout=timeout_seconds)
                validated_provider = True
                break

    snapshot = _ui_snapshot_from_apptest(app)
    visible_text = _visible_text_from_snapshot(snapshot)
    common.write_json(output_dir / "ui_snapshot.json", snapshot)
    common.write_text(output_dir / "visible_text.txt", visible_text)

    if not provider_health.get("ok", False):
        summary = (
            "Rendered the UI and captured the unreachable-provider state through "
            "the Streamlit AppTest fallback."
        )
        return {
            "name": "browser_smoke",
            "kind": "browser",
            "provider_mode": provider_mode.value,
            "automation_backend": "streamlit_apptest",
            "provider_validation_attempted": validated_provider,
            "provider_health": provider_health,
            "status": common.SCENARIO_STATUS_INFRA,
            "summary": summary,
            "checks": checks,
            "artifacts": {
                "config": str(config_path),
                "ui_snapshot": str(output_dir / "ui_snapshot.json"),
                "visible_text": str(output_dir / "visible_text.txt"),
                "screenshot": None,
            },
            "ui_snapshot": snapshot,
        }

    (
        snapshot,
        visible_text,
        status_updates_observed,
        chat_response_visible,
    ) = _apptest_send_chat_prompt(
        app,
        prompt=CHAT_SMOKE_PROMPT,
        marker=CHAT_SMOKE_MARKER,
        timeout_seconds=timeout_seconds,
    )
    _apptest_wait_for_turn_idle(app, timeout_seconds=timeout_seconds)
    (
        snapshot,
        visible_text,
        followup_status_updates_observed,
        followup_response_visible,
    ) = _apptest_send_chat_prompt(
        app,
        prompt=CHAT_FOLLOWUP_PROMPT,
        marker=CHAT_FOLLOWUP_MARKER,
        timeout_seconds=timeout_seconds,
    )

    for text_area in app.text_area:
        if text_area.label == "Create a research task":
            text_area.set_value(
                "Investigate how the Streamlit assistant research lane differs from "
                "normal chat in this repository."
            ).run(timeout=timeout_seconds)
            break
    for button in app.button:
        if button.label == "Start research task":
            button.click().run(timeout=timeout_seconds)
            break
    for _ in range(6):
        app.run(timeout=timeout_seconds)

    snapshot = _ui_snapshot_from_apptest(app)
    visible_text = _visible_text_from_snapshot(snapshot)
    common.write_json(output_dir / "ui_snapshot.json", snapshot)
    common.write_text(output_dir / "visible_text.txt", visible_text)
    checks.update(
        {
            "provider_validation_attempted": validated_provider,
            "chat_interaction_attempted": True,
            "chat_response_visible": chat_response_visible,
            "chat_followup_response_visible": followup_response_visible,
            "status_updates_observed": (
                status_updates_observed or followup_status_updates_observed
            ),
            "research_interaction_attempted": True,
            "visible_activity_recorded": bool(snapshot["chat_message_count"])
            or bool(snapshot["warnings"])
            or any("Research session:" in text for text in snapshot["markdown"]),
        }
    )
    return {
        "name": "browser_smoke",
        "kind": "browser",
        "provider_mode": provider_mode.value,
        "automation_backend": "streamlit_apptest",
        "provider_health": provider_health,
        "status": (
            common.SCENARIO_STATUS_PASSED
            if all(checks.values())
            else common.SCENARIO_STATUS_FAILED
        ),
        "summary": (
            "Rendered the app, completed a two-turn chat exchange with status "
            "updates, and attempted one research task through the Streamlit AppTest fallback."
        ),
        "checks": checks,
        "artifacts": {
            "config": str(config_path),
            "ui_snapshot": str(output_dir / "ui_snapshot.json"),
            "visible_text": str(output_dir / "visible_text.txt"),
            "screenshot": None,
        },
        "ui_snapshot": snapshot,
    }


def _wait_for_http_ready(url: str, *, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):  # noqa: S310
                return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    return False


def _wait_for_browser_chat_activity(
    page: Any,
    *,
    marker: str,
    timeout_seconds: float,
) -> tuple[str, bool, bool]:
    deadline = time.time() + timeout_seconds
    last_visible_text = ""
    status_updates_observed = False
    while time.time() < deadline:
        last_visible_text = page.locator("body").inner_text()
        status_updates_observed = (
            status_updates_observed or _status_updates_visible(last_visible_text)
        )
        if marker in last_visible_text:
            return last_visible_text, status_updates_observed, True
        page.wait_for_timeout(500)
    return last_visible_text, status_updates_observed, False


def _wait_for_browser_turn_idle(
    page: Any,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        visible_text = page.locator("body").inner_text()
        if "Queue follow-up" not in visible_text and "Stop current turn" in visible_text:
            return True
        page.wait_for_timeout(500)
    return False


def _run_playwright_smoke(
    *,
    provider_mode: ProviderModeStrategy,
    workspace: Path,
    config_path: Path,
    state_dir: Path,
    provider_health: dict[str, Any],
    timeout_seconds: float,
    output_dir: Path,
    port: int,
    headless: bool,
) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    server_log_path = output_dir / "streamlit-server.log"
    env = dict(os.environ)
    env["LLM_TOOLS_STREAMLIT_ASSISTANT_STATE_DIR"] = str(state_dir)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(common.SRC_ROOT)
        if not current_pythonpath
        else f"{common.SRC_ROOT}{os.pathsep}{current_pythonpath}"
    )
    app_path = (
        common.REPO_ROOT
        / "src"
        / "llm_tools"
        / "apps"
        / "streamlit_assistant"
        / "app.py"
    )
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--",
        str(workspace),
        "--config",
        str(config_path),
    ]
    with server_log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=common.REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    url = f"http://127.0.0.1:{port}"
    try:
        if not _wait_for_http_ready(url, timeout_seconds=timeout_seconds):
            raise RuntimeError("Streamlit app did not become ready before the timeout.")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=headless)
            page = browser.new_page()
            page.goto(
                url, wait_until="networkidle", timeout=int(timeout_seconds * 1000)
            )
            page.get_by_role("button", name="Validate provider connection").click()
            page.wait_for_timeout(1000)

            screenshot_path = output_dir / "browser-smoke.png"
            dom_path = output_dir / "dom_snapshot.html"
            visible_text_path = output_dir / "visible_text.txt"

            if provider_health.get("ok", False):
                page.get_by_label("Message the assistant").fill(
                    CHAT_SMOKE_PROMPT
                )
                page.get_by_role("button", name="Send").click()
                visible_text, status_updates_observed, chat_response_visible = (
                    _wait_for_browser_chat_activity(
                        page,
                        marker=CHAT_SMOKE_MARKER,
                        timeout_seconds=min(timeout_seconds, 90.0),
                    )
                )
                _wait_for_browser_turn_idle(
                    page,
                    timeout_seconds=min(timeout_seconds, 30.0),
                )
                page.get_by_label("Message the assistant").fill(CHAT_FOLLOWUP_PROMPT)
                page.get_by_role("button", name="Send").click()
                (
                    visible_text,
                    followup_status_updates_observed,
                    followup_response_visible,
                ) = _wait_for_browser_chat_activity(
                    page,
                    marker=CHAT_FOLLOWUP_MARKER,
                    timeout_seconds=min(timeout_seconds, 90.0),
                )
                page.get_by_text("Research tasks", exact=True).click()
                page.wait_for_timeout(500)
                page.get_by_label("Create a research task").fill(
                    "Investigate how the Streamlit assistant research lane differs "
                    "from normal chat in this repository."
                )
                page.get_by_role("button", name="Start research task").click()
                page.wait_for_timeout(3000)
            else:
                visible_text = page.locator("body").inner_text()
                status_updates_observed = False
                chat_response_visible = False
                followup_status_updates_observed = False
                followup_response_visible = False

            page.screenshot(path=str(screenshot_path), full_page=True)
            dom_path.write_text(page.content(), encoding="utf-8")
            visible_text = page.locator("body").inner_text()
            visible_text_path.write_text(visible_text, encoding="utf-8")
            browser.close()

        checks = {
            "ui_rendered": True,
            "workspace_visible": str(workspace) in visible_text
            or "workspace:" in visible_text.lower(),
            "research_visible": "Create a research task" in visible_text,
            "chat_composer_visible": "Message the assistant" in visible_text,
            "provider_validation_attempted": True,
        }
        if provider_health.get("ok", False):
            checks["chat_interaction_attempted"] = True
            checks["chat_response_visible"] = chat_response_visible
            checks["chat_followup_response_visible"] = followup_response_visible
            checks["status_updates_observed"] = (
                status_updates_observed or followup_status_updates_observed
            )
            checks["research_interaction_attempted"] = True
            status = (
                common.SCENARIO_STATUS_PASSED
                if all(checks.values())
                else common.SCENARIO_STATUS_FAILED
            )
            summary = (
                "Rendered the app, completed a two-turn chat exchange with status "
                "updates, and attempted one research action through Playwright."
            )
        else:
            status = common.SCENARIO_STATUS_INFRA
            summary = (
                "Rendered the app and captured the unreachable-provider state "
                "through Playwright."
            )
        return {
            "name": "browser_smoke",
            "kind": "browser",
            "provider_mode": provider_mode.value,
            "automation_backend": "playwright",
            "provider_health": provider_health,
            "status": status,
            "summary": summary,
            "checks": checks,
            "artifacts": {
                "config": str(config_path),
                "server_log": str(server_log_path),
                "screenshot": str(screenshot_path),
                "dom_snapshot": str(dom_path),
                "visible_text": str(visible_text_path),
            },
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    """Run the UI smoke probe."""
    args = build_parser().parse_args(argv)
    selected = common.select_scenarios(
        ALL_SCENARIOS,
        scenario_args=args.scenario,
        scenarios_csv=args.scenarios,
    )
    selected_modes = common.select_provider_modes(
        mode_args=args.provider_mode,
        modes_csv=args.provider_modes,
    )
    workspace = args.workspace.expanduser().resolve()
    output_dir = common.resolve_output_dir(args.output_dir, kind="browser")

    results: list[dict[str, Any]] = []
    provider_health_reports: list[dict[str, Any]] = []
    per_mode_runs: list[dict[str, Any]] = []
    playwright_available = _playwright_available()
    for provider_mode in selected_modes:
        mode_output_dir = common.mode_output_dir(output_dir, provider_mode)
        config = common.build_assistant_config(
            workspace=workspace,
            output_dir=mode_output_dir,
            ollama_base_url=args.ollama_base_url,
            model=args.model,
            provider_mode=provider_mode,
            timeout_seconds=args.timeout_seconds,
        )
        config_path = mode_output_dir / "assistant-config.yaml"
        state_dir = mode_output_dir / "streamlit-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        common.write_generated_config(config, config_path)
        runtime = common.build_runtime_config(config, workspace=workspace)
        provider_health = common.build_provider_health(config, runtime)
        provider_health["provider_mode"] = provider_mode.value
        provider_health_reports.append(provider_health)

        mode_results: list[dict[str, Any]] = []
        for _name in selected:
            try:
                if playwright_available:
                    port = _choose_port(args.port)
                    result = _run_playwright_smoke(
                        provider_mode=provider_mode,
                        workspace=workspace,
                        config_path=config_path,
                        state_dir=state_dir,
                        provider_health=provider_health,
                        timeout_seconds=args.timeout_seconds,
                        output_dir=mode_output_dir,
                        port=port,
                        headless=args.headless,
                    )
                else:
                    result = _run_apptest_smoke(
                        provider_mode=provider_mode,
                        workspace=workspace,
                        config_path=config_path,
                        state_dir=state_dir,
                        provider_health=provider_health,
                        timeout_seconds=args.timeout_seconds,
                        output_dir=mode_output_dir,
                    )
            except Exception as exc:  # pragma: no cover - probe failure path
                result = {
                    "name": "browser_smoke",
                    "kind": "browser",
                    "provider_mode": provider_mode.value,
                    "provider_health": provider_health,
                    "status": common.SCENARIO_STATUS_FAILED,
                    "summary": "Browser smoke execution raised an exception.",
                    "failure": common.failure_payload(exc),
                    "artifacts": {"config": str(config_path)},
                }
            mode_results.append(result)
            results.append(result)
            common.write_json(mode_output_dir / "browser_smoke.json", result)

        mode_payload = {
            "run_kind": "browser_smoke",
            "workspace": str(workspace),
            "output_dir": str(mode_output_dir),
            "selected_scenarios": selected,
            "selected_provider_modes": [provider_mode.value],
            "provider_health": provider_health,
            "config_path": str(config_path),
            "results": mode_results,
        }
        per_mode_runs.append(mode_payload)
        common.write_json(mode_output_dir / "results.json", mode_payload)
        common.write_text(
            mode_output_dir / "summary.md",
            common.results_markdown(
                title=f"Browser Assistant E2E Probe ({provider_mode.value})",
                results=mode_results,
                provider_health=provider_health,
            ),
        )

    common.write_json(
        output_dir / "results.json",
        {
            "run_kind": "browser_smoke",
            "workspace": str(workspace),
            "output_dir": str(output_dir),
            "selected_scenarios": selected,
            "selected_provider_modes": [mode.value for mode in selected_modes],
            "provider_health": provider_health_reports,
            "mode_runs": per_mode_runs,
            "results": results,
        },
    )
    common.write_text(
        output_dir / "summary.md",
        common.results_markdown(
            title="Browser Assistant E2E Probe",
            results=results,
            provider_health=provider_health_reports,
        ),
    )
    print(f"Browser probe artifacts written to {output_dir}")
    return common.final_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())

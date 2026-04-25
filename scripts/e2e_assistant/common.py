"""Shared helpers for temporary assistant E2E probes."""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_tools.apps.assistant_config import (  # noqa: E402
    StreamlitAssistantConfig,
    load_streamlit_assistant_config,
)
from llm_tools.apps.streamlit_assistant.runtime import (  # noqa: E402
    _create_provider_for_runtime,
    _llm_config_for_runtime,
)
from llm_tools.apps.streamlit_models import StreamlitRuntimeConfig  # noqa: E402
from llm_tools.llm_providers import ProviderModeStrategy  # noqa: E402
from llm_tools.tool_api import SideEffectClass  # noqa: E402

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_PROVIDER_MODES = [
    ProviderModeStrategy.TOOLS,
    ProviderModeStrategy.JSON,
    ProviderModeStrategy.PROMPT_TOOLS,
]
DEFAULT_ENABLED_TOOLS = [
    "list_directory",
    "find_files",
    "get_file_info",
    "read_file",
    "search_text",
    "run_git_status",
    "run_git_diff",
    "run_git_log",
]
SCENARIO_STATUS_PASSED = "passed"
SCENARIO_STATUS_FAILED = "failed"
SCENARIO_STATUS_INFRA = "infra_unreachable"
UNAVAILABLE_LANGUAGE_HINTS = (
    "cannot access",
    "can't access",
    "do not have access",
    "don't have access",
    "unable to access",
    "not available",
    "unavailable",
    "missing credentials",
    "missing access",
    "not configured",
)


def now_slug() -> str:
    """Return a filesystem-friendly UTC timestamp."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_output_dir(output_dir: str | None, *, kind: str) -> Path:
    """Return the output directory for one probe run."""
    if output_dir is not None:
        path = Path(output_dir).expanduser().resolve()
    else:
        path = (
            Path(gettempdir()) / "llm-tools-assistant-e2e" / now_slug() / kind
        ).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_assistant_config(
    *,
    workspace: Path,
    output_dir: Path,
    ollama_base_url: str,
    model: str,
    provider_mode: ProviderModeStrategy,
    timeout_seconds: float,
) -> StreamlitAssistantConfig:
    """Build the local-only assistant config used by the probe scripts."""
    template = load_streamlit_assistant_config(
        REPO_ROOT / "examples" / "assistant_configs" / "harness-research-chat.yaml"
    )
    return template.model_copy(
        update={
            "llm": template.llm.model_copy(
                update={
                    "api_base_url": ollama_base_url,
                    "model_name": model,
                    "provider_mode_strategy": provider_mode,
                    "timeout_seconds": timeout_seconds,
                }
            ),
            "policy": template.policy.model_copy(
                update={"enabled_tools": list(DEFAULT_ENABLED_TOOLS)}
            ),
            "workspace": template.workspace.model_copy(
                update={"default_root": str(workspace.resolve())}
            ),
            "research": template.research.model_copy(
                update={
                    "enabled": True,
                    "default_max_turns": 1,
                    "default_max_tool_invocations": 4,
                    "default_max_elapsed_seconds": 45,
                    "max_recent_sessions": 4,
                    "store_dir": str((output_dir / "research-store").resolve()),
                    "include_replay_by_default": True,
                }
            ),
            "protection": template.protection.model_copy(update={"enabled": False}),
        },
        deep=True,
    )


def build_runtime_config(
    config: StreamlitAssistantConfig,
    *,
    workspace: Path,
) -> StreamlitRuntimeConfig:
    """Build the runtime controls used by the temporary probes."""
    default_approvals = set(config.policy.require_approval_for).union(
        {SideEffectClass.LOCAL_WRITE, SideEffectClass.EXTERNAL_WRITE}
    )
    return StreamlitRuntimeConfig(
        provider=config.llm.provider,
        provider_mode_strategy=config.llm.provider_mode_strategy,
        model_name=config.llm.model_name,
        api_base_url=config.llm.api_base_url,
        temperature=config.llm.temperature,
        timeout_seconds=config.llm.timeout_seconds,
        root_path=str(workspace.resolve()),
        default_workspace_root=str(workspace.resolve()),
        enabled_tools=list(DEFAULT_ENABLED_TOOLS),
        require_approval_for=default_approvals,
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=True,
        inspector_open=True,
        show_token_usage=config.ui.show_token_usage,
        show_footer_help=config.ui.show_footer_help,
        session_config=config.session.model_copy(deep=True),
        tool_limits=config.tool_limits.model_copy(deep=True),
        research=config.research.model_copy(deep=True),
        protection=config.protection.model_copy(deep=True),
    )


def select_provider_modes(
    *,
    mode_args: Sequence[str],
    modes_csv: str | None,
) -> list[ProviderModeStrategy]:
    """Return the selected provider modes in stable order."""
    chosen: list[ProviderModeStrategy] = []
    raw_values: list[str] = []
    for raw in mode_args:
        value = raw.strip()
        if value:
            raw_values.append(value)
    if modes_csv:
        for raw in modes_csv.split(","):
            value = raw.strip()
            if value:
                raw_values.append(value)
    if not raw_values:
        return list(DEFAULT_PROVIDER_MODES)
    for raw in raw_values:
        try:
            chosen.append(ProviderModeStrategy(raw))
        except ValueError as exc:
            valid = ", ".join(mode.value for mode in DEFAULT_PROVIDER_MODES)
            raise ValueError(
                f"Unknown provider mode '{raw}'. Expected one of: {valid}"
            ) from exc
    deduped: list[ProviderModeStrategy] = []
    seen: set[ProviderModeStrategy] = set()
    for mode in chosen:
        if mode in seen:
            continue
        deduped.append(mode)
        seen.add(mode)
    return deduped


def mode_output_dir(parent: Path, mode: ProviderModeStrategy) -> Path:
    """Return the artifact directory for one provider mode."""
    path = (parent / mode.value).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_provider_health(
    config: StreamlitAssistantConfig,
    runtime: StreamlitRuntimeConfig,
) -> dict[str, Any]:
    """Return provider preflight details for the configured runtime."""
    llm_config = _llm_config_for_runtime(config, runtime)
    provider = _create_provider_for_runtime(
        llm_config,
        runtime,
        api_key=None,
        model_name=runtime.model_name,
    )
    preflight = provider.preflight(
        request_params={"temperature": config.llm.temperature}
    )
    payload = preflight.model_dump(mode="json")
    payload.update(
        {
            "base_url": runtime.api_base_url,
            "model": runtime.model_name,
            "provider": runtime.provider.value,
        }
    )
    return payload


def select_scenarios(
    available: Sequence[str],
    *,
    scenario_args: Sequence[str],
    scenarios_csv: str | None,
) -> list[str]:
    """Return the selected scenario names in stable order."""
    chosen: list[str] = []
    for raw in scenario_args:
        name = raw.strip()
        if name:
            chosen.append(name)
    if scenarios_csv:
        for raw in scenarios_csv.split(","):
            name = raw.strip()
            if name:
                chosen.append(name)
    if not chosen:
        return list(available)
    available_set = set(available)
    missing = [name for name in chosen if name not in available_set]
    if missing:
        raise ValueError("Unknown scenarios: " + ", ".join(sorted(missing)))
    deduped: list[str] = []
    seen: set[str] = set()
    for name in chosen:
        if name in seen:
            continue
        deduped.append(name)
        seen.add(name)
    return deduped


def write_json(path: Path, payload: Any) -> None:
    """Write one JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    """Write one text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_generated_config(config: StreamlitAssistantConfig, path: Path) -> None:
    """Persist one reusable config file for debugging the probe run."""
    payload = config.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def dump_model(value: Any) -> Any:
    """Return a JSON-friendly representation for models and common objects."""
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return value


def failure_payload(exc: BaseException) -> dict[str, Any]:
    """Return a compact structured failure payload."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc().strip(),
    }


def final_exit_code(results: Sequence[dict[str, Any]]) -> int:
    """Return the process exit code for one probe run."""
    if not results:
        return 1
    statuses = [str(result.get("status", "")) for result in results]
    if any(status == SCENARIO_STATUS_FAILED for status in statuses):
        return 1
    if all(status == SCENARIO_STATUS_INFRA for status in statuses):
        return 2
    return 0


def results_markdown(
    *,
    title: str,
    results: Sequence[dict[str, Any]],
    provider_health: dict[str, Any] | Sequence[dict[str, Any]],
) -> str:
    """Render a human-readable summary for one probe run."""
    grouped: dict[str, list[dict[str, Any]]] = {
        SCENARIO_STATUS_PASSED: [],
        SCENARIO_STATUS_FAILED: [],
        SCENARIO_STATUS_INFRA: [],
    }
    for result in results:
        grouped.setdefault(str(result.get("status")), []).append(result)

    reports = (
        list(provider_health)
        if not isinstance(provider_health, dict)
        else [provider_health]
    )
    lines = [f"# {title}", "", "## Provider health", ""]
    for report in reports:
        mode_label = report.get("provider_mode")
        if mode_label:
            lines.append(f"### `{mode_label}`")
            lines.append("")
        lines.extend(
            [
                f"- ok: `{report.get('ok')}`",
                f"- provider: `{report.get('provider')}`",
                f"- model: `{report.get('model')}`",
                f"- base URL: `{report.get('base_url')}`",
                f"- actionable message: {report.get('actionable_message', '')}",
            ]
        )
        error_message = report.get("error_message")
        if error_message:
            lines.append(f"- error: {error_message}")
        lines.append("")
    if lines[-1] == "":
        lines.pop()

    for section, heading in (
        (SCENARIO_STATUS_PASSED, "Passed"),
        (SCENARIO_STATUS_FAILED, "Failed"),
        (SCENARIO_STATUS_INFRA, "Infra Unreachable"),
    ):
        items = grouped.get(section, [])
        lines.extend(["", f"## {heading}", ""])
        if not items:
            lines.append("- none")
            continue
        for item in items:
            summary = (
                item.get("summary") or item.get("failure", {}).get("message") or ""
            )
            mode_suffix = (
                f", mode={item.get('provider_mode')}"
                if item.get("provider_mode")
                else ""
            )
            lines.append(
                f"- `{item.get('name')}` ({item.get('kind')}{mode_suffix}): {summary}"
            )
    return "\n".join(lines).rstrip() + "\n"


def transcript_entry(role: str, text: str, **extra: Any) -> dict[str, Any]:
    """Create one serialized transcript entry."""
    payload: dict[str, Any] = {"role": role, "text": text}
    payload.update(extra)
    return payload


def tool_sequence_from_inspector(
    tool_executions: Sequence[dict[str, Any]],
) -> list[str]:
    """Return tool names in execution order from inspector payloads."""
    sequence: list[str] = []
    for record in tool_executions:
        tool_name = record.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            sequence.append(tool_name)
    return sequence


def only_local_tools(tool_sequence: Sequence[str]) -> bool:
    """Return whether the executed tool sequence stayed inside the local allowlist."""
    return all(tool_name in DEFAULT_ENABLED_TOOLS for tool_name in tool_sequence)


def looks_like_remote_unavailable(answer_text: str | None) -> bool:
    """Return whether an answer clearly says remote data is unavailable."""
    if not answer_text:
        return False
    lowered = answer_text.lower()
    return any(hint in lowered for hint in UNAVAILABLE_LANGUAGE_HINTS)


def visible_text_block(parts: Sequence[str]) -> str:
    """Return a compact visible-text artifact."""
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(cleaned) + ("\n" if cleaned else "")

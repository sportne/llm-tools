from __future__ import annotations

import sys
from pathlib import Path

from llm_tools.llm_providers import ProviderModeStrategy

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "e2e_assistant"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common  # noqa: E402
import run_backend_matrix as backend_matrix  # noqa: E402


def _backend_probe_config(tmp_path: Path) -> tuple[object, object]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text(
        "# Probe Project\n\nThis workspace exists for backend matrix tests.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"
    config = common.build_assistant_config(
        workspace=workspace,
        output_dir=output_dir,
        ollama_base_url=common.DEFAULT_OLLAMA_BASE_URL,
        model=common.DEFAULT_MODEL,
        provider_mode=ProviderModeStrategy.JSON,
        timeout_seconds=5.0,
    )
    runtime = common.build_runtime_config(config, workspace=workspace)
    return config, runtime


def test_backend_matrix_protection_demo_uses_complete_protection_config(
    tmp_path: Path,
) -> None:
    config, runtime = _backend_probe_config(tmp_path)

    result = backend_matrix._run_chat_protection_demo(
        provider_mode=ProviderModeStrategy.JSON,
        provider_health={"ok": True},
        config=config,
        runtime=runtime,
        mode_output_dir=tmp_path / "mode",
    )

    assert result["status"] == common.SCENARIO_STATUS_PASSED
    assert all(result["checks"].values())
    assert result["protection_debug"]["protection_ready"] is True
    assert result["protection_debug"]["configured_allowed_labels"] == ["public"]
    assert result["protection_debug"]["configured_category_labels"] == [
        "public",
        "restricted",
    ]
    assert result["turns"][0]["final_response"] is not None
    assert result["turns"][0]["session_state"]["pending_protection_prompt"] is not None
    assert result["corrections_entries"][-1]["expected_sensitivity_label"] == "public"


def test_backend_matrix_research_approval_uses_controller_compatible_provider(
    tmp_path: Path,
) -> None:
    config, runtime = _backend_probe_config(tmp_path)

    result = backend_matrix._run_research_approval_resume_write_flow(
        provider_mode=ProviderModeStrategy.JSON,
        provider_health={"ok": True},
        config=config,
        runtime=runtime,
        mode_output_dir=tmp_path / "mode",
    )

    assert result["status"] == common.SCENARIO_STATUS_PASSED
    assert all(result["checks"].values())
    assert result["deny_flow"]["file_exists"] is False
    assert result["approve_flow"]["file_content"] == "approved research output\n"
    assert result["approve_flow"]["tool_name_counts"]["write_file"] == 2

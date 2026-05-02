from __future__ import annotations

import sys
from pathlib import Path

from llm_tools.llm_providers import ResponseModeStrategy

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "e2e_assistant"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common  # noqa: E402
import run_gemini_agent_eval as gemini_eval  # noqa: E402


def test_gemini_default_provider_modes_include_native_tools() -> None:
    parser = gemini_eval.build_parser()
    args = parser.parse_args([])

    assert args.provider_modes.split(",") == [
        "tools",
        "json",
        "prompt_tools",
    ]


def test_backend_matrix_config_supports_custom_openai_provider(tmp_path: Path) -> None:
    config = common.build_assistant_config(
        workspace=tmp_path,
        output_dir=tmp_path / "out",
        ollama_base_url="http://127.0.0.1:11434/v1",
        api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-3-flash-preview",
        response_mode=ResponseModeStrategy.JSON,
        timeout_seconds=30.0,
        requires_bearer_token=True,
    )

    assert config.llm.provider_protocol.value == "openai_api"
    assert (
        config.llm.provider_connection.api_base_url
        == "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    assert config.llm.provider_connection.requires_bearer_token is True
    assert config.llm.selected_model == "gemini-3-flash-preview"


def test_gemini_discovery_filters_and_normalizes_text_models() -> None:
    native_payload = {
        "models": [
            {
                "name": "models/gemini-3.1-flash-lite-preview",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-3-flash-preview",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-3.1-pro-preview",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-3.1-pro-preview-customtools",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-3-pro-image-preview",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-3-count-only",
                "supportedGenerationMethods": ["countTokens"],
            },
        ]
    }
    openai_payload = {
        "data": [
            {"id": "models/gemini-3.1-flash-lite-preview"},
            {"id": "models/gemini-3-flash-preview"},
            {"id": "models/gemini-3.1-pro-preview"},
            {"id": "models/gemini-3.1-flash-image-preview"},
            {"id": "models/gemini-3.1-flash-live-preview"},
            {"id": "models/gemini-2.5-pro"},
        ]
    }

    native_ids = gemini_eval._native_text_model_ids(native_payload)
    openai_ids = gemini_eval._openai_text_model_ids(openai_payload)
    available_ids = sorted(set(native_ids).intersection(openai_ids))

    assert native_ids == [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
    ]
    assert openai_ids == [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
    ]
    assert gemini_eval._default_selected_ids(available_ids) == [
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
    ]


def test_gemini_selected_profiles_use_discovered_defaults() -> None:
    profiles = gemini_eval._selected_profiles(
        [],
        None,
        discovery={
            "selected_default_model_ids": [
                "gemini-3.1-flash-lite-preview",
                "gemini-3-flash-preview",
            ]
        },
    )

    assert [profile["model"] for profile in profiles] == [
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
    ]
    assert profiles[0]["profile"] == "small/gemini-3/reasoning"


def test_gemini_selected_profiles_honor_cli_models() -> None:
    profiles = gemini_eval._selected_profiles(
        ["gemini-3-flash-preview,gemini-3.1-pro-preview"],
        None,
        discovery={"selected_default_model_ids": ["gemini-3.1-flash-lite-preview"]},
    )

    assert [profile["model"] for profile in profiles] == [
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
    ]
    assert {profile["profile"] for profile in profiles} == {"custom"}

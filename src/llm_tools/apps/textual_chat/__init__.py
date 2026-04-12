"""Interactive repository chat app built on top of llm-tools."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_tools.apps.textual_chat.app import run_chat_app
from llm_tools.apps.textual_chat.config import load_textual_chat_config
from llm_tools.apps.textual_chat.models import ProviderPreset, TextualChatConfig

__all__ = ["main", "run_chat_app"]


def _resolve_chat_config(args: argparse.Namespace) -> TextualChatConfig:
    base_config = load_textual_chat_config(args.config)
    raw = base_config.model_dump(mode="python")
    raw.setdefault("llm", {})
    raw.setdefault("session", {})
    raw.setdefault("tool_limits", {})
    if args.provider is not None:
        raw["llm"]["provider"] = args.provider
    if args.model is not None:
        raw["llm"]["model_name"] = args.model
    if args.temperature is not None:
        raw["llm"]["temperature"] = args.temperature
    if args.api_base_url is not None:
        raw["llm"]["api_base_url"] = args.api_base_url
    for field_name in (
        "max_context_tokens",
        "max_tool_round_trips",
        "max_tool_calls_per_round",
        "max_total_tool_calls_per_turn",
    ):
        value = getattr(args, field_name)
        if value is not None:
            raw["session"][field_name] = value
    for field_name in (
        "max_entries_per_call",
        "max_recursive_depth",
        "max_search_matches",
        "max_read_lines",
        "max_file_size_characters",
        "max_tool_result_chars",
    ):
        value = getattr(args, field_name)
        if value is not None:
            raw["tool_limits"][field_name] = value
    return TextualChatConfig.model_validate(raw)


def _run_chat_interactive(args: argparse.Namespace) -> int:
    return run_chat_app(
        root_path=args.directory.resolve(),
        config=_resolve_chat_config(args),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-tools-chat",
        description="Interactive directory-scoped chat over repository files.",
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--provider",
        choices=[preset.value for preset in ProviderPreset],
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--api-base-url", type=str)
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--max-tool-round-trips", type=int)
    parser.add_argument("--max-tool-calls-per-round", type=int)
    parser.add_argument("--max-total-tool-calls-per-turn", type=int)
    parser.add_argument("--max-entries-per-call", type=int)
    parser.add_argument("--max-recursive-depth", type=int)
    parser.add_argument("--max-search-matches", type=int)
    parser.add_argument("--max-read-lines", type=int)
    parser.add_argument("--max-file-size-characters", type=int)
    parser.add_argument("--max-tool-result-chars", type=int)
    parser.set_defaults(run=_run_chat_interactive)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and launch the Textual chat app."""
    args = build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    return int(args.run(args))

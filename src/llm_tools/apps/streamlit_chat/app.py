"""Streamlit app shell for interactive repository chat."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llm_tools.apps.chat_config import (
    ProviderPreset,
    TextualChatConfig,
    load_textual_chat_config,
)
from llm_tools.apps.chat_presentation import format_citation
from llm_tools.apps.chat_prompts import build_chat_system_prompt
from llm_tools.apps.chat_runtime import (
    build_available_tool_names,
    build_chat_context,
    build_chat_executor,
    create_provider,
)
from llm_tools.tool_api import SideEffectClass, ToolPolicy
from llm_tools.workflow_api import (
    ChatFinalResponse,
    ChatSessionState,
    ChatTokenUsage,
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowTurnResult,
    run_interactive_chat_session_turn,
)
from llm_tools.workflow_api.chat_session import ModelTurnProvider

_TRANSCRIPT_STATE_SLOT = "llm_tools_streamlit_chat_transcript"  # noqa: S105
_SESSION_STATE_SLOT = "llm_tools_streamlit_chat_session_state"  # noqa: S105
_TOKEN_USAGE_STATE_SLOT = "llm_tools_streamlit_chat_token_usage"  # noqa: S105
_API_KEY_STATE_SLOT = "llm_tools_streamlit_chat_api_key"  # noqa: S105


@dataclass(slots=True)
class StreamlitTranscriptEntry:
    """One rendered transcript entry persisted in Streamlit session state."""

    role: Literal["user", "assistant", "system", "error"]
    text: str
    final_response: ChatFinalResponse | None = None


@dataclass(slots=True)
class StreamlitTurnOutcome:
    """One processed user turn plus any transcript-side system messages."""

    session_state: ChatSessionState
    transcript_entries: list[StreamlitTranscriptEntry]
    token_usage: ChatTokenUsage | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by the bootstrap and script entrypoints."""
    parser = argparse.ArgumentParser(
        prog="llm-tools-streamlit-chat",
        description="Streamlit directory-scoped chat over repository files.",
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
    return parser


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


def resolve_enabled_tool_names(config: TextualChatConfig) -> set[str]:
    """Return the session-visible fixed read-only tool set."""
    available_tools = build_available_tool_names()
    configured_tools = config.policy.enabled_tools
    if configured_tools is None:
        return available_tools
    return {tool_name for tool_name in configured_tools if tool_name in available_tools}


def process_streamlit_chat_turn(
    *,
    root_path: Path,
    config: TextualChatConfig,
    provider: ModelTurnProvider,
    session_state: ChatSessionState,
    user_message: str,
) -> StreamlitTurnOutcome:
    """Execute one repository-chat turn for the Streamlit app."""
    enabled_tools = resolve_enabled_tool_names(config)
    policy = ToolPolicy(
        allowed_tools=set(enabled_tools),
        allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
        require_approval_for=set(config.policy.require_approval_for),
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
        redaction=config.policy.redaction.model_copy(deep=True),
    )
    registry, executor = build_chat_executor(policy=policy)
    runner = run_interactive_chat_session_turn(
        user_message=user_message,
        session_state=session_state,
        executor=executor,
        provider=provider,
        system_prompt=build_chat_system_prompt(
            tool_registry=registry,
            tool_limits=config.tool_limits,
            enabled_tool_names=enabled_tools,
        ),
        base_context=build_chat_context(
            root_path=root_path,
            config=config,
            app_name="streamlit-chat",
        ),
        session_config=config.session,
        tool_limits=config.tool_limits,
        redaction_config=config.policy.redaction,
        temperature=config.llm.temperature,
    )

    transcript_entries: list[StreamlitTranscriptEntry] = []
    token_usage: ChatTokenUsage | None = None
    updated_session_state = session_state

    for event in runner:
        if isinstance(event, ChatWorkflowApprovalEvent):
            transcript_entries.append(
                StreamlitTranscriptEntry(
                    role="system",
                    text=(
                        f"Approval requested for {event.approval.tool_name}. "
                        "The Streamlit lane denies interactive approvals."
                    ),
                )
            )
            runner.resolve_pending_approval(False)
            continue
        if isinstance(event, ChatWorkflowApprovalResolvedEvent):
            resolution = {
                "approved": "Approved pending approval request.",
                "denied": "Denied pending approval request.",
                "timed_out": "Pending approval request timed out.",
                "cancelled": "Pending approval request was cancelled.",
            }[event.resolution]
            transcript_entries.append(
                StreamlitTranscriptEntry(role="system", text=resolution)
            )
            continue
        if not isinstance(event, ChatWorkflowResultEvent):
            continue

        result = ChatWorkflowTurnResult.model_validate(event.result)
        updated_session_state = result.session_state or updated_session_state
        token_usage = result.token_usage
        if result.context_warning:
            transcript_entries.append(
                StreamlitTranscriptEntry(role="system", text=result.context_warning)
            )
        if result.status == "needs_continuation" and result.continuation_reason:
            transcript_entries.append(
                StreamlitTranscriptEntry(
                    role="system",
                    text=result.continuation_reason,
                )
            )
        if result.final_response is not None:
            transcript_entries.append(
                StreamlitTranscriptEntry(
                    role="assistant",
                    text=result.final_response.answer,
                    final_response=result.final_response,
                )
            )
        elif result.status == "interrupted" and result.interruption_reason:
            transcript_entries.append(
                StreamlitTranscriptEntry(
                    role="system",
                    text=result.interruption_reason,
                )
            )
    return StreamlitTurnOutcome(
        session_state=updated_session_state,
        transcript_entries=transcript_entries,
        token_usage=token_usage,
    )


def _streamlit_module() -> Any:
    import streamlit as streamlit

    return streamlit


def _launch_streamlit_app(script_args: Sequence[str]) -> int:
    from streamlit.web import cli as streamlit_cli

    previous_argv = list(sys.argv)
    try:
        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--",
            *list(script_args),
        ]
        return int(streamlit_cli.main())
    finally:
        sys.argv = previous_argv


def _build_startup_entry(
    root_path: Path, config: TextualChatConfig
) -> StreamlitTranscriptEntry:
    return StreamlitTranscriptEntry(
        role="system",
        text=(
            f"Root: {root_path}\n"
            f"Model: {config.llm.model_name}\n"
            "Ask grounded questions about the selected repository."
        ),
    )


def _ensure_session_state(root_path: Path, config: TextualChatConfig) -> None:
    st = _streamlit_module()
    if _TRANSCRIPT_STATE_SLOT not in st.session_state:
        st.session_state[_TRANSCRIPT_STATE_SLOT] = [
            _build_startup_entry(root_path, config)
        ]
    if _SESSION_STATE_SLOT not in st.session_state:
        st.session_state[_SESSION_STATE_SLOT] = ChatSessionState()
    if _TOKEN_USAGE_STATE_SLOT not in st.session_state:
        st.session_state[_TOKEN_USAGE_STATE_SLOT] = None
    st.session_state.setdefault(_API_KEY_STATE_SLOT, "")


def _reset_session_state(root_path: Path, config: TextualChatConfig) -> None:
    st = _streamlit_module()
    st.session_state[_TRANSCRIPT_STATE_SLOT] = [_build_startup_entry(root_path, config)]
    st.session_state[_SESSION_STATE_SLOT] = ChatSessionState()
    st.session_state[_TOKEN_USAGE_STATE_SLOT] = None
    st.session_state[_API_KEY_STATE_SLOT] = ""


def _resolve_api_key(config: TextualChatConfig) -> str | None:
    st = _streamlit_module()
    metadata = config.llm.credential_prompt_metadata()
    if not metadata.expects_api_key:
        return None

    env_var = config.llm.api_key_env_var or "OPENAI_API_KEY"
    env_value = os.getenv(env_var)
    if env_value:
        return env_value

    st.sidebar.subheader("Credentials")
    api_key = st.sidebar.text_input(
        f"Enter {env_var}",
        value=st.session_state[_API_KEY_STATE_SLOT],
        type="password",
    )
    api_key_text = str(api_key)
    st.session_state[_API_KEY_STATE_SLOT] = api_key_text
    cleaned_api_key = api_key_text.strip()
    if cleaned_api_key:
        st.session_state[_API_KEY_STATE_SLOT] = ""
        return cleaned_api_key
    st.info(f"Set {env_var} or enter it in the sidebar to start chatting.")
    return None


def _render_sidebar(root_path: Path, config: TextualChatConfig) -> None:
    st = _streamlit_module()
    with st.sidebar:
        st.subheader("Session")
        st.markdown(f"**Root**: `{root_path}`")
        st.markdown(f"**Model**: `{config.llm.model_name}`")
        token_usage = st.session_state[_TOKEN_USAGE_STATE_SLOT]
        if isinstance(token_usage, ChatTokenUsage):
            st.caption(
                "Session tokens: "
                f"{token_usage.session_tokens or '-'} | "
                f"Active context: {token_usage.active_context_tokens or '-'}"
            )
        enabled_tools = sorted(resolve_enabled_tool_names(config))
        st.caption("Tools: " + ", ".join(enabled_tools))
        if st.button("Clear chat", use_container_width=True):
            _reset_session_state(root_path, config)
            st.rerun()


def _render_final_response(response: ChatFinalResponse) -> None:
    st = _streamlit_module()
    st.markdown(response.answer)
    if response.confidence is not None:
        st.caption(f"Confidence: {response.confidence:.2f}")
    if response.citations:
        st.markdown("**Citations**")
        for citation in response.citations:
            st.markdown(f"- `{format_citation(citation)}`")
            if citation.excerpt:
                st.code(citation.excerpt)
    if response.uncertainty:
        st.markdown("**Uncertainty**")
        for item in response.uncertainty:
            st.markdown(f"- {item}")
    if response.missing_information:
        st.markdown("**Missing Information**")
        for item in response.missing_information:
            st.markdown(f"- {item}")
    if response.follow_up_suggestions:
        st.markdown("**Follow-up Suggestions**")
        for item in response.follow_up_suggestions:
            st.markdown(f"- {item}")


def _render_transcript_entry(entry: StreamlitTranscriptEntry) -> None:
    st = _streamlit_module()
    if entry.role == "user":
        with st.chat_message("user"):
            st.markdown(entry.text)
        return
    with st.chat_message("assistant"):
        if entry.role == "system":
            st.caption("System")
            st.markdown(entry.text)
            return
        if entry.role == "error":
            st.caption("Error")
            st.error(entry.text)
            return
        if entry.final_response is not None:
            _render_final_response(entry.final_response)
            return
        st.markdown(entry.text)


def run_streamlit_chat_app(*, root_path: Path, config: TextualChatConfig) -> None:
    """Render the Streamlit repository chat UI."""
    st = _streamlit_module()
    st.set_page_config(page_title="llm-tools Streamlit Chat", layout="wide")
    st.title("llm-tools Streamlit Chat")
    _ensure_session_state(root_path, config)
    _render_sidebar(root_path, config)

    api_key = _resolve_api_key(config)
    if config.llm.credential_prompt_metadata().expects_api_key and api_key is None:
        for entry in st.session_state[_TRANSCRIPT_STATE_SLOT]:
            _render_transcript_entry(entry)
        return

    for entry in st.session_state[_TRANSCRIPT_STATE_SLOT]:
        _render_transcript_entry(entry)

    prompt = st.chat_input("Ask about this repository")
    if not prompt or not prompt.strip():
        return

    provider = create_provider(
        config.llm,
        api_key=api_key,
        model_name=config.llm.model_name,
    )
    st.session_state[_TRANSCRIPT_STATE_SLOT].append(
        StreamlitTranscriptEntry(role="user", text=prompt.strip())
    )
    with st.spinner("Thinking"):
        try:
            outcome = process_streamlit_chat_turn(
                root_path=root_path,
                config=config,
                provider=provider,
                session_state=st.session_state[_SESSION_STATE_SLOT],
                user_message=prompt.strip(),
            )
        except Exception as exc:
            st.session_state[_TRANSCRIPT_STATE_SLOT].append(
                StreamlitTranscriptEntry(role="error", text=str(exc))
            )
        else:
            st.session_state[_SESSION_STATE_SLOT] = outcome.session_state
            st.session_state[_TOKEN_USAGE_STATE_SLOT] = outcome.token_usage
            st.session_state[_TRANSCRIPT_STATE_SLOT].extend(outcome.transcript_entries)
    st.rerun()


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the Streamlit chat app through the Streamlit CLI."""
    script_args = list(argv) if argv is not None else list(sys.argv[1:])
    return _launch_streamlit_app(script_args)


def _run_streamlit_script(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    run_streamlit_chat_app(
        root_path=args.directory.resolve(),
        config=_resolve_chat_config(args),
    )


if __name__ == "__main__":
    _run_streamlit_script()

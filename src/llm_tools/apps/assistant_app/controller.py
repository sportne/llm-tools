"""Compatibility facade for assistant app controller internals."""

from __future__ import annotations

import sys

from llm_tools.apps.assistant_app import controller_core as _controller_core
from llm_tools.apps.assistant_app.controller_core import (
    DEFAULT_SESSION_SECRET_TTL_SECONDS,
    PROVIDER_API_KEY_FIELD,
    NiceGUIActiveTurnHandle,
    NiceGUIChatController,
    NiceGUIQueuedEvent,
    NiceGUISessionSecret,
    NiceGUITurnState,
    SessionSecretState,
    _effective_assistant_config,
    _exposed_tool_names_for_runtime,
    _interaction_protocol,
    _nicegui_protection_is_ready,
    _remember_status,
    _seconds_between_iso,
    _serialize_workflow_event,
    _workbench_inspector_title,
    _worker_resume_harness,
    _worker_run_harness,
    _worker_run_turn,
    default_runtime_config,
)

__all__ = [
    "DEFAULT_SESSION_SECRET_TTL_SECONDS",
    "PROVIDER_API_KEY_FIELD",
    "NiceGUIActiveTurnHandle",
    "NiceGUIChatController",
    "NiceGUIQueuedEvent",
    "NiceGUISessionSecret",
    "NiceGUITurnState",
    "SessionSecretState",
    "default_runtime_config",
    "_effective_assistant_config",
    "_exposed_tool_names_for_runtime",
    "_interaction_protocol",
    "_nicegui_protection_is_ready",
    "_remember_status",
    "_seconds_between_iso",
    "_serialize_workflow_event",
    "_workbench_inspector_title",
    "_worker_resume_harness",
    "_worker_run_harness",
    "_worker_run_turn",
]

sys.modules[__name__] = _controller_core

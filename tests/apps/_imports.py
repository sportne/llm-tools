"""Helpers for importing app-facing modules with a lightweight provider stub."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


class _FakeProviderModeStrategy(StrEnum):
    AUTO = "auto"
    TOOLS = "tools"
    JSON = "json"
    MD_JSON = "md_json"


class _FakeOpenAICompatibleProvider:
    """Small provider shim that supports the app and example smoke tests."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        mode_strategy: _FakeProviderModeStrategy | str = _FakeProviderModeStrategy.AUTO,
        default_request_params: dict[str, Any] | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.mode_strategy = _FakeProviderModeStrategy(mode_strategy)
        self.default_request_params = dict(default_request_params or {})
        self._client = client
        self._async_client = async_client

    @classmethod
    def for_openai(cls, **kwargs: Any) -> _FakeOpenAICompatibleProvider:
        return cls(**kwargs)

    @classmethod
    def for_ollama(cls, **kwargs: Any) -> _FakeOpenAICompatibleProvider:
        return cls(**kwargs)

    def list_available_models(self) -> list[str]:
        if self.model.strip():
            return [self.model]
        return ["demo-model"]

    def run(
        self,
        *,
        adapter: Any,
        messages: Sequence[dict[str, Any]],
        response_model: type[Any],
        request_params: dict[str, Any] | None = None,
    ) -> Any:
        if self._client is None:
            raise RuntimeError("Fake provider requires a sync client.")
        payload = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            response_model=response_model,
            **self._merged_request_params(request_params),
        )
        return adapter.parse_model_output(payload, response_model=response_model)

    async def run_async(
        self,
        *,
        adapter: Any,
        messages: Sequence[dict[str, Any]],
        response_model: type[Any],
        request_params: dict[str, Any] | None = None,
    ) -> Any:
        if self._async_client is None:
            raise RuntimeError("Fake provider requires an async client.")
        payload = await self._async_client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            response_model=response_model,
            **self._merged_request_params(request_params),
        )
        return adapter.parse_model_output(payload, response_model=response_model)

    def _merged_request_params(
        self, request_params: dict[str, Any] | None
    ) -> dict[str, Any]:
        merged = dict(self.default_request_params)
        if request_params is not None:
            merged.update(request_params)
        return merged


def _fake_provider_module() -> ModuleType:
    module = ModuleType("llm_tools.llm_providers")
    module.OpenAICompatibleProvider = _FakeOpenAICompatibleProvider
    module.ProviderModeStrategy = _FakeProviderModeStrategy
    module.__all__ = ["OpenAICompatibleProvider", "ProviderModeStrategy"]
    return module


@contextmanager
def fake_llm_providers() -> Any:
    """Temporarily replace the provider package with a lightweight stub."""
    original = sys.modules.get("llm_tools.llm_providers")
    sys.modules["llm_tools.llm_providers"] = _fake_provider_module()
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("llm_tools.llm_providers", None)
        else:
            sys.modules["llm_tools.llm_providers"] = original


def _import_modules(
    module_names: Sequence[str],
    *,
    reset_modules: Sequence[str],
) -> tuple[ModuleType, ...]:
    for module_name in reset_modules:
        sys.modules.pop(module_name, None)
    with fake_llm_providers():
        return tuple(
            importlib.import_module(module_name) for module_name in module_names
        )


_TEXTUAL_CHAT_MODULES: SimpleNamespace | None = None
_STREAMLIT_CHAT_MODULES: SimpleNamespace | None = None
_STREAMLIT_ASSISTANT_MODULES: SimpleNamespace | None = None
_TEXTUAL_WORKBENCH_MODULES: SimpleNamespace | None = None


def import_textual_chat_modules() -> SimpleNamespace:
    """Import chat modules while shielding them from the heavy provider import."""
    global _TEXTUAL_CHAT_MODULES
    if _TEXTUAL_CHAT_MODULES is None:
        package, app, controller, main = _import_modules(
            (
                "llm_tools.apps.textual_chat",
                "llm_tools.apps.textual_chat.app",
                "llm_tools.apps.textual_chat.controller",
                "llm_tools.apps.textual_chat.__main__",
            ),
            reset_modules=(
                "llm_tools.apps.textual_chat",
                "llm_tools.apps.textual_chat.app",
                "llm_tools.apps.textual_chat.controller",
                "llm_tools.apps.textual_chat.__main__",
            ),
        )
        _TEXTUAL_CHAT_MODULES = SimpleNamespace(
            package=package,
            app=app,
            controller=controller,
            main=main,
        )
    return _TEXTUAL_CHAT_MODULES


def import_textual_workbench_modules() -> SimpleNamespace:
    """Import workbench modules while shielding them from the heavy provider import."""
    global _TEXTUAL_WORKBENCH_MODULES
    if _TEXTUAL_WORKBENCH_MODULES is None:
        package, app, controller, main = _import_modules(
            (
                "llm_tools.apps.textual_workbench",
                "llm_tools.apps.textual_workbench.app",
                "llm_tools.apps.textual_workbench.controller",
                "llm_tools.apps.textual_workbench.__main__",
            ),
            reset_modules=(
                "llm_tools.apps.textual_workbench",
                "llm_tools.apps.textual_workbench.app",
                "llm_tools.apps.textual_workbench.controller",
                "llm_tools.apps.textual_workbench.__main__",
            ),
        )
        _TEXTUAL_WORKBENCH_MODULES = SimpleNamespace(
            package=package,
            app=app,
            controller=controller,
            main=main,
        )
    return _TEXTUAL_WORKBENCH_MODULES


def import_streamlit_chat_modules() -> SimpleNamespace:
    """Import Streamlit chat modules while shielding heavy provider imports."""
    global _STREAMLIT_CHAT_MODULES
    if _STREAMLIT_CHAT_MODULES is None:
        package, app, main = _import_modules(
            (
                "llm_tools.apps.streamlit_chat",
                "llm_tools.apps.streamlit_chat.app",
                "llm_tools.apps.streamlit_chat.__main__",
            ),
            reset_modules=(
                "llm_tools.apps.streamlit_chat",
                "llm_tools.apps.streamlit_chat.app",
                "llm_tools.apps.streamlit_chat.__main__",
            ),
        )
        _STREAMLIT_CHAT_MODULES = SimpleNamespace(
            package=package,
            app=app,
            main=main,
        )
    return _STREAMLIT_CHAT_MODULES


def import_streamlit_assistant_modules() -> SimpleNamespace:
    """Import Streamlit assistant modules while shielding heavy provider imports."""
    global _STREAMLIT_ASSISTANT_MODULES
    if _STREAMLIT_ASSISTANT_MODULES is None:
        package, app, main = _import_modules(
            (
                "llm_tools.apps.streamlit_assistant",
                "llm_tools.apps.streamlit_assistant.app",
                "llm_tools.apps.streamlit_assistant.__main__",
            ),
            reset_modules=(
                "llm_tools.apps.streamlit_assistant",
                "llm_tools.apps.streamlit_assistant.app",
                "llm_tools.apps.streamlit_assistant.__main__",
            ),
        )
        _STREAMLIT_ASSISTANT_MODULES = SimpleNamespace(
            package=package,
            app=app,
            main=main,
        )
    return _STREAMLIT_ASSISTANT_MODULES


def load_module_from_path(
    path: Path,
    *,
    module_name: str,
    fake_providers_enabled: bool = False,
) -> ModuleType:
    """Load a module from a file path, optionally under the fake provider stub."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        if fake_providers_enabled:
            with fake_llm_providers():
                spec.loader.exec_module(module)
        else:
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module

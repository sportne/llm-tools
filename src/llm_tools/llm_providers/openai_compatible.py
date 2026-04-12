"""OpenAI-compatible provider client implemented with Instructor."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse

try:
    import instructor as _instructor
except Exception:  # pragma: no cover - import-time fallback path
    _instructor = None


class _InstructorModeShim(str, Enum):  # noqa: UP042
    TOOLS = "TOOLS"
    JSON = "JSON"
    MD_JSON = "MD_JSON"


class _InstructorShim:
    """Best-effort local shim for environments missing instructor."""

    Mode = _InstructorModeShim

    @staticmethod
    def from_openai(client: Any, *, mode: Any) -> Any:
        del mode
        return client


_INSTRUCTOR_FALLBACK = _InstructorShim()


class ProviderModeStrategy(str, Enum):  # noqa: UP042
    """Instructor mode strategy for provider calls."""

    AUTO = "auto"
    TOOLS = "tools"
    JSON = "json"
    MD_JSON = "md_json"


class OpenAICompatibleProvider:
    """Call an OpenAI-compatible endpoint through Instructor."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        mode_strategy: ProviderModeStrategy | str = ProviderModeStrategy.AUTO,
        default_request_params: dict[str, Any] | None = None,
        client: OpenAI | Any | None = None,
        async_client: AsyncOpenAI | Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.mode_strategy = ProviderModeStrategy(mode_strategy)
        self.default_request_params = dict(default_request_params or {})
        self.last_mode_used: ProviderModeStrategy | None = None
        self._api_key = api_key
        self._client = client
        self._async_client = async_client
        self._instructor_sync_clients: dict[ProviderModeStrategy, Any] = {}
        self._instructor_async_clients: dict[ProviderModeStrategy, Any] = {}

    @classmethod
    def for_openai(
        cls,
        *,
        model: str,
        api_key: str | None = None,
        mode_strategy: ProviderModeStrategy | str = ProviderModeStrategy.AUTO,
        default_request_params: dict[str, Any] | None = None,
        client: OpenAI | Any | None = None,
        async_client: AsyncOpenAI | Any | None = None,
    ) -> OpenAICompatibleProvider:
        """Return a provider configured for OpenAI's default endpoint."""
        return cls(
            model=model,
            api_key=api_key,
            mode_strategy=mode_strategy,
            default_request_params=default_request_params,
            client=client,
            async_client=async_client,
        )

    @classmethod
    def for_ollama(
        cls,
        *,
        model: str = "gemma4:26b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        mode_strategy: ProviderModeStrategy | str = ProviderModeStrategy.AUTO,
        default_request_params: dict[str, Any] | None = None,
        client: OpenAI | Any | None = None,
        async_client: AsyncOpenAI | Any | None = None,
    ) -> OpenAICompatibleProvider:
        """Return a provider configured for Ollama's compatibility endpoint."""
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            mode_strategy=mode_strategy,
            default_request_params=default_request_params,
            client=client,
            async_client=async_client,
        )

    def run(
        self,
        *,
        adapter: ActionEnvelopeAdapter,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Run one model turn and normalize it through the action adapter."""
        payload = self._run_with_fallback(
            messages=messages,
            response_model=response_model,
            request_params=request_params,
        )
        return adapter.parse_model_output(payload, response_model=response_model)

    def list_available_models(self) -> list[str]:
        """Return discoverable model ids from the active endpoint."""
        page = self._sync_client.models.list()
        data = getattr(page, "data", None)
        items = data if isinstance(data, list) else list(page)
        model_ids = {
            model_id.strip()
            for item in items
            for model_id in [getattr(item, "id", None)]
            if isinstance(model_id, str) and model_id.strip()
        }
        return sorted(model_ids)

    async def run_async(
        self,
        *,
        adapter: ActionEnvelopeAdapter,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Run one model turn asynchronously through Instructor."""
        payload = await self._run_with_fallback_async(
            messages=messages,
            response_model=response_model,
            request_params=request_params,
        )
        return adapter.parse_model_output(payload, response_model=response_model)

    def _run_with_fallback(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> object:
        failures: list[tuple[ProviderModeStrategy, Exception]] = []
        for mode in self._candidate_modes():
            client = self._instructor_sync_client(mode)
            try:
                payload = client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    response_model=response_model,
                    **self._merged_request_params(request_params),
                )
                self.last_mode_used = mode
                return payload
            except Exception as exc:
                if self.mode_strategy is not ProviderModeStrategy.AUTO:
                    raise
                failures.append((mode, exc))

        raise ValueError(self._fallback_error_message(failures)) from failures[-1][1]

    async def _run_with_fallback_async(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> object:
        failures: list[tuple[ProviderModeStrategy, Exception]] = []
        for mode in self._candidate_modes():
            client = self._instructor_async_client(mode)
            try:
                payload = await client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    response_model=response_model,
                    **self._merged_request_params(request_params),
                )
                self.last_mode_used = mode
                return payload
            except Exception as exc:
                if self.mode_strategy is not ProviderModeStrategy.AUTO:
                    raise
                failures.append((mode, exc))

        raise ValueError(self._fallback_error_message(failures)) from failures[-1][1]

    def _candidate_modes(self) -> list[ProviderModeStrategy]:
        if self.mode_strategy is ProviderModeStrategy.AUTO:
            return [
                ProviderModeStrategy.TOOLS,
                ProviderModeStrategy.JSON,
                ProviderModeStrategy.MD_JSON,
            ]
        return [self.mode_strategy]

    def _instructor_sync_client(self, mode: ProviderModeStrategy) -> Any:
        if mode not in self._instructor_sync_clients:
            instructor_module = self._require_instructor()
            base_client = self._sync_client
            if self._should_bypass_instructor_sync(base_client, instructor_module):
                self._instructor_sync_clients[mode] = base_client
            else:
                wrapped_client = instructor_module.from_openai(
                    base_client,
                    mode=self._resolve_instructor_mode(mode),
                )
                self._instructor_sync_clients[mode] = (
                    wrapped_client
                    if self._is_compatible_chat_client(wrapped_client)
                    else base_client
                )
        return self._instructor_sync_clients[mode]

    def _instructor_async_client(self, mode: ProviderModeStrategy) -> Any:
        if mode not in self._instructor_async_clients:
            instructor_module = self._require_instructor()
            base_client = self._async_client_instance
            if self._should_bypass_instructor_async(base_client, instructor_module):
                self._instructor_async_clients[mode] = base_client
            else:
                wrapped_client = instructor_module.from_openai(
                    base_client,
                    mode=self._resolve_instructor_mode(mode),
                )
                self._instructor_async_clients[mode] = (
                    wrapped_client
                    if self._is_compatible_chat_client(wrapped_client)
                    else base_client
                )
        return self._instructor_async_clients[mode]

    def _resolve_instructor_mode(self, mode: ProviderModeStrategy) -> Any:
        instructor_module = self._require_instructor()
        mode_name = {
            ProviderModeStrategy.TOOLS: "TOOLS",
            ProviderModeStrategy.JSON: "JSON",
            ProviderModeStrategy.MD_JSON: "MD_JSON",
        }[mode]
        try:
            return getattr(instructor_module.Mode, mode_name)
        except AttributeError as exc:  # pragma: no cover - defensive path
            raise ValueError(
                f"Instructor mode '{mode_name}' is not available."
            ) from exc

    def _require_instructor(self) -> Any:
        if _instructor is not None:
            return _instructor
        return _INSTRUCTOR_FALLBACK

    def _should_bypass_instructor_sync(
        self, client: Any, instructor_module: Any
    ) -> bool:
        # Real instructor only supports OpenAI SDK clients. Keep tests/examples
        # with fake local clients working by bypassing wrapping.
        return self._is_real_instructor_module(instructor_module) and not isinstance(
            client, OpenAI
        )

    def _should_bypass_instructor_async(
        self, client: Any, instructor_module: Any
    ) -> bool:
        # Real instructor only supports OpenAI SDK clients. Keep tests/examples
        # with fake local clients working by bypassing wrapping.
        return self._is_real_instructor_module(instructor_module) and not isinstance(
            client, AsyncOpenAI
        )

    @staticmethod
    def _is_real_instructor_module(instructor_module: Any) -> bool:
        return getattr(instructor_module, "__name__", "") == "instructor"

    @staticmethod
    def _is_compatible_chat_client(client: Any) -> bool:
        chat = getattr(client, "chat", None)
        if chat is None:
            return False
        completions = getattr(chat, "completions", None)
        if completions is None:
            return False
        return callable(getattr(completions, "create", None))

    def _fallback_error_message(
        self, failures: list[tuple[ProviderModeStrategy, Exception]]
    ) -> str:
        details = ", ".join(
            f"{mode.value}: {type(exc).__name__}" for mode, exc in failures
        )
        return f"All provider mode attempts failed ({details})."

    def _merged_request_params(
        self, request_params: dict[str, Any] | None
    ) -> dict[str, Any]:
        merged = dict(self.default_request_params)
        if request_params is not None:
            merged.update(request_params)
        return merged

    @property
    def _sync_client(self) -> OpenAI | Any:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        return self._client

    @property
    def _async_client_instance(self) -> AsyncOpenAI | Any:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
            )
        return self._async_client

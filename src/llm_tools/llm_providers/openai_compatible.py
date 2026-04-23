"""OpenAI-compatible provider client implemented with Instructor."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from enum import Enum
from typing import Any

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field, ValidationError

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


class ProviderPreflightResult(BaseModel):
    """Typed result for one provider connectivity and mode probe."""

    ok: bool
    connection_succeeded: bool = False
    model_accepted: bool = False
    selected_mode_supported: bool = False
    model_listing_supported: bool = False
    available_models: list[str] = Field(default_factory=list)
    resolved_mode: ProviderModeStrategy | None = None
    actionable_message: str
    error_message: str | None = None


class _ProviderPreflightResponse(BaseModel):
    status: str


class StructuredOutputValidationError(ValueError):
    """Structured output could not be parsed or validated."""

    def __init__(self, message: str, *, invalid_payload: object | None = None) -> None:
        super().__init__(message)
        self.invalid_payload = invalid_payload


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
        provider_family: str = "generic",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.mode_strategy = ProviderModeStrategy(mode_strategy)
        self.default_request_params = dict(default_request_params or {})
        self.provider_family = provider_family
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
            provider_family="openai",
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
            provider_family="ollama",
        )

    def run_structured(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> object:
        """Run one structured model call without adapter-level normalization."""
        return self._run_with_fallback(
            messages=messages,
            response_model=response_model,
            request_params=request_params,
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
        payload = self.run_structured(
            messages=messages,
            response_model=response_model,
            request_params=request_params,
        )
        return adapter.parse_model_output(payload, response_model=response_model)

    def prefers_simplified_json_schema_contract(self) -> bool:
        """Return whether JSON mode should use the simplified action envelope."""
        return self._should_use_native_json_schema(ProviderModeStrategy.JSON)

    def uses_staged_schema_protocol(self) -> bool:
        """Return whether structured interaction should use staged strict schemas."""
        if (
            self.mode_strategy is ProviderModeStrategy.AUTO
            and self.provider_family == "ollama"
        ):
            return True
        return self.mode_strategy in {
            ProviderModeStrategy.JSON,
            ProviderModeStrategy.MD_JSON,
        }

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

    def preflight(
        self,
        *,
        request_params: dict[str, Any] | None = None,
    ) -> ProviderPreflightResult:
        """Probe one provider configuration for connection, model, and mode support."""
        available_models: list[str] = []
        model_listing_supported = False
        connection_succeeded = False
        error_message: str | None = None

        try:
            available_models = self.list_available_models()
            model_listing_supported = True
            connection_succeeded = True
        except Exception as exc:
            error_message = self._exception_summary(exc)

        if self.mode_strategy is ProviderModeStrategy.AUTO:
            try:
                self.run_structured(
                    messages=self._preflight_messages(),
                    response_model=_ProviderPreflightResponse,
                    request_params=request_params,
                )
            except Exception as exc:
                connection_succeeded = connection_succeeded or self._looks_connected(
                    exc
                )
                error_message = self._exception_summary(exc)
                return ProviderPreflightResult(
                    ok=False,
                    connection_succeeded=connection_succeeded,
                    model_accepted=not self._looks_like_model_error(exc),
                    selected_mode_supported=False,
                    model_listing_supported=model_listing_supported,
                    available_models=available_models,
                    resolved_mode=None,
                    actionable_message=self._preflight_error_message(
                        exc,
                        available_models=available_models,
                        model_listing_supported=model_listing_supported,
                    ),
                    error_message=error_message,
                )
            return ProviderPreflightResult(
                ok=True,
                connection_succeeded=True,
                model_accepted=True,
                selected_mode_supported=True,
                model_listing_supported=model_listing_supported,
                available_models=available_models,
                resolved_mode=self.last_mode_used,
                actionable_message=(
                    "Model connection is ready for this session. "
                    f"Resolved provider mode: {(self.last_mode_used or self.mode_strategy).value}."
                ),
                error_message=error_message,
            )

        selected_mode = self.mode_strategy
        try:
            self._run_in_mode(
                mode=selected_mode,
                messages=self._preflight_messages(),
                response_model=_ProviderPreflightResponse,
                request_params=request_params,
            )
        except Exception as exc:
            connection_succeeded = connection_succeeded or self._looks_connected(exc)
            error_message = self._exception_summary(exc)
            return ProviderPreflightResult(
                ok=False,
                connection_succeeded=connection_succeeded,
                model_accepted=not self._looks_like_model_error(exc),
                selected_mode_supported=False,
                model_listing_supported=model_listing_supported,
                available_models=available_models,
                resolved_mode=None,
                actionable_message=self._preflight_error_message(
                    exc,
                    available_models=available_models,
                    model_listing_supported=model_listing_supported,
                    selected_mode=selected_mode,
                ),
                error_message=error_message,
            )
        self.last_mode_used = selected_mode
        return ProviderPreflightResult(
            ok=True,
            connection_succeeded=True,
            model_accepted=True,
            selected_mode_supported=True,
            model_listing_supported=model_listing_supported,
            available_models=available_models,
            resolved_mode=selected_mode,
            actionable_message=(
                "Model connection is ready for this session. "
                f"Configured provider mode '{selected_mode.value}' works with this endpoint."
            ),
            error_message=error_message,
        )

    async def run_structured_async(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> object:
        """Run one structured model call asynchronously without adapter parsing."""
        return await self._run_with_fallback_async(
            messages=messages,
            response_model=response_model,
            request_params=request_params,
        )

    async def run_async(
        self,
        *,
        adapter: ActionEnvelopeAdapter,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Run one model turn asynchronously through Instructor."""
        payload = await self.run_structured_async(
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
        previous_client: Any | None = None
        for mode in self._candidate_modes():
            client = self._sync_execution_client(mode)
            if failures and not self._mode_attempts_are_distinct(
                previous_client, client
            ):
                break

            try:
                payload = self._create_sync_completion(
                    mode=mode,
                    client=client,
                    messages=messages,
                    response_model=response_model,
                    request_params=request_params,
                )
                self.last_mode_used = mode
                return payload
            except Exception as exc:
                if self.mode_strategy is not ProviderModeStrategy.AUTO:
                    raise
                if not self._should_retry_mode_failure(exc):
                    raise
                failures.append((mode, exc))
                previous_client = client

        raise ValueError(self._fallback_error_message(failures)) from failures[-1][1]

    async def _run_with_fallback_async(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> object:
        failures: list[tuple[ProviderModeStrategy, Exception]] = []
        previous_client: Any | None = None
        for mode in self._candidate_modes():
            client = self._async_execution_client(mode)
            if failures and not self._mode_attempts_are_distinct(
                previous_client, client
            ):
                break

            try:
                payload = await self._create_async_completion(
                    mode=mode,
                    client=client,
                    messages=messages,
                    response_model=response_model,
                    request_params=request_params,
                )
                self.last_mode_used = mode
                return payload
            except Exception as exc:
                if self.mode_strategy is not ProviderModeStrategy.AUTO:
                    raise
                if not self._should_retry_mode_failure(exc):
                    raise
                failures.append((mode, exc))
                previous_client = client

        raise ValueError(self._fallback_error_message(failures)) from failures[-1][1]

    def _run_in_mode(
        self,
        *,
        mode: ProviderModeStrategy,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> object:
        client = self._sync_execution_client(mode)
        payload = self._create_sync_completion(
            mode=mode,
            client=client,
            messages=messages,
            response_model=response_model,
            request_params=request_params,
        )
        self.last_mode_used = mode
        return payload

    def _create_sync_completion(
        self,
        *,
        mode: ProviderModeStrategy,
        client: Any,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> object:
        if self._should_use_native_json_schema(mode):
            return self._create_native_json_schema_sync(
                messages=messages,
                response_model=response_model,
                request_params=request_params,
            )
        if self._should_use_local_md_json(mode):
            return self._create_local_md_json_sync(
                messages=messages,
                response_model=response_model,
                request_params=request_params,
            )
        return client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            response_model=response_model,
            **self._merged_request_params(request_params),
        )

    async def _create_async_completion(
        self,
        *,
        mode: ProviderModeStrategy,
        client: Any,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> object:
        if self._should_use_native_json_schema(mode):
            return await self._create_native_json_schema_async(
                messages=messages,
                response_model=response_model,
                request_params=request_params,
            )
        if self._should_use_local_md_json(mode):
            return await self._create_local_md_json_async(
                messages=messages,
                response_model=response_model,
                request_params=request_params,
            )
        return await client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            response_model=response_model,
            **self._merged_request_params(request_params),
        )

    def _create_native_json_schema_sync(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> BaseModel:
        response = self._sync_client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            response_format=self._native_json_schema_payload(response_model),
            **self._merged_request_params(request_params),
        )
        return self._parse_native_json_schema_response(
            response=response,
            response_model=response_model,
        )

    def _create_local_md_json_sync(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> BaseModel:
        response = self._sync_client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            **self._merged_request_params(request_params),
        )
        return self._parse_markdown_json_response(
            response=response,
            response_model=response_model,
        )

    async def _create_native_json_schema_async(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> BaseModel:
        response = await self._async_client_instance.chat.completions.create(
            model=self.model,
            messages=list(messages),
            response_format=self._native_json_schema_payload(response_model),
            **self._merged_request_params(request_params),
        )
        return self._parse_native_json_schema_response(
            response=response,
            response_model=response_model,
        )

    async def _create_local_md_json_async(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> BaseModel:
        response = await self._async_client_instance.chat.completions.create(
            model=self.model,
            messages=list(messages),
            **self._merged_request_params(request_params),
        )
        return self._parse_markdown_json_response(
            response=response,
            response_model=response_model,
        )

    @staticmethod
    def _preflight_messages() -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "Return structured output with status set to 'ok'.",
            },
            {
                "role": "user",
                "content": "Validate this endpoint, model, and response mode.",
            },
        ]

    def _preflight_error_message(
        self,
        exc: Exception,
        *,
        available_models: list[str],
        model_listing_supported: bool,
        selected_mode: ProviderModeStrategy | None = None,
    ) -> str:
        if self._looks_like_model_error(exc):
            if model_listing_supported and available_models:
                return (
                    f"The endpoint rejected model '{self.model}'. "
                    f"Choose one of the listed models: {', '.join(available_models)}."
                )
            return (
                f"The endpoint rejected model '{self.model}'. "
                "Check the configured model name for this provider."
            )
        if selected_mode is not None and self._should_retry_mode_failure(exc):
            return (
                f"The endpoint did not accept provider mode '{selected_mode.value}' for model '{self.model}'. "
                "Choose a different provider mode for this endpoint."
            )
        return (
            "Unable to validate this provider configuration. "
            f"{self._exception_summary(exc)}"
        )

    @staticmethod
    def _looks_like_model_error(exc: Exception) -> bool:
        message = str(exc).lower()
        markers = (
            "model",
            "not found",
            "unknown model",
            "does not exist",
            "invalid model",
        )
        return any(marker in message for marker in markers)

    @staticmethod
    def _looks_connected(exc: Exception) -> bool:
        message = str(exc).lower()
        disconnected_markers = (
            "connection",
            "connect",
            "timed out",
            "timeout",
            "name resolution",
            "dns",
            "refused",
            "unreachable",
        )
        return not any(marker in message for marker in disconnected_markers)

    def _candidate_modes(self) -> list[ProviderModeStrategy]:
        if self.mode_strategy is ProviderModeStrategy.AUTO:
            if self.provider_family == "ollama":
                return [
                    ProviderModeStrategy.JSON,
                    ProviderModeStrategy.MD_JSON,
                    ProviderModeStrategy.TOOLS,
                ]
            return [
                ProviderModeStrategy.TOOLS,
                ProviderModeStrategy.JSON,
                ProviderModeStrategy.MD_JSON,
            ]
        return [self.mode_strategy]

    def _should_use_native_json_schema(self, mode: ProviderModeStrategy) -> bool:
        return (
            self.provider_family == "ollama"
            and mode is ProviderModeStrategy.JSON
        )

    @staticmethod
    def _should_use_local_md_json(mode: ProviderModeStrategy) -> bool:
        return mode is ProviderModeStrategy.MD_JSON

    def _sync_execution_client(self, mode: ProviderModeStrategy) -> Any:
        if self._should_use_native_json_schema(mode):
            return self._sync_client
        return self._instructor_sync_client(mode)

    def _async_execution_client(self, mode: ProviderModeStrategy) -> Any:
        if self._should_use_native_json_schema(mode):
            return self._async_client_instance
        return self._instructor_async_client(mode)

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

    @staticmethod
    def _mode_attempts_are_distinct(
        previous_client: Any | None, current_client: Any
    ) -> bool:
        return previous_client is None or previous_client is not current_client

    @staticmethod
    def _should_retry_mode_failure(exc: Exception) -> bool:
        retry_markers = (
            "validation",
            "parse",
            "json",
            "schema",
            "retry",
            "incomplete",
        )
        for candidate in OpenAICompatibleProvider._iter_exception_chain(exc):
            if isinstance(candidate, (ValidationError, json.JSONDecodeError)):
                return True

            candidate_type = type(candidate)
            module_name = candidate_type.__module__
            if module_name.startswith("openai"):
                continue

            if module_name.startswith("pydantic"):
                return True

            if not module_name.startswith("instructor"):
                continue

            class_name = candidate_type.__name__.lower()
            if any(marker in class_name for marker in retry_markers):
                return True

            message = str(candidate).lower()
            if any(marker in message for marker in retry_markers):
                return True

        return False

    @staticmethod
    def _iter_exception_chain(exc: Exception) -> list[BaseException]:
        seen: set[int] = set()
        pending: list[BaseException] = [exc]
        chain: list[BaseException] = []
        while pending:
            current = pending.pop()
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)
            chain.append(current)

            cause = getattr(current, "__cause__", None)
            if isinstance(cause, BaseException):
                pending.append(cause)

            context = getattr(current, "__context__", None)
            if isinstance(context, BaseException):
                pending.append(context)

        return chain

    @classmethod
    def _failure_category(cls, exc: Exception) -> str:
        return (
            "schema/parse-related"
            if cls._should_retry_mode_failure(exc)
            else "transport-related"
        )

    @staticmethod
    def _exception_summary(exc: Exception) -> str:
        message = str(exc).strip()
        if not message:
            return type(exc).__name__
        return f"{type(exc).__name__}: {message}"

    def _fallback_error_message(
        self, failures: list[tuple[ProviderModeStrategy, Exception]]
    ) -> str:
        categories = {self._failure_category(exc) for _, exc in failures}
        overall_category = categories.pop() if len(categories) == 1 else "mixed"
        details = "; ".join(
            (
                f"{mode.value}: {self._failure_category(exc)} "
                f"({self._exception_summary(exc)})"
            )
            for mode, exc in failures
        )
        return (
            "All provider mode attempts failed. "
            f"Overall failure type: {overall_category}. "
            f"Tried modes: {details}."
        )

    def _merged_request_params(
        self, request_params: dict[str, Any] | None
    ) -> dict[str, Any]:
        merged = dict(self.default_request_params)
        if request_params is not None:
            merged.update(request_params)
        return merged

    @staticmethod
    def _native_json_schema_payload(
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        schema_name = response_model.__name__ or "StructuredResponse"
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_model.model_json_schema(),
            },
        }

    @classmethod
    def _parse_native_json_schema_response(
        cls,
        *,
        response: Any,
        response_model: type[BaseModel],
    ) -> BaseModel:
        raw_text = cls._native_response_text(response)
        try:
            return response_model.model_validate_json(raw_text)
        except ValidationError as exc:
            raise StructuredOutputValidationError(
                str(exc),
                invalid_payload=raw_text,
            ) from exc

    @classmethod
    def _parse_markdown_json_response(
        cls,
        *,
        response: Any,
        response_model: type[BaseModel],
    ) -> BaseModel:
        raw_text = cls._structured_response_text(response)
        try:
            return response_model.model_validate_json(raw_text)
        except ValidationError as exc:
            candidate = cls._extract_markdown_json_candidate(raw_text)
            if candidate is None or candidate.strip() == raw_text.strip():
                raise StructuredOutputValidationError(
                    str(exc),
                    invalid_payload=raw_text,
                ) from exc
            try:
                return response_model.model_validate_json(candidate)
            except ValidationError as candidate_exc:
                raise StructuredOutputValidationError(
                    str(candidate_exc),
                    invalid_payload=candidate,
                ) from candidate_exc

    @staticmethod
    def _native_response_text(response: Any) -> str:
        return OpenAICompatibleProvider._structured_response_text(response)

    @staticmethod
    def _structured_response_text(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("Structured provider response did not include choices.")
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise ValueError("Structured provider response did not include a message.")
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            text = "".join(parts).strip()
            if text:
                return text
        raise ValueError("Structured provider response did not include JSON content.")

    @classmethod
    def _extract_markdown_json_candidate(cls, text: str) -> str | None:
        for match in reversed(
            list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL))
        ):
            candidate = match.group(1).strip()
            balanced = cls._find_balanced_json_slice(candidate)
            if balanced is not None:
                return balanced
            if candidate:
                return candidate
        return cls._find_balanced_json_slice(text)

    @staticmethod
    def _find_balanced_json_slice(text: str) -> str | None:
        for start_index, char in enumerate(text):
            if char not in "[{":
                continue
            candidate = OpenAICompatibleProvider._balanced_json_from_index(
                text, start_index
            )
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _balanced_json_from_index(text: str, start_index: int) -> str | None:
        stack: list[str] = []
        in_string = False
        escaping = False
        for index in range(start_index, len(text)):
            char = text[index]
            if in_string:
                if escaping:
                    escaping = False
                    continue
                if char == "\\":
                    escaping = True
                    continue
                if char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                stack.append("}")
                continue
            if char == "[":
                stack.append("]")
                continue
            if char in "}]":
                if not stack or char != stack.pop():
                    return None
                if not stack:
                    return text[start_index : index + 1]
        return None

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

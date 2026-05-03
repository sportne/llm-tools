"""Native Ollama provider client."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import ollama
from pydantic import BaseModel, ValidationError

from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.llm_providers.openai_compatible_models import (
    ProviderPreflightResult,
    ResponseModeStrategy,
    _ProviderPreflightResponse,
)


class StructuredOutputValidationError(ValueError):
    """Structured output could not be parsed or validated."""

    def __init__(self, message: str, *, invalid_payload: object | None = None) -> None:
        super().__init__(message)
        self.invalid_payload = invalid_payload


class OllamaNativeProvider:
    """Call Ollama's native chat API."""

    def __init__(
        self,
        *,
        model: str,
        host: str | None = None,
        response_mode_strategy: ResponseModeStrategy | str = ResponseModeStrategy.AUTO,
        default_request_params: dict[str, Any] | None = None,
        client: ollama.Client | Any | None = None,
        async_client: ollama.AsyncClient | Any | None = None,
    ) -> None:
        self.model = model
        self.host = host
        self.response_mode_strategy = ResponseModeStrategy(response_mode_strategy)
        self.default_request_params = dict(default_request_params or {})
        self.last_mode_used: ResponseModeStrategy | None = None
        self._client = client
        self._async_client = async_client

    @property
    def _sync_client(self) -> Any:
        if self._client is None:
            self._client = ollama.Client(
                host=self.host, timeout=self.default_request_params.get("timeout")
            )
        return self._client

    @property
    def _async_client_instance(self) -> Any:
        if self._async_client is None:
            self._async_client = ollama.AsyncClient(
                host=self.host, timeout=self.default_request_params.get("timeout")
            )
        return self._async_client

    def list_available_models(self) -> list[str]:
        """Return discoverable model ids from the active Ollama host."""
        response = self._sync_client.list()
        models = getattr(response, "models", [])
        names = {
            name.strip()
            for item in models
            for name in [getattr(item, "model", None) or getattr(item, "name", None)]
            if isinstance(name, str) and name.strip()
        }
        return sorted(names)

    def preflight(
        self,
        *,
        request_params: dict[str, Any] | None = None,
    ) -> ProviderPreflightResult:
        """Probe one Ollama provider configuration."""
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

        try:
            if self.response_mode_strategy is ResponseModeStrategy.PROMPT_TOOLS:
                self._run_prompt_tools_preflight(request_params=request_params)
                resolved_mode = ResponseModeStrategy.PROMPT_TOOLS
            else:
                self.run_structured(
                    messages=self._preflight_messages(),
                    response_model=_ProviderPreflightResponse,
                    request_params=request_params,
                )
                resolved_mode = self.last_mode_used or ResponseModeStrategy.JSON
        except Exception as exc:
            error_message = self._exception_summary(exc)
            return ProviderPreflightResult(
                ok=False,
                connection_succeeded=connection_succeeded or self._looks_connected(exc),
                model_accepted=not self._looks_like_model_error(exc),
                selected_mode_supported=False,
                model_listing_supported=model_listing_supported,
                available_models=available_models,
                resolved_mode=None,
                actionable_message=self._preflight_error_message(exc),
                error_message=error_message,
            )

        return ProviderPreflightResult(
            ok=True,
            connection_succeeded=True,
            model_accepted=True,
            selected_mode_supported=True,
            model_listing_supported=model_listing_supported,
            available_models=available_models,
            resolved_mode=resolved_mode,
            actionable_message=(
                "Model connection is ready for this session. "
                f"Resolved response mode: {resolved_mode.value}."
            ),
            error_message=error_message,
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
        failures: list[tuple[ResponseModeStrategy, Exception]] = []
        for mode in self._candidate_modes():
            try:
                if mode is ResponseModeStrategy.TOOLS:
                    return self._run_native_tools(
                        adapter=adapter,
                        messages=messages,
                        response_model=response_model,
                        request_params=request_params,
                    )
                if mode is ResponseModeStrategy.JSON:
                    payload = self._run_json_schema(
                        messages=messages,
                        response_model=response_model,
                        request_params=request_params,
                    )
                    return adapter.parse_model_output(
                        payload, response_model=response_model
                    )
                text = self.run_text(messages=messages, request_params=request_params)
                return ParsedModelResponse(final_response=text)
            except Exception as exc:
                if self.response_mode_strategy is not ResponseModeStrategy.AUTO:
                    raise
                if not self._should_retry_mode_failure(exc):
                    raise
                failures.append((mode, exc))
        raise ValueError(self._fallback_error_message(failures)) from failures[-1][1]

    async def run_async(
        self,
        *,
        adapter: ActionEnvelopeAdapter,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Run one model turn asynchronously."""
        failures: list[tuple[ResponseModeStrategy, Exception]] = []
        for mode in self._candidate_modes():
            try:
                if mode is ResponseModeStrategy.TOOLS:
                    return await self._run_native_tools_async(
                        adapter=adapter,
                        messages=messages,
                        response_model=response_model,
                        request_params=request_params,
                    )
                if mode is ResponseModeStrategy.JSON:
                    payload = await self._run_json_schema_async(
                        messages=messages,
                        response_model=response_model,
                        request_params=request_params,
                    )
                    return adapter.parse_model_output(
                        payload, response_model=response_model
                    )
                text = await self.run_text_async(
                    messages=messages, request_params=request_params
                )
                return ParsedModelResponse(final_response=text)
            except Exception as exc:
                if self.response_mode_strategy is not ResponseModeStrategy.AUTO:
                    raise
                if not self._should_retry_mode_failure(exc):
                    raise
                failures.append((mode, exc))
        raise ValueError(self._fallback_error_message(failures)) from failures[-1][1]

    def run_structured(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> object:
        """Run one structured model call without adapter-level normalization."""
        if self.response_mode_strategy is ResponseModeStrategy.PROMPT_TOOLS:
            text = self.run_text(
                messages=self._structured_json_prompt_messages(
                    messages=messages, response_model=response_model
                ),
                request_params=request_params,
            )
            return self._parse_json_text(text, response_model=response_model)
        return self._run_json_schema(
            messages=messages,
            response_model=response_model,
            request_params=request_params,
        )

    async def run_structured_async(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> object:
        """Run one structured model call asynchronously."""
        if self.response_mode_strategy is ResponseModeStrategy.PROMPT_TOOLS:
            text = await self.run_text_async(
                messages=self._structured_json_prompt_messages(
                    messages=messages, response_model=response_model
                ),
                request_params=request_params,
            )
            return self._parse_json_text(text, response_model=response_model)
        return await self._run_json_schema_async(
            messages=messages,
            response_model=response_model,
            request_params=request_params,
        )

    def run_text(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        request_params: dict[str, Any] | None = None,
    ) -> str:
        """Run one plain Ollama chat completion and return assistant text."""
        response = self._sync_client.chat(
            model=self.model,
            messages=list(messages),
            stream=False,
            **self._chat_kwargs(request_params),
        )
        self.last_mode_used = ResponseModeStrategy.PROMPT_TOOLS
        return self._response_text(response)

    async def run_text_async(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        request_params: dict[str, Any] | None = None,
    ) -> str:
        """Run one plain Ollama chat completion asynchronously."""
        response = await self._async_client_instance.chat(
            model=self.model,
            messages=list(messages),
            stream=False,
            **self._chat_kwargs(request_params),
        )
        self.last_mode_used = ResponseModeStrategy.PROMPT_TOOLS
        return self._response_text(response)

    def prefers_simplified_json_schema_contract(self) -> bool:
        """Return whether JSON mode should use the simplified action envelope."""
        return False

    def uses_staged_schema_protocol(self) -> bool:
        """Return whether structured interaction should use staged strict schemas."""
        return self.response_mode_strategy is ResponseModeStrategy.JSON

    def uses_prompt_tool_protocol(self) -> bool:
        """Return whether agent turns should use prompt-emitted tool calls."""
        return self.response_mode_strategy is ResponseModeStrategy.PROMPT_TOOLS

    def can_fallback_to_prompt_tools(self, exc: Exception) -> bool:
        """Return whether a native-mode failure can fall back to prompt tools."""
        return (
            self.response_mode_strategy is ResponseModeStrategy.AUTO
            and self._should_retry_mode_failure(exc)
        )

    async def can_fallback_to_prompt_tools_async(self, exc: Exception) -> bool:
        """Return whether async native-mode failure can fall back to prompt tools."""
        return self.can_fallback_to_prompt_tools(exc)

    def _run_native_tools(
        self,
        *,
        adapter: ActionEnvelopeAdapter,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> ParsedModelResponse:
        response = self._sync_client.chat(
            model=self.model,
            messages=list(messages),
            tools=self._ollama_tools_from_response_model(response_model),
            stream=False,
            **self._chat_kwargs(request_params),
        )
        self.last_mode_used = ResponseModeStrategy.TOOLS
        payload = self._tool_response_payload(response)
        return adapter.parse_model_output(payload, response_model=response_model)

    async def _run_native_tools_async(
        self,
        *,
        adapter: ActionEnvelopeAdapter,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> ParsedModelResponse:
        response = await self._async_client_instance.chat(
            model=self.model,
            messages=list(messages),
            tools=self._ollama_tools_from_response_model(response_model),
            stream=False,
            **self._chat_kwargs(request_params),
        )
        self.last_mode_used = ResponseModeStrategy.TOOLS
        payload = self._tool_response_payload(response)
        return adapter.parse_model_output(payload, response_model=response_model)

    def _run_json_schema(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> BaseModel:
        response = self._sync_client.chat(
            model=self.model,
            messages=self._structured_json_prompt_messages(
                messages=messages, response_model=response_model
            ),
            format=response_model.model_json_schema(),
            stream=False,
            **self._chat_kwargs(request_params),
        )
        self.last_mode_used = ResponseModeStrategy.JSON
        return self._parse_json_text(
            self._response_text(response), response_model=response_model
        )

    async def _run_json_schema_async(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None,
    ) -> BaseModel:
        response = await self._async_client_instance.chat(
            model=self.model,
            messages=self._structured_json_prompt_messages(
                messages=messages, response_model=response_model
            ),
            format=response_model.model_json_schema(),
            stream=False,
            **self._chat_kwargs(request_params),
        )
        self.last_mode_used = ResponseModeStrategy.JSON
        return self._parse_json_text(
            self._response_text(response), response_model=response_model
        )

    def _candidate_modes(self) -> list[ResponseModeStrategy]:
        if self.response_mode_strategy is ResponseModeStrategy.AUTO:
            return [
                ResponseModeStrategy.TOOLS,
                ResponseModeStrategy.JSON,
                ResponseModeStrategy.PROMPT_TOOLS,
            ]
        return [self.response_mode_strategy]

    def _chat_kwargs(self, request_params: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(self.default_request_params)
        if request_params is not None:
            merged.update(request_params)
        timeout = merged.pop("timeout", None)
        if timeout is not None:
            # Timeout is configured on the client, not per native chat request.
            pass
        temperature = merged.pop("temperature", None)
        raw_options = merged.pop("options", None)
        options = dict(cast(Mapping[str, Any], raw_options or {}))
        if temperature is not None:
            options["temperature"] = temperature
        kwargs = dict(merged)
        if options:
            kwargs["options"] = options
        return kwargs

    @staticmethod
    def _response_text(response: object) -> str:
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(message, Mapping):
            raw_content = message.get("content")
            if isinstance(raw_content, str):
                return raw_content
        raise StructuredOutputValidationError(
            "Ollama response did not contain assistant text.",
            invalid_payload=response,
        )

    @staticmethod
    def _response_tool_calls(response: object) -> Sequence[object]:
        message = getattr(response, "message", None)
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None and isinstance(message, Mapping):
            tool_calls = message.get("tool_calls")
        if tool_calls is None:
            return []
        if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, str):
            return tool_calls
        raise StructuredOutputValidationError(
            "Ollama tool_calls payload was not a list.",
            invalid_payload=response,
        )

    @classmethod
    def _tool_response_payload(cls, response: object) -> dict[str, object]:
        tool_calls = cls._response_tool_calls(response)
        text = cls._response_text_or_none(response)
        if tool_calls:
            if text and text.strip():
                raise StructuredOutputValidationError(
                    "Ollama returned both tool calls and final-answer text.",
                    invalid_payload=response,
                )
            return {
                "actions": [
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                    }
                    for tool_name, arguments in [
                        cls._tool_call_fields(tool_call) for tool_call in tool_calls
                    ]
                ],
                "final_response": None,
            }
        if not text or not text.strip():
            raise StructuredOutputValidationError(
                "Ollama returned neither tool calls nor final-answer text.",
                invalid_payload=response,
            )
        return {"actions": [], "final_response": text}

    @staticmethod
    def _response_text_or_none(response: object) -> str | None:
        try:
            return OllamaNativeProvider._response_text(response)
        except StructuredOutputValidationError:
            return None

    @staticmethod
    def _tool_call_fields(tool_call: object) -> tuple[str, dict[str, Any]]:
        function = getattr(tool_call, "function", None)
        if function is None and isinstance(tool_call, Mapping):
            function = tool_call.get("function")
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments")
        if not isinstance(name, str) or not name.strip():
            raise StructuredOutputValidationError(
                "Ollama tool call did not include a function name.",
                invalid_payload=tool_call,
            )
        if not isinstance(arguments, Mapping):
            raise StructuredOutputValidationError(
                "Ollama tool call arguments were not an object.",
                invalid_payload=tool_call,
            )
        return name, dict(arguments)

    @staticmethod
    def _parse_json_text(text: str, *, response_model: type[BaseModel]) -> BaseModel:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StructuredOutputValidationError(
                "Ollama response did not contain valid JSON.",
                invalid_payload=text,
            ) from exc
        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            raise StructuredOutputValidationError(
                "Ollama JSON response did not match the requested schema.",
                invalid_payload=payload,
            ) from exc

    @classmethod
    def _ollama_tools_from_response_model(
        cls, response_model: type[BaseModel]
    ) -> list[dict[str, Any]]:
        schema = response_model.model_json_schema()
        defs = schema.get("$defs", {})
        if not isinstance(defs, Mapping):
            return []
        tools: list[dict[str, Any]] = []
        for definition in defs.values():
            if not isinstance(definition, Mapping):
                continue
            properties = definition.get("properties")
            if not isinstance(properties, Mapping):
                continue
            tool_name_schema = properties.get("tool_name")
            arguments_schema = properties.get("arguments")
            if not isinstance(tool_name_schema, Mapping) or not isinstance(
                arguments_schema, Mapping
            ):
                continue
            tool_name = tool_name_schema.get("const") or tool_name_schema.get("default")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            parameters = cls._resolve_schema_ref(arguments_schema, defs)
            if not isinstance(parameters, Mapping):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Invoke the {tool_name} tool.",
                        "parameters": dict(parameters),
                    },
                }
            )
        return tools

    @staticmethod
    def _resolve_schema_ref(
        schema: Mapping[str, Any], defs: Mapping[str, object]
    ) -> object:
        ref = schema.get("$ref")
        if not isinstance(ref, str):
            return schema
        prefix = "#/$defs/"
        if not ref.startswith(prefix):
            return schema
        return defs.get(ref.removeprefix(prefix), schema)

    @staticmethod
    def _structured_json_prompt_messages(
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> list[dict[str, Any]]:
        schema = json.dumps(
            response_model.model_json_schema(),
            indent=2,
            sort_keys=True,
            default=str,
        )
        return [
            *list(messages),
            {
                "role": "system",
                "content": (
                    "Return a JSON object that satisfies this schema. "
                    "Do not include prose outside the JSON object.\n"
                    f"{schema}"
                ),
            },
        ]

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

    def _run_prompt_tools_preflight(
        self,
        *,
        request_params: dict[str, Any] | None,
    ) -> None:
        text = self.run_text(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return exactly this text and nothing else: PROMPT_TOOLS_OK"
                    ),
                },
                {
                    "role": "user",
                    "content": "Validate this endpoint and model.",
                },
            ],
            request_params=request_params,
        )
        if "PROMPT_TOOLS_OK" not in text:
            raise StructuredOutputValidationError(
                "Prompt-tools preflight did not return the expected text.",
                invalid_payload=text,
            )

    @staticmethod
    def _looks_like_model_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "model",
                "not found",
                "unknown model",
                "does not exist",
                "invalid model",
            )
        )

    @staticmethod
    def _looks_connected(exc: Exception) -> bool:
        message = str(exc).lower()
        return not any(
            marker in message
            for marker in (
                "connection",
                "connect",
                "timed out",
                "timeout",
                "name resolution",
                "dns",
                "refused",
                "unreachable",
            )
        )

    @staticmethod
    def _should_retry_mode_failure(exc: Exception) -> bool:
        if isinstance(exc, (StructuredOutputValidationError, ValidationError)):
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "tool",
                "tools",
                "validation",
                "parse",
                "json",
                "schema",
                "format",
            )
        )

    @staticmethod
    def _exception_summary(exc: Exception) -> str:
        message = str(exc).strip()
        if not message:
            return type(exc).__name__
        return f"{type(exc).__name__}: {message}"

    def _preflight_error_message(self, exc: Exception) -> str:
        if self._looks_like_model_error(exc):
            return (
                f"The endpoint rejected model '{self.model}'. "
                "Check the configured model name for this Ollama host."
            )
        return (
            "Unable to validate this Ollama provider configuration. "
            f"{self._exception_summary(exc)}"
        )

    def _fallback_error_message(
        self, failures: list[tuple[ResponseModeStrategy, Exception]]
    ) -> str:
        details = "; ".join(
            f"{mode.value}: {self._exception_summary(exc)}" for mode, exc in failures
        )
        return f"All Ollama response mode attempts failed. Tried modes: {details}."


__all__ = ["OllamaNativeProvider", "StructuredOutputValidationError"]

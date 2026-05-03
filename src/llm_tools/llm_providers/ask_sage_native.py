"""Native Ask Sage provider client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError

from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.llm_providers.openai_compatible_models import (
    ProviderPreflightResult,
    ResponseModeStrategy,
    _ProviderPreflightResponse,
)


class AskSageNativeProviderError(ValueError):
    """Ask Sage native provider request or response failed."""

    def __init__(self, message: str, *, invalid_payload: object | None = None) -> None:
        super().__init__(message)
        self.invalid_payload = invalid_payload


class AskSageNativeProvider:
    """Call Ask Sage's native server API."""

    def __init__(
        self,
        *,
        model: str,
        access_token: str,
        base_url: str = "https://api.asksage.ai/server",
        response_mode_strategy: ResponseModeStrategy | str = ResponseModeStrategy.AUTO,
        request_settings: Mapping[str, Any] | None = None,
        default_request_params: dict[str, Any] | None = None,
        post_json: Callable[[str, dict[str, Any], float | None], object] | None = None,
    ) -> None:
        self.model = model
        self.access_token = access_token
        self.base_url = base_url.rstrip("/")
        self.response_mode_strategy = ResponseModeStrategy(response_mode_strategy)
        self.request_settings = dict(request_settings or {})
        self.default_request_params = dict(default_request_params or {})
        self.last_mode_used: ResponseModeStrategy | None = None
        self._post_json_override = post_json

    def list_available_models(self) -> list[str]:
        """Return discoverable model ids from the active Ask Sage server API."""
        payload = self._post_json("/get-models", {}, self._timeout_seconds())
        return self._extract_model_names(payload)

    def preflight(
        self,
        *,
        request_params: dict[str, Any] | None = None,
    ) -> ProviderPreflightResult:
        """Probe one Ask Sage provider configuration."""
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
            elif self.response_mode_strategy is ResponseModeStrategy.TOOLS:
                raise AskSageNativeProviderError(
                    "Ask Sage native protocol does not support native tools mode."
                )
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
                if mode is ResponseModeStrategy.JSON:
                    payload = self.run_structured(
                        messages=messages,
                        response_model=response_model,
                        request_params=request_params,
                    )
                    return adapter.parse_model_output(
                        payload, response_model=response_model
                    )
                if mode is ResponseModeStrategy.PROMPT_TOOLS:
                    return ParsedModelResponse(
                        final_response=self.run_text(
                            messages=messages, request_params=request_params
                        )
                    )
                raise AskSageNativeProviderError(
                    "Ask Sage native protocol does not support native tools mode."
                )
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
        return await asyncio.to_thread(
            self.run,
            adapter=adapter,
            messages=messages,
            response_model=response_model,
            request_params=request_params,
        )

    def run_structured(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> object:
        """Run one structured model call without adapter-level normalization."""
        text = self.run_text(
            messages=self._structured_json_prompt_messages(
                messages=messages,
                response_model=response_model,
            ),
            request_params=request_params,
        )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AskSageNativeProviderError(
                "Ask Sage response did not contain valid JSON.",
                invalid_payload=text,
            ) from exc
        try:
            result = response_model.model_validate(payload)
        except ValidationError as exc:
            raise AskSageNativeProviderError(
                "Ask Sage JSON response did not match the requested schema.",
                invalid_payload=payload,
            ) from exc
        self.last_mode_used = ResponseModeStrategy.JSON
        return result

    async def run_structured_async(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> object:
        """Run one structured model call asynchronously."""
        return await asyncio.to_thread(
            self.run_structured,
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
        """Run one Ask Sage query and return assistant text."""
        payload = self._query_payload(messages=messages, request_params=request_params)
        response = self._post_json("/query", payload, self._timeout_seconds())
        text = self._extract_response_text(response)
        self.last_mode_used = ResponseModeStrategy.PROMPT_TOOLS
        return text

    async def run_text_async(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        request_params: dict[str, Any] | None = None,
    ) -> str:
        """Run one Ask Sage query asynchronously."""
        return await asyncio.to_thread(
            self.run_text,
            messages=messages,
            request_params=request_params,
        )

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

    def _query_payload(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        request_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(self.default_request_params)
        if request_params is not None:
            merged.update(request_params)
        temperature = merged.pop("temperature", None)
        payload = dict(self.request_settings)
        payload.setdefault("persona", 1)
        payload.setdefault("dataset", "none")
        payload["message"] = self._render_messages(messages)
        payload["model"] = self.model
        if temperature is not None:
            payload["temperature"] = temperature
        for key in ("limit_references", "live"):
            if key in merged:
                payload[key] = merged[key]
        return payload

    @staticmethod
    def _render_messages(messages: Sequence[dict[str, Any]]) -> str:
        rendered: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip() or "user"
            content = str(message.get("content", "")).strip()
            if content:
                rendered.append(f"{role}: {content}")
        return "\n\n".join(rendered)

    def _post_json(
        self, path: str, payload: dict[str, Any], timeout: float | None
    ) -> object:
        if self._post_json_override is not None:
            return self._post_json_override(path, payload, timeout)
        url = urljoin(f"{self.base_url}/", path.lstrip("/"))
        request = Request(  # noqa: S310
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-access-tokens": self.access_token,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout or 60.0) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise AskSageNativeProviderError(
                "Ask Sage native API request failed."
            ) from exc

    def _timeout_seconds(self) -> float | None:
        timeout = self.default_request_params.get("timeout")
        if isinstance(timeout, int | float):
            return float(timeout)
        return None

    @staticmethod
    def _extract_model_names(payload: object) -> list[str]:
        candidates: list[object]
        if isinstance(payload, Mapping):
            raw = (
                payload.get("models")
                or payload.get("data")
                or payload.get("model_names")
                or payload.get("result")
            )
            candidates = raw if isinstance(raw, list) else [raw]
        elif isinstance(payload, list):
            candidates = payload
        else:
            candidates = []
        names: set[str] = set()
        for item in candidates:
            if isinstance(item, str) and item.strip():
                names.add(item.strip())
            if isinstance(item, Mapping):
                raw_name = item.get("model") or item.get("name") or item.get("id")
                if isinstance(raw_name, str) and raw_name.strip():
                    names.add(raw_name.strip())
        return sorted(names)

    @staticmethod
    def _extract_response_text(payload: object) -> str:
        if isinstance(payload, str) and payload.strip():
            return payload
        if isinstance(payload, Mapping):
            for key in ("response", "answer", "message", "result", "output", "text"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        raise AskSageNativeProviderError(
            "Ask Sage response did not contain assistant text.",
            invalid_payload=payload,
        )

    def _candidate_modes(self) -> list[ResponseModeStrategy]:
        if self.response_mode_strategy is ResponseModeStrategy.AUTO:
            return [ResponseModeStrategy.JSON, ResponseModeStrategy.PROMPT_TOOLS]
        return [self.response_mode_strategy]

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
            raise AskSageNativeProviderError(
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
        if isinstance(exc, (AskSageNativeProviderError, ValidationError)):
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in ("validation", "parse", "json", "schema", "format")
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
                "Check the configured model name for Ask Sage."
            )
        return (
            "Unable to validate this Ask Sage provider configuration. "
            f"{self._exception_summary(exc)}"
        )

    def _fallback_error_message(
        self, failures: list[tuple[ResponseModeStrategy, Exception]]
    ) -> str:
        details = "; ".join(
            f"{mode.value}: {self._exception_summary(exc)}" for mode, exc in failures
        )
        return f"All Ask Sage response mode attempts failed. Tried modes: {details}."


__all__ = ["AskSageNativeProvider", "AskSageNativeProviderError"]

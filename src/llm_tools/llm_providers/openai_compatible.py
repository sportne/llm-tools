"""OpenAI-compatible provider client built on the OpenAI Python SDK."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from llm_tools.llm_adapters import (
    LLMAdapter,
    NativeToolCallingAdapter,
    ParsedModelResponse,
    PromptSchemaAdapter,
    StructuredOutputAdapter,
)
from llm_tools.tool_api import ToolRegistry, ToolSpec


class OpenAICompatibleProvider:
    """Call an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        default_request_params: dict[str, Any] | None = None,
        client: OpenAI | Any | None = None,
        async_client: AsyncOpenAI | Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.default_request_params = dict(default_request_params or {})
        self._api_key = api_key
        self._client = client
        self._async_client = async_client

    @classmethod
    def for_openai(
        cls,
        *,
        model: str,
        api_key: str | None = None,
        default_request_params: dict[str, Any] | None = None,
        client: OpenAI | Any | None = None,
        async_client: AsyncOpenAI | Any | None = None,
    ) -> OpenAICompatibleProvider:
        """Return a provider configured for OpenAI's default endpoint."""
        return cls(
            model=model,
            api_key=api_key,
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
        default_request_params: dict[str, Any] | None = None,
        client: OpenAI | Any | None = None,
        async_client: AsyncOpenAI | Any | None = None,
    ) -> OpenAICompatibleProvider:
        """Return a provider configured for Ollama's compatibility endpoint."""
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            default_request_params=default_request_params,
            client=client,
            async_client=async_client,
        )

    def run_native_tool_calling(
        self,
        *,
        adapter: NativeToolCallingAdapter,
        messages: Sequence[dict[str, Any]],
        registry: ToolRegistry | None = None,
        tool_descriptions: object | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Call the model in native tool-calling mode and parse the result."""
        tools = self._resolve_tool_descriptions(
            adapter=adapter,
            registry=registry,
            tool_descriptions=tool_descriptions,
        )
        completions: Any = self._sync_client.chat.completions
        response = completions.create(
            model=self.model,
            messages=list(messages),
            tools=tools,
            **self._merged_request_params(request_params),
        )
        return adapter.parse_model_output(self._extract_message(response))

    async def run_native_tool_calling_async(
        self,
        *,
        adapter: NativeToolCallingAdapter,
        messages: Sequence[dict[str, Any]],
        registry: ToolRegistry | None = None,
        tool_descriptions: object | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Asynchronously call native tool-calling mode and parse the result."""
        tools = self._resolve_tool_descriptions(
            adapter=adapter,
            registry=registry,
            tool_descriptions=tool_descriptions,
        )
        completions: Any = self._async_client_instance.chat.completions
        response = await completions.create(
            model=self.model,
            messages=list(messages),
            tools=tools,
            **self._merged_request_params(request_params),
        )
        return adapter.parse_model_output(self._extract_message(response))

    def run_structured_output(
        self,
        *,
        adapter: StructuredOutputAdapter,
        messages: Sequence[dict[str, Any]],
        registry: ToolRegistry | None = None,
        tool_descriptions: object | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Call the model in structured-output mode and parse the result."""
        schema = self._resolve_tool_descriptions(
            adapter=adapter,
            registry=registry,
            tool_descriptions=tool_descriptions,
        )
        completions: Any = self._sync_client.chat.completions
        response = completions.create(
            model=self.model,
            messages=list(messages),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output_envelope",
                    "schema": schema,
                },
            },
            **self._merged_request_params(request_params),
        )
        return adapter.parse_model_output(self._extract_content(response))

    async def run_structured_output_async(
        self,
        *,
        adapter: StructuredOutputAdapter,
        messages: Sequence[dict[str, Any]],
        registry: ToolRegistry | None = None,
        tool_descriptions: object | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Asynchronously call structured-output mode and parse the result."""
        schema = self._resolve_tool_descriptions(
            adapter=adapter,
            registry=registry,
            tool_descriptions=tool_descriptions,
        )
        completions: Any = self._async_client_instance.chat.completions
        response = await completions.create(
            model=self.model,
            messages=list(messages),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output_envelope",
                    "schema": schema,
                },
            },
            **self._merged_request_params(request_params),
        )
        return adapter.parse_model_output(self._extract_content(response))

    def run_prompt_schema(
        self,
        *,
        adapter: PromptSchemaAdapter,
        messages: Sequence[dict[str, Any]],
        registry: ToolRegistry | None = None,
        tool_descriptions: object | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Call the model in prompt-schema mode and parse the result."""
        prompt = self._resolve_tool_descriptions(
            adapter=adapter,
            registry=registry,
            tool_descriptions=tool_descriptions,
        )
        completions: Any = self._sync_client.chat.completions
        response = completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}, *list(messages)],
            **self._merged_request_params(request_params),
        )
        return adapter.parse_model_output(self._extract_content(response))

    async def run_prompt_schema_async(
        self,
        *,
        adapter: PromptSchemaAdapter,
        messages: Sequence[dict[str, Any]],
        registry: ToolRegistry | None = None,
        tool_descriptions: object | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Asynchronously call prompt-schema mode and parse the result."""
        prompt = self._resolve_tool_descriptions(
            adapter=adapter,
            registry=registry,
            tool_descriptions=tool_descriptions,
        )
        completions: Any = self._async_client_instance.chat.completions
        response = await completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}, *list(messages)],
            **self._merged_request_params(request_params),
        )
        return adapter.parse_model_output(self._extract_content(response))

    def _resolve_tool_descriptions(
        self,
        *,
        adapter: LLMAdapter,
        registry: ToolRegistry | None,
        tool_descriptions: object | None,
    ) -> object:
        if tool_descriptions is not None:
            return tool_descriptions
        if registry is None:
            raise ValueError("Either tool_descriptions or registry must be provided.")
        return self._export_tools(adapter=adapter, registry=registry)

    def _export_tools(
        self,
        *,
        adapter: LLMAdapter,
        registry: ToolRegistry,
    ) -> object:
        tools = registry.list_registered_tools()
        specs: list[ToolSpec] = [tool.spec for tool in tools]
        input_models: dict[str, type[BaseModel]] = {
            tool.spec.name: tool.input_model for tool in tools
        }
        return adapter.export_tool_descriptions(specs, input_models)

    def _extract_message(self, response: object) -> object:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ValueError("Provider response is missing choices.")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("Provider response is missing a message.")
        return message

    def _extract_content(self, response: object) -> object:
        message = self._extract_message(response)
        if isinstance(message, (str, list)):
            return message

        if isinstance(message, BaseModel):
            return message.model_dump(mode="json", exclude_none=True).get("content")

        model_dump = getattr(message, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json", exclude_none=True).get("content")

        if isinstance(message, dict):
            return message.get("content")

        return getattr(message, "content", None)

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

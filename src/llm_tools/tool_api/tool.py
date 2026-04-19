"""Base class contract for concrete tools."""

from __future__ import annotations

import inspect
from abc import ABC
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from llm_tools.tool_api.execution import (
    ToolExecutionContext,
    _context_accepts_permit,
    _ExecutionPermit,
)
from llm_tools.tool_api.models import ToolSpec

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)

_DIRECT_EXECUTION_ERROR = (
    "Direct tool execution is disabled. Use ToolRuntime to execute tools."
)


class Tool(ABC, Generic[InputT, OutputT]):
    """Base class for all concrete tools."""

    spec: ClassVar[ToolSpec]
    input_model: ClassVar[type[InputT]]
    output_model: ClassVar[type[OutputT]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if inspect.isabstract(cls):
            return

        cls._validate_tool_subclass()

    @classmethod
    def _validate_tool_subclass(cls) -> None:
        spec = getattr(cls, "spec", None)
        if spec is None:
            raise TypeError(
                f"Concrete tool subclass {cls.__name__} must define class attribute "
                "'spec'."
            )
        if not isinstance(spec, ToolSpec):
            raise TypeError(
                f"Concrete tool subclass {cls.__name__} must define 'spec' as a "
                "ToolSpec instance."
            )

        input_model = getattr(cls, "input_model", None)
        if input_model is None:
            raise TypeError(
                f"Concrete tool subclass {cls.__name__} must define class attribute "
                "'input_model'."
            )
        if not isinstance(input_model, type) or not issubclass(input_model, BaseModel):
            raise TypeError(
                f"Concrete tool subclass {cls.__name__} must define 'input_model' as "
                "a BaseModel subclass."
            )

        output_model = getattr(cls, "output_model", None)
        if output_model is None:
            raise TypeError(
                f"Concrete tool subclass {cls.__name__} must define class attribute "
                "'output_model'."
            )
        if not isinstance(output_model, type) or not issubclass(
            output_model, BaseModel
        ):
            raise TypeError(
                f"Concrete tool subclass {cls.__name__} must define 'output_model' as "
                "a BaseModel subclass."
            )
        if not cls._has_sync_implementation() and not cls._has_async_implementation():
            raise TypeError(
                f"Concrete tool subclass {cls.__name__} must implement at least one "
                "execution method: '_invoke_impl' or '_ainvoke_impl'."
            )

    def invoke(
        self,
        context: ToolExecutionContext,
        args: InputT,
        *,
        _permit: _ExecutionPermit | None = None,
    ) -> OutputT:
        """Execute the tool through a runtime-issued execution context."""
        self._require_runtime_permit(context, _permit)
        return self._invoke_impl(context, args)

    async def ainvoke(
        self,
        context: ToolExecutionContext,
        args: InputT,
        *,
        _permit: _ExecutionPermit | None = None,
    ) -> OutputT:
        """Asynchronously execute the tool through the runtime boundary."""
        self._require_runtime_permit(context, _permit)
        return await self._ainvoke_impl(context, args)

    def _invoke_impl(self, context: ToolExecutionContext, args: InputT) -> OutputT:
        """Execute the tool using validated input and return the declared model."""
        del context, args
        raise NotImplementedError

    async def _ainvoke_impl(
        self,
        context: ToolExecutionContext,
        args: InputT,
    ) -> OutputT:
        """Asynchronously execute the tool and return the declared output model."""
        del context, args
        raise NotImplementedError

    @classmethod
    def _has_sync_implementation(cls) -> bool:
        """Whether the class provides a concrete synchronous implementation."""
        return cls._invoke_impl is not Tool._invoke_impl

    @classmethod
    def _has_async_implementation(cls) -> bool:
        """Whether the class provides a concrete asynchronous implementation."""
        return cls._ainvoke_impl is not Tool._ainvoke_impl

    @staticmethod
    def _require_runtime_permit(
        context: ToolExecutionContext,
        permit: _ExecutionPermit | None,
    ) -> None:
        if not _context_accepts_permit(context, permit):
            raise RuntimeError(_DIRECT_EXECUTION_ERROR)

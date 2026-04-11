"""Base class contract for concrete tools."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from llm_tools.tool_api.models import ToolContext, ToolSpec

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


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

    @abstractmethod
    def invoke(self, context: ToolContext, args: InputT) -> OutputT:
        """Execute the tool using validated input and return the declared model."""
        raise NotImplementedError

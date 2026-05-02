"""Read-file filesystem tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.filesystem._ops import read_file_impl
from llm_tools.tools.filesystem._shared import (
    append_local_source_provenance,
    require_repository_metadata,
)
from llm_tools.tools.filesystem.models import FileReadResult


class ReadFileInput(BaseModel):
    path: str
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class ReadFileOutput(FileReadResult):
    pass


class ReadFileTool(Tool[ReadFileInput, ReadFileOutput]):
    spec = ToolSpec(
        name="read_file",
        description="Read text or converted markdown content for one file.",
        tags=["filesystem", "read"],
        side_effects=SideEffectClass.LOCAL_READ,
        requires_filesystem=True,
        writes_internal_workspace_cache=True,
    )
    input_model = ReadFileInput
    output_model = ReadFileOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: ReadFileInput,
    ) -> ReadFileOutput:
        from llm_tools.tools.filesystem import tools as filesystem_tools

        root_path, _, tool_limits = require_repository_metadata(context)
        result = read_file_impl(
            root_path,
            args.path,
            tool_limits=tool_limits,
            start_char=args.start_char,
            end_char=args.end_char,
            cache_root=filesystem_tools._get_read_file_cache_root(root_path),
        )
        context.log(f"Read file '{result.resolved_path}'.")
        context.add_artifact(result.resolved_path)
        append_local_source_provenance(context, relative_path=result.resolved_path)
        return ReadFileOutput.model_validate(result.model_dump(mode="json"))

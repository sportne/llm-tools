"""Write-file filesystem tool."""

from __future__ import annotations

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tool_api.execution import get_workspace_root
from llm_tools.tools._path_utils import relative_display_path
from llm_tools.tools.filesystem._paths import resolve_writable_file_path
from llm_tools.tools.filesystem.write_file_models import (
    WriteFileInput as WriteFileInput,
)
from llm_tools.tools.filesystem.write_file_models import (
    WriteFileOutput as WriteFileOutput,
)


class WriteFileTool(Tool[WriteFileInput, WriteFileOutput]):
    spec = ToolSpec(
        name="write_file",
        description="Write text content to a file inside the workspace.",
        tags=["filesystem", "write"],
        side_effects=SideEffectClass.LOCAL_WRITE,
        idempotent=False,
        requires_filesystem=True,
    )
    input_model = WriteFileInput
    output_model = WriteFileOutput

    def _invoke_impl(
        self,
        context: ToolExecutionContext,
        args: WriteFileInput,
    ) -> WriteFileOutput:
        root_path = get_workspace_root(context)
        resolved_request = resolve_writable_file_path(root_path, args.path)
        target = resolved_request.resolved
        created = not target.exists()
        if target.exists() and not args.overwrite:
            raise FileExistsError(
                f"Path '{resolved_request.resolved_path}' already exists."
            )

        if not target.parent.exists():
            if not args.create_parents:
                parent_path = relative_display_path(root_path, target.parent)
                raise FileNotFoundError(
                    f"Parent directory '{parent_path}' does not exist."
                )
            target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(args.content, encoding=args.encoding)
        bytes_written = len(args.content.encode(args.encoding))
        context.log(f"Wrote file '{resolved_request.resolved_path}'.")
        context.add_artifact(resolved_request.resolved_path)
        return WriteFileOutput(
            path=args.path,
            resolved_path=resolved_request.resolved_path,
            bytes_written=bytes_written,
            created=created,
        )

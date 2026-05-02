"""Bitbucket file-read tool."""

from __future__ import annotations

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.atlassian._shared import (
    _BITBUCKET_ENV_KEYS,
    _REMOTE_TOOL_TIMEOUT_SECONDS,
    _append_remote_source_provenance,
    _bitbucket_file_to_text,
    _build_text_read_result,
    _get_tool_limits,
    _normalize_remote_exception,
)
from llm_tools.tools.atlassian.read_bitbucket_file_models import (
    ReadBitbucketFileInput as ReadBitbucketFileInput,
)
from llm_tools.tools.atlassian.read_bitbucket_file_models import (
    ReadBitbucketFileOutput as ReadBitbucketFileOutput,
)


class ReadBitbucketFileTool(Tool[ReadBitbucketFileInput, ReadBitbucketFileOutput]):
    spec = ToolSpec(
        name="read_bitbucket_file",
        description="Read one UTF-8 text file from Bitbucket Server/DC.",
        tags=["atlassian", "bitbucket", "read", "file"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=_REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(_BITBUCKET_ENV_KEYS),
    )
    input_model = ReadBitbucketFileInput
    output_model = ReadBitbucketFileOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: ReadBitbucketFileInput
    ) -> ReadBitbucketFileOutput:
        client = context.services.require_bitbucket().client
        tool_limits = _get_tool_limits(context)
        effective_ref = args.ref or "HEAD"
        try:
            raw_content = client.get_content_of_file(
                args.project_key,
                args.repository_slug,
                args.path,
                at=effective_ref,
            )
        except Exception as exc:
            raise _normalize_remote_exception(exc) from exc
        content, file_size_bytes, error_message = _bitbucket_file_to_text(raw_content)
        requested_path = args.path
        resolved_path = (
            f"{args.project_key}/{args.repository_slug}@{effective_ref}:{args.path}"
        )
        result = _build_text_read_result(
            requested_path=requested_path,
            resolved_path=resolved_path,
            tool_limits=tool_limits,
            content=content,
            file_size_bytes=file_size_bytes,
            status="ok" if content is not None else "unsupported",
            read_kind="text" if content is not None else "unsupported",
            error_message=error_message,
            start_char=args.start_char,
            end_char=args.end_char,
        )
        context.log(
            "Read Bitbucket file "
            f"'{args.path}' from '{args.project_key}/{args.repository_slug}'."
        )
        context.add_artifact(resolved_path)
        _append_remote_source_provenance(
            context,
            source_kind="bitbucket_file",
            source_id=resolved_path,
        )
        return ReadBitbucketFileOutput(
            project_key=args.project_key,
            repository_slug=args.repository_slug,
            ref=effective_ref,
            **result.model_dump(mode="json"),
        )

"""GitLab file-read tool."""

from __future__ import annotations

from typing import cast

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.filesystem._content import (
    effective_full_read_char_limit,
    estimate_token_count,
    is_within_character_limit,
    line_range_for_character_range,
    normalize_range,
)
from llm_tools.tools.gitlab._shared import (
    GITLAB_ENV_KEYS,
    REMOTE_TOOL_TIMEOUT_SECONDS,
    append_remote_source_provenance,
    decode_gitlab_file_content,
    get_gitlab_project,
    get_project_file,
    get_tool_limits,
    get_value,
    normalize_project_name,
)
from llm_tools.tools.gitlab.read_gitlab_file_models import (
    ReadGitLabFileInput as ReadGitLabFileInput,
)
from llm_tools.tools.gitlab.read_gitlab_file_models import (
    ReadGitLabFileOutput as ReadGitLabFileOutput,
)


class ReadGitLabFileTool(Tool[ReadGitLabFileInput, ReadGitLabFileOutput]):
    spec = ToolSpec(
        name="read_gitlab_file",
        description="Read one UTF-8 text file from a GitLab project.",
        tags=["gitlab", "read", "file"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(GITLAB_ENV_KEYS),
    )
    input_model = ReadGitLabFileInput
    output_model = ReadGitLabFileOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: ReadGitLabFileInput
    ) -> ReadGitLabFileOutput:
        client = context.services.require_gitlab().client
        project = get_gitlab_project(client, args.project)
        project_name = normalize_project_name(project, args.project)
        tool_limits = get_tool_limits(context)
        effective_ref = (
            args.ref or cast(str | None, get_value(project, "default_branch")) or "HEAD"
        )
        full_read_char_limit = effective_full_read_char_limit(tool_limits)
        file_obj = get_project_file(project, args.file_path, ref=effective_ref)
        content, file_size_bytes, error_message = decode_gitlab_file_content(file_obj)
        resolved_path = f"{project_name}@{effective_ref}:{args.file_path}"

        if content is None:
            return ReadGitLabFileOutput(
                project=project_name,
                ref=effective_ref,
                requested_path=args.file_path,
                resolved_path=resolved_path,
                read_kind="unsupported",
                status="unsupported",
                content=None,
                file_size_bytes=file_size_bytes,
                max_read_input_bytes=tool_limits.max_read_input_bytes,
                max_file_size_characters=tool_limits.max_file_size_characters,
                full_read_char_limit=full_read_char_limit,
                error_message=error_message,
            )

        character_count = len(content)
        if not is_within_character_limit(content, tool_limits=tool_limits):
            return ReadGitLabFileOutput(
                project=project_name,
                ref=effective_ref,
                requested_path=args.file_path,
                resolved_path=resolved_path,
                read_kind="text",
                status="too_large",
                content=None,
                character_count=character_count,
                file_size_bytes=file_size_bytes,
                max_read_input_bytes=tool_limits.max_read_input_bytes,
                max_file_size_characters=tool_limits.max_file_size_characters,
                full_read_char_limit=full_read_char_limit,
                estimated_token_count=estimate_token_count(content),
                error_message="File exceeds the configured readable character limit",
            )

        normalized_start, normalized_end = normalize_range(
            start_char=args.start_char,
            end_char=args.end_char,
            character_count=character_count,
        )
        truncated_end = min(normalized_end, normalized_start + full_read_char_limit)
        content_slice = content[normalized_start:truncated_end]
        truncated = truncated_end < character_count or truncated_end < normalized_end
        line_start, line_end = line_range_for_character_range(
            content,
            start_char=normalized_start,
            end_char=truncated_end,
        )

        context.log(
            f"Read GitLab file '{args.file_path}' from project '{args.project}'."
        )
        context.add_artifact(resolved_path)
        append_remote_source_provenance(
            context,
            source_kind="gitlab_file",
            source_id=resolved_path,
        )
        return ReadGitLabFileOutput(
            project=project_name,
            ref=effective_ref,
            requested_path=args.file_path,
            resolved_path=resolved_path,
            read_kind="text",
            status="ok",
            content=content_slice,
            truncated=truncated,
            content_char_count=len(content_slice),
            character_count=character_count,
            start_char=normalized_start,
            end_char=truncated_end,
            line_start=line_start,
            line_end=line_end,
            file_size_bytes=file_size_bytes,
            max_read_input_bytes=tool_limits.max_read_input_bytes,
            max_file_size_characters=tool_limits.max_file_size_characters,
            full_read_char_limit=full_read_char_limit,
            estimated_token_count=estimate_token_count(content_slice),
        )

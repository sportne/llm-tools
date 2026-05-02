"""Confluence attachment-read tool."""

from __future__ import annotations

from typing import cast

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.atlassian import _shared
from llm_tools.tools.atlassian._shared import (
    _CONFLUENCE_ENV_KEYS,
    _REMOTE_TOOL_TIMEOUT_SECONDS,
    _absolute_url,
    _append_remote_source_provenance,
    _build_attachment_read_result,
    _ensure_cached_confluence_attachment,
    _get_tool_limits,
    _get_value,
    _normalize_remote_exception,
    _resolve_confluence_attachment,
)
from llm_tools.tools.atlassian.read_confluence_attachment_models import (
    ReadConfluenceAttachmentInput as ReadConfluenceAttachmentInput,
)
from llm_tools.tools.atlassian.read_confluence_attachment_models import (
    ReadConfluenceAttachmentOutput as ReadConfluenceAttachmentOutput,
)


class ReadConfluenceAttachmentTool(
    Tool[ReadConfluenceAttachmentInput, ReadConfluenceAttachmentOutput]
):
    spec = ToolSpec(
        name="read_confluence_attachment",
        description="Read one Confluence attachment from a page.",
        tags=["atlassian", "confluence", "read", "attachment"],
        side_effects=SideEffectClass.LOCAL_WRITE,
        requires_network=True,
        requires_filesystem=True,
        writes_internal_workspace_cache=True,
        timeout_seconds=_REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(_CONFLUENCE_ENV_KEYS),
    )
    input_model = ReadConfluenceAttachmentInput
    output_model = ReadConfluenceAttachmentOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: ReadConfluenceAttachmentInput
    ) -> ReadConfluenceAttachmentOutput:
        from llm_tools.tools.atlassian import tools as atlassian_tools

        _shared._get_confluence_attachment_cache_root = (
            atlassian_tools._get_confluence_attachment_cache_root
        )
        client = context.services.require_confluence().client
        base_url = context.secrets.get_required("CONFLUENCE_BASE_URL")
        tool_limits = _get_tool_limits(context)
        try:
            page = client.get_page_by_id(args.page_id, expand="space,_links")
            attachment = _resolve_confluence_attachment(
                client,
                page_id=args.page_id,
                attachment_id=args.attachment_id,
                attachment_filename=args.attachment_filename,
            )
            cached_path = _ensure_cached_confluence_attachment(
                client,
                base_url=base_url,
                page_id=args.page_id,
                attachment=attachment,
            )
        except Exception as exc:
            raise _normalize_remote_exception(exc) from exc
        page_links = _get_value(page, "_links", {})
        page_web_url = _absolute_url(
            base_url,
            cast(
                str | None,
                _get_value(page_links, "webui", _get_value(page_links, "tinyui")),
            ),
        )
        page_title = cast(str | None, _get_value(page, "title"))
        space_key = cast(str | None, _get_value(_get_value(page, "space", {}), "key"))
        attachment_title = cast(str | None, _get_value(attachment, "title"))
        attachment_links = _get_value(attachment, "_links", {})
        attachment_url = _absolute_url(
            base_url,
            cast(str | None, _get_value(attachment_links, "download")),
        )
        result = _build_attachment_read_result(
            requested_path=(
                f"page:{args.page_id}/attachment:{args.attachment_filename or args.attachment_id}"
            ),
            resolved_path=attachment_url or cached_path.name,
            tool_limits=tool_limits,
            cached_path=cached_path,
            start_char=args.start_char,
            end_char=args.end_char,
        )
        attachment_source_id = (
            attachment_url
            or f"confluence:page:{args.page_id}:attachment:{attachment_title}"
        )
        context.log(
            "Read Confluence attachment "
            f"'{attachment_title or args.attachment_id}' from page '{args.page_id}'."
        )
        context.add_artifact(attachment_source_id)
        _append_remote_source_provenance(
            context,
            source_kind="confluence_attachment",
            source_id=attachment_source_id,
        )
        return ReadConfluenceAttachmentOutput(
            page_id=args.page_id,
            title=page_title,
            space_key=space_key,
            web_url=page_web_url,
            attachment_id=cast(str | None, _get_value(attachment, "id")),
            attachment_filename=attachment_title,
            **result.model_dump(mode="json"),
        )

"""Confluence page-read tool."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.atlassian._shared import (
    _CONFLUENCE_ENV_KEYS,
    _REMOTE_TOOL_TIMEOUT_SECONDS,
    _absolute_url,
    _append_remote_source_provenance,
    _build_text_read_result,
    _confluence_page_body,
    _get_tool_limits,
    _get_value,
    _normalize_remote_exception,
)
from llm_tools.tools.filesystem.models import FileReadResult


class ReadConfluencePageInput(BaseModel):
    page_id: str
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class ReadConfluencePageOutput(FileReadResult):
    page_id: str
    title: str | None = None
    space_key: str | None = None
    web_url: str | None = None
    representation: str | None = None


class ReadConfluencePageTool(Tool[ReadConfluencePageInput, ReadConfluencePageOutput]):
    spec = ToolSpec(
        name="read_confluence_page",
        description="Read one Confluence page body.",
        tags=["atlassian", "confluence", "read", "page"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=_REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(_CONFLUENCE_ENV_KEYS),
    )
    input_model = ReadConfluencePageInput
    output_model = ReadConfluencePageOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: ReadConfluencePageInput
    ) -> ReadConfluencePageOutput:
        client = context.services.require_confluence().client
        base_url = context.secrets.get_required("CONFLUENCE_BASE_URL")
        tool_limits = _get_tool_limits(context)
        try:
            page = client.get_page_by_id(
                args.page_id,
                expand="body.storage,space,_links",
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
        body_content, representation, body_error = _confluence_page_body(page)
        result = _build_text_read_result(
            requested_path=f"page:{args.page_id}",
            resolved_path=page_web_url or f"confluence:page:{args.page_id}",
            tool_limits=tool_limits,
            content=body_content if body_error is None else None,
            file_size_bytes=len(body_content.encode()),
            status="ok" if body_error is None else "error",
            read_kind="text",
            error_message=body_error,
            start_char=args.start_char,
            end_char=args.end_char,
        )
        page_source_id = page_web_url or f"confluence:page:{args.page_id}"
        context.log(f"Read Confluence page '{args.page_id}'.")
        context.add_artifact(page_source_id)
        _append_remote_source_provenance(
            context,
            source_kind="confluence_page",
            source_id=page_source_id,
        )
        return ReadConfluencePageOutput(
            page_id=args.page_id,
            title=page_title,
            space_key=space_key,
            web_url=page_web_url,
            representation=representation,
            **result.model_dump(mode="json"),
        )

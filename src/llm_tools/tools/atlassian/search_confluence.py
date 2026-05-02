"""Confluence search tool."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.atlassian._shared import (
    _CONFLUENCE_ENV_KEYS,
    _REMOTE_COLLECTION_LIMIT,
    _REMOTE_TOOL_TIMEOUT_SECONDS,
    _absolute_url,
    _extract_collection,
    _get_value,
    _normalize_remote_exception,
    _search_fetch_limit,
)


class SearchConfluenceInput(BaseModel):
    cql: str
    limit: int = Field(default=20, ge=1, le=_REMOTE_COLLECTION_LIMIT)


class ConfluenceSearchMatch(BaseModel):
    content_id: str
    title: str | None = None
    content_type: str | None = None
    space_key: str | None = None
    excerpt: str | None = None
    web_url: str | None = None


class SearchConfluenceOutput(BaseModel):
    cql: str
    matches: list[ConfluenceSearchMatch] = Field(default_factory=list)
    truncated: bool = False


class SearchConfluenceTool(Tool[SearchConfluenceInput, SearchConfluenceOutput]):
    spec = ToolSpec(
        name="search_confluence",
        description="Search Confluence with CQL.",
        tags=["atlassian", "confluence", "search", "read"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=_REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(_CONFLUENCE_ENV_KEYS),
    )
    input_model = SearchConfluenceInput
    output_model = SearchConfluenceOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: SearchConfluenceInput
    ) -> SearchConfluenceOutput:
        client = context.services.require_confluence().client
        base_url = context.secrets.get_required("CONFLUENCE_BASE_URL")
        fetch_limit = _search_fetch_limit(args.limit)
        try:
            payload = client.cql(args.cql, limit=fetch_limit, excerpt="highlight")
        except Exception as exc:
            raise _normalize_remote_exception(exc) from exc
        raw_matches = _extract_collection(payload)
        matches = []
        for item in raw_matches[: args.limit]:
            content = _get_value(item, "content", item)
            links = _get_value(content, "_links", _get_value(item, "_links", {}))
            matches.append(
                ConfluenceSearchMatch(
                    content_id=str(
                        _get_value(content, "id", _get_value(item, "id", ""))
                    ),
                    title=cast(
                        str | None,
                        _get_value(content, "title", _get_value(item, "title")),
                    ),
                    content_type=cast(
                        str | None,
                        _get_value(content, "type", _get_value(item, "type")),
                    ),
                    space_key=cast(
                        str | None,
                        _get_value(_get_value(content, "space", {}), "key"),
                    ),
                    excerpt=cast(str | None, _get_value(item, "excerpt")),
                    web_url=_absolute_url(
                        base_url,
                        cast(
                            str | None,
                            _get_value(links, "webui", _get_value(links, "tinyui")),
                        ),
                    ),
                )
            )
        context.log(f"Ran Confluence search for CQL '{args.cql}'.")
        return SearchConfluenceOutput(
            cql=args.cql,
            matches=matches,
            truncated=len(raw_matches) > args.limit,
        )

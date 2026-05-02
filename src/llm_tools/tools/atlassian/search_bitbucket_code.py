"""Bitbucket code-search tool."""

from __future__ import annotations

from typing import cast

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.atlassian._shared import (
    _BITBUCKET_ENV_KEYS,
    _REMOTE_TOOL_TIMEOUT_SECONDS,
    _extract_collection,
    _get_value,
    _normalize_remote_exception,
    _search_fetch_limit,
)
from llm_tools.tools.atlassian.search_bitbucket_code_models import (
    BitbucketCodeMatch as BitbucketCodeMatch,
)
from llm_tools.tools.atlassian.search_bitbucket_code_models import (
    SearchBitbucketCodeInput as SearchBitbucketCodeInput,
)
from llm_tools.tools.atlassian.search_bitbucket_code_models import (
    SearchBitbucketCodeOutput as SearchBitbucketCodeOutput,
)


class SearchBitbucketCodeTool(
    Tool[SearchBitbucketCodeInput, SearchBitbucketCodeOutput]
):
    spec = ToolSpec(
        name="search_bitbucket_code",
        description="Search Bitbucket Server/DC code within one project.",
        tags=["atlassian", "bitbucket", "search", "read"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=_REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(_BITBUCKET_ENV_KEYS),
    )
    input_model = SearchBitbucketCodeInput
    output_model = SearchBitbucketCodeOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: SearchBitbucketCodeInput
    ) -> SearchBitbucketCodeOutput:
        client = context.services.require_bitbucket().client
        fetch_limit = _search_fetch_limit(args.limit)
        try:
            payload = client.search_code(
                args.project_key, args.query, limit=fetch_limit
            )
        except Exception as exc:
            raise _normalize_remote_exception(exc) from exc
        raw_matches = _extract_collection(payload)
        matches = []
        for item in raw_matches[: args.limit]:
            matches.append(
                BitbucketCodeMatch(
                    repository_slug=cast(
                        str | None,
                        _get_value(
                            _get_value(item, "repository", {}),
                            "slug",
                            _get_value(item, "repository_slug"),
                        ),
                    ),
                    path=str(
                        _get_value(item, "path")
                        or _get_value(item, "file")
                        or _get_value(item, "filename")
                        or ""
                    ),
                    line_number=cast(
                        int | None,
                        _get_value(item, "line", _get_value(item, "lineNumber")),
                    ),
                    snippet=cast(
                        str | None,
                        _get_value(item, "content", _get_value(item, "snippet")),
                    ),
                )
            )

        context.log(
            f"Ran Bitbucket code search for '{args.query}' in project '{args.project_key}'."
        )
        return SearchBitbucketCodeOutput(
            project_key=args.project_key,
            query=args.query,
            matches=matches,
            truncated=len(raw_matches) > args.limit,
        )

"""GitLab code-search tool."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.gitlab._shared import (
    GITLAB_ENV_KEYS,
    REMOTE_TOOL_TIMEOUT_SECONDS,
    get_gitlab_project,
    get_value,
    normalize_project_name,
    search_fetch_limit,
    search_project_code,
)


class SearchGitLabCodeInput(BaseModel):
    project: str
    query: str
    ref: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


class GitLabCodeSearchMatch(BaseModel):
    project: str
    path: str
    name: str
    ref: str | None = None
    start_line: int | None = None
    snippet: str | None = None


class SearchGitLabCodeOutput(BaseModel):
    project: str
    query: str
    ref: str | None = None
    matches: list[GitLabCodeSearchMatch] = Field(default_factory=list)
    truncated: bool = False


class SearchGitLabCodeTool(Tool[SearchGitLabCodeInput, SearchGitLabCodeOutput]):
    spec = ToolSpec(
        name="search_gitlab_code",
        description="Search one GitLab project for code matches.",
        tags=["gitlab", "search", "read", "code"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(GITLAB_ENV_KEYS),
    )
    input_model = SearchGitLabCodeInput
    output_model = SearchGitLabCodeOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: SearchGitLabCodeInput
    ) -> SearchGitLabCodeOutput:
        client = context.services.require_gitlab().client
        project = get_gitlab_project(client, args.project)
        project_name = normalize_project_name(project, args.project)
        raw_matches = search_project_code(
            project,
            args.query,
            ref=args.ref,
            limit=search_fetch_limit(args.limit),
        )
        matches = [
            GitLabCodeSearchMatch(
                project=project_name,
                path=str(
                    get_value(raw, "path")
                    or get_value(raw, "filename")
                    or get_value(raw, "file_path")
                    or ""
                ),
                name=Path(
                    str(
                        get_value(raw, "path")
                        or get_value(raw, "filename")
                        or get_value(raw, "file_path")
                        or ""
                    )
                ).name,
                ref=cast(str | None, get_value(raw, "ref", args.ref)),
                start_line=cast(
                    int | None,
                    get_value(raw, "startline", get_value(raw, "start_line")),
                ),
                snippet=cast(
                    str | None,
                    get_value(raw, "data", get_value(raw, "snippet")),
                ),
            )
            for raw in raw_matches[: args.limit]
        ]
        context.log(
            f"Ran GitLab code search for '{args.query}' in project '{args.project}'."
        )
        return SearchGitLabCodeOutput(
            project=project_name,
            query=args.query,
            ref=args.ref,
            matches=matches,
            truncated=len(raw_matches) > args.limit,
        )

"""GitLab merge-request read tool."""

from __future__ import annotations

from typing import cast

from llm_tools.tool_api import SideEffectClass, Tool, ToolExecutionContext, ToolSpec
from llm_tools.tools.gitlab._shared import (
    GITLAB_ENV_KEYS,
    REMOTE_TOOL_TIMEOUT_SECONDS,
    get_gitlab_project,
    get_merge_request,
    get_merge_request_changes,
    get_merge_request_commits,
    get_value,
    normalize_project_name,
)
from llm_tools.tools.gitlab.read_gitlab_merge_request_models import (
    GitLabMergeRequestChange as GitLabMergeRequestChange,
)
from llm_tools.tools.gitlab.read_gitlab_merge_request_models import (
    GitLabMergeRequestCommit as GitLabMergeRequestCommit,
)
from llm_tools.tools.gitlab.read_gitlab_merge_request_models import (
    ReadGitLabMergeRequestInput as ReadGitLabMergeRequestInput,
)
from llm_tools.tools.gitlab.read_gitlab_merge_request_models import (
    ReadGitLabMergeRequestOutput as ReadGitLabMergeRequestOutput,
)


class ReadGitLabMergeRequestTool(
    Tool[ReadGitLabMergeRequestInput, ReadGitLabMergeRequestOutput]
):
    spec = ToolSpec(
        name="read_gitlab_merge_request",
        description="Read one GitLab merge request by IID.",
        tags=["gitlab", "read", "merge_request"],
        side_effects=SideEffectClass.EXTERNAL_READ,
        requires_network=True,
        timeout_seconds=REMOTE_TOOL_TIMEOUT_SECONDS,
        required_secrets=list(GITLAB_ENV_KEYS),
    )
    input_model = ReadGitLabMergeRequestInput
    output_model = ReadGitLabMergeRequestOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: ReadGitLabMergeRequestInput
    ) -> ReadGitLabMergeRequestOutput:
        client = context.services.require_gitlab().client
        project = get_gitlab_project(client, args.project)
        project_name = normalize_project_name(project, args.project)
        merge_request = get_merge_request(project, args.merge_request_iid)
        raw_commits = get_merge_request_commits(merge_request)
        raw_changes = get_merge_request_changes(merge_request)

        commits = [
            GitLabMergeRequestCommit(
                id=cast(str | None, get_value(commit, "id")),
                short_id=cast(str | None, get_value(commit, "short_id")),
                title=cast(
                    str | None,
                    get_value(commit, "title", get_value(commit, "message")),
                ),
                author_name=cast(str | None, get_value(commit, "author_name")),
            )
            for commit in raw_commits[: args.commit_limit]
        ]
        changed_files = [
            GitLabMergeRequestChange(
                old_path=cast(str | None, get_value(change, "old_path")),
                new_path=cast(str | None, get_value(change, "new_path")),
                new_file=bool(get_value(change, "new_file", False)),
                renamed_file=bool(get_value(change, "renamed_file", False)),
                deleted_file=bool(get_value(change, "deleted_file", False)),
                diff_excerpt=(cast(str | None, get_value(change, "diff")) or "")[:400]
                or None,
            )
            for change in raw_changes[: args.change_limit]
        ]

        context.log(
            "Read GitLab merge request "
            f"'{args.merge_request_iid}' from project '{args.project}'."
        )
        return ReadGitLabMergeRequestOutput(
            project=project_name,
            merge_request_iid=args.merge_request_iid,
            title=cast(str | None, get_value(merge_request, "title")),
            description=cast(str | None, get_value(merge_request, "description")),
            state=cast(str | None, get_value(merge_request, "state")),
            author=cast(
                str | None,
                get_value(
                    get_value(merge_request, "author", {}),
                    "name",
                    get_value(
                        get_value(merge_request, "author", {}),
                        "username",
                    ),
                ),
            ),
            source_branch=cast(str | None, get_value(merge_request, "source_branch")),
            target_branch=cast(str | None, get_value(merge_request, "target_branch")),
            web_url=cast(str | None, get_value(merge_request, "web_url")),
            commits=commits,
            commits_truncated=len(raw_commits) > args.commit_limit,
            changed_files=changed_files,
            changed_files_truncated=len(raw_changes) > args.change_limit,
        )

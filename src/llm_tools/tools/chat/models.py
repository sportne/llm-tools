"""Typed models for repository-chat tool configuration and outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DirectoryEntryType = Literal["file", "directory", "symlink", "other"]
FileReadKind = Literal["text", "markitdown", "unsupported"]
FileInfoStatus = Literal["ok", "unsupported", "error"]
FileReadStatus = Literal["ok", "too_large", "unsupported", "error"]


class ChatSourceFilters(BaseModel):
    """Directory discovery filters applied inside the configured root."""

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    include_hidden: bool = False

    @field_validator("include", "exclude")
    @classmethod
    def validate_glob_patterns(cls, value: list[str]) -> list[str]:
        cleaned = [entry.strip() for entry in value]
        if any(not entry for entry in cleaned):
            raise ValueError("source filter patterns must not be empty")
        return cleaned


class ChatSessionConfig(BaseModel):
    """Per-session context and tool-call safety limits."""

    max_context_tokens: int = 24000
    max_tool_round_trips: int = 8
    max_tool_calls_per_round: int = 4
    max_total_tool_calls_per_turn: int = 12

    @field_validator(
        "max_context_tokens",
        "max_tool_round_trips",
        "max_tool_calls_per_round",
        "max_total_tool_calls_per_turn",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chat session limits must be positive integers")
        return value


class ChatToolLimits(BaseModel):
    """Deterministic upper bounds for repository chat tools."""

    max_entries_per_call: int = 200
    max_recursive_depth: int = 12
    max_search_matches: int = 50
    max_read_lines: int = 200
    max_file_size_characters: int = 262144
    max_read_file_chars: int | None = None
    max_tool_result_chars: int = 24000

    @field_validator(
        "max_entries_per_call",
        "max_recursive_depth",
        "max_search_matches",
        "max_read_lines",
        "max_file_size_characters",
        "max_tool_result_chars",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chat tool limits must be positive integers")
        return value

    @field_validator("max_read_file_chars")
    @classmethod
    def validate_optional_positive_int(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("chat tool limits must be positive integers")
        return value


class DirectoryEntry(BaseModel):
    """One filesystem entry exposed by a chat listing tool."""

    path: str
    name: str
    entry_type: DirectoryEntryType
    depth: int
    is_hidden: bool
    is_symlink: bool


class DirectoryListingResult(BaseModel):
    """Structured result for direct or recursive directory listing."""

    requested_path: str
    resolved_path: str
    recursive: bool
    max_depth_applied: int
    entries: list[DirectoryEntry] = Field(default_factory=list)
    truncated: bool = False


class FileMatch(BaseModel):
    """One matched file returned by the find-files tool."""

    path: str
    name: str
    parent_path: str
    is_hidden: bool


class FileSearchResult(BaseModel):
    """Structured result for deterministic file search."""

    requested_path: str
    resolved_path: str
    pattern: str
    matches: list[FileMatch] = Field(default_factory=list)
    truncated: bool = False


class TextSearchMatch(BaseModel):
    """One matching line returned by the text-search tool."""

    path: str
    line_number: int
    line_text: str
    is_hidden: bool


class TextSearchResult(BaseModel):
    """Structured result for deterministic text search."""

    requested_path: str
    resolved_path: str
    query: str
    matches: list[TextSearchMatch] = Field(default_factory=list)
    truncated: bool = False


class FileInfoResult(BaseModel):
    """Structured metadata for one root-confined file."""

    requested_path: str
    resolved_path: str
    name: str
    size_bytes: int
    is_hidden: bool
    is_symlink: bool
    read_kind: FileReadKind
    status: FileInfoStatus = "ok"
    estimated_token_count: int | None = None
    character_count: int | None = None
    line_count: int | None = None
    max_file_size_characters: int
    within_size_limit: bool
    full_read_char_limit: int
    can_read_full: bool
    error_message: str | None = None


class FileInfoBatchResult(BaseModel):
    """Structured metadata for multiple root-confined files."""

    results: list[FileInfoResult] = Field(default_factory=list)


class FileReadResult(BaseModel):
    """Structured content read result for one root-confined file."""

    requested_path: str
    resolved_path: str
    read_kind: FileReadKind
    status: FileReadStatus = "ok"
    content: str | None = None
    truncated: bool = False
    content_char_count: int = 0
    character_count: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    file_size_bytes: int
    max_file_size_characters: int
    full_read_char_limit: int
    estimated_token_count: int | None = None
    error_message: str | None = None


class GetFileInfoInputShape(BaseModel):
    """Reusable validation shape for single or batch file-info arguments."""

    path: str | None = None
    paths: list[str] | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("path must not be empty")
        return cleaned

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("paths must not be empty")
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("paths must not contain empty values")
        return cleaned

    @model_validator(mode="after")
    def validate_path_selection(self) -> GetFileInfoInputShape:
        if (self.path is None) == (self.paths is None):
            raise ValueError("provide exactly one of path or paths")
        return self

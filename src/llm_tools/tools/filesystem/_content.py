"""Readable-content and file-metadata helpers for repository filesystem tools."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from llm_tools.tools.filesystem._paths import is_hidden
from llm_tools.tools.filesystem.models import (
    FileInfoResult,
    FileInfoStatus,
    FileReadKind,
    ToolLimits,
)

MARKITDOWN_EXTENSIONS = {
    ".doc",
    ".docx",
    ".epub",
    ".html",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rtf",
    ".xls",
    ".xlsx",
}
PROJECT_EXTENSIONS = {
    ".mpp",
    ".mpt",
}
DEFAULT_MAX_READ_FILE_CHARS = 4000
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
ISO_DATETIME_SEPARATOR_RE = re.compile(r"(?<=\d)T(?=\d)")
_LOG4J_SIMPLE_LOGGER_ARG = (
    "-Dlog4j2.loggerContextFactory="
    "org.apache.logging.log4j.simple.SimpleLoggerContextFactory"
)
_MPXJ_JVM_LOCK = threading.Lock()
_MPXJ_JVM_READY = False
_MPXJ_READER_CLASS: Any | None = None


@dataclass(frozen=True, slots=True)
class LoadedReadableContent:
    """Deterministic text representation for one readable file."""

    read_kind: FileReadKind
    status: FileInfoStatus
    content: str | None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ConversionBackend:
    """One non-text file conversion backend."""

    read_kind: FileReadKind
    convert: Callable[[Path], str]


def _get_read_file_cache_root() -> Path:
    """Return the cache root for converted file content."""
    return Path(tempfile.gettempdir()) / "llm_tools" / "read_file_cache"


def _get_cached_conversion_paths(resolved: Path) -> tuple[Path, Path]:
    """Return the markdown and metadata cache paths for a source file."""
    cache_key = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
    cache_dir = _get_read_file_cache_root() / cache_key
    return cache_dir / "content.md", cache_dir / "metadata.json"


def _read_cached_conversion(resolved: Path) -> str | None:
    """Return cached converted markdown when the source has not changed."""
    content_path, metadata_path = _get_cached_conversion_paths(resolved)
    if not content_path.exists() or not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stat = resolved.stat()
    if (
        metadata.get("mtime_ns") != stat.st_mtime_ns
        or metadata.get("size_bytes") != stat.st_size
    ):
        return None
    return content_path.read_text(encoding="utf-8")


def _write_cached_conversion(resolved: Path, markdown: str) -> Path:
    """Persist converted markdown and source metadata for future reads."""
    content_path, metadata_path = _get_cached_conversion_paths(resolved)
    content_path.parent.mkdir(parents=True, exist_ok=True)
    content_path.write_text(markdown, encoding="utf-8")
    stat = resolved.stat()
    metadata_path.write_text(
        json.dumps({"mtime_ns": stat.st_mtime_ns, "size_bytes": stat.st_size}),
        encoding="utf-8",
    )
    return content_path


def _get_conversion_backend(path: Path) -> ConversionBackend | None:
    """Return the non-text conversion backend for one file, if any."""
    suffix = path.suffix.lower()
    if suffix in PROJECT_EXTENSIONS:
        return ConversionBackend(read_kind="project", convert=convert_with_mpxj)
    if suffix in MARKITDOWN_EXTENSIONS:
        return ConversionBackend(
            read_kind="markitdown", convert=convert_with_markitdown
        )
    return None


def read_searchable_text(path: Path) -> str | None:
    """Read text content when it is UTF-8-like and non-binary."""
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if "\x00" in content:
        return None
    return content


def load_readable_content(path: Path) -> LoadedReadableContent:
    """Return deterministic readable content for one file."""
    text_content = read_searchable_text(path)
    if text_content is not None:
        return LoadedReadableContent(
            read_kind="text", status="ok", content=text_content
        )

    backend = _get_conversion_backend(path)
    if backend is None:
        return LoadedReadableContent(
            read_kind="unsupported",
            status="unsupported",
            content=None,
            error_message="File type is not supported for repository reads",
        )

    cached = _read_cached_conversion(path)
    if cached is not None:
        return LoadedReadableContent(
            read_kind=backend.read_kind,
            status="ok",
            content=cached,
        )

    try:
        converted = backend.convert(path)
    except Exception as exc:
        return LoadedReadableContent(
            read_kind=backend.read_kind,
            status="error",
            content=None,
            error_message=str(exc),
        )

    _write_cached_conversion(path, converted)
    return LoadedReadableContent(
        read_kind=backend.read_kind,
        status="ok",
        content=converted,
    )


def convert_with_markitdown(path: Path) -> str:
    """Convert a supported non-text document into markdown text."""
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(path))
    if isinstance(result, str):
        return result
    for attribute in ("text_content", "markdown", "text"):
        value = getattr(result, attribute, None)
        if isinstance(value, str):
            return value
    raise RuntimeError("markitdown conversion did not return readable markdown text")


def convert_with_mpxj(path: Path) -> str:
    """Convert a Microsoft Project file into deterministic markdown text."""
    reader_class = _get_mpxj_reader_class()
    project = reader_class().read(str(path))
    return render_project_as_markdown(project)


def _get_mpxj_reader_class() -> Any:
    """Return the MPXJ reader class after starting the JVM once."""
    global _MPXJ_JVM_READY, _MPXJ_READER_CLASS

    if _MPXJ_READER_CLASS is not None:
        return _MPXJ_READER_CLASS

    with _MPXJ_JVM_LOCK:
        if _MPXJ_READER_CLASS is not None:
            return _MPXJ_READER_CLASS

        try:
            jpype = _import_mpxj_runtime()
        except ImportError as exc:
            raise RuntimeError(
                "Microsoft Project support requires the 'mpxj' package and JPype."
            ) from exc

        if not _MPXJ_JVM_READY:
            try:
                if not jpype.isJVMStarted():
                    jpype.startJVM(_LOG4J_SIMPLE_LOGGER_ARG)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to start the JVM required for Microsoft Project reads."
                ) from exc
            _MPXJ_JVM_READY = True

        try:
            _MPXJ_READER_CLASS = _import_mpxj_reader_class()
        except ImportError as exc:
            raise RuntimeError(
                "Microsoft Project support requires the MPXJ reader classes."
            ) from exc
        return _MPXJ_READER_CLASS


def _import_mpxj_runtime() -> Any:
    """Import the MPXJ Python package and JPype runtime."""
    jpype = importlib.import_module("jpype")
    importlib.import_module("mpxj")
    return jpype


def _import_mpxj_reader_class() -> Any:
    """Import the MPXJ universal reader after the JVM is running."""
    module = importlib.import_module("org.mpxj.reader")
    return module.UniversalProjectReader


def render_project_as_markdown(project: object) -> str:
    """Render an MPXJ project file as deterministic markdown text."""
    properties = _first_value(project, "getProjectProperties")
    title = _format_project_value(
        _first_present_value(properties, "getProjectTitle", "getTitle")
    )
    name = _format_project_value(
        _first_present_value(properties, "getProjectName", "getName")
    )
    manager = _format_project_value(_first_present_value(properties, "getManager"))
    company = _format_project_value(_first_present_value(properties, "getCompany"))
    start = _format_project_value(
        _first_present_value(properties, "getStartDate", "getPlannedStart")
    )
    finish = _format_project_value(
        _first_present_value(properties, "getFinishDate", "getPlannedFinish")
    )
    status_date = _format_project_value(
        _first_present_value(properties, "getStatusDate")
    )
    default_calendar = _format_project_value(
        _first_present_value(
            _first_present_value(project, "getDefaultCalendar"),
            "getName",
        )
    )

    project_names = {value for value in (title, name) if value != "-"}
    raw_tasks = _iter_collection(_first_present_value(project, "getTasks"))
    tasks = [
        task
        for index, task in enumerate(raw_tasks)
        if task is not None
        and not _is_placeholder_task(
            task,
            index=index,
            project_names=project_names,
        )
    ]
    raw_resources = _iter_collection(_first_present_value(project, "getResources"))
    resources = [
        resource
        for index, resource in enumerate(raw_resources)
        if resource is not None and not _is_placeholder_resource(resource, index=index)
    ]

    lines = [
        "# Project",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Title | {_escape_markdown_cell(title)} |",
        f"| Name | {_escape_markdown_cell(name)} |",
        f"| Manager | {_escape_markdown_cell(manager)} |",
        f"| Company | {_escape_markdown_cell(company)} |",
        f"| Start | {_escape_markdown_cell(start)} |",
        f"| Finish | {_escape_markdown_cell(finish)} |",
        f"| Status Date | {_escape_markdown_cell(status_date)} |",
        f"| Default Calendar | {_escape_markdown_cell(default_calendar)} |",
        f"| Task Count | {len(tasks)} |",
        f"| Resource Count | {len(resources)} |",
        "",
        "## Tasks",
        "",
    ]
    lines.extend(
        _render_markdown_table(
            headers=[
                "ID",
                "WBS",
                "Name",
                "Start",
                "Finish",
                "Duration",
                "% Complete",
                "Predecessors",
                "Resources",
                "Critical",
            ],
            rows=[
                [
                    _format_project_value(
                        _first_present_value(task, "getID", "getUniqueID")
                    ),
                    _format_project_value(_first_present_value(task, "getWBS")),
                    _format_project_value(_first_present_value(task, "getName")),
                    _format_project_value(_first_present_value(task, "getStart")),
                    _format_project_value(_first_present_value(task, "getFinish")),
                    _format_project_value(_first_present_value(task, "getDuration")),
                    _format_project_value(
                        _first_present_value(
                            task,
                            "getPercentageComplete",
                            "getPercentComplete",
                        )
                    ),
                    _format_predecessors(task),
                    _format_task_resources(task),
                    _format_project_value(_first_present_value(task, "getCritical")),
                ]
                for task in tasks
            ],
            empty_message="_No tasks found._",
        )
    )
    lines.extend(["", "## Resources", ""])
    lines.extend(
        _render_markdown_table(
            headers=["ID", "Name", "Type", "Max Units", "Calendar"],
            rows=[
                [
                    _format_project_value(
                        _first_present_value(resource, "getID", "getUniqueID")
                    ),
                    _format_project_value(_first_present_value(resource, "getName")),
                    _format_project_value(_first_present_value(resource, "getType")),
                    _format_project_value(
                        _first_present_value(resource, "getMaxUnits")
                    ),
                    _format_project_value(
                        _first_present_value(
                            _first_present_value(resource, "getCalendar"),
                            "getName",
                        )
                    ),
                ]
                for resource in resources
            ],
            empty_message="_No resources found._",
        )
    )

    note_lines = _render_notes_section(
        project_notes=_extract_project_notes(properties),
        tasks=tasks,
        resources=resources,
    )
    if note_lines:
        lines.extend(["", *note_lines])
    return "\n".join(lines)


def _render_markdown_table(
    *,
    headers: list[str],
    rows: list[list[str]],
    empty_message: str,
) -> list[str]:
    """Render a deterministic markdown table or an empty-state line."""
    if not rows:
        return [empty_message]

    header_row = "| " + " | ".join(headers) + " |"
    divider_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| "
        + " | ".join(_escape_markdown_cell(_format_project_value(cell)) for cell in row)
        + " |"
        for row in rows
    ]
    return [header_row, divider_row, *body_rows]


def _render_notes_section(
    *,
    project_notes: str | None,
    tasks: list[object],
    resources: list[object],
) -> list[str]:
    """Render the optional notes section."""
    task_notes = [
        (
            _format_project_value(_first_present_value(task, "getID", "getUniqueID")),
            _format_project_value(_first_present_value(task, "getName")),
            note,
        )
        for task in tasks
        if (note := _extract_notes(task)) is not None
    ]
    resource_notes = [
        (
            _format_project_value(
                _first_present_value(resource, "getID", "getUniqueID")
            ),
            _format_project_value(_first_present_value(resource, "getName")),
            note,
        )
        for resource in resources
        if (note := _extract_notes(resource)) is not None
    ]
    if project_notes is None and not task_notes and not resource_notes:
        return []

    lines = ["## Notes", ""]
    if project_notes is not None:
        lines.extend(["### Project", "", project_notes, ""])
    if task_notes:
        lines.extend(["### Tasks", ""])
        lines.extend(
            f"- [{task_id}] {task_name}: {note}"
            for task_id, task_name, note in task_notes
        )
        lines.append("")
    if resource_notes:
        lines.extend(["### Resources", ""])
        lines.extend(
            f"- [{resource_id}] {resource_name}: {note}"
            for resource_id, resource_name, note in resource_notes
        )
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _extract_project_notes(properties: object | None) -> str | None:
    """Extract project-level notes text when present."""
    for method_name in ("getComments", "getNotes"):
        note = _normalize_note(_first_present_value(properties, method_name))
        if note is not None:
            return note
    return None


def _extract_notes(entity: object | None) -> str | None:
    """Extract note text from one task or resource."""
    return _normalize_note(_first_present_value(entity, "getNotes"))


def _normalize_note(value: object | None) -> str | None:
    """Normalize note text to a deterministic single-line form."""
    if value is None:
        return None
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if not parts:
        return None
    return _escape_markdown_cell(" / ".join(parts))


def _format_predecessors(task: object) -> str:
    """Format one task's predecessor list."""
    predecessors: list[str] = []
    for relation in _iter_collection(_first_present_value(task, "getPredecessors")):
        target = _first_present_value(relation, "getTargetTask")
        target_id = _format_project_value(
            _first_present_value(target, "getID", "getUniqueID")
        )
        relation_type = _format_project_value(_first_present_value(relation, "getType"))
        if target_id == "-":
            continue
        if relation_type == "-":
            predecessors.append(target_id)
            continue
        predecessors.append(f"{target_id} ({relation_type})")
    return ", ".join(predecessors) if predecessors else "-"


def _format_task_resources(task: object) -> str:
    """Format one task's assigned resource list."""
    names: list[str] = []
    for assignment in _iter_collection(
        _first_present_value(task, "getResourceAssignments")
    ):
        resource = _first_present_value(assignment, "getResource")
        name = _first_present_value(resource, "getName")
        if name is None:
            name = _first_present_value(assignment, "getResourceName")
        normalized = _format_project_value(name)
        if normalized != "-":
            names.append(normalized)
    return ", ".join(names) if names else "-"


def _is_placeholder_task(
    task: object,
    *,
    index: int,
    project_names: set[str],
) -> bool:
    """Return whether the task matches MPXJ's documented synthetic summary task."""
    if index != 0 or _first_present_value(task, "getOutlineLevel") != 0:
        return False
    task_name = _format_project_value(_first_present_value(task, "getName"))
    return task_name != "-" and task_name in project_names


def _is_placeholder_resource(resource: object, *, index: int) -> bool:
    """Return whether the resource matches MPXJ's documented synthetic resource."""
    return (
        index == 0
        and _normalize_scalar_text(_first_present_value(resource, "getName")) is None
    )


def _first_present_value(entity: object | None, *method_names: str) -> object | None:
    """Return the first non-None zero-arg getter result."""
    for method_name in method_names:
        value = _first_value(entity, method_name)
        if value is not None:
            return value
    return None


def _first_value(entity: object | None, method_name: str) -> object | None:
    """Return one zero-argument getter result when available."""
    if entity is None:
        return None
    method = getattr(entity, method_name, None)
    if not callable(method):
        return None
    return cast(object | None, method())


def _iter_collection(value: object | None) -> list[object]:
    """Coerce a collection-like value into a Python list."""
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    try:
        return [item for item in cast(Any, value) if item is not None]
    except TypeError:
        return [value]


def _format_project_value(value: object | None) -> str:
    """Format one scalar project value for markdown output."""
    normalized = _normalize_scalar_text(value)
    if normalized is None:
        return "-"
    return normalized


def _normalize_scalar_text(value: object | None) -> str | None:
    """Normalize a scalar into stable single-line text."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, "g")

    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if not parts:
        return None
    normalized = ISO_DATETIME_SEPARATOR_RE.sub(" ", " / ".join(parts))
    lowered = normalized.lower()
    if lowered == "true":
        return "Yes"
    if lowered == "false":
        return "No"
    return normalized


def _escape_markdown_cell(text: str) -> str:
    """Escape markdown table cell separators."""
    return text.replace("\\", "\\\\").replace("|", "\\|")


def count_lines(text: str, *, max_read_lines: int) -> int | None:
    """Count lines, returning None when the count exceeds the configured cap."""
    if not text:
        return 0
    line_count = text.count("\n")
    if not text.endswith("\n"):
        line_count += 1
    if line_count > max_read_lines:
        return None
    return line_count


def estimate_token_count(text: str) -> int:
    """Estimate tokens with a small deterministic lexical tokenizer."""
    return len([match.group(0).lower() for match in TOKEN_RE.finditer(text)])


def is_within_character_limit(
    content: str | None,
    *,
    tool_limits: ToolLimits,
) -> bool:
    """Return whether readable content fits within the configured size cap."""
    return content is not None and len(content) <= tool_limits.max_file_size_characters


def effective_full_read_char_limit(tool_limits: ToolLimits) -> int:
    """Return the full-read cap derived from effective tool limits."""
    configured_limit = tool_limits.max_read_file_chars
    if configured_limit is not None:
        return configured_limit
    return DEFAULT_MAX_READ_FILE_CHARS


def normalize_range(
    *,
    start_char: int | None,
    end_char: int | None,
    character_count: int,
) -> tuple[int, int]:
    """Validate and clamp a character-range request."""
    normalized_start = 0 if start_char is None else start_char
    normalized_end = character_count if end_char is None else end_char
    if normalized_start < 0:
        raise ValueError("start_char must be greater than or equal to 0")
    if normalized_end < 0:
        raise ValueError("end_char must be greater than or equal to 0")
    if end_char is not None and normalized_end <= normalized_start:
        raise ValueError("end_char must be greater than start_char")
    if normalized_start > character_count:
        raise ValueError("start_char must not exceed character_count")
    return normalized_start, min(normalized_end, character_count)


def build_file_info_result(
    *,
    requested_path: str,
    resolved_path: str,
    candidate_file: Path,
    resolved_file: Path,
    relative_candidate_path: Path,
    tool_limits: ToolLimits,
    loaded_content: LoadedReadableContent,
) -> FileInfoResult:
    """Return deterministic metadata for one root-confined file."""
    full_read_char_limit = effective_full_read_char_limit(tool_limits)
    size_bytes = resolved_file.stat().st_size
    content = loaded_content.content
    character_count = len(content) if content is not None else None
    within_size_limit = is_within_character_limit(content, tool_limits=tool_limits)
    return FileInfoResult(
        requested_path=requested_path,
        resolved_path=resolved_path,
        name=candidate_file.name,
        size_bytes=size_bytes,
        is_hidden=is_hidden(relative_candidate_path),
        is_symlink=candidate_file.is_symlink(),
        read_kind=loaded_content.read_kind,
        status=loaded_content.status,
        estimated_token_count=(
            estimate_token_count(content) if content is not None else None
        ),
        character_count=character_count,
        line_count=(
            count_lines(content, max_read_lines=tool_limits.max_read_lines)
            if content is not None
            else None
        ),
        max_file_size_characters=tool_limits.max_file_size_characters,
        within_size_limit=within_size_limit,
        full_read_char_limit=full_read_char_limit,
        can_read_full=within_size_limit
        and character_count is not None
        and character_count <= full_read_char_limit,
        error_message=loaded_content.error_message,
    )


def dump_json(payload: object) -> str:
    """Return a stable compact JSON string."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

# ruff: noqa: N802
"""Tests for MPXJ-backed Microsoft Project reads."""

from __future__ import annotations

from pathlib import Path

import pytest

import llm_tools.tools.filesystem._content as filesystem_content
from llm_tools.tool_api import ToolContext
from llm_tools.tools.filesystem import ReadFileTool
from llm_tools.tools.filesystem._content import (
    _extract_project_notes,
    _first_value,
    _format_predecessors,
    _format_project_value,
    _format_task_resources,
    _get_mpxj_reader_class,
    _import_mpxj_reader_class,
    _import_mpxj_runtime,
    _iter_collection,
    _normalize_note,
    _normalize_scalar_text,
    _render_markdown_table,
    _render_notes_section,
    convert_with_mpxj,
    load_readable_content,
)
from llm_tools.tools.filesystem._ops import get_file_info_impl
from llm_tools.tools.filesystem.models import SourceFilters, ToolLimits
from llm_tools.tools.text._ops import search_text_impl


class FakeJPype:
    def __init__(self, *, started: bool = False, fail_on_start: bool = False) -> None:
        self.started = started
        self.fail_on_start = fail_on_start
        self.start_calls: list[tuple[object, ...]] = []

    def isJVMStarted(self) -> bool:
        return self.started

    def startJVM(self, *args: object) -> None:
        if self.fail_on_start:
            raise OSError("JVM missing")
        self.start_calls.append(args)
        self.started = True


class FakeCalendar:
    def __init__(self, name: str) -> None:
        self._name = name

    def getName(self) -> str:
        return self._name


class FakeProperties:
    def getProjectTitle(self) -> str:
        return "Migration Plan"

    def getProjectName(self) -> str:
        return "Platform Upgrade"

    def getManager(self) -> str:
        return "Alex Manager"

    def getCompany(self) -> str:
        return "Example Co"

    def getStartDate(self) -> str:
        return "2026-01-01T08:00"

    def getFinishDate(self) -> str:
        return "2026-01-31T17:00"

    def getStatusDate(self) -> str:
        return "2026-01-15T12:00"

    def getComments(self) -> str:
        return "Program level note"


class FakeTaskReference:
    def __init__(self, task_id: int) -> None:
        self._task_id = task_id

    def getID(self) -> int:
        return self._task_id


class FakeRelation:
    def __init__(self, task_id: int, relation_type: str) -> None:
        self._task_id = task_id
        self._relation_type = relation_type

    def getTargetTask(self) -> FakeTaskReference:
        return FakeTaskReference(self._task_id)

    def getType(self) -> str:
        return self._relation_type


class FakeResource:
    def __init__(
        self,
        *,
        resource_id: int,
        name: str | None,
        resource_type: str,
        max_units: float,
        calendar_name: str,
        notes: str | None = None,
    ) -> None:
        self._resource_id = resource_id
        self._name = name
        self._resource_type = resource_type
        self._max_units = max_units
        self._calendar = FakeCalendar(calendar_name)
        self._notes = notes

    def getID(self) -> int:
        return self._resource_id

    def getName(self) -> str | None:
        return self._name

    def getType(self) -> str:
        return self._resource_type

    def getMaxUnits(self) -> float:
        return self._max_units

    def getCalendar(self) -> FakeCalendar:
        return self._calendar

    def getNotes(self) -> str | None:
        return self._notes


class FakeAssignment:
    def __init__(self, resource: FakeResource) -> None:
        self._resource = resource

    def getResource(self) -> FakeResource:
        return self._resource

    def getResourceName(self) -> str | None:
        return self._resource.getName()


class FakeTask:
    def __init__(
        self,
        *,
        task_id: int,
        name: str,
        outline_level: int,
        wbs: str,
        start: str,
        finish: str,
        duration: str,
        percent_complete: int,
        predecessors: list[FakeRelation] | None = None,
        assignments: list[FakeAssignment] | None = None,
        critical: bool = False,
        notes: str | None = None,
    ) -> None:
        self._task_id = task_id
        self._name = name
        self._outline_level = outline_level
        self._wbs = wbs
        self._start = start
        self._finish = finish
        self._duration = duration
        self._percent_complete = percent_complete
        self._predecessors = predecessors or []
        self._assignments = assignments or []
        self._critical = critical
        self._notes = notes

    def getID(self) -> int:
        return self._task_id

    def getUniqueID(self) -> int:
        return self._task_id

    def getName(self) -> str:
        return self._name

    def getOutlineLevel(self) -> int:
        return self._outline_level

    def getWBS(self) -> str:
        return self._wbs

    def getStart(self) -> str:
        return self._start

    def getFinish(self) -> str:
        return self._finish

    def getDuration(self) -> str:
        return self._duration

    def getPercentageComplete(self) -> int:
        return self._percent_complete

    def getPredecessors(self) -> list[FakeRelation]:
        return self._predecessors

    def getResourceAssignments(self) -> list[FakeAssignment]:
        return self._assignments

    def getCritical(self) -> bool:
        return self._critical

    def getNotes(self) -> str | None:
        return self._notes


class FakeProject:
    def __init__(self) -> None:
        engineer = FakeResource(
            resource_id=1,
            name="Engineer",
            resource_type="WORK",
            max_units=1.0,
            calendar_name="Standard",
            notes="Resource note",
        )
        self._properties = FakeProperties()
        self._default_calendar = FakeCalendar("Standard")
        self._tasks = [
            FakeTask(
                task_id=0,
                name="Platform Upgrade",
                outline_level=0,
                wbs="0",
                start="2026-01-01T08:00",
                finish="2026-01-31T17:00",
                duration="21d",
                percent_complete=0,
            ),
            FakeTask(
                task_id=1,
                name="Build API",
                outline_level=1,
                wbs="1.1",
                start="2026-01-05T09:00",
                finish="2026-01-10T17:00",
                duration="5d",
                percent_complete=50,
                predecessors=[FakeRelation(7, "FS")],
                assignments=[FakeAssignment(engineer)],
                critical=True,
                notes="Task note",
            ),
        ]
        self._resources = [
            FakeResource(
                resource_id=0,
                name=None,
                resource_type="WORK",
                max_units=0.0,
                calendar_name="Standard",
            ),
            engineer,
        ]

    def getProjectProperties(self) -> FakeProperties:
        return self._properties

    def getDefaultCalendar(self) -> FakeCalendar:
        return self._default_calendar

    def getTasks(self) -> list[FakeTask]:
        return self._tasks

    def getResources(self) -> list[FakeResource]:
        return self._resources


class FakeReader:
    read_calls: list[str] = []
    project_factory = FakeProject
    read_exception: Exception | None = None

    def read(self, raw_path: str) -> FakeProject:
        type(self).read_calls.append(raw_path)
        if type(self).read_exception is not None:
            raise type(self).read_exception
        return type(self).project_factory()


class FailingReader(FakeReader):
    pass


def _reset_mpxj_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(filesystem_content, "_MPXJ_JVM_READY", False)
    monkeypatch.setattr(filesystem_content, "_MPXJ_READER_CLASS", None)


def _install_fake_mpxj(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reader_class: type[FakeReader] = FakeReader,
    jpype: FakeJPype | None = None,
) -> FakeJPype:
    fake_jpype = jpype or FakeJPype()
    _reset_mpxj_state(monkeypatch)
    reader_class.read_calls = []
    reader_class.read_exception = None
    monkeypatch.setattr(filesystem_content, "_import_mpxj_runtime", lambda: fake_jpype)
    monkeypatch.setattr(
        filesystem_content,
        "_import_mpxj_reader_class",
        lambda: reader_class,
    )
    return fake_jpype


def test_convert_with_mpxj_initializes_jvm_once_and_renders_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "plan.mpp"
    file_path.write_bytes(bytes.fromhex("fffe0010"))
    fake_jpype = _install_fake_mpxj(monkeypatch)

    first = convert_with_mpxj(file_path)
    second = convert_with_mpxj(file_path)

    assert first == second
    assert len(fake_jpype.start_calls) == 1
    assert FakeReader.read_calls == [str(file_path), str(file_path)]
    assert "# Project" in first
    assert "| Task Count | 1 |" in first
    assert "| Resource Count | 1 |" in first
    assert "| 1 | 1.1 | Build API | 2026-01-05 09:00 |" in first
    assert "Engineer" in first
    assert "Program level note" in first
    assert "Task note" in first
    assert "Resource note" in first
    assert "Platform Upgrade | 2026-01-01 08:00" not in first


def test_load_readable_content_converts_project_files_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "plan.mpp"
    file_path.write_bytes(bytes.fromhex("fffe0010"))
    fake_jpype = _install_fake_mpxj(monkeypatch)
    monkeypatch.setattr(
        filesystem_content,
        "_get_read_file_cache_root",
        lambda: tmp_path / "cache-root",
    )

    first = load_readable_content(file_path)
    second = load_readable_content(file_path)

    assert first.read_kind == "project"
    assert first.status == "ok"
    assert second.content == first.content
    assert FakeReader.read_calls == [str(file_path)]
    assert len(fake_jpype.start_calls) == 1


def test_load_readable_content_surfaces_project_runtime_and_parse_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "plan.mpp"
    file_path.write_bytes(bytes.fromhex("fffe0010"))

    _reset_mpxj_state(monkeypatch)
    monkeypatch.setattr(
        filesystem_content,
        "_import_mpxj_runtime",
        lambda: (_ for _ in ()).throw(ImportError("missing mpxj")),
    )
    missing_runtime = load_readable_content(file_path)
    assert missing_runtime.read_kind == "project"
    assert missing_runtime.status == "error"
    assert missing_runtime.error_message is not None
    assert "mpxj" in missing_runtime.error_message.lower()

    fake_jpype = _install_fake_mpxj(
        monkeypatch,
        jpype=FakeJPype(fail_on_start=True),
    )
    jvm_error = load_readable_content(file_path)
    assert jvm_error.status == "error"
    assert (
        jvm_error.error_message
        == "Failed to start the JVM required for Microsoft Project reads."
    )
    assert fake_jpype.start_calls == []

    _install_fake_mpxj(monkeypatch, reader_class=FailingReader)
    FailingReader.read_exception = RuntimeError("File is password protected")
    parse_error = load_readable_content(file_path)
    assert parse_error.status == "error"
    assert parse_error.error_message == "File is password protected"


def test_read_file_tool_applies_ranges_and_limits_to_project_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "plan.mpp"
    file_path.write_bytes(bytes.fromhex("fffe0010"))
    _install_fake_mpxj(monkeypatch)

    full_content = convert_with_mpxj(file_path)
    result = ReadFileTool().invoke(
        ToolContext(
            invocation_id="inv-project",
            workspace=str(tmp_path),
            metadata={"tool_limits": {"max_read_file_chars": 40}},
        ),
        ReadFileTool.input_model(path="plan.mpp", start_char=2, end_char=80),
    )

    assert result.read_kind == "project"
    assert result.content == full_content[2:42]
    assert result.start_char == 2
    assert result.end_char == 42
    assert result.truncated is True


def test_get_file_info_and_search_text_support_project_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "plan.mpp"
    file_path.write_bytes(bytes.fromhex("fffe0010"))
    _install_fake_mpxj(monkeypatch)
    monkeypatch.setattr(
        filesystem_content,
        "_get_read_file_cache_root",
        lambda: tmp_path / "cache-root",
    )

    info = get_file_info_impl(
        tmp_path,
        "plan.mpp",
        tool_limits=ToolLimits(max_file_size_characters=100_000),
    )
    assert info.read_kind == "project"
    assert info.status == "ok"
    assert info.character_count is not None and info.character_count > 0
    assert info.can_read_full is True

    search = search_text_impl(
        tmp_path,
        "Build API",
        "plan.mpp",
        source_filters=SourceFilters(),
        tool_limits=ToolLimits(max_file_size_characters=100_000),
    )
    assert len(search.matches) >= 1
    assert all(match.path == "plan.mpp" for match in search.matches)
    assert any("| 1 | 1.1 | Build API |" in match.line_text for match in search.matches)


def test_convert_with_mpxj_keeps_literal_t_characters_in_non_date_strings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LiteralTProject(FakeProject):
        def __init__(self) -> None:
            tester = FakeResource(
                resource_id=1,
                name="Tester",
                resource_type="WORK",
                max_units=1.0,
                calendar_name="Test Calendar",
                notes="Test note",
            )
            self._properties = FakeProperties()
            self._default_calendar = FakeCalendar("Test Calendar")
            self._tasks = [
                FakeTask(
                    task_id=0,
                    name="Platform Upgrade",
                    outline_level=0,
                    wbs="0",
                    start="2026-01-01T08:00",
                    finish="2026-01-31T17:00",
                    duration="21d",
                    percent_complete=0,
                ),
                FakeTask(
                    task_id=1,
                    name="Test Task",
                    outline_level=1,
                    wbs="1.1",
                    start="2026-01-05T09:00",
                    finish="2026-01-10T17:00",
                    duration="5d",
                    percent_complete=50,
                    assignments=[FakeAssignment(tester)],
                    notes="Task note",
                ),
            ]
            self._resources = [
                FakeResource(
                    resource_id=0,
                    name=None,
                    resource_type="WORK",
                    max_units=0.0,
                    calendar_name="Standard",
                ),
                tester,
            ]

    file_path = tmp_path / "plan.mpp"
    file_path.write_bytes(bytes.fromhex("fffe0010"))
    monkeypatch.setattr(FakeReader, "project_factory", LiteralTProject)
    _install_fake_mpxj(monkeypatch)

    rendered = convert_with_mpxj(file_path)

    assert "Test Task" in rendered
    assert "Tester" in rendered
    assert "Test Calendar" in rendered
    assert " est  ask" not in rendered
    assert " ester" not in rendered


def test_convert_with_mpxj_keeps_real_top_level_first_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RealTopLevelTaskProject(FakeProject):
        def __init__(self) -> None:
            engineer = FakeResource(
                resource_id=1,
                name="Engineer",
                resource_type="WORK",
                max_units=1.0,
                calendar_name="Standard",
            )
            self._properties = FakeProperties()
            self._default_calendar = FakeCalendar("Standard")
            self._tasks = [
                FakeTask(
                    task_id=1,
                    name="Kickoff",
                    outline_level=0,
                    wbs="1",
                    start="2026-01-01T08:00",
                    finish="2026-01-02T17:00",
                    duration="2d",
                    percent_complete=0,
                    assignments=[FakeAssignment(engineer)],
                ),
                FakeTask(
                    task_id=2,
                    name="Build API",
                    outline_level=1,
                    wbs="1.1",
                    start="2026-01-05T09:00",
                    finish="2026-01-10T17:00",
                    duration="5d",
                    percent_complete=50,
                ),
            ]
            self._resources = [engineer]

    file_path = tmp_path / "plan.mpp"
    file_path.write_bytes(bytes.fromhex("fffe0010"))
    monkeypatch.setattr(FakeReader, "project_factory", RealTopLevelTaskProject)
    _install_fake_mpxj(monkeypatch)

    rendered = convert_with_mpxj(file_path)

    assert "| Task Count | 2 |" in rendered
    assert "| 1 | 1 | Kickoff | 2026-01-01 08:00 |" in rendered


def test_project_content_helpers_cover_remaining_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LockThatPublishesReader:
        def __enter__(self) -> None:
            filesystem_content._MPXJ_READER_CLASS = "published-reader"
            return None

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    _reset_mpxj_state(monkeypatch)
    monkeypatch.setattr(
        filesystem_content, "_MPXJ_JVM_LOCK", _LockThatPublishesReader()
    )
    assert _get_mpxj_reader_class() == "published-reader"

    class _StartedJPype:
        def isJVMStarted(self) -> bool:
            return True

        def startJVM(self, *args: object) -> None:
            raise AssertionError("startJVM should not be called")

    _reset_mpxj_state(monkeypatch)
    monkeypatch.setattr(
        filesystem_content, "_MPXJ_JVM_LOCK", filesystem_content.threading.Lock()
    )
    monkeypatch.setattr(
        filesystem_content,
        "_import_mpxj_runtime",
        lambda: _StartedJPype(),
    )
    monkeypatch.setattr(
        filesystem_content,
        "_import_mpxj_reader_class",
        lambda: (_ for _ in ()).throw(ImportError("missing reader module")),
    )
    with pytest.raises(RuntimeError, match="reader classes"):
        _get_mpxj_reader_class()

    class _ReaderModule:
        UniversalProjectReader = "reader-class"

    modules = {
        "jpype": "jpype-module",
        "mpxj": "mpxj-module",
        "org.mpxj.reader": _ReaderModule(),
    }
    monkeypatch.setattr(
        filesystem_content.importlib,
        "import_module",
        lambda name: modules[name],
    )
    assert _import_mpxj_runtime() == "jpype-module"
    assert _import_mpxj_reader_class() == "reader-class"

    assert _render_markdown_table(headers=["A"], rows=[], empty_message="none") == [
        "none"
    ]
    assert _render_notes_section(project_notes=None, tasks=[], resources=[]) == []
    assert _extract_project_notes(object()) is None
    assert _normalize_note("  \n  ") is None

    class _MissingTargetRelation:
        def getTargetTask(self) -> object:
            return object()

        def getType(self) -> str:
            return "FS"

    predecessors = _format_predecessors(
        type(
            "TaskWithMixedPredecessors",
            (),
            {
                "getPredecessors": lambda self: [
                    _MissingTargetRelation(),
                    FakeRelation(3, "-"),
                ]
            },
        )()
    )
    assert predecessors == "3"

    fallback_resources = _format_task_resources(
        type(
            "TaskWithFallbackAssignments",
            (),
            {
                "getResourceAssignments": lambda self: [
                    type(
                        "FallbackAssignment",
                        (),
                        {
                            "getResource": lambda self: object(),
                            "getResourceName": lambda self: "Fallback Tester",
                        },
                    )()
                ]
            },
        )()
    )
    assert fallback_resources == "Fallback Tester"

    class _HasAttribute:
        attr = 1

    assert _first_value(None, "getName") is None
    assert _first_value(_HasAttribute(), "attr") is None
    assert _iter_collection(None) == []
    assert _iter_collection("demo") == ["demo"]
    sentinel = object()
    assert _iter_collection(sentinel) == [sentinel]
    assert _format_project_value(None) == "-"
    assert _normalize_scalar_text("  \n ") is None
    assert _normalize_scalar_text("true") == "Yes"
    assert _normalize_scalar_text("false") == "No"

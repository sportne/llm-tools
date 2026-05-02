"""Persistence helpers for workflow protection state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml  # type: ignore[import-untyped]

from llm_tools.tools.filesystem.models import ToolLimits
from llm_tools.tools.filesystem.readable_content import (
    CONVERTIBLE_DOCUMENT_EXTENSIONS,
    LoadedReadableContent,
    get_conversion_backend,
    get_read_file_cache_root,
    load_readable_content,
    read_cached_conversion,
    write_cached_conversion,
)
from llm_tools.workflow_api.protection_models import (
    ProtectionConfig,
    ProtectionCorpus,
    ProtectionDocument,
    ProtectionFeedbackEntry,
    ProtectionFeedbackFile,
)
from llm_tools.workflow_api.protection_store_models import (
    ProtectionCorpusLoadIssue as ProtectionCorpusLoadIssue,
)
from llm_tools.workflow_api.protection_store_models import (
    ProtectionCorpusLoadReport as ProtectionCorpusLoadReport,
)

_ALLOWED_PROTECTION_DOCUMENT_SUFFIXES = {
    ".adoc",
    ".csv",
    ".json",
    ".log",
    ".md",
    ".rst",
    ".text",
    ".txt",
    ".yaml",
    ".yml",
}
_CONVERTED_PROTECTION_DOCUMENT_SUFFIXES = CONVERTIBLE_DOCUMENT_EXTENSIONS
_PROTECTION_SOURCE_MANIFEST_FILENAMES = {
    ".llm-tools-protection-sources.json",
    ".llm-tools-protection-sources.yaml",
    ".llm-tools-protection-sources.yml",
    "llm-tools-protection-sources.json",
    "llm-tools-protection-sources.yaml",
    "llm-tools-protection-sources.yml",
}
_PROTECTION_METADATA_FILENAMES = _PROTECTION_SOURCE_MANIFEST_FILENAMES | {
    ".llm-tools-protection-corrections.json",
    ".llm-tools-protection-corrections.yaml",
    ".llm-tools-protection-corrections.yml",
}
_FRONT_MATTER_CATEGORY_KEYS = {
    "category",
    "sensitivity",
    "sensitivity_category",
    "sensitivity_label",
}


@dataclass(frozen=True, slots=True)
class _SourceCategoryRule:
    """One manifest-provided category rule scoped to a corpus directory."""

    base_dir: Path
    pattern: str
    category: str
    source: str

    def matches(self, path: Path) -> bool:
        try:
            relative_path = path.relative_to(self.base_dir)
        except ValueError:
            return False
        candidate = relative_path.as_posix()
        return candidate == self.pattern or fnmatchcase(candidate, self.pattern)


class ProtectionFeedbackStore:
    """Read and write the structured corrections sidecar file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load_entries(self) -> list[ProtectionFeedbackEntry]:
        if not self._path.exists():
            return []
        raw = self._read_payload()
        data = raw if isinstance(raw, dict) else {"entries": raw}
        return ProtectionFeedbackFile.model_validate(data).entries

    def save_entries(self, entries: list[ProtectionFeedbackEntry]) -> None:
        payload = ProtectionFeedbackFile(entries=entries)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_payload(payload.model_dump(mode="json"))

    def append_entry(
        self, entry: ProtectionFeedbackEntry
    ) -> list[ProtectionFeedbackEntry]:
        entries = self.load_entries()
        entries.append(entry)
        self.save_entries(entries)
        return entries

    def _read_payload(self) -> Any:
        suffix = self._path.suffix.lower()
        text = self._path.read_text(encoding="utf-8")
        if suffix == ".json":
            return json.loads(text)
        if suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(text)
            return {} if loaded is None else loaded
        raise ValueError(f"Unsupported protection feedback file type: {self._path}")

    def _write_payload(self, payload: dict[str, Any]) -> None:
        suffix = self._path.suffix.lower()
        if suffix == ".json":
            self._path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            return
        if suffix in {".yaml", ".yml"}:
            self._path.write_text(
                yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
            )
            return
        raise ValueError(f"Unsupported protection feedback file type: {self._path}")


def inspect_protection_corpus(
    config: ProtectionConfig,
    *,
    feedback_store: ProtectionFeedbackStore | None = None,
) -> ProtectionCorpusLoadReport:
    """Load the configured documents and report skipped or invalid entries."""
    documents: list[ProtectionDocument] = []
    issues: list[ProtectionCorpusLoadIssue] = []
    source_rules = _load_source_category_rules(config, issues=issues)
    corrections_path = (
        Path(config.corrections_path).expanduser().resolve()
        if config.corrections_path is not None
        else None
    )
    for configured_path in config.document_paths:
        resolved = Path(configured_path).expanduser()
        if not resolved.exists():
            issues.append(
                ProtectionCorpusLoadIssue(
                    path=str(configured_path),
                    message="Path does not exist.",
                )
            )
            continue
        if resolved.is_dir():
            for candidate in sorted(resolved.rglob("*")):
                if not candidate.is_file():
                    continue
                document = _load_protection_document(
                    candidate,
                    corpus_root=resolved,
                    config=config,
                    corrections_path=corrections_path,
                    source_rules=source_rules,
                    issues=issues,
                )
                if document is not None:
                    documents.append(document)
            continue
        document = _load_protection_document(
            resolved,
            corpus_root=resolved.parent,
            config=config,
            corrections_path=corrections_path,
            source_rules=source_rules,
            issues=issues,
        )
        if document is not None:
            documents.append(document)
    feedback_entries: list[ProtectionFeedbackEntry] = []
    if feedback_store is not None:
        feedback_entries = feedback_store.load_entries()
    return ProtectionCorpusLoadReport(
        corpus=ProtectionCorpus(
            documents=documents,
            feedback_entries=feedback_entries,
        ),
        issues=issues,
    )


def load_protection_corpus(
    config: ProtectionConfig,
    *,
    feedback_store: ProtectionFeedbackStore | None = None,
) -> ProtectionCorpus:
    """Load the configured unstructured documents and structured feedback entries."""
    return inspect_protection_corpus(
        config,
        feedback_store=feedback_store,
    ).corpus


def _document_id_for_path(path: str) -> str:
    return Path(path).name or f"document-{uuid4().hex}"


def _load_protection_document(
    path: Path,
    *,
    corpus_root: Path,
    config: ProtectionConfig,
    corrections_path: Path | None,
    source_rules: list[_SourceCategoryRule],
    issues: list[ProtectionCorpusLoadIssue],
) -> ProtectionDocument | None:
    resolved = path.expanduser()
    try:
        resolved_for_compare = resolved.resolve()
    except OSError:
        resolved_for_compare = resolved
    if _is_internal_protection_metadata(
        resolved_for_compare, corrections_path=corrections_path
    ):
        return None

    suffix = resolved.suffix.lower()
    if (
        suffix not in _ALLOWED_PROTECTION_DOCUMENT_SUFFIXES
        and suffix not in _CONVERTED_PROTECTION_DOCUMENT_SUFFIXES
    ):
        issues.append(
            ProtectionCorpusLoadIssue(
                path=str(resolved),
                message="Unsupported file type for protection corpus.",
            )
        )
        return None

    cache_root = (
        get_read_file_cache_root(corpus_root) if config.cache_documents else None
    )
    loaded = _load_protection_readable_content(
        resolved,
        suffix=suffix,
        cache_root=cache_root,
    )
    if loaded.status != "ok" or loaded.content is None:
        issues.append(
            ProtectionCorpusLoadIssue(
                path=str(resolved),
                message=_protection_document_load_error(
                    loaded.error_message, suffix=suffix
                ),
            )
        )
        return None

    category, category_source = _resolve_document_category(
        path=resolved,
        corpus_root=corpus_root,
        content=loaded.content,
        config=config,
        source_rules=source_rules,
    )
    return ProtectionDocument(
        document_id=_document_id_for_path_with_root(resolved, corpus_root=corpus_root),
        path=str(resolved),
        content=loaded.content,
        display_name=resolved.name,
        read_kind=loaded.read_kind,
        content_hash=sha256(loaded.content.encode("utf-8")).hexdigest(),
        sensitivity_label=category,
        sensitivity_label_source=category_source,
    )


def _protection_document_load_error(error_message: str | None, *, suffix: str) -> str:
    if error_message == "File exceeds the configured readable byte limit":
        return "Protection document exceeds the readable byte limit."
    if error_message == "File type is not supported for repository reads":
        if suffix in _ALLOWED_PROTECTION_DOCUMENT_SUFFIXES:
            return "Unable to read protection document."
        return "Unsupported file type for protection corpus."
    if error_message:
        return f"Unable to convert protection document: {error_message}"
    return "Unable to read protection document."


def _load_protection_readable_content(
    path: Path,
    *,
    suffix: str,
    cache_root: Path | None,
) -> LoadedReadableContent:
    if suffix not in _CONVERTED_PROTECTION_DOCUMENT_SUFFIXES:
        return load_readable_content(
            path,
            tool_limits=ToolLimits(),
            cache_root=cache_root,
        )
    backend = get_conversion_backend(path)
    if backend is None:
        return load_readable_content(
            path,
            tool_limits=ToolLimits(),
            cache_root=cache_root,
        )
    if path.stat().st_size > ToolLimits().max_read_input_bytes:
        return LoadedReadableContent(
            read_kind=backend.read_kind,
            status="too_large",
            content=None,
            error_message="File exceeds the configured readable byte limit",
        )
    if cache_root is not None:
        cached = read_cached_conversion(path, cache_root=cache_root)
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
            error_message=str(exc) or "conversion failed",
        )
    if cache_root is not None:
        write_cached_conversion(path, converted, cache_root=cache_root)
    return LoadedReadableContent(
        read_kind=backend.read_kind,
        status="ok",
        content=converted,
    )


def _is_internal_protection_metadata(
    path: Path, *, corrections_path: Path | None
) -> bool:
    if ".llm_tools" in path.parts:
        return True
    if path.name in _PROTECTION_METADATA_FILENAMES:
        return True
    return corrections_path is not None and path == corrections_path


def _document_id_for_path_with_root(path: Path, *, corpus_root: Path) -> str:
    try:
        return path.relative_to(corpus_root).as_posix()
    except ValueError:
        return _document_id_for_path(str(path))


def _load_source_category_rules(
    config: ProtectionConfig,
    *,
    issues: list[ProtectionCorpusLoadIssue],
) -> list[_SourceCategoryRule]:
    roots: set[Path] = set()
    for configured_path in config.document_paths:
        path = Path(configured_path).expanduser()
        roots.add(path if path.is_dir() else path.parent)
    rules: list[_SourceCategoryRule] = []
    for root in sorted(roots):
        for filename in sorted(_PROTECTION_SOURCE_MANIFEST_FILENAMES):
            manifest = root / filename
            if manifest.is_file():
                rules.extend(_read_source_category_manifest(manifest, issues=issues))
    return rules


def _read_source_category_manifest(
    path: Path,
    *,
    issues: list[ProtectionCorpusLoadIssue],
) -> list[_SourceCategoryRule]:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text)
    except Exception as exc:
        issues.append(
            ProtectionCorpusLoadIssue(
                path=str(path),
                message=(
                    "Unable to read protection source metadata: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        )
        return []

    rules: list[_SourceCategoryRule] = []
    for pattern, category in _iter_manifest_category_entries(payload):
        if not pattern or not category:
            issues.append(
                ProtectionCorpusLoadIssue(
                    path=str(path),
                    message="Skipped invalid protection source metadata entry.",
                )
            )
            continue
        rules.append(
            _SourceCategoryRule(
                base_dir=path.parent,
                pattern=pattern,
                category=category,
                source=f"manifest:{path.name}",
            )
        )
    return rules


def _iter_manifest_category_entries(payload: object) -> list[tuple[str, str]]:
    if isinstance(payload, dict):
        raw_entries = (
            payload.get("sources")
            or payload.get("documents")
            or payload.get("files")
            or payload
        )
    else:
        raw_entries = payload

    entries: list[tuple[str, str]] = []
    if isinstance(raw_entries, dict):
        for raw_pattern, raw_value in raw_entries.items():
            if isinstance(raw_value, str):
                entries.append((str(raw_pattern).strip(), raw_value.strip()))
            elif isinstance(raw_value, dict):
                entries.append((str(raw_pattern).strip(), _entry_category(raw_value)))
        return entries
    if isinstance(raw_entries, list):
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            pattern = (
                raw_entry.get("path")
                or raw_entry.get("pattern")
                or raw_entry.get("glob")
                or raw_entry.get("document")
            )
            entries.append((str(pattern or "").strip(), _entry_category(raw_entry)))
    return entries


def _entry_category(entry: dict[Any, Any]) -> str:
    raw_category = (
        entry.get("category")
        or entry.get("sensitivity")
        or entry.get("sensitivity_label")
        or entry.get("sensitivity_category")
    )
    return str(raw_category or "").strip()


def _resolve_document_category(
    *,
    path: Path,
    corpus_root: Path,
    content: str,
    config: ProtectionConfig,
    source_rules: list[_SourceCategoryRule],
) -> tuple[str | None, str | None]:
    front_matter_category = _front_matter_category(content)
    if front_matter_category:
        return _canonical_category(front_matter_category, config), "front_matter"
    for rule in source_rules:
        if rule.matches(path):
            return _canonical_category(rule.category, config), rule.source
    folder_category = _folder_category(path, corpus_root=corpus_root, config=config)
    if folder_category:
        return folder_category, "folder"
    return None, None


def _front_matter_category(content: str) -> str | None:
    if not content.startswith("---"):
        return None
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            try:
                payload = yaml.safe_load("\n".join(lines[1:index]))
            except yaml.YAMLError:
                return None
            if not isinstance(payload, dict):
                return None
            for key in _FRONT_MATTER_CATEGORY_KEYS:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None
    return None


def _folder_category(
    path: Path,
    *,
    corpus_root: Path,
    config: ProtectionConfig,
) -> str | None:
    try:
        relative_parts = path.relative_to(corpus_root).parts[:-1]
    except ValueError:
        relative_parts = path.parts[:-1]
    for part in relative_parts:
        if _known_category(part, config):
            return _canonical_category(part, config)
    return None


def _canonical_category(label: str, config: ProtectionConfig) -> str:
    cleaned = label.strip()
    if not cleaned:
        return cleaned
    by_casefold: dict[str, str] = {
        category.label.casefold(): category.label
        for category in config.sensitivity_categories
    }
    for category in config.sensitivity_categories:
        for alias in category.aliases:
            by_casefold[alias.casefold()] = category.label
    for allowed_label in config.allowed_sensitivity_labels:
        by_casefold.setdefault(allowed_label.casefold(), allowed_label)
    return by_casefold.get(cleaned.casefold(), cleaned)


def _known_category(label: str, config: ProtectionConfig) -> bool:
    canonical = _canonical_category(label, config).casefold()
    known = {item.label.casefold() for item in config.sensitivity_categories}
    known.update(item.casefold() for item in config.allowed_sensitivity_labels)
    return canonical in known


__all__ = [
    "ProtectionCorpusLoadIssue",
    "ProtectionCorpusLoadReport",
    "ProtectionFeedbackStore",
    "inspect_protection_corpus",
    "load_protection_corpus",
]

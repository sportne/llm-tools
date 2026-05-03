"""Generate dependency inventory tables and CycloneDX-style SBOM output."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

try:  # pragma: no cover - fallback is covered by direct helper tests.
    from packaging.markers import default_environment
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name as _packaging_canonicalize_name
except Exception:  # pragma: no cover
    default_environment = None
    InvalidRequirement = ValueError
    Requirement = None
    _packaging_canonicalize_name = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYPROJECT = PROJECT_ROOT / "pyproject.toml"
DEFAULT_PURPOSES = PROJECT_ROOT / "docs" / "dependency-purposes.toml"
GROUPS = ("runtime", "dev")
DIRECT_ONLY_FORMATS = {"markdown", "json"}
TRANSITIVE_FORMATS = {"cyclonedx-json"}
ALL_FORMATS = DIRECT_ONLY_FORMATS | TRANSITIVE_FORMATS


@dataclass(frozen=True, slots=True)
class DirectDependency:
    """One direct dependency declared by the project."""

    group: str
    requirement: str
    name: str
    canonical_name: str


@dataclass(frozen=True, slots=True)
class PackageMetadata:
    """Installed distribution metadata used by inventory output."""

    canonical_name: str
    name: str
    version: str | None
    summary: str | None
    license: str | None
    provenance_url: str | None
    requires: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectInventoryRow:
    """One row in the human-facing dependency table."""

    group: str
    name: str
    requirement: str
    installed_version: str | None
    purpose: str | None
    provenance_url: str | None
    license: str | None


def canonicalize_name(name: str) -> str:
    """Return a normalized package key."""
    if _packaging_canonicalize_name is not None:
        return str(_packaging_canonicalize_name(name))
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    """Return the distribution name from a dependency requirement string."""
    if Requirement is not None:
        try:
            return str(Requirement(requirement).name)
        except InvalidRequirement:
            pass
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        raise ValueError(f"Could not parse dependency requirement: {requirement!r}")
    return match.group(1)


def requirement_applies(requirement: str) -> bool:
    """Return whether a requirement marker applies to the current environment."""
    if Requirement is None or default_environment is None:
        return True
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        return True
    if parsed.marker is None:
        return True
    environment = default_environment()
    environment.setdefault("extra", "")
    try:
        return bool(parsed.marker.evaluate(environment))
    except Exception:
        return True


def load_direct_dependencies(pyproject_path: Path) -> list[DirectDependency]:
    """Load direct runtime and development dependencies from pyproject.toml."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    optional = project.get("optional-dependencies", {})
    raw_by_group = {
        "runtime": project.get("dependencies", []),
        "dev": optional.get("dev", []),
    }
    dependencies: list[DirectDependency] = []
    for group in GROUPS:
        raw_dependencies = raw_by_group[group]
        if not isinstance(raw_dependencies, list):
            raise ValueError(f"Expected {group} dependencies to be a list.")
        for raw_requirement in raw_dependencies:
            if not isinstance(raw_requirement, str):
                raise ValueError(f"Expected {group} dependency to be a string.")
            name = requirement_name(raw_requirement)
            dependencies.append(
                DirectDependency(
                    group=group,
                    requirement=raw_requirement,
                    name=name,
                    canonical_name=canonicalize_name(name),
                )
            )
    return dependencies


def load_dependency_purposes(path: Path) -> dict[str, dict[str, str]]:
    """Load curated dependency purpose text."""
    if not path.exists():
        return {group: {} for group in GROUPS}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    purposes: dict[str, dict[str, str]] = {group: {} for group in GROUPS}
    for group in GROUPS:
        group_data = data.get(group, {})
        if not isinstance(group_data, dict):
            raise ValueError(f"Expected [{group}] dependency purposes table.")
        for name, purpose in group_data.items():
            if not isinstance(purpose, str) or not purpose.strip():
                raise ValueError(f"Purpose for {group}.{name} must be non-empty.")
            purposes[group][canonicalize_name(str(name))] = purpose.strip()
    return purposes


def installed_distributions() -> dict[str, PackageMetadata]:
    """Return installed distributions keyed by canonical name."""
    packages: dict[str, PackageMetadata] = {}
    for distribution in metadata.distributions():
        package = package_metadata_from_distribution(distribution)
        packages[package.canonical_name] = package
    return packages


def package_metadata_from_distribution(
    distribution: metadata.Distribution,
) -> PackageMetadata:
    """Convert importlib distribution metadata into an inventory record."""
    raw = distribution.metadata
    name = raw.get("Name") or distribution.name
    return PackageMetadata(
        canonical_name=canonicalize_name(name),
        name=name,
        version=distribution.version,
        summary=_clean_optional(raw.get("Summary")),
        license=_metadata_license(raw),
        provenance_url=_metadata_provenance_url(raw),
        requires=tuple(distribution.requires or ()),
    )


def direct_inventory_rows(
    dependencies: list[DirectDependency],
    purposes: dict[str, dict[str, str]],
    installed: dict[str, PackageMetadata],
    *,
    groups: set[str],
) -> list[DirectInventoryRow]:
    """Build direct dependency rows for human documentation."""
    rows: list[DirectInventoryRow] = []
    for dependency in dependencies:
        if dependency.group not in groups:
            continue
        package = installed.get(dependency.canonical_name)
        rows.append(
            DirectInventoryRow(
                group=dependency.group,
                name=package.name if package is not None else dependency.name,
                requirement=dependency.requirement,
                installed_version=package.version if package is not None else None,
                purpose=purposes.get(dependency.group, {}).get(
                    dependency.canonical_name
                ),
                provenance_url=(
                    package.provenance_url if package is not None else None
                ),
                license=package.license if package is not None else None,
            )
        )
    return rows


def transitive_package_closure(
    dependencies: list[DirectDependency],
    installed: dict[str, PackageMetadata],
    *,
    groups: set[str],
) -> tuple[list[PackageMetadata], dict[str, set[str]]]:
    """Return installed package closure and group memberships."""
    group_by_package: dict[str, set[str]] = {}
    queue: deque[str] = deque()
    for dependency in dependencies:
        if dependency.group not in groups:
            continue
        group_by_package.setdefault(dependency.canonical_name, set()).add(
            dependency.group
        )
        queue.append(dependency.canonical_name)

    seen: set[str] = set()
    ordered: list[PackageMetadata] = []
    while queue:
        package_name = queue.popleft()
        if package_name in seen:
            continue
        seen.add(package_name)
        package = installed.get(package_name)
        if package is None:
            continue
        ordered.append(package)
        parent_groups = group_by_package.get(package_name, set())
        for requirement in package.requires:
            if not requirement_applies(requirement):
                continue
            child_name = canonicalize_name(requirement_name(requirement))
            child_groups = group_by_package.setdefault(child_name, set())
            child_groups.update(parent_groups)
            if child_name not in seen:
                queue.append(child_name)
    ordered.sort(key=lambda package: package.canonical_name)
    return ordered, group_by_package


def render_markdown(rows: list[DirectInventoryRow]) -> str:
    """Render direct dependencies as a Markdown table grouped by scope."""
    lines: list[str] = []
    for group in GROUPS:
        group_rows = [row for row in rows if row.group == group]
        if not group_rows:
            continue
        title = "Runtime Dependencies" if group == "runtime" else "Development Dependencies"
        if lines:
            lines.append("")
        lines.extend(
            [
                f"### {title}",
                "",
                "| Package | Constraint | Installed | Purpose | Provenance | License |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in sorted(group_rows, key=lambda item: canonicalize_name(item.name)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(row.name),
                        _markdown_cell(row.requirement),
                        _markdown_cell(row.installed_version or "unknown"),
                        _markdown_cell(row.purpose or "unknown"),
                        _markdown_cell(row.provenance_url or "unknown"),
                        _markdown_cell(row.license or "unknown"),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def render_json(rows: list[DirectInventoryRow]) -> str:
    """Render direct dependency rows as simplified JSON."""
    payload = [
        {
            "group": row.group,
            "name": row.name,
            "requirement": row.requirement,
            "installed_version": row.installed_version,
            "purpose": row.purpose,
            "provenance_url": row.provenance_url,
            "license": row.license,
        }
        for row in rows
    ]
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_cyclonedx_json(
    packages: list[PackageMetadata],
    group_by_package: dict[str, set[str]],
) -> str:
    """Render a CycloneDX 1.5 JSON SBOM for the installed dependency closure."""
    components: list[dict[str, Any]] = []
    for package in packages:
        component: dict[str, Any] = {
            "type": "library",
            "bom-ref": f"pkg:pypi/{package.canonical_name}@{package.version or 'unknown'}",
            "name": package.name,
            "version": package.version or "unknown",
            "purl": f"pkg:pypi/{package.canonical_name}@{package.version or 'unknown'}",
            "scope": _cyclonedx_scope(group_by_package.get(package.canonical_name, set())),
            "properties": [
                {
                    "name": "llm-tools:dependency-groups",
                    "value": ",".join(
                        sorted(group_by_package.get(package.canonical_name, set()))
                    ),
                }
            ],
        }
        if package.summary:
            component["description"] = package.summary
        if package.license:
            component["licenses"] = [{"license": {"name": package.license}}]
        if package.provenance_url:
            component["externalReferences"] = [
                {"type": "website", "url": package.provenance_url}
            ]
        components.append(component)

    payload = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "tools": [
                {
                    "vendor": "llm-tools",
                    "name": "generate_dependency_inventory.py",
                }
            ],
            "component": {
                "type": "application",
                "name": "llm-tools",
            },
        },
        "components": components,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def validate_inventory(
    dependencies: list[DirectDependency],
    purposes: dict[str, dict[str, str]],
    installed: dict[str, PackageMetadata],
    *,
    groups: set[str],
    require_installed: bool,
    require_purposes: bool,
) -> list[str]:
    """Return validation messages for missing metadata."""
    errors: list[str] = []
    for dependency in dependencies:
        if dependency.group not in groups:
            continue
        if require_installed and dependency.canonical_name not in installed:
            errors.append(f"Missing installed metadata for {dependency.requirement}.")
        if (
            require_purposes
            and dependency.canonical_name not in purposes.get(dependency.group, {})
        ):
            errors.append(
                f"Missing dependency purpose for "
                f"{dependency.group}.{dependency.canonical_name}."
            )
    return errors


def _metadata_license(raw: metadata.PackageMetadata) -> str | None:
    expression = _clean_optional(raw.get("License-Expression"))
    if expression:
        return expression
    classifiers = raw.get_all("Classifier") or []
    license_classifiers = [
        classifier.split("::")[-1].strip()
        for classifier in classifiers
        if classifier.startswith("License ::")
    ]
    if license_classifiers:
        return ", ".join(dict.fromkeys(license_classifiers))
    license_text = _clean_optional(raw.get("License"))
    if license_text is None:
        return None
    first_line = license_text.splitlines()[0].strip()
    return first_line[:120] if first_line else None


def _metadata_provenance_url(raw: metadata.PackageMetadata) -> str | None:
    project_urls = raw.get_all("Project-URL") or []
    preferred_labels = (
        "source",
        "repository",
        "homepage",
        "documentation",
        "changelog",
    )
    parsed_urls: list[tuple[str, str]] = []
    for entry in project_urls:
        if "," not in entry:
            continue
        label, url = entry.split(",", 1)
        parsed_urls.append((label.strip().lower(), url.strip()))
    for preferred in preferred_labels:
        for label, url in parsed_urls:
            if preferred in label and url:
                return url
    homepage = _clean_optional(raw.get("Home-page"))
    if homepage:
        return homepage
    if parsed_urls:
        return parsed_urls[0][1]
    return None


def _cyclonedx_scope(groups: set[str]) -> str:
    if groups == {"runtime"}:
        return "required"
    return "optional"


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    return cleaned or None


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _selected_groups(value: str) -> set[str]:
    if value == "all":
        return set(GROUPS)
    return {value}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate llm-tools dependency inventory output."
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help="Path to pyproject.toml.",
    )
    parser.add_argument(
        "--purposes",
        type=Path,
        default=DEFAULT_PURPOSES,
        help="Path to curated dependency purpose TOML.",
    )
    parser.add_argument(
        "--group",
        choices=("runtime", "dev", "all"),
        default="all",
        help="Dependency group to include.",
    )
    parser.add_argument(
        "--format",
        choices=sorted(ALL_FORMATS),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when direct dependency metadata or curated purposes are missing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write output to this path instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the dependency inventory command."""
    args = _parse_args(list(argv or sys.argv[1:]))
    groups = _selected_groups(args.group)
    dependencies = load_direct_dependencies(args.pyproject)
    purposes = load_dependency_purposes(args.purposes)
    installed = installed_distributions()

    errors = validate_inventory(
        dependencies,
        purposes,
        installed,
        groups=groups,
        require_installed=args.strict or args.format in TRANSITIVE_FORMATS,
        require_purposes=args.strict,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    if args.format in DIRECT_ONLY_FORMATS:
        rows = direct_inventory_rows(
            dependencies,
            purposes,
            installed,
            groups=groups,
        )
        output = render_markdown(rows)
    elif args.format == "json":
        rows = direct_inventory_rows(
            dependencies,
            purposes,
            installed,
            groups=groups,
        )
        output = render_json(rows)
    else:
        packages, group_by_package = transitive_package_closure(
            dependencies,
            installed,
            groups=groups,
        )
        output = render_cyclonedx_json(packages, group_by_package)

    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

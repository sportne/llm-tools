from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_dependency_inventory as inventory  # noqa: E402


def test_load_direct_dependencies_groups_runtime_and_dev(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
dependencies = ["Runtime-Pkg>=1"]

[project.optional-dependencies]
dev = ["Dev_Pkg>=2"]
""".strip(),
        encoding="utf-8",
    )

    dependencies = inventory.load_direct_dependencies(pyproject)

    assert [
        (dependency.group, dependency.name, dependency.canonical_name)
        for dependency in dependencies
    ] == [
        ("runtime", "Runtime-Pkg", "runtime-pkg"),
        ("dev", "Dev_Pkg", "dev-pkg"),
    ]


def test_markdown_renders_runtime_and_dev_tables() -> None:
    rows = [
        inventory.DirectInventoryRow(
            group="runtime",
            name="Runtime Package",
            requirement="runtime-package>=1",
            installed_version="1.2.3",
            purpose="Runtime behavior.",
            provenance_url="https://example.com/runtime",
            license="MIT",
        ),
        inventory.DirectInventoryRow(
            group="dev",
            name="Dev Package",
            requirement="dev-package>=2",
            installed_version=None,
            purpose=None,
            provenance_url=None,
            license=None,
        ),
    ]

    markdown = inventory.render_markdown(rows)

    assert "### Runtime Dependencies" in markdown
    assert "### Development Dependencies" in markdown
    assert "Runtime behavior." in markdown
    assert (
        "| Dev Package | dev-package>=2 | unknown | unknown | unknown | unknown |"
        in markdown
    )


def test_cyclonedx_output_contains_transitive_components() -> None:
    packages = [
        inventory.PackageMetadata(
            canonical_name="runtime-package",
            name="Runtime Package",
            version="1.2.3",
            summary="Runtime summary.",
            license="MIT",
            provenance_url="https://example.com/runtime",
            requires=(),
        )
    ]

    payload = json.loads(
        inventory.render_cyclonedx_json(packages, {"runtime-package": {"runtime"}})
    )

    assert payload["bomFormat"] == "CycloneDX"
    assert payload["specVersion"] == "1.5"
    assert payload["components"][0]["name"] == "Runtime Package"
    assert payload["components"][0]["scope"] == "required"


def test_strict_validation_reports_missing_purpose_and_metadata() -> None:
    dependencies = [
        inventory.DirectDependency(
            group="runtime",
            requirement="missing-package>=1",
            name="missing-package",
            canonical_name="missing-package",
        )
    ]

    errors = inventory.validate_inventory(
        dependencies,
        purposes={"runtime": {}, "dev": {}},
        installed={},
        groups={"runtime"},
        require_installed=True,
        require_purposes=True,
    )

    assert errors == [
        "Missing installed metadata for missing-package>=1.",
        "Missing dependency purpose for runtime.missing-package.",
    ]

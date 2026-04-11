"""Fail if any measured file falls below a minimum coverage threshold."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate per-file coverage from coverage.py JSON output."
    )
    parser.add_argument(
        "--input",
        default="coverage.json",
        help="Path to coverage.py JSON output.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Minimum required per-file coverage percentage.",
    )
    return parser.parse_args()


def _load_coverage_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _percent_covered(file_report: dict[str, Any]) -> float:
    summary = file_report.get("summary", {})
    percent = summary.get("percent_covered")
    if isinstance(percent, (int, float)):
        return float(percent)

    displayed = summary.get("percent_covered_display")
    if isinstance(displayed, str):
        return float(displayed)

    raise ValueError("Coverage JSON is missing per-file coverage summary data.")


def main() -> int:
    args = _parse_args()
    report = _load_coverage_report(Path(args.input))
    files = report.get("files", {})

    if not isinstance(files, dict) or not files:
        raise ValueError("Coverage JSON does not contain any measured files.")

    failures: list[tuple[str, float]] = []
    threshold = float(args.threshold)

    for filename, file_report in sorted(files.items()):
        if not isinstance(file_report, dict):
            continue
        percent = _percent_covered(file_report)
        if percent + 1e-9 < threshold:
            failures.append((filename, percent))

    if failures:
        print(f"Per-file coverage check failed: minimum required is {threshold:.1f}%")
        for filename, percent in failures:
            rounded = math.floor(percent * 10 + 0.5) / 10
            print(f"  {filename}: {rounded:.1f}%")
        return 1

    print(f"Per-file coverage check passed: all files are at least {threshold:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Print a compact summary from GTest JUnit XML files."""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _message(element: ET.Element) -> str:
    message = element.attrib.get("message", "")
    text = element.text or ""
    return " ".join((message or text).split())


def summarize(path: Path, max_failures: int) -> int:
    if not path.exists():
        print(f"{path}: report not found")
        return 1

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        print(f"{path}: failed to parse JUnit XML: {exc}")
        return 1

    failures: list[str] = []
    for testcase in root.iter("testcase"):
        test_name = f"{testcase.attrib.get('classname', '')}.{testcase.attrib.get('name', '')}".strip(
            "."
        )
        location = ""
        if "file" in testcase.attrib:
            location = testcase.attrib["file"]
            if "line" in testcase.attrib:
                location += f":{testcase.attrib['line']}"

        for tag in ("failure", "error"):
            for element in testcase.findall(tag):
                detail = _message(element)
                suffix = f" ({location})" if location else ""
                failures.append(f"{test_name}{suffix}: {detail}")

    print(f"JUnit summary for {path}:")
    if failures:
        print(f"  failed/error tests: {len(failures)}")
        for failure in failures[:max_failures]:
            print(f"  - {failure}")
        if len(failures) > max_failures:
            print(f"  ... {len(failures) - max_failures} more failures omitted")
    else:
        print("  failed/error tests: 0")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--max-failures", type=int, default=50)
    args = parser.parse_args()

    status = 0
    for report in args.reports:
        status |= summarize(report, args.max_failures)
    return status


if __name__ == "__main__":
    sys.exit(main())

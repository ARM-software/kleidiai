#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Sort marked lines; exit 1 if unsorted, 2 on errors, otherwise 0.

Run without arguments to discover tracked core source/configuration files, or
pass filenames for a targeted run. Region markers contain ``kai-keep-sorted-``
followed by ``begin`` or ``end``, using any comment style.
Use --check to report unsorted files without modifying them.
Each line must be an independent entry: multiline entries, attached comments,
and positional or order-dependent lists must remain outside marked regions.
"""
import argparse
import re
import sys
from pathlib import Path

import utils.git as git_utils
from utils.file_lists import gather_sortable_files

MARKER = re.compile(r"kai-keep-sorted-(begin|end)")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="files to sort")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report unsorted files without modifying them",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=git_utils.repo_root(),
        help="repository root for file discovery",
    )
    return parser.parse_args()


def sort_regions(content: str) -> str:
    """Sort complete lines, validating all markers before returning content."""
    lines = content.splitlines(keepends=True)
    begin = None
    for index, line in enumerate(lines):
        marker = MARKER.search(line)
        if marker is None:
            continue
        kind = marker.group(1)
        if kind == "begin":
            if begin is not None:
                raise ValueError(f"{index + 1}: nested {marker.group(0)}")
            begin = index
        else:
            if begin is None:
                raise ValueError(f"{index + 1}: unmatched {marker.group(0)}")
            lines[begin + 1 : index] = sorted(lines[begin + 1 : index])
            begin = None
    if begin is not None:
        raise ValueError(
            f"{begin + 1}: unmatched {MARKER.search(lines[begin]).group(0)}"
        )
    return "".join(lines)


def main() -> int:
    args = parse_arguments()
    try:
        files = args.files or gather_sortable_files(args.repo_root)
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 2

    result = 0
    for path in files:
        try:
            with path.open(encoding="utf-8", newline="") as source:
                content = source.read()
            sorted_content = sort_regions(content)
            if sorted_content != content:
                if args.check:
                    print(f"Unsorted {path}")
                else:
                    with path.open("w", encoding="utf-8", newline="") as destination:
                        destination.write(sorted_content)
                    print(f"Sorted {path}")
                result = max(result, 1)
        except (OSError, UnicodeError, ValueError) as error:
            print(f"{path}:{error}", file=sys.stderr)
            result = 2
    return result


if __name__ == "__main__":
    sys.exit(main())

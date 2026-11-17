#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Validate that implemented micro-kernels are listed in docs/microkernel_tables.md.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Set

import utils.file_lists as file_lists
import utils.git as git_utils

KERNEL_NAME_RE = re.compile(r"`(kai_[A-Za-z0-9_]+)`")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")


def resolve_path(repo_root: str, rel_path: str) -> str:
    path = os.path.join(repo_root, rel_path)
    return os.path.abspath(path)


def split_table_row(line: str) -> list[str]:
    content = line.strip()
    if content.startswith("|"):
        content = content[1:]
    if content.endswith("|"):
        content = content[:-1]
    return [cell.strip() for cell in content.split("|")]


def gather_documented_kernels(table_file: str) -> Set[str]:
    try:
        with open(table_file, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError as err:
        raise SystemExit(f"Cannot read table file '{table_file}': {err}") from err

    documented: Set[str] = set()
    index = 0
    while index + 1 < len(lines):
        # Find the start of the next table in the table file
        header_line = lines[index]
        separator_line = lines[index + 1]

        if not header_line.lstrip().startswith("|") or not TABLE_SEPARATOR_RE.match(
            separator_line
        ):
            index += 1
            continue

        # Find the index of the column that lists the micro-kernel names
        headers = split_table_row(header_line)
        microkernel_column = None
        for column_index, header in enumerate(headers):
            if header == "Micro-kernel":
                microkernel_column = column_index
                break

        index += 2
        while index < len(lines):
            # Iterate over the entries in the table and add the micro-kernel names to the set that is used
            # to keep track of the documented micro-kernels
            row = lines[index]
            if not row.lstrip().startswith("|"):
                break
            cells = split_table_row(row)
            if (
                microkernel_column is not None
                and microkernel_column < len(cells)
                and cells[microkernel_column]
            ):
                documented.update(KERNEL_NAME_RE.findall(cells[microkernel_column]))
            index += 1

    return documented


def main() -> int:
    root_dir = git_utils.repo_root()
    ukernels_dir = resolve_path(root_dir, "kai/ukernels")
    table_file = resolve_path(root_dir, "docs/microkernel_tables.md")

    assert os.path.isdir(
        ukernels_dir
    ), f"Micro-kernel directory not found: {ukernels_dir}"
    assert os.path.isfile(table_file), f"Table file not found: {table_file}"

    implemented = {
        kernel.name for kernel in file_lists.gather_microkernels(ukernels_dir)
    }
    documented = gather_documented_kernels(table_file)

    missing = sorted(implemented - documented)
    if missing:
        print(f"Missing micro-kernels in {table_file}:")
        for kernel_name in missing:
            print(kernel_name)
        return 1

    non_public = sorted(documented - implemented)
    if non_public:
        print(f"Non-public micro-kernels in {table_file}:")
        for kernel_name in non_public:
            print(kernel_name)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Validate that every matmul, imatmul, or dwconv micro-kernel header has a benchmark registration.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Optional
from typing import Sequence
from typing import Set

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]')
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)

SRC_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}


def strip_comments(text: str) -> str:
    text = BLOCK_COMMENT_RE.sub("", text)
    text = LINE_COMMENT_RE.sub("", text)
    return text


def iter_files(root: str):
    for dirpath, _, files in os.walk(root):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in SRC_EXTS:
                yield os.path.join(dirpath, filename)


def classify_kernel(path: str) -> Optional[str]:
    parts = path.split("/")
    if not parts:
        return None
    head = parts[0]
    if head == "dwconv":
        return "dwconv"
    if head != "matmul":
        return None
    if len(parts) > 1 and parts[1].startswith("imatmul"):
        return "imatmul"
    return "matmul"


def gather_includes(
    src_dir: str, include_prefix: str, kernel_types: Set[str]
) -> Set[str]:
    used: Set[str] = set()
    # Normalize include prefix for comparison so that Windows and Unix paths match
    prefix = include_prefix.replace("\\", "/").rstrip("/")
    plen = len(prefix)
    for path in iter_files(src_dir):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = strip_comments(fh.read())
        except OSError:
            continue
        for line in content.splitlines():
            match = INCLUDE_RE.match(line)
            if not match:
                continue
            inc = match.group(1).replace("\\", "/")
            if not inc.startswith(prefix):
                continue
            rel = inc[plen + 1 :]
            kernel_type = classify_kernel(rel)
            if kernel_type in kernel_types:
                used.add(rel)
    return used


def list_present(ukernels_dir: str, kernel_types: Set[str]) -> Set[str]:
    present: Set[str] = set()
    base = os.path.abspath(ukernels_dir)
    for dirpath, _, files in os.walk(base):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() not in {".h", ".hh", ".hpp", ".hxx"}:
                continue
            rel_path = os.path.relpath(os.path.join(dirpath, filename), base).replace(
                "\\", "/"
            )
            if rel_path.startswith(".."):  # Skip anything outside the tree
                continue
            # Skip packing kernels and interface headers, as benchmarks only cover concrete micro-kernels
            if "/pack/" in rel_path:
                continue
            if rel_path.endswith("_interface.h"):
                continue
            kernel_type = classify_kernel(rel_path)
            if kernel_type in kernel_types:
                present.add(rel_path)
    return present


def parse_kernel_types(values: Sequence[str]) -> Set[str]:
    allowed = {"matmul", "imatmul", "dwconv"}
    requested = set(value.lower() for value in values)
    unknown = requested.difference(allowed)
    if unknown:
        raise SystemExit(
            f"Unknown kernel types requested: {', '.join(sorted(unknown))}"
        )
    return requested


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        default="benchmark",
        help="Directory to scan for benchmark micro-kernel registrations (default: %(default)s)",
    )
    parser.add_argument(
        "--ukernels-dir",
        default="kai/ukernels",
        help="Directory containing micro-kernel headers (default: %(default)s)",
    )
    parser.add_argument(
        "--kernel-types",
        nargs="+",
        default=["matmul", "imatmul", "dwconv"],
        help="Kernel families to verify (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    kernel_types = parse_kernel_types(args.kernel_types)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    def resolve_dir(value: str, default: str) -> str:
        raw = os.path.expanduser(value)
        if value == default:
            raw = os.path.join(repo_root, default)
        return os.path.abspath(raw)

    benchmark_dir = resolve_dir(args.benchmark_dir, parser.get_default("benchmark_dir"))
    ukernels_dir = resolve_dir(args.ukernels_dir, parser.get_default("ukernels_dir"))

    if not os.path.isdir(benchmark_dir):
        raise SystemExit(f"Benchmark directory not found: {benchmark_dir}")
    if not os.path.isdir(ukernels_dir):
        raise SystemExit(f"Micro-kernel directory not found: {ukernels_dir}")

    present = list_present(ukernels_dir, kernel_types)

    include_prefix = os.path.relpath(ukernels_dir, repo_root).replace("\\", "/")
    if include_prefix.startswith(".."):
        raise SystemExit(
            f"Micro-kernel directory must reside within the repo: {ukernels_dir}"
        )

    used = gather_includes(benchmark_dir, include_prefix, kernel_types)

    unused = sorted(present - used)
    if unused:
        grouped = defaultdict(list)
        for rel in unused:
            grouped[classify_kernel(rel)].append(rel)
        print("Missing benchmark registrations for the following micro-kernel headers:")
        for kernel_type in sorted(grouped):
            print(f"{kernel_type}:")
            for rel in grouped[kernel_type]:
                print(f"  - {rel}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

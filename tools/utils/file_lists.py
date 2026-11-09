#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers for gathering lists."""
from dataclasses import dataclass
from pathlib import Path

import utils.git as git_utils


def gather_sortable_files(repo_root: Path) -> list[Path]:
    """
    Discover tracked core source/configuration files for marked sorting.
    """
    extensions = {
        # kai-keep-sorted-begin
        ".c",
        ".clang-format",
        ".clang-tidy",
        ".cmake",
        ".cpp",
        ".gitignore",
        ".h",
        ".hpp",
        ".md",
        ".py",
        ".toml",
        ".txt",
        # kai-keep-sorted-end
    }

    exclude_dirs = {"experimental", "third_party", "LICENSES"}

    files = []

    for candidate in git_utils.get_versioned_files(root_dir=repo_root):
        if candidate.parts[0] in exclude_dirs:
            continue

        if candidate.suffix not in extensions:
            continue

        path = repo_root / candidate
        if not path.is_symlink():
            files.append(path)

    return files


@dataclass(frozen=True, order=True)
class Microkernel:
    operation: str
    directory: str
    name: str


@dataclass(frozen=True)
class MicrokernelHeader:
    """A micro-kernel family API header."""

    # Micro-kernel family directory name, for example matmul or dwconv.
    family: str
    # Header is for compute micro-kernels
    is_compute: bool
    # Path to the family API header.
    path: Path


def gather_microkernel_headers(ukernels_dir: Path) -> list[MicrokernelHeader]:
    """Gather sorted family API headers."""
    headers: list[MicrokernelHeader] = []
    for directory in sorted(ukernels_dir.iterdir()):
        if not directory.is_dir():
            continue
        for header in sorted(directory.glob("kai_*.h")):
            if header.stem.endswith("_types"):
                continue
            headers.append(
                MicrokernelHeader(
                    family=directory.name,
                    is_compute="pack" not in header.stem.split("_"),
                    path=header,
                )
            )
    return headers


def gather_microkernels(ukernels_dir: str) -> set[Microkernel]:
    """Gather micro-kernels implemented as .c files under a ukernels directory."""
    root = Path(ukernels_dir)
    microkernels: set[Microkernel] = set()

    for path in root.rglob("*.c"):
        relative_path = path.relative_to(root)
        if len(relative_path.parts) < 2:
            continue

        microkernels.add(
            Microkernel(
                operation=relative_path.parts[0],
                directory=path.parent.name,
                name=path.stem,
            )
        )

    return microkernels

#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers for discovering implemented micro-kernels."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class Microkernel:
    operation: str
    directory: str
    name: str


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

#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Check the names of all microkernels."""
import argparse
from pathlib import Path

import naming.rules as naming
import utils.git as git_utils
import utils.microkernels as microkernels
from naming.grammar import MatchFailure
from naming.issues import KNOWN_DIRECTORY_PROBLEMS
from naming.issues import KNOWN_UKERNEL_PROBLEMS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # REVISIT: Add mode for generating new naming documentation

    parser.add_argument(
        "--ukernels-dir",
        type=Path,
        help="ukernels directory to scan (default: (root)/kai/ukernels)",
        default=git_utils.repo_root() / "kai" / "ukernels",
    )

    parser.add_argument(
        "--dump-grammar",
        action="store_true",
        help="dump the naming grammar and exit",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first invalid kernel or directory name",
    )

    return parser.parse_args()


def report_invalid_name(name: str, match: MatchFailure) -> None:
    failure = match.furthest_failure()
    assert failure is not None

    print(f"  {name}")
    print(f"  {' ' * failure.pos}^")
    print(f"  expected: {failure.expected}")
    print(f"  remaining: {name[failure.pos:]}")


def report_invalid_kernel(
    kernel: microkernels.Microkernel, match: MatchFailure
) -> None:
    print(
        "Invalid micro-kernel name in "
        f"{kernel.operation}/{kernel.directory}: {kernel.name}"
    )
    report_invalid_name(kernel.name, match)


def report_invalid_directory(
    kernel: microkernels.Microkernel, match: MatchFailure
) -> None:
    print(f"Invalid micro-kernel directory in {kernel.operation}: {kernel.directory}")
    report_invalid_name(kernel.directory, match)


def check_directory_tree(directory, fail_fast: bool) -> int:
    assert directory.is_dir(), f"{directory} not found"
    kernels = sorted(microkernels.gather_microkernels(directory))
    misnamed_kernels = 0
    misnamed_directories = 0
    for kernel in kernels:
        is_known_ukernel_issue = kernel.name in KNOWN_UKERNEL_PROBLEMS
        match_result = naming.match_kernel_name(kernel.name)
        if not match_result.matched and not is_known_ukernel_issue:
            report_invalid_kernel(kernel, match_result)
            misnamed_kernels += 1
            if fail_fast:
                return 1
        elif is_known_ukernel_issue and match_result.matched:
            print(
                f"WARNING: {kernel.name} is listed as a known issue, but passes name check"
            )

        is_known_directory_issue = kernel.directory in KNOWN_DIRECTORY_PROBLEMS
        directory_match_result = naming.match_directory_name(kernel.directory)
        if not directory_match_result.matched and not is_known_directory_issue:
            report_invalid_directory(kernel, directory_match_result)
            misnamed_directories += 1
            if fail_fast:
                return 1
        elif is_known_directory_issue and directory_match_result.matched:
            print(
                f"WARNING: {kernel.directory} is listed as a known issue, but passes name check"
            )

        # REVISIT, in the future we could compare directory name with microkernel name

    print(f"There are {misnamed_kernels} kernels with invalid names")
    print(f"There are {misnamed_directories} kernels in invalid directories")
    return 0 if misnamed_kernels == 0 and misnamed_directories == 0 else 1


def main() -> int:
    args = parse_arguments()

    if args.dump_grammar:
        print(naming.grammar.to_grammar())
        return 0

    return check_directory_tree(args.ukernels_dir, args.fail_fast)


if __name__ == "__main__":
    raise SystemExit(main())

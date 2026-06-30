#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Check the names of all microkernels."""
import argparse
import sys
from pathlib import Path

import naming.documentation as naming_documentation
import naming.rules as naming
import utils.git as git_utils
import utils.microkernels as microkernels
from naming.grammar import ParseFailure
from naming.issues import KNOWN_DIRECTORY_PROBLEMS
from naming.issues import KNOWN_UKERNEL_PROBLEMS
from naming.issues import KnownIssue
from utils.annotate import annotate


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = git_utils.repo_root()

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dump-grammar",
        action="store_true",
        help="dump the naming grammar and exit",
    )
    mode.add_argument(
        "--generate-documentation",
        nargs="?",
        type=Path,
        const=repo_root / "docs" / "microkernel_names.md",
        metavar="FILE",
        help=(
            "generate micro-kernel naming documentation and exit "
            "(default: (repo_root)/docs/microkernel_names.md). Using `-` as path writes to stdout."
        ),
    )
    mode.add_argument(
        "--describe-ukernel",
        metavar="NAME",
        help="describe a micro-kernel name and exit",
    )

    parser.add_argument(
        "--ukernels-dir",
        type=Path,
        help="ukernels directory to scan (default: (repo_root)/kai/ukernels)",
        default=repo_root / "kai" / "ukernels",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first invalid kernel or directory name",
    )
    parser.add_argument(
        "--report-all",
        "--report-known-issues",
        action="store_true",
        dest="report_known_issues",
        help="report known invalid names instead of disregarding them",
    )

    return parser.parse_args()


def report_invalid_name(name: str, parse_result: ParseFailure) -> None:
    failure = parse_result.furthest_failure()
    assert failure is not None

    print(f"  {name}")
    print(f"  {' ' * failure.pos}^")
    print(f"  expected: {failure.expected}")
    print(f"  remaining: {name[failure.pos:]}")


def report_known_issue(issue: KnownIssue) -> None:
    print(f"  known issue: {issue.description}")
    print(f"  expected name: {issue.expected}")


def report_invalid_kernel(
    kernel: microkernels.Microkernel, parse_result: ParseFailure
) -> None:
    print(
        "Invalid micro-kernel name in "
        f"{kernel.operation}/{kernel.directory}: {kernel.name}"
    )
    report_invalid_name(kernel.name, parse_result)


def report_invalid_directory(
    kernel: microkernels.Microkernel, parse_result: ParseFailure
) -> None:
    print(f"Invalid micro-kernel directory in {kernel.operation}: {kernel.directory}")
    report_invalid_name(kernel.directory, parse_result)


def check_directory_tree(
    directory: Path, fail_fast: bool, report_known_issues: bool
) -> int:
    assert directory.is_dir(), f"{directory} not found"
    kernels = sorted(microkernels.gather_microkernels(directory))
    misnamed_kernels = 0
    misnamed_directories = 0
    for kernel in kernels:
        known_ukernel_issue = KNOWN_UKERNEL_PROBLEMS.get(kernel.name)
        parse_result = naming.parse_kernel_name(kernel.name)
        if not parse_result.parsed_ok and (
            known_ukernel_issue is None or report_known_issues
        ):
            report_invalid_kernel(kernel, parse_result)
            if known_ukernel_issue is not None:
                report_known_issue(known_ukernel_issue)
            misnamed_kernels += 1
            if fail_fast:
                return 1
        elif known_ukernel_issue is not None and parse_result.parsed_ok:
            print(
                f"WARNING: {kernel.name} is listed as a known issue, but passes name check"
            )

        known_directory_issue = KNOWN_DIRECTORY_PROBLEMS.get(kernel.directory)
        directory_parse_result = naming.parse_directory_name(kernel.directory)
        if not directory_parse_result.parsed_ok and (
            known_directory_issue is None or report_known_issues
        ):
            report_invalid_directory(kernel, directory_parse_result)
            if known_directory_issue is not None:
                report_known_issue(known_directory_issue)
            misnamed_directories += 1
            if fail_fast:
                return 1
        elif known_directory_issue is not None and directory_parse_result.parsed_ok:
            print(
                f"WARNING: {kernel.directory} is listed as a known issue, but passes name check"
            )

        # REVISIT, in the future we could compare directory name with microkernel name

    print(f"There are {len(kernels)} checked kernels")
    print(f"There are {misnamed_kernels} kernels with invalid names")
    print(f"There are {misnamed_directories} kernels in invalid directories")
    return 0 if misnamed_kernels == 0 and misnamed_directories == 0 else 1


def generate_documentation(path: Path) -> None:
    if path == Path("-"):
        sys.stdout.write(naming_documentation.render_documentation())
        return

    path.write_text(naming_documentation.render_documentation(), encoding="utf-8")
    print(f"Generated micro-kernel naming documentation: {path}")


def describe_ukernel_name(name: str) -> int:
    parse_result = naming.parse_kernel_name(name)
    if not parse_result.parsed_ok:
        print("Invalid micro-kernel name:")
        report_invalid_name(name, parse_result)
        return 1

    print(annotate(name, parse_result.descriptions))
    return 0


def main() -> int:
    args = parse_arguments()

    if args.dump_grammar:
        print(naming.grammar.to_grammar())
        return 0

    if args.generate_documentation is not None:
        generate_documentation(args.generate_documentation)
        return 0

    if args.describe_ukernel is not None:
        return describe_ukernel_name(args.describe_ukernel)

    return check_directory_tree(
        args.ukernels_dir, args.fail_fast, args.report_known_issues
    )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Verifies that every header file under kai/ can be included on its own.

This script does this by collecting all headers under the kai/ directory and
for each header, generates a temporary .c/.cpp file that includes the header
and compiles the source file with the flag -fsyntax-only. It does this with
-std=c99 and -std=c++17.

If any headers fail, the script prints the failing headers, language mode,
compiler command and compiler stderr. The script then exits by returning 1.

If all headers pass, the script prints the number of checked headers and exits
by returning 0.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from typing import Sequence

import utils.git as git_utils

HEADER_EXTS = {".h"}


class Language(Enum):
    C = "c"
    CPP = "c++"

    def suffix(self) -> str:
        return {Language.C: ".c", Language.CPP: ".cpp"}[self]

    def standard(self) -> str:
        return {Language.C: "c99", Language.CPP: "c++17"}[self]


LANGUAGES = [Language.C, Language.CPP]


@dataclass(frozen=True)
class CheckFailure:
    header: str
    language: str
    command: Sequence[str]
    stderr: str


def default_target() -> str:
    if sys.platform == "darwin":
        return "arm64-apple-darwin"
    return "aarch64-linux-gnu"


def resolve_compiler(name: Optional[str]) -> str:
    """Returns the full path of the compiler binary. If no compiler is
    specified as argument to this function, it attempts to read the
    environment variable CC. If that also fails, the function falls
    back to assuming clang."""
    compiler = name or os.environ.get("CC") or "clang"
    path = shutil.which(compiler)
    assert path, f"Compiler not found: {compiler}"
    return path


def iter_headers(kai_dir: str) -> list[str]:
    headers: list[str] = []
    for dirpath, _, filenames in os.walk(kai_dir):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in HEADER_EXTS:
                headers.append(os.path.join(dirpath, filename))
    return sorted(headers)


def check_header(
    compiler: str,
    repo_dir: str,
    header: str,
    language: Language,
    target: Optional[str],
    verbose: bool,
) -> Optional[CheckFailure]:
    """Verifies that header can be included standalone by a c/cpp file.
    This is done by creating a temporary c/cpp file that just includes
    the header file and then compiled by the specified compiler using
    the -fsyntax-only flag."""
    rel_header = os.path.relpath(header, repo_dir).replace(os.sep, "/")
    suffix = language.suffix()
    standard = language.standard()

    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, f"include_check{suffix}")
        with open(source, "w", encoding="utf-8") as source_file:
            source_file.write(f'#include "{rel_header}"\n')

        command = [compiler, "-fsyntax-only", f"-std={standard}", "-I", repo_dir]
        if target:
            command.extend(["-target", target])
        command.append(source)

        if verbose:
            print(f"{command}")

        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    if result.returncode == 0:
        return None

    return CheckFailure(
        header=rel_header,
        language=language.value,
        command=command,
        stderr=result.stderr.strip(),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler", help="Compiler to use for syntax checks (default: CC or clang)"
    )
    parser.add_argument(
        "--target",
        default=default_target(),
        help=(
            "Compiler target triple. Pass an empty string to use the compiler "
            "default (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each compiler invocation before running it",
    )
    args = parser.parse_args(argv)

    repo_dir = git_utils.repo_root()
    kai_dir = os.path.abspath(os.path.join(repo_dir, "kai"))

    if not os.path.isdir(kai_dir):
        raise SystemExit(f"Micro-kernel directory not found: {kai_dir}")

    compiler = resolve_compiler(args.compiler)
    target = args.target or None
    headers = iter_headers(kai_dir)

    failures: list[CheckFailure] = []
    for header in headers:
        for language in LANGUAGES:
            failure = check_header(
                compiler, repo_dir, header, language, target, args.verbose
            )
            if failure is not None:
                failures.append(failure)

    if failures:
        print("Headers that failed standalone include checks:")
        for failure in failures:
            print(f"\n{failure.header} ({failure.language})")
            print("Command:")
            print(f"   {failure.command}")
            if failure.stderr:
                print("Compiler output:")
                print(failure.stderr)
        return 1

    print(
        f"Checked {len(headers)} header(s) under {os.path.relpath(kai_dir, repo_dir)} "
        f"as {', '.join([lang.value for lang in LANGUAGES])}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

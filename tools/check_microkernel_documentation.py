#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
"""Check function documentation in kai/ukernels/<family>/kai_*.h API headers.

Each kernel should be documented using the following pattern

    /// <Title>
    ///
    /// (Optional) <Detailed description>
    ///
    /// (Optional) Required CPU features:
    ///   <Bullet list>
    ///
    /// (Optional) Configuration parameters:
    ///   <Bullet list>
    ///
    /// (Optional) Required operands:
    ///   <Bullet list>
    ///
    /// (Optional) Optional arguments:
    ///   <Bullet list>
    ///
    /// (Optional) Accumulation: <Description>
    ///
    /// (Optional) Supported flags:
    ///   <Bullet list>
    ///
    /// @return The micro-kernel API.
    struct kai_<family>_uker_api kai_<family>_<description>(void);
"""
import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import naming.rules as naming
import utils.git as git_utils
import utils.microkernels as microkernels
from naming.issues import KNOWN_UKERNEL_PROBLEMS

RETURN = "@return The micro-kernel API."
HEADING_RE = re.compile(r"[^:]+:(?: .*)?")
TOP_LEVEL_BULLET_RE = re.compile(r"  \* \S.*")
NESTED_BULLET_RE = re.compile(r" {4,}[*-] \S.*")

# Known declarations using a function prefix that differs from their API family.
KNOWN_DECLARATION_PROBLEMS = {
    "struct kai_matmul_pack_rhs_uker_api kai_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve(void);": (
        "struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_kxn_x16p16vsx2bx16_x16_x16_sve(void);"
    ),
}


@dataclass
class SourceLine:
    """A line of source code."""

    # One-based line number in the source file.
    offset: int
    # Source text without the trailing newline.
    text: str


@dataclass
class DocumentedFunction:
    """A function declaration and its preceding documentation lines."""

    declaration: SourceLine
    documentation: list[SourceLine]


def is_heading(text: str) -> bool:
    """Return whether unindented text has a heading and optional text after ':'."""
    return text == text.lstrip() and HEADING_RE.fullmatch(text) is not None


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments and return their namespace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=git_utils.repo_root(),
        help="Repository directory to check (default: detected repository root)",
    )
    return parser.parse_args()


class HeaderChecker:
    """Check one API header and collect its diagnostics."""

    def __init__(self, path: Path, api_name: str) -> None:
        """Initialize the header location, API name and per-file errors."""
        self.path = path
        self.api_name = api_name
        self._declaration_re = re.compile(
            rf"struct kai_{api_name}_uker_api kai_{api_name}_[A-Za-z0-9_]+\(void\);"
        )
        self.errors: list[str] = []
        self.function_count = 0

    def _report_error(self, message: str, line: SourceLine | None = None) -> None:
        """Collect a diagnostic with the header's path and optional line number."""
        if line is not None:
            message = f"{line.offset}: {message}"
        self.errors.append(f"{self.path}:{message}")

    def _split_documentation(self, lines: list[SourceLine]) -> list[list[SourceLine]]:
        """Remove /// prefixes and return blocks separated by blank /// lines."""
        content = []
        for line in lines:
            assert line.text.startswith("///"), "Expected validated /// prefixes"
            text = line.text[3:]
            if text and not text.startswith(" "):
                self._report_error("Expected a space after ///", line)
            content.append(
                SourceLine(line.offset, text[1:] if text.startswith(" ") else text)
            )
        return [
            list(group)
            for nonempty, group in groupby(
                content, key=lambda line: bool(line.text.strip())
            )
            if nonempty
        ]

    def _check_paragraph(self, lines: list[SourceLine]) -> None:
        """Check prose, allowing colons within sentences but no bullets or directives."""
        for line in lines:
            if line.text.endswith(":") or line.text.startswith("@"):
                self._report_error(f"Expected paragraph text: {line.text}", line)
            if line.text.lstrip().startswith(("*", "-", "+")):
                self._report_error(
                    f"Bullets must belong to a list section: {line.text}", line
                )

    def _check_bullet_list(self, lines: list[SourceLine], heading: str) -> None:
        """Check a specific heading followed by an indented bullet list."""
        first = lines[0]
        if first.text != heading:
            self._report_error(f"Expected '{heading}': {first.text}", first)
        if len(lines) == 1:
            self._report_error(f"List section requires bullets: {heading}", first)
            return
        if not TOP_LEVEL_BULLET_RE.fullmatch(lines[1].text):
            self._report_error(
                f"Expected a top-level * bullet: {lines[1].text}", lines[1]
            )
        for line in lines[1:]:
            # A new top-level bullet starts with two spaces and *
            if TOP_LEVEL_BULLET_RE.fullmatch(line.text):
                continue
            # A nested bullet starts with at least four spaces and * or -
            if NESTED_BULLET_RE.fullmatch(line.text):
                continue
            # A continuation has at least four spaces and no bullet marker
            if not line.text.startswith("    ") or line.text.lstrip().startswith(
                ("*", "-", "+")
            ):
                self._report_error(
                    f"Expected an indented bullet or continuation: {line.text}", line
                )

    def _check_description(self, lines: list[SourceLine], heading: str) -> None:
        """Check a heading with inline description text and continuation lines."""
        first = lines[0]
        if not first.text.startswith(heading):
            self._report_error(f"Expected '{heading}': {first.text}", first)
        if not first.text[len(heading) :].strip():
            self._report_error(
                f"'{heading}' requires text after the colon: {first.text}", first
            )
        self._check_paragraph(lines[1:])

    def _check_documentation(self, lines: list[SourceLine]) -> None:
        """Check a title, description, ordered optional sections, and final @return."""
        if not lines:
            self._report_error("Missing function documentation")
            return
        blocks = self._split_documentation(lines)
        if not blocks:
            self._report_error("Empty documentation block", lines[0])
            return

        sections: tuple[tuple[str, Callable[[list[SourceLine], str], None]], ...] = (
            ("Required CPU features:", self._check_bullet_list),
            ("Configuration parameters:", self._check_bullet_list),
            ("Required operands:", self._check_bullet_list),
            ("Optional arguments:", self._check_bullet_list),
            ("Accumulation:", self._check_description),
            ("Supported flags:", self._check_bullet_list),
        )

        # Check the title first
        if is_heading(blocks[0][0].text):
            self._report_error("Expected a title before headings", blocks[0][0])
        self._check_paragraph(blocks[0])

        # Check description paragraphs before the first parameter or @return block.
        # A description sentence may itself contain a colon.
        headings = [heading for heading, _ in sections]
        index = 1
        while index < len(blocks):
            text = blocks[index][0].text
            if text.partition(":")[0] + ":" in headings or text.startswith("@"):
                break
            self._check_paragraph(blocks[index])
            index += 1

        # Consume each parameter section at most once, in the required order
        for heading, checker in sections:
            if index == len(blocks):
                break
            text = blocks[index][0].text
            if not text.startswith(heading):
                continue
            if not is_heading(text):
                self._report_error(f"Expected a heading: {text}", blocks[index][0])
            checker(blocks[index], heading)
            index += 1

        # Only the exact @return block may remain
        if index != len(blocks) - 1:
            self._report_error(f"Expected one final '{RETURN}' block", lines[-1])
        for block in blocks[index:]:
            if len(block) != 1 or block[0].text != RETURN:
                self._report_error(
                    f"Expected exactly '{RETURN}': {block[0].text}", block[0]
                )

    def _check_section(self, function: DocumentedFunction) -> None:
        """Check one declaration and its preceding documentation."""
        line = function.declaration.text

        # Read the declaration, but allow pre-existing known problems
        declaration = self._declaration_re.fullmatch(
            KNOWN_DECLARATION_PROBLEMS.get(line, line),
        )
        if declaration is None:
            self._report_error(
                f"Expected 'struct kai_{self.api_name}_uker_api "
                f"kai_{self.api_name}_<description>(void);': {line}",
                function.declaration,
            )
        else:
            name = line.split()[2].removesuffix("(void);")
            parse_result = naming.parse_kernel_name(name)
            if not parse_result.parsed_ok and name not in KNOWN_UKERNEL_PROBLEMS:
                failure = parse_result.furthest_failure()
                assert failure is not None
                self._report_error(
                    f"Invalid micro-kernel name '{name}': "
                    f"expected {failure.expected} at character {failure.pos + 1}; "
                    f"remaining: {name[failure.pos:]}",
                    function.declaration,
                )
        if not function.documentation:
            self._report_error("Missing /// documentation", function.declaration)
            return
        valid_prefixes = True
        for line in function.documentation:
            if not line.text.startswith("///"):
                self._report_error(
                    f"Expected /// documentation before declaration: {line.text}", line
                )
                valid_prefixes = False
        if valid_prefixes:
            self._check_documentation(function.documentation)

    def check_header(self) -> list[str]:
        """Check the API header and return its collected diagnostics."""
        # Attach one-based source offsets before trimming the file
        try:
            lines = [
                SourceLine(offset, text)
                for offset, text in enumerate(
                    self.path.read_text(encoding="utf-8").splitlines(), start=1
                )
            ]
        except (OSError, UnicodeError) as error:
            self._report_error(f" {error}")
            return self.errors

        # Skip the preamble, until first declaration.
        start = 0
        while start < len(lines):
            line = lines[start].text
            if (
                not line.strip()
                or line.startswith(("//", "#"))
                or line == 'extern "C" {'
            ):
                start += 1
            else:
                break

        # Rewind to put `start` at beginning of first declaration documentation
        while start and lines[start - 1].text.startswith("///"):
            start -= 1

        # Trim end until last declaration
        end = len(lines)
        while end and not lines[end - 1].text.strip():
            end -= 1
        if [line.text for line in lines[end - 3 : end]] == [
            "#ifdef __cplusplus",
            '}  // extern "C"',
            "#endif",
        ]:
            end -= 3

        # Split on empty lines and omit the groups containing only empty lines
        sections = [
            list(group)
            for nonempty, group in groupby(
                lines[start:end], key=lambda line: bool(line.text.strip())
            )
            if nonempty
        ]
        if not sections:
            self._report_error(" No micro-kernel API declarations found")

        # Each section must contain documentation followed by one declaration
        for section in sections:
            function = DocumentedFunction(section[-1], section[:-1])
            self._check_section(function)
        self.function_count = len(sections)
        return self.errors


def check_headers(ukernels_dir: Path) -> tuple[int, int, list[str]]:
    """Check family API headers and return header/function counts and errors."""
    errors: list[str] = []
    if not ukernels_dir.is_dir():
        errors.append(f"{ukernels_dir}: Micro-kernel directory not found")
        return 0, 0, errors
    headers = 0
    functions = 0
    try:
        for header in microkernels.gather_microkernel_headers(ukernels_dir):
            headers += 1
            checker = HeaderChecker(
                path=header.path,
                api_name=header.path.stem.removeprefix("kai_"),
            )
            errors.extend(checker.check_header())
            functions += checker.function_count
    except OSError as error:
        errors.append(str(error))
        return headers, functions, errors
    if not headers:
        errors.append(f"{ukernels_dir}: No micro-kernel API headers found")
    return headers, functions, errors


def main() -> int:
    """Run the checks and return exit status 0 on success."""
    args = parse_arguments()
    headers, functions, errors = check_headers(args.repo_root / "kai/ukernels")
    print(f"Checked {headers} header(s), {functions} function(s).")
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

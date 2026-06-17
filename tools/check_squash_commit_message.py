#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Check that the merge request title is valid for use as a squash/merge commit message.
"""
import os
import re
import sys

ALLOWED_TYPES = "major, feat, fix, docs, chore"
TITLE_RE = re.compile(r"^(?:major|feat|fix|docs|chore): \S(?:.*\S)?$")


def main() -> int:
    assert (
        "CI_MERGE_REQUEST_TITLE" in os.environ
    ), "CI_MERGE_REQUEST_TITLE environment variable is not set."
    title = os.environ["CI_MERGE_REQUEST_TITLE"]

    if os.environ["CI_MERGE_REQUEST_DRAFT"] == "true":
        print("Merge request squash-commit validation skipped for draft merge request.")
        raise SystemExit(0)

    if not TITLE_RE.fullmatch(title.strip()):
        print("Merge request squash-commit validation failed:")
        print("- Merge request title must match '<type>: <description>'.")
        print(f"- Allowed types: {ALLOWED_TYPES}.")
        print(f"- Received title: {title!r}")
        raise SystemExit(1)

    print("Merge request squash-commit validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

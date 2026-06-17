#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return_code, output = git_run(
        ["rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
    )
    if return_code != 0:
        return Path(__file__).resolve().parent.parent.parent
    return Path(output)


def is_shallow_repo(root_dir: Path | None = None) -> bool:
    cwd = root_dir or repo_root()
    return_code, output = git_run(
        ["rev-parse", "--is-shallow-repository"],
        cwd=cwd,
    )
    if return_code != 0:
        return False
    return output == "true"


def fetch_tags(root_dir: Path | None = None) -> tuple[int, str]:
    cwd = root_dir or repo_root()
    fetch_args = [
        "fetch",
        "--force",
        "--prune",
        "-q",
        "origin",
        "refs/tags/*:refs/tags/*",
    ]
    if is_shallow_repo(cwd):
        fetch_args.insert(1, "--unshallow")
    return git_run(fetch_args, cwd=cwd)


def has_local_changes(
    paths: list[str] | None = None, *, root_dir: Path | None = None
) -> bool:
    cwd = root_dir or repo_root()
    args = ["status", "--porcelain"]
    if paths:
        args.extend(["--", *paths])
    _, output = git_run(args, cwd=cwd)
    return bool(output.strip())


def get_commits(
    start_ref: str,
    end_ref: str = "HEAD",
    *,
    root_dir: Path | None = None,
) -> list[str]:
    cwd = root_dir or repo_root()
    _, output = git_run(
        ["log", "--format=%s", f"{start_ref}..{end_ref}"],
        cwd=cwd,
    )
    if not output:
        return []
    return output.splitlines()


def latest_tag_matching(pattern: str, *, root_dir: Path | None = None) -> str | None:
    cwd = root_dir or repo_root()
    return_code, output = git_run(["tag", "--sort=-version:refname"], cwd=cwd)
    if return_code != 0:
        return None

    compiled_pattern = re.compile(pattern)
    for tag in output.splitlines():
        if compiled_pattern.match(tag):
            return tag
    return None


def git_run(args: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    cmd = ["git", *args]
    print(f"Running command: {shlex.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip()

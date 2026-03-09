#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Brief: check_api_compat.py — build current sources using test/benchmark from the
#        latest tagged release to detect API breakage.
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import utils.git as git_utils


ROOT_DIR = git_utils.repo_root()
BUILD_DIR = "build-api-compat"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"Running command: {shlex.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        check=check,
        text=True,
    )


def major_breaking_since(tag: str) -> bool:
    headlines = git_utils.get_commits(tag, "HEAD", root_dir=ROOT_DIR)
    return any(headline.lower().startswith("major:") for headline in headlines)


def build_jobs() -> str:
    if os.environ.get("PARALLEL_JOBS"):
        return os.environ["PARALLEL_JOBS"]
    try:
        return str(os.cpu_count() or 1)
    except Exception:
        return "1"


def gtest_args() -> list[str]:
    args = [
        "--gtest_brief=1",
        f"--gtest_random_seed={os.environ.get('GLOBAL_TEST_SEED', '42')}",
    ]
    gtest_filter = os.environ.get("GTEST_FILTER")
    if gtest_filter:
        args.append(f"--gtest_filter={gtest_filter}")
    return args


def main() -> int:
    fetch_return_code, _ = git_utils.fetch_tags(ROOT_DIR)
    if fetch_return_code != 0:
        print("Failed to fetch tags from origin; unable to run API compatibility test.")
        return 1

    latest_tag = git_utils.latest_tag_matching(
        r"^v[0-9]+\.[0-9]+\.[0-9]+$", root_dir=ROOT_DIR
    )
    if not latest_tag:
        print("No semver tags found; unable to run API compatibility test.")
        return 1

    if major_breaking_since(latest_tag):
        print(
            "Major-breaking change detected since latest tag; "
            f"API breakage allowed. Skipping API compatibility test against {latest_tag}."
        )
        return 0

    if git_utils.has_local_changes(
        ["CMakeLists.txt", "test", "benchmark"], root_dir=ROOT_DIR
    ):
        print(
            "Repository has local changes in CMakeLists.txt, test/, or benchmark/. "
            "Refusing to overwrite local work."
        )
        return 1

    print(f"Checking out CMakeLists.txt, test/, and benchmark/ from {latest_tag}.")
    # CI job only: this intentionally mutates the workspace for the remainder of the script;
    # the job workspace is ephemeral and cleaned up by CI.
    for path in (ROOT_DIR / "test", ROOT_DIR / "benchmark"):
        if path.exists():
            shutil.rmtree(path)
    run(
        [
            "git",
            "checkout",
            "-q",
            latest_tag,
            "--",
            "CMakeLists.txt",
            "test",
            "benchmark",
        ]
    )

    build_path = ROOT_DIR / BUILD_DIR
    if build_path.exists():
        shutil.rmtree(build_path)

    if not (ROOT_DIR / "CMakePresets.json").is_file():
        raise SystemExit(
            "CMakePresets.json is required for the API compatibility check."
        )

    run(["cmake", "--preset", "default-config", "-B", BUILD_DIR], check=True)
    run(
        [
            "cmake",
            "--build",
            BUILD_DIR,
            f"-j{build_jobs()}",
            "--verbose",
        ],
        check=True,
    )

    run([str(Path(BUILD_DIR) / "kleidiai_test"), *gtest_args()], check=True)
    run(
        [str(Path(BUILD_DIR) / "kleidiai_benchmark"), "--benchmark_list_tests"],
        check=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

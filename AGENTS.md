<!--
    SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# KleidiAI Repo Cheatsheet

## Purpose

- KleidiAI delivers optimized Arm® micro-kernels (packing + matmul) for
  AI/ML frameworks that already manage scheduling, threading, and memory.

## Key Layout

- `kai/` – Common headers plus micro-kernel families grouped by operator (for
  example `ukernels/matmul`). Each family follows naming rules documented in
  per-directory READMEs and implements the standard interfaces in
  `ukernels/matmul/*.h`.
- `test/` – GoogleTest unit suites covering micro-kernel correctness and API
  guarantees.
- `benchmark/` – CMake targets that time kernels across Arm variants.
- `examples/` – Minimal builds that exercise the library as an external
  dependency and serve as smoke tests.
- `docs/` – Task-focused guides (packing/matmul intros, indirect matmul
  walkthroughs, framework integration examples, external patches).
- `docker/` - Contains containers used in CI.

## Build & Run

KleidiAI uses two build systems; CMake and Bazel.

- Native Arm build (default): `cmake -S . -B build && cmake --build build`
  - Test with `build/kleidiai_test`
- Bazel build and test `bazelisk test //test:kleidiai_test`

### CI/CD

The testing pipeline is described in `.gitlab-ci.yml`, which does make use of
container described in `docker/Dockerfile`. This utilizes FVP, which enables
testing on different HW configurations.

## Working Notes for Agents

- Typical flow: pack LHS → pack RHS → invoke matmul kernel.

## Development and Review Checklist

Use these items as a checklist when reviewing or making changes:

- Follow the [KleidiAI coding conventions](docs/coding_conventions.md).
- Follow the [KleidiAI micro-kernel requirements](docs/microkernel_requirements.md).
- Follow the [KleidiAI contribution guidelines](CONTRIBUTING.md).

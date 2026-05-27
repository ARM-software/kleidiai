<!--
    SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Contribution Guidelines

This purpose of this document is to describe how to contribute to Arm® KleidiAI™ and
define what is expected from a contribution. It does not cover every corner case;
the policy will evolve over time.

# Before you start

Before starting non-trivial work, please read:

- [README.md](README.md) for project scope, supported instructions, build
  requirements, and support expectations.
- [docs/README.md](docs/README.md) for the documentation hub, examples, and
  repository layout.
- [kai/ukernels/matmul/README.md](kai/ukernels/matmul/README.md) for matmul
  micro-kernel naming and layout rules.
- [docs/microkernel_testing.md](docs/microkernel_testing.md) for adding a
  micro-kernel to the NextGen test suite.
- [benchmark/README.md](benchmark/README.md) for benchmark build and usage.

Reach out to the maintainers by creating a new issue on the official [KleidiAI
GitLab](https://gitlab.arm.com/kleidi/kleidiai) before investing significant
time in:

- New micro-kernels.
- New public APIs.
- Large rewrites, broad refactors, or changes that alter established
  micro-kernel behavior.

## Accepted contributions

The following areas are generally suitable for external contributions:

- New micro-kernels
- Bug fixes
- Example improvements and new examples that demonstrate supported usage.
- Build-system fixes for supported build systems.
- Test and benchmark improvements that follow the existing framework.

## Rejected contributions

The following directories are not open to external contributions and
contributions will be rejected without prior maintainer agreement:

- `experimental`

KleidiAI does not accept contributions that add support for:

- Non-Arm CPU architectures.
- AArch32 execution state.
- Unsupported operating systems, compiler families, or build systems unless
  maintainers have agreed to expand the support policy.
- Breaking changes to the public API.

# General contribution requirements

All contributions must:

- Use one of the project licenses in the [LICENSES](./LICENSES/) sub-directory.
- Include a Developer Certificate of Origin sign-off.
- Preserve or add correct copyright and license metadata.
- Follow the [Coding standard and convention](docs/coding_conventions.md).
- Build with all relevant supported build systems.
- Include appropriate tests cases.
- Update relevant documentation.
- Include appropriate updates to `CHANGELOG.md`.

For contributions, a Developer Certificate of Origin (DCO) is required to
certify origin. The process is managed by
[DCO v1.1](https://developercertificate.org/).

The agreement to the DCO is indicated by a `Signed-off-by` line in the commit
message using your real name and email address:

```text
Signed-off-by: Name <name@example.com>
```

Contributors are responsible for ensuring their copyright notices are correct.
Third party contributors should add their own copyright notice applicable to
their own modifications in existing files. For new files, if they include any
code copied from existing files, preserve the existing copyright as is (do
not modify it in any way - for example changing the year) and amend as required
with third party copyright notice. If the file is entirely new and does not
contain Arm copyrightable content, there is typically no need to include Arm
copyright.

Do not contribute code with unclear provenance and do not include code copied
from licenses that are incompatible with KleidiAI distribution. Disclose
third-party sources in the merge request.

# Micro-kernel contributions

Discuss new micro-kernels with maintainers before implementation. The discussion
should cover:

- The operation and data formats.
- The target architecture feature or microarchitecture.
- Expected integration use case.
- Expected performance benefit.
- Whether the micro-kernel needs to expand the public API.
- Required packing micro-kernels.
- Test, benchmark, and documentation requirements.

New micro-kernels must:

- Follow the established naming convention or explicitly justify an extension to
  it.
- Use the [current micro-kernel API](kai/ukernels/matmul/kai_matmul.h) style
  unless maintainers explicitly agree to a different interface.
- Provide matching interface headers where required.
- Preserve public symbol naming consistency.
- Use ACLE feature test macros for CPU-feature-dependent code where applicable.
- Maintain the advertised behavior, including clamp behavior and partial-output
  behavior.
- Include unit tests. Must use the NextGen test framework which is described in
  [docs/microkernel_testing.md](docs/microkernel_testing.md).
- Include benchmark test.
- Be added to the micro-kernel tables in [docs/microkernel_tables.md](docs/microkernel_tables.md).

# Getting access and submitting a merge request

Contributions are submitted as GitLab merge requests (MR). The process to submit a MR is:

1. Create an account on the [Arm hosted GitLab instance](https://gitlab.arm.com/).
1. Request project fork permission. See
   [Arm GitLab contribution documentation](https://gitlab.arm.com/documentation/contributions).
1. Fork the KleidiAI repository.
1. Make the smallest coherent change that solves the problem.
1. Run the relevant tests locally and fix any issues.
1. Push the patch-set to your forked repository.
1. Submit a MR on the official KleidiAI GitLab repository with a clear description.

The description of a MR must contain:

- Prefix the first line with type of contribution: major, feat, fix, docs or chore
- What changed.
- Why the change is needed.
- How it was tested.
- Any remaining limitations or follow-up work.

# Testing requirements

Never skip tests unless there is a clear unavoidable reason.

For normal validation, run one of:

```sh
cmake -S . -B build
cmake --build build
build/kleidiai_test
```

```sh
bazelisk test //test:kleidiai_test
```

The NextGen test framework uses deterministic seeds by default. For new
micro-kernels, run with several explicit seeds to cover more generated
shapes and values.

# Benchmarking and performance evidence

Performance-sensitive contributions must include benchmark evidence. Build and
run the benchmark tool as described in
[benchmark/README.md](benchmark/README.md).

Performance evidence should include:

- The exact hardware platform.
- Operating system and toolchain version.
- Build system and build flags.
- Benchmark command lines.
- Shape set used for testing.
- Baseline commit or baseline micro-kernel.
- Results for the changed implementation and the baseline.

Use shapes that represent the intended integration use case. When possible,
choose shapes from existing model workloads rather than synthetic shapes only.

# Examples and documentation

Add a new, or update one of the existing, standalone C++ example application(s)
in the [examples](examples/) sub-directory when a change introduces a new
concept users of the micro-kernel must understand in order to use the
micro-kernel. Examples of when this is needed is when the contribution adds:

- A new integration pattern.
- A new operator flow not covered by existing examples.
- A new public API that is not self-explanatory from existing documentation.

Before adding new explanatory text, check whether existing documentation can be
referenced instead:

- [docs/README.md](docs/README.md) for the documentation hub and example list.
- [kai/ukernels/matmul/pack/README.md](kai/ukernels/matmul/pack/README.md) for
  matmul packing.
- [kai/ukernels/matmul/README.md](kai/ukernels/matmul/README.md) for matmul
  naming and layout.
- [docs/microkernel_testing.md](docs/microkernel_testing.md) for NextGen
  testing.
- [benchmark/README.md](benchmark/README.md) for benchmark usage.

# Pre-commit checks

Install and run the configured pre-commit hooks before submitting a merge
request:

```sh
pre-commit install
pre-commit run --all-files
```

The hook configuration is in `.pre-commit-config.yaml`.

If a hook fails, fix the underlying issue rather than bypassing the hook. If a
hook is not applicable to a specific change, explain that in the merge request.

# Supported environments

KleidiAI support is intentionally bounded. Contributions must preserve support
for the environments targeted by the project.

Library C sources must remain C99-compatible, with ACLE and assembly extensions
where required for Arm architecture features. C++ code in tests and tooling must
remain C++17-compatible unless maintainers agree otherwise.

The project supports CMake 3.16 and Bazel 6.5 or later. Contributions that add
files or change build behavior must keep both build systems up to date.

| OS / environment | Supported versions | Compiler / toolchain | Supported versions / policy |
|---|---|---|---|
| Ubuntu | Current LTS versions | GCC | Default GCC installation versions and all more recent supported releases on gcc.gnu.org |
| Ubuntu | Current LTS versions | Clang | Default Clang installation versions and latest upstream Clang release |
| Ubuntu | Supported Ubuntu versions where available | Arm Compiler for Linux | All ACfL releases from the previous 13 months |
| Windows® | Windows releases marked (E) or (W) in Active Support: Windows 11 26H1 E/W, 25H2 E/W, 24H2 E/W, 23H2 E | MSVC / Visual Studio® | Visual Studio Stable Channel |
| macOS® operating system software | macOS 26, 15, 14 | Xcode® developer software | Most recent minor release of the most recent major Xcode release |
| Android™ |  | Android NDK | Current Android NDK LTS |

> macOS and Xcode are trademarks of Apple Inc., registered in the U.S. and other countries and regions.<br>
> Android is a trademark of Google LLC.<br>
> Visual Studio and Windows are trademarks of the Microsoft group of companies.

# Review and acceptance process

Once a MR is uploaded, the KleidiAI maintainers will review it and trigger the
CI pipeline. Maintainers review merge requests for:

- Project fit.
- Code maintainability.
- Correctness and API behavior.
- Test coverage.
- Benchmark and performance evidence where relevant.
- Build-system coverage.
- Coding convention compliance.
- Documentation quality.
- Licensing and provenance.

Maintainers may ask for additional explanation or rewriting when the code
is hard to review, inconsistent with project conventions, or has unclear
origin.

Review comments must be addressed before merge. Keep updates focused and avoid
unrelated refactoring while responding to review.

Acceptance of a contribution does not imply acceptance of follow-up work, new
support scope, or future micro-kernels in the same area. Discuss major follow-up
work with the maintainers.

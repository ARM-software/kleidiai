<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Micro-kernel requirements

KleidiAI micro-kernels must follow the API, implementation, integration, and
testing requirements described in this document. Code-writing conventions are
described in [Coding standard and conventions](coding_conventions.md).

## Source organization

Micro-kernel bundles (`.c`, `.h`, and `.S`) live under `kai/` and depend only on
`kai_common.h`. Packing micro-kernels live in `pack` directories, while matmul
micro-kernels live in operator-specific directories and have matching interface
headers.

Micro-kernels must follow [the micro-kernel naming
scheme](microkernel_names.md).

## Public API and compatibility

When making changes, preserve public symbol names, or make it clearly
intentional by tagging commit message with `major:` and have `CHANGELOG.md`
describe the change.

Parameter names may be changed for consistency. Preserve micro-kernel
functionality, or have commit message clearly describe the change.

## Architecture feature guards

Guard CPU-feature-dependent code with
[ACLE feature test macros](https://arm-software.github.io/acle/main/acle.html#feature-test-macros).
`__ARM_FEATURE_SME` is often an exception.

At the top of `.c` files, assert the required architecture extensions. For
example:

```c
#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD)) && \
    !defined(_M_ARM64)
```

## Assembly code

Pure assembly micro-kernels must:

- Conform to [AAPCS64](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst).
- Preserve `d8`–`d15` and `x19`–`x28` when modifying them.
- Not use register `x18`.
- Emit exactly one `ret`.
- Avoid calls with `bl`, except for approved
  [SME support routines](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#sme-support-routines)
  with proper `LR`/`FP` preservation and `__ARM_FEATURE_SME` guards.
- Implement the advertised behavior, including clamp functionality where
  applicable.

Avoid using inline assembly, as compiler support is not standardized across the
supported compilers.

## Build integration

New source files must be added to all relevant build scripts. CMake source lists
are named `KLEIDIAI_FILES_<TECH>[_<FEAT>]*[_ASM]`. Bazel source lists are named
`<TECH>[_<FEAT>]*_KERNELS[_ASM]`. Keep file lists sorted when adding files.

Add each micro-kernel to the list matching its required technology and features.

Kernels that use inline assembly belong in the non-`_ASM` list. Kernels that do
not use inline assembly normally belong in an `_ASM` list, which is preferred
for compiler support.

## Testing

New unit tests must use the NextGen test framework which is described in
[Testing a new micro-kernel in the test suite](microkernel_testing.md).

Cache expensive reference-data generation where appropriate to keep the CI
pipeline execution time low, using utilities such as `test/common/cache.hpp`.

Tests must behave deterministically. Randomize a seed only when explicitly
requested by a runtime parameter.

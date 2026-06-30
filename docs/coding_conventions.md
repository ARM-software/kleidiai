<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Coding standard and conventions

KleidiAI source code must follow the project coding convention described in this
section. The convention is intentionally small and relies on the repository's
tooling as the baseline enforcement mechanism.

## clang-format and clang-tidy

Follow the formatting and static-analysis rules configured in `.clang-format`
and `.clang-tidy`.

The clang-format configuration is based on Google style with project-specific
adjustments. Deviation from the base format should be minimal and justified.

The clang-tidy configuration enables the checks that are relevant to KleidiAI
and disables unsuitable checks explicitly.

## Comments and documentation

Use line comments for both code comments and API documentation:

- Use `//` for ordinary code comments.
- Use `///` for documentation comments.
- Do not use block comments for normal source documentation.

Write comments in descriptive third person when describing what code does.
Imperative comments are acceptable when describing a future action, for example
in a `TODO`.

Example:

```cpp
/// Performs softmax activation function.
///
/// @param[out] dst Output data buffer.
/// @param[in] src Input data buffer.
/// @param[in] length Number of elements.
void softmax(float* dst, const float* src, size_t length) {
    // Finds max.
    // Regularizes.
    // Normalizes.
}
```

## Integer data types

Use `size_t` for sizes and fixed-width integer types, e.g. `int32_t`,
for integer values.

In C++ code, do not qualify `size_t` with the `std` namespace.

## Data pointers

Use `void*` and `const void*` for data pointers in micro-kernel APIs. This
keeps the public API consistent for cases when C and C++ do not provide native
types for the stored format.

## Code structure

Use blank lines to separate blocks of distinct functionality. This is not
enforced automatically, but it should be used where it makes the structure of a
function easier to read.

Use `const` for variables and parameters that are not modified.

For classes where member-name shadowing is an issue, use a leading `m_` for
member variables.

Example:

```cpp
struct Foo {
public:
    Foo(int x, int y) : m_x{x}, m_y{y} {
    }

    void set(int x, int y) {
        m_x = x;
        m_y = y;
    }

private:
    int m_x;
    int m_y;
};
```

## Assertions and assumptions

Use `KAI_ASSUME` for expected preconditions and `KAI_ASSERT` for invariants that
must hold when those preconditions are true. Function parameter requirements
should normally be expressed as assumptions rather than runtime `if` checks.

Example:

```cpp
/// Performs softmax activation function.
///
/// @param[out] dst Output data buffer.
/// @param[in] src Input data buffer.
/// @param[in] length Number of elements.
void softmax(float* dst, const float* src, size_t length) {
    KAI_ASSUME(dst != NULL);
    KAI_ASSUME(src != NULL);
    KAI_ASSUME(length > 0);

    float max = -INFINITY;
    for (size_t i = 0; i < length; ++i) {
        if (src[i] > max) {
            max = src[i];
        }
    }

    float sum = 0;
    for (size_t i = 0; i < length; ++i) {
        const float tmp = exp(src[i] - max);
        dst[i] = tmp;
        sum += tmp;
    }

    KAI_ASSERT(sum > 0);

    for (size_t i = 0; i < length; ++i) {
        dst[i] = dst[i] / sum;
    }
}
```

Do not assert or assume values for unused parameters. Mark unused parameters
with `KAI_UNUSED` instead. Requiring specific values for parameters that are not
used places an unnecessary burden on integrators.

Example:

```c
void kai_run_...(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr,
    size_t rhs_stride_row, const void* rhs, const void* bias, const void* scale,
    void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_UNUSED(num_groups);
    KAI_UNUSED(nr);
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_UNUSED(scale);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_UNUSED(extra_bytes);
    KAI_UNUSED(params);

    KernelArgs args;
    args.bias_ptr = bias;
    args.height = k;
    args.width = n;
    args.in = rhs;
    args.out = rhs_packed;
    args.in_stride = rhs_stride_row;
    kai_kernel_...(&args);
}
```

Test code must use `KAI_ASSUME_ALWAYS(expr)` and `KAI_ASSERT_ALWAYS(expr)` where
the check must not be optimized away in release builds.

## Assembly code

Pure assembly micro-kernels must:

- Conform to [AAPCS64](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst).
- Emit exactly one `ret`.
- Avoid calls with `bl`.

Avoid using inline assembly, as compiler support is not standardized across the
supported compilers.

## Naming and file layout

Follow the established naming, directory, CMake, and Bazel conventions used by
the surrounding code. Micro-kernel naming is described in
[docs/microkernel_names.md](microkernel_names.md).

## Build scripts

New source files must be added to all relevant build scripts. CMake source lists
are named `KLEIDIAI_FILES_<TECH>[_<FEAT>]*[_ASM]`. Bazel source lists are named
`<TECH>[_<FEAT>]*_KERNELS[_ASM]`. Keep file lists sorted when adding files.

Kernels that use inline assembly belong in the non-`_ASM` list. Kernels that do
not use inline assembly normally belong in an `_ASM` list, which is preferred
for compiler support.

## Test code

New unit tests must use the NextGen test framework which is described in
[docs/microkernel_testing.md](docs/microkernel_testing.md).

Cache expensive reference-data generation where appropriate to keep the CI
pipeline execution time low.

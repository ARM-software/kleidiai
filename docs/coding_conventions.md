<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Coding standard and conventions

KleidiAI source code must follow the project coding convention described in this
section. The convention is intentionally small and relies on the repository's
tooling as the baseline enforcement mechanism.

Requirements for micro-kernel APIs, implementations, build integration, and
testing are described in
[Micro-kernel requirements](microkernel_requirements.md).

## clang-format and clang-tidy

Follow the formatting and static-analysis rules configured in `.clang-format`
and `.clang-tidy`.

The clang-format configuration is based on Google style with project-specific
adjustments. Deviation from the base format should be minimal and justified.

The clang-tidy configuration enables the checks that are relevant to KleidiAI
and disables unsuitable checks explicitly. Every disabled check must have a
justification in a comment in `.clang-tidy`.

## Comments and documentation

Use line comments for both code comments and API documentation:

- Use `//` for ordinary code comments.
- Use `///` for documentation comments.
- Do not use block comments for normal source documentation.

Write comments in descriptive third person when describing what code does.
Imperative comments are acceptable when describing a future action, for example
in a `TODO`.

Document every public function clearly with documentation comments, `///`.
Include a brief one-line description, any necessary longer description, all
parameters with their directions, and the return value when applicable.

Functions with static linkage should have a documentation comment. For
trivial functions it's sufficient with only a brief one-line description.

Remove commented-out code. Use `TODO` comments sparingly and explain the
required follow-up.

Example:

```cpp
/// Performs softmax activation function.
///
/// The softmax activation takes a `src` array of `length` elements, and
/// writes the resulting values to the `dst` array of the same length.
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
/// The softmax activation takes a `src` array of `length` elements, and
/// writes the resulting values to the `dst` array of the same length.
///
/// @param[out] dst Output data buffer.
/// @param[in] src Input data buffer.
/// @param[in] length Number of elements.
void softmax(float* dst, const float* src, size_t length) {
    KAI_ASSUME(dst != NULL);
    KAI_ASSUME(src != NULL);
    KAI_ASSUME(length > 0);

    // Finds max.
    float max = -INFINITY;
    for (size_t i = 0; i < length; ++i) {
        KAI_ASSUME(!isnan(src[i]));
        KAI_ASSUME(!isinf(src[i]));

        if (src[i] > max) {
            max = src[i];
        }
    }

    // Regularizes.
    float sum = 0;
    for (size_t i = 0; i < length; ++i) {
        const float tmp = exp(src[i] - max);
        dst[i] = tmp;
        sum += tmp;
    }

    KAI_ASSERT(sum > 0);

    // Normalizes.
    for (size_t i = 0; i < length; ++i) {
        dst[i] = dst[i] / sum;
    }
}
```

Mark unused parameters with `KAI_UNUSED`. You may use `KAI_ASSUME` on unused
parameters in order to allow detection when kernel is used with an unsupported
configuration.

Example:

```c
void kai_run_...(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr,
    size_t rhs_stride_row, const void* rhs, const void* bias, const void* scale,
    void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_UNUSED(num_groups);
    KAI_ASSUME(nr == 4);
    KAI_ASSUME(kr == 16);
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

## Naming and terminology

Use the term _micro-kernel_ rather than kernel, ukernel, or function when
referring to a micro-kernel.

Follow the established naming conventions used by the surrounding code. Follow
the existing convention when naming a new micro-kernel, or explicitly extend
it. Micro-kernel naming is described in
[docs/microkernel_names.md](microkernel_names.md).

<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
    <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Testing a new micro-kernel in the test suite

This guide explains how to add a new micro-kernel to the test suite. The process
is general, but the concrete example assumes an F16 matmul micro-kernel with F16
input and F16 output, that has a RHS packing kernel, no LHS packing, and per $N$
bias. Input matrices are assumed to be row major format. Do note that this is
intended to be a guiding example, and there might be aspects of the examples
here that are not yet implemented.

The test suite is built around a small set of components and entry points:

- Operator registration and test enumeration in
  `test/nextgen/operators/matmul/matmul_operator.cpp` and
  `test/nextgen/tests/matmul_test_next.cpp`.
- Wrapper classes (specializations using the `KernelWrapper` interface) in
  subdirectories of `test/nextgen/operators/matmul`. The files here adapt the
  micro-kernel C API into the common test harness API. Micro-kernel factory
  declarations are listed in wrapper interface headers, e.g.,
  `test/nextgen/operators/matmul/matmul/matmul_wrapper.hpp`.
- A test bench in `test/nextgen/operators/matmul/matmul_tb.cpp` that generates
  inputs, runs the implementation, and compares to a reference.

This is the second version of the testing framework, which is why there are a
lot of references to _NextGen_. This naming might change in the future, once the
transition to _NextGen_ is completely done.

## Glossary

- *LHS* - Left-hand side. The $A$ matrix in $C = A * B + β$.
- *MatMul* - Matrix multiplication.
- *operator* - a set of micro-kernels that together are used to accelerate an
  operation. For example, RHS packing and matmul micro-kernels.
- *Packing* - reordering of input matrices to enable efficient memory access
  patterns.
- *portion* - Refers to test cases where only partial output is calculated,
  as to test tiling functionality.
- *RHS* - Right-hand side. The $B$ matrix in $C = A * B + β$.
- *tensor slot* - a named entry in `MatMulSlot` used by `MatMulTb` tensor
  storage to hold configuration/arguments and tensor data (source, derived, packed, and
  destination). Wrappers declare required slots via `run_inputs()` and
  `ref_inputs()`.
- *Wrapper* - a wrapper around the micro-kernel as to expose a standardized API
  to the test framework.

## The testing framework API

Two utility types appear throughout the test harness API: `Span<T>` and
`Poly<Base>`. Both are local backports of standard-library functionality from
C++20 and later, and are used to keep wrapper and format interfaces compact.

`Span<T>` is a non-owning view over contiguous memory, similar to `std::span`.
The harness uses it for shapes, tile coordinates, and tensor data buffers.

`Poly<Base>` is an owning, copyable wrapper for an object derived from `Base`,
similar to later standard-library polymorphic value types. In the harness it is
mainly used as `Poly<Format>` so tensors and wrappers can store concrete format
objects by value while using the common `Format` interface.

## Test flow overview

`MatMulNext` is the matmul test suite registered in
`test/nextgen/tests/matmul_test_next.cpp`.
Its flow is:

1. Enumerate available operators from `get_available_matmul_operators()`.
1. Skip operators that current CPU doesn't support. This is checked by calling
   `is_cpu_supported()` on operator.
1. For each operator, sample random shapes and keep only shapes that satisfy
   `is_shape_suitable(shape_m, shape_n, shape_k, portion)` on operator.
1. Register tests for LHS pack, RHS pack, and matmul.
1. Build a cached `MatMulTb` per fixture, generate test data once, then run
   tile-level tests against precomputed references.

Inside `MatMulTb::generate_test_data()`, the sequence is:

1. Populate `CONFIG`, then collect the required tensor slots from each
   wrapper's `run_inputs()` and `ref_inputs()`.
1. Let wrappers populate constant slot metadata and argument payloads such as
   `PACK_ARGS`, `MATMUL_ARGS`, and packed tensor formats.
1. Generate the source tensors (`LHS_DATA`, `RHS_DATA`, `BIAS_DATA`), then
   derive any always-needed source variants such as `RHS_T_DATA`.
1. Apply optional quantization (`lhs_quant`, `rhs_quant`, `bias_quant`).
1. Compute extra derived tensors required by wrappers (for example
   `LHS_QZP_NEG`, `RHS_T_QDATA_SIGN`, `RHS_T_QDATA_SIGN_SUM`).
1. Compute required packed reference LHS and RHS input, and reference
   destination data.

Conceptually, `MatMulTb` owns a tensor pool indexed by `MatMulSlot`. Each slot
names one piece of state in the test flow: source tensors, derived tensors,
packed tensors, or non-array argument blobs such as `PACK_ARGS` and
`MATMUL_ARGS`.

Wrappers declare which slots they need by returning slot IDs from `run_inputs()`
for the implementation path and `ref_inputs()` for the reference path.
`determine_required_tensors()` takes the union of those declarations and uses it
to decide which data must exist before any wrapper runs.

This is how a floating-point matmul wrapper can simply ask for `LHS_PACKED`,
`RHS_PACKED`, and `MATMUL_ARGS`, while an RHS pack wrapper can ask for
`RHS_DATA` or `RHS_T_DATA` depending on layout and still use `RHS_T_DATA` for
reference packing. Optional derived slots such as `LHS_QZP_NEG` or
`RHS_T_QDATA_SIGN_SUM` are only computed when some wrapper has declared that it
needs them.

## Adding a new micro-kernel to the test suite

This section outlines the steps needed to be able to add a new micro-kernel to
the test suite.

High-level steps:

- Add wrapper factory functions for any required pack micro-kernels and the
  matmul micro-kernel.
- Register a `MatMulOperator` entry pointing at those wrappers and setting
  data types, bias modes, quantizers, and shape suitability.

### Step 1: Add or extend matmul wrapper factories

Wrappers adapt the micro-kernel C API to the harness API (`KernelWrapper`). For
matmul, the relevant interfaces are:

- `test/nextgen/operators/matmul/matmul/matmul_interface.hpp`
- `test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_interface.hpp`
- `test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_interface.hpp`

For a simple F16 example, with no LHS packing:

- Use `MatMulFpWrapper` for the matmul micro-kernel.
- Use `MatMulPackRhsFpNtWrapper` for RHS packing.

As an example, the matmul factory can be implemented as follows:

```cpp
// test/nextgen/operators/matmul/matmul/matmul_wrapper.cpp
std::unique_ptr<KernelWrapper<MatMulShape>>
create_matmul_clamp_f16_f16_f16p_neon() {
    return std::make_unique<MatMulFpWrapper>(
        "matmul_clamp_f16_f16_f16p_neon",
        MatMulFpInterface{
            kai_get_m_step_matmul_clamp_f16_f16_f16p_neon,
            ...
            kai_get_dst_size_matmul_clamp_f16_f16_f16p_neon,
            kai_run_matmul_clamp_f16_f16_f16p_neon,
        },
        make_poly<PlainFormat>(DataType::FP16),  // LHS input format
        make_poly<Block2dRowFormat>(
            16, 1, 1, false, DataType::FP16,
            std::array{DataType::FP16},
            std::array<DataType, 0>{}),           // Packed RHS input format
        make_poly<PlainFormat>(DataType::FP16));  // Output format
}
```

For this example, the matmul wrapper uses unpacked FP16 LHS, packed FP16 RHS
with block size of $16 ⨯ 1$, and a plain FP16 destination format. Use existing
factories in `matmul_wrapper.cpp` and packer wrappers as a model.

Make sure there is a `create_matmul_...` factory declaration added to
`test/nextgen/operators/matmul/matmul/matmul_wrapper.hpp` for the new micro-kernel.

### Step 2: Add packing micro-kernel wrapper factories

For the F16 example, only RHS packing is needed. Add or reuse the RHS
packer wrapper in:

- `test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper.cpp`

If another micro-kernel also packs the LHS, add or reuse a packer wrapper in
`test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_wrapper.cpp`.

Example skeleton for RHS packing:

```cpp
// test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper.cpp
std::unique_ptr<KernelWrapper<MatShape>>
create_matmul_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon() {
    return std::make_unique<MatMulPackRhsFpNtWrapper>(
        "matmul_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon",
        MatMulPackRhsFpInterface{
            kai_get_n_step_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
            kai_get_rhs_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
            kai_get_bias_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
            kai_get_rhs_packed_stride_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
            kai_get_rhs_packed_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
            kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
            kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
        },
        make_poly<PlainFormat>(DataType::FP16),  // RHS input format
        make_poly<PlainFormat>(DataType::FP16),  // Bias input format
        make_poly<Block2dRowFormat>(
            16, 1, 1, false, DataType::FP16,
            std::array{DataType::FP16},
            std::array<DataType, 0>{}));  // Packed RHS output format
}
```

In this example the wrapper is for a non-transposed RHS packing micro-kernel,
which is why the RHS input matrix is consumed from `RHS_DATA` in plain row
major $K × N$ layout, while the bias row vector is consumed from `BIAS_DATA` in
plain row major $N$ layout. The packed output is a bit more complex, as it uses
`Block2dRowFormat` to describe a $16 × 1$ packing format with the per-$N$ bias
stored alongside the packed RHS data.

If the packing micro-kernel instead expects a transposed RHS, the source format
is still `PlainFormat`, but the logical shape changes to plain row major $N × K$
and the wrapper should consume `RHS_T_DATA` rather than `RHS_DATA`. The test
bench already derives `RHS_T_DATA` by transposing the generated $K × N$ RHS, so
the operator registration does not need any extra transpose step. In practice,
`...kxn...` packers use the non-transposed path shown above, while `...nxk...`
packers should follow the transposed wrapper pattern and read from the
transposed RHS slot. The bias handling, data types, and packed output format
description stay the same; only the selected pack wrapper/factory and matching
shape-suitability helper change.

### Step 3: Register the operator

Register the micro-kernel in `test/nextgen/operators/matmul/matmul_operator.cpp`. Each
operator declares:

- Micro-kernel availability (`is_cpu_supported`).
- Shape suitability for matrix portions (`is_shape_suitable(...)`).
- Supported bias modes.
- Input/output/accumulator data types.
- Optional quantizers (for quantized micro-kernels).
- Wrapper factories for any required packers and matmul.

Example skeleton for operator

```cpp
// test/nextgen/operators/matmul/matmul_operator.cpp
operators[i].name = "matmul_clamp_f16_f16_f16p_neon";

operators[i].is_cpu_supported = []() { return cpu_has_fp16(); };
operators[i].is_shape_suitable =
    [](size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
        return is_shape_suitable_rhs_f16_example(shape_m, shape_n, shape_k, portion);
    };

operators[i].supported_bias_modes = { MatMulBiasMode::PER_N, };

operators[i].lhs_dtype = DataType::FP16;
operators[i].rhs_dtype = DataType::FP16;
operators[i].bias_dtype = DataType::FP16;
operators[i].acc_dtype = DataType::FP16;
operators[i].dst_dtype = DataType::FP16;

operators[i].pack_rhs = create_matmul_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon();
operators[i].matmul = create_matmul_clamp_f16_f16_f16p_neon();
```

## Build and run tests

Tests are compiled via the main test targets. Use a filter to focus on matmul
tests.

CMake:

```bash
cmake -S . -B build
cmake --build build
build/kleidiai_test --gtest_filter='MatMulNext*'
```

Bazel:

```bash
bazelisk test //test:kleidiai_test --test_filter='MatMulNext*'
```

`MatMulNext` is deterministic by default because the test harness uses a fixed
seed unless one is explicitly provided. When adding a new micro-kernel, do not
stop at a single default-seed run; it is good practice to rerun the filtered
tests with several explicit seeds to cover more generated shapes and values.

Examples:

```bash
build/kleidiai_test --gtest_filter='MatMulNext*' --gtest_random_seed=<seed>
bazelisk test //test:kleidiai_test --test_filter='MatMulNext*' --test_arg=--gtest_random_seed=<seed>
```

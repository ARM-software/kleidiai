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
intended to be a guiding example.

The test framework supports testing different micro-kernel C APIs, including
earlier interfaces that pass arguments individually. New micro-kernels are
expected to use the struct-based API, which groups related arguments into
C structs. The examples below use its wrapper adapters.

The test suite is built around a small set of components and entry points:

- Operator registration and test enumeration in
  `test/nextgen/operators/matmul/matmul_operator.cpp` and
  `test/nextgen/tests/matmul_test_next.cpp`.
- Wrapper classes (specializations using the `KernelWrapper` interface) in
  subdirectories of `test/nextgen/operators/matmul`. The files here adapt the
  micro-kernel C API into the common test harness API. Micro-kernel factory
  declarations are listed in wrapper interface headers, e.g.,
  `test/nextgen/operators/matmul/matmul/matmul_wrapper_registry.hpp`.
- A test bench in `test/nextgen/operators/matmul/matmul_tb.cpp` that generates
  inputs, runs the implementation, and compares to a reference.

This is the second version of the testing framework, which is why there are a
lot of references to _NextGen_. This naming might change in the future, once the
transition to _NextGen_ is completely done.

## Glossary

- *Compute kernels* - Micro-kernels that perform numerical operations, such as
  softmax, matrix multiplication (matmul), and depthwise convolution (dwconv).
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
  storage to hold configuration/arguments and tensor data (source, derived,
  packed, bias, and destination). Wrappers declare required slots via
  `run_inputs()` and `ref_inputs()`.
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
1. For each operator and tested output portion, sample random shapes and keep
   only shapes that satisfy
   `is_shape_suitable(shape_m, shape_n, shape_k, portion)` on operator.
1. Select a supported bias mode set and clamp configuration for the operator.
1. Register tests for each component present on the operator: LHS pack, RHS
   pack, and matmul.
1. Build a cached `MatMulTb` per fixture, generate test data once, then run
   tile-level tests against precomputed references.

Inside `MatMulTb::generate_test_data()`, the sequence is:

1. Populate `CONFIG`, then collect the required tensor slots from each
   wrapper's `run_inputs()` and `ref_inputs()`.
1. Have wrappers populate constant slot metadata and payloads, such as
   `PACK_ARGS`, `MATMUL_ARGS`, and packed tensor formats. This is done by
   calling `populate_constant_info()` on each wrapper in matmul, LHS pack,
   and RHS pack order.
1. Generate the source tensors (`LHS_DATA`, `RHS_DATA`, and any required
   logical bias or scale slots), then derive any always-needed source variants
   such as `RHS_T_DATA`.
1. Apply optional quantization (`lhs_quant`, `rhs_quant`, `bias_quant`).
1. Compute extra derived tensors required by wrappers (for example,
   `LHS_QZP_NEG`, `RHS_T_QDATA_SIGN`, `RHS_T_QDATA_SIGN_SUM`).
1. Compute required packed reference LHS and RHS inputs, reference destination
   data, and any additional kernel configuration, such as clamp parameters.

Conceptually, `MatMulTb` owns a tensor pool indexed by `MatMulSlot`. Each slot
names one piece of state in the test flow: source tensors, derived tensors,
packed tensors, or non-array argument values such as `PACK_ARGS` and
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

Bias is configured at operator level with `MatMulBiasModeSet`. Each
`MatMulOperator` lists the supported sets in `supported_bias_mode_sets`; an
empty set means no bias. The current logical bias modes are
`ACCUMULATION_PER_M` for row bias, `ACCUMULATION_PER_N` for column bias, and
`SCALE_BIAS_PER_N` for column bias applied after accumulator scaling. The test
framework uses these modes to model the operator result. It does not matter to
the high-level flow which component consumes or precomputes a particular value.

The reference path computes the operator result in the same logical order:
matmul, accumulation-stage bias, destination conversion when needed,
accumulator scaling, scaled bias, and finally clamping. The exact tensors are
generated only when the selected wrappers declare them as required inputs.

Operators also declare clamping support with `MatMulClampMode`. Use
`UNSUPPORTED` for micro-kernels with no clamp arguments, `OPTIONAL` for
micro-kernels that can run with or without clamp arguments, and `REQUIRED` for
micro-kernels whose API always requires clamp limits. Current matmul clamp
arguments are carried in `MATMUL_ARGS` as `std::optional<MatMulClampArgsF32>`.
For `REQUIRED` operators, the test bench supplies a full output-range clamp when
the random test case does not select dynamic clamping.

Compute comparisons use `make_compute_mismatch_handler()` from
`test/common/compare.cpp`, deriving tolerances from the participating data
types; quantized-output precision is supplied by
`Quantizer::make_mismatch_handler()`.
Tolerance caps can only tighten these defaults, and no mismatches beyond the
thresholds are allowed; packing comparisons remain exact and format-aware.

## Adding a new micro-kernel to the test suite

This section outlines the steps needed to be able to add a new micro-kernel to
the test suite.

High-level steps:

- Add wrapper factory functions for any required pack micro-kernels and the
  matmul micro-kernel.
- Register a `MatMulOperator` entry pointing at those wrappers and setting
  data types, supported bias mode sets, clamp mode, quantizers, and shape
  suitability.

A wrapper factory is a function that creates a wrapper for a micro-kernel.
Use `template <DataType DType>` when the wrapper factory needs to support
multiple data types such as signed and unsigned integer types. The RHS
packing example in Step 2 uses one factory for FP16 and BF16; its
`static_assert` checks that the chosen type is 16 bits. A template is not
required when the wrapper factory uses fixed data types. The data types can
be specified directly, as shown in Step 1, where the inputs use FP16 and the
accumulator uses FP32. The input and output types can differ, provided each
is fixed.

A fixed-type wrapper factory may still use a template to match related
factory interfaces. In that case, use a `static_assert` to allow only the
supported type, for example `static_assert(DType == DataType::FP32)`.
When a template factory is defined in a registry `.cpp` file and used by an
operator in another file, add an explicit instantiation in the registry
`.cpp` for each data type the operator uses.

### Step 1: Add or extend matmul wrapper factories

Wrappers adapt the micro-kernel C API to the harness API (`KernelWrapper`). For
matmul, the struct-based API adapters are:

- `test/nextgen/operators/matmul/matmul/matmul_ukerapi_wrapper.hpp`
- `test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_ukerapi_wrapper.hpp`
- `test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_ukerapi_wrapper.hpp`

For a simple F16 example, with no LHS packing:

- Use `MatMulUkerApiWrapper` for the matmul micro-kernel.
- Use `MatMulPackRhsUkerApiWrapper` for RHS packing.

As an example, the matmul factory can be implemented as follows, in the
registry's existing `kai::test` namespace and include context.
`MatMulKernelPtr` owns a pointer to a `KernelWrapper<MatMulShape, MatMulSlot>`;
`MatMulPackKernelPtr` is the equivalent alias for packing with `MatShape`.

```cpp
// test/nextgen/operators/matmul/matmul/matmul_wrapper_registry.cpp
MatMulKernelPtr create_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot",      // name
        kai_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot(),  // kernel
        MatMulSlot::LHS_DATA,                                      // lhs input slot
        format::plain<DataType::FP16>(),                           // lhs format
        format::x16pbx16<DataType::FP16, sme::VS<4>, 2>(),         // rhs format
        format::plain<DataType::FP16>(),                           // dst format
        DataType::FP32,                                            // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),           // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                 // bias stage
}
```

For this example, the matmul wrapper uses unpacked FP16 LHS, packed FP16 RHS
with block size of $4VS × 2$, and a plain FP16 destination format with FP32
accumulation. Shared format helpers come from
`test/nextgen/format/formats.hpp` in `kai::test::format`.
Use existing factories in `matmul_wrapper_registry.cpp` and packer wrappers
as a model.

Make sure there is a `create_matmul_...` factory declaration added to
`test/nextgen/operators/matmul/matmul/matmul_wrapper_registry.hpp` for the new
micro-kernel.

### Step 2: Add packing micro-kernel wrapper factories

For the F16 example, only RHS packing is needed. Add or reuse the RHS
packer wrapper in:

- `test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper_registry.cpp`

If another micro-kernel also packs the LHS, add or reuse a packer wrapper in
`test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_wrapper_registry.cpp`.

Example skeleton for RHS packing:

```cpp
// test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper_registry.cpp
template <DataType DType>
MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme() {
    static_assert(data_type_size_in_bits(DType) == 16);

    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    const auto dt = data_type_uid(DType);
    std::string name = "matmul_pack_rhs_kxn_" + dt + "p4vsx2b" + dt + "_" + dt + "_" + dt + "_sme";

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        name,                                                               // name
        kai_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme(),                // kernel
        format::plain<DType>(),                                             // src data format
        format::plain<DType>(),                                             // src bias format
        format::x16pbx16<DType, sme::VS<4>, 2>(),                           // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        MatMulSlot::RHS_DATA,                                               // rhs slot
        operand_slots,                                                      // operand slots
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});  // reference slots
}
```

In this example, the wrapper is for a non-transposed RHS packing micro-kernel,
which is why the RHS input matrix is consumed from `RHS_DATA` in plain row
major $K × N$ layout, while the logical per-$N$ accumulation bias is represented
by `ACC_BIAS_N_DATA` in plain row major $N$ layout. The packed output is a bit
more complex, as `format::x16pbx16` uses `Block2dRowFormat` to describe a
$4VS × 2$ packing format with the per-$N$ bias stored alongside the packed RHS
data. Shared formats must model the packer's padding, including repeated rows
where required; padding is not universally zero-filled.

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

Register the micro-kernel in
`test/nextgen/operators/matmul/matmul_operator.cpp`. Each operator declares:

- Micro-kernel availability (`is_cpu_supported`).
- Shape suitability for matrix portions (`is_shape_suitable(...)`).
- Supported bias mode sets.
- Clamp support mode.
- Input/output/accumulator data types.
- Optional quantizers (for quantized micro-kernels).
- Wrapper factories for any required packers and matmul.

Example skeleton for an operator (`acc_bias_per_n` and
`is_shape_suitable_lhs_vector` are existing helpers in this file):

```cpp
// test/nextgen/operators/matmul/matmul_operator.cpp
MatMulOperator create_operator_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<   //
        is_shape_suitable_lhs_vector,  //
        is_shape_suitable_rhs_kxn_x16p4vsx2bx16_x16_x16_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::FP16;
    op.rhs_dtype = DataType::FP16;
    op.bias_dtype = DataType::FP16;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP16;

    op.pack_lhs = std::nullopt;
    op.pack_rhs = create_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme<DataType::FP16>();
    op.matmul = create_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot();
    return op;
}
```

Add the factory call to the list in `get_available_matmul_operators()` to
register the operator.

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
Random sampling does not guarantee coverage of every tail or exact K value;
partial portions follow wrapper steps and typically still process full K.

Examples:

```bash
build/kleidiai_test --gtest_filter='MatMulNext*' --gtest_random_seed=<seed>
bazelisk test //test:kleidiai_test --test_filter='MatMulNext*' --test_arg=--gtest_random_seed=<seed>
```

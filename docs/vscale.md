<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Vector scale

This document describes why the KleidiAI API uses `vs` (_vector scale_, `vscale`)
instead of `vl` (_vector length_) for new micro-kernel names and packing formats.

## What is vscale?

`vs` or `vscale` is a datatype-independent scale factor based on a 128-bit vector length.
`1vs` (i.e. `1 * vscale`) refers to 1 element when the vector length is 128-bits. The number of elements scales with the runtime vector length.

In KleidiAI, one vscale unit is 16 bytes as defined by `KAI_VSCALE_UNIT_BYTES` in
[`kai/kai_common.h`](../kai/kai_common.h).

The 128-bit unit corresponds to the Advanced SIMD vector length baseline. For scalable-vector technologies such as SVE and SME, the `vs` factor scales with the runtime vector length. For example, with a 512-bit vector length, `1vs` equals 4, and `8vs` equals 32.

## Where is vscale applicable?

`vscale` is applicable to scalable-vector micro-kernels where the runtime
vector length affects the tile shape, packing shape, or scheduling step. This
includes SVE, SME, and SME2 micro-kernels.

`vscale` is not used for Advanced SIMD or NEON micro-kernels because their
vector length is fixed and does not vary at runtime.

## Why use vscale instead of vector length?

`vl` refer to the number of elements of a given type that fit into one vector.
That means `vl` changes depending on the element type. For example, assuming a 512-bit
vector, `1vl` has the following element counts:

| Element Datatype | Elements in `1vl` |
|------------------|-------------------|
|    8-bit         |    64             |
|    16-bit        |    32             |
|    32-bit        |    16             |

This makes `vl` useful when a format is explicitly tied to a specific element
type, but it can make API names harder to compare across data types. It can
also hide extra scaling factors that are part of the micro-kernel layout.

### Example of `vl` Confusion

For KleidiAI micro-kernels, packing formats are specified in the micro-kernel name using `<type>p<MR>x<KR>`.

For example, the format `f16p2vlx2` from
[`kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.c`](../kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.c#L34)
means that `f16` (16-bit) elements are packed in `2vl` by 2 blocks. At first glance, this can look like a 64x2 block on a 512-bit configuration, because `2vl` of 16-bit elements is 64.

However, the implementation derives the `M` and `N` block sizes as `kai_mr *  kai_get_sme_vector_length_u16() / kai_kr`. In this case, that is `2 *  kai_get_sme_vector_length_u16() / 2` giving 32 elements. The format `f16p2vlx2` thus refers to packing in 32x2 blocks.
The calculation in the code is as follows:

```c
enum {
    MAX_MR = MR * (KAI_SME_VEC_LENGTH_MAX_BYTES / sizeof(int8_t)) / KR,
}

size_t kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa(void) {
    // Code inlined for brevity
    return kai_mr * kai_get_sme_vector_length_u16() / kai_kr;
}
```

The `vl` spelling therefore does not fully describe the relationship between the accumulation
type and the input type that is encoded by the scale factor `KR`. `KR` is dictated by the
number of accumulation per instruction rather than by the input and accumulation types.

## What are the benefits of vscale?

The benefit of using `vs` is simplicity, as `vs` indicates the vector length in a __datatype-independent__ fashion.

`vs` uses the smallest possible integer such that any concrete vector length will always be a multiple of `vs`. In SME with 512-bit vector configuration, `1vs` always indicates 4 elements,
`4vs` always 16. There is no extra considerations.

Using the example `f16p2vlx2` and renaming it with `vs` would give `kai_matmul_clamp_f16_f16p8vsx2_f16p8vsx2b_..._sme_mopa`, thus mapping directly to `8 * vscale = 32` elements.

This also simplifies the logic inside the kernel itself, as the `mr` function gets simplified.

```c
enum {
    MAX_MR = MR_VSCALE * KAI_VSCALE_MAX,
}

size_t kai_get_mr_matmul_clamp_f16_f16p8vsx2_f16p8vsx2b_8vsx8vcs_sme_mopa(void) {
    // Code inlined for brevity
    return MR_VSCALE * kai_get_sme_vscale();
}
```

Existing micro-kernels that use `vl` in their names are not modified. New kernel should be based on `vs` rather than `vl` and use the `vs` logic as it is objectively simpler.

## Examples of micro-kernels using vscale

The following micro-kernels show the preferred `vs` naming style and how the
implementation derives runtime dimensions from `kai_get_sme_vscale()`:

- [`kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa.c`](../kai/ukernels/matmul/matmul_clamp_f32_u8p_u8p/kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa.c#L27)
  uses `MR_VSCALE`, `NR_VSCALE`, `M_STEP_VSCALE`, and `N_STEP_VSCALE` for an
  `8vsx8vs` tile.

- [`kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa.c`](../kai/ukernels/matmul/matmul_i32_u8p_u8p/kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa.c#L25)
  uses the same `8vsx8vs` style for an integer-accumulation micro-kernel.

- [`kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_sme2_dot.c`](../kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_sme2_dot.c#L26)
  keeps the M dimension fixed at 1 and derives the N dimension from
  `NR_VSCALE * kai_get_sme_vscale()`.

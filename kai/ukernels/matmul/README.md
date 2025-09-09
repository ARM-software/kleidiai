<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# About

This document contains information related to matrix-multiplication (matmul)
micro kernels. At the moment there are two main types kernels matrix
multiplication kernels, and indirect matrix multiplication kernels. The indirect
kernels are denoted _imatmul_.

# Matmul

Matmul kernels operate directly on matrices stored in memory buffers, where the
buffers are normally first packed into a more efficient layout.

# Indirect Matmul

The indirect matmul kernels operate on indirection buffers, matrices of pointers
to actual data.

# Naming convention

## Microkernel naming

The naming of microkernels must follow the convention below. Unless explicitly specified, arguments are mandatory.

`kai_<op>_<fused_ops>_<dst_info>_<input_0_info, input_1_info, ...>_<m_step x n_step>_<simd_engine>_<feature>_<instruction>_<uarch>`

| Syntax                          | Description                                                                                                                        | Example                                                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| op                              | The primary operation of the microkernel                                                                                           | `matmul`, `imatmul `                                                                                                                                                        |
| fused_ops                       | (Optional) Information on applied fused operations, e.g., activation functions                                                     | `clamp`                                                                                                                                                                     |
| dst_info                        | Description of the destination buffer                                                                                              | See Buffer descriptors section                                                                                                                                              |
| input_0_info, input_1_info, ... | Description of input buffers to the microkernel                                                                                    | In `matmul` routines, the LHS precedes the RHS.                                                                                                                             |                                                                                                                                                                                    |
| m_step x n_step                 | Minimum tile size computed by the microkernel                                                                                      | `6x32` where the tile size is 6 rows by 32 columns; `2vlx2vl` where the tile size is equivalent to twice the hardware-defined vector length in the row and column dimensions |
| simd_engine                     | SIMD engine used to drive the computation                                                                                          | `neon`, `sme`, `sme2`                                                                                                                                                       |
| feature                         | (Optional) Further information about the Arm architecture feature used, often referred to as `FEAT_<feature>` in the specification | `dotprod`, `i8mm `                                                                                                                                                          |
| instruction                     | (Optional) Predominant SIMD instruction used in the microkernel                                                                    | `mla`, `mopa`, `sdot`                                                                                                                                                       |
| uarch                           | (Optional) Microarchitecture for which the microkernel has been optimized for                                                      | `cortexa55` to represent the Arm® Cortex®-A55 processor                                                                                                                     |

## Buffer descriptors

Input and output buffers can be described using the following form:

| Syntax   | Description                                                                                      |
|----------|--------------------------------------------------------------------------------------------------|
| f32      | Single-precision floating-point                                                                  |
| f16      | Half-precision floating-point                                                                    |
| bf16     | Brain floating-point                                                                             |
| x        | Data type agnostic. Usually used when describing moving data around like in packing microkernels |
| qs       | Quantized symmetric                                                                              |
| qa       | Quantized asymmetric                                                                             |
| i        | Signed integer                                                                                   |
| u        | Unsigned integer                                                                                 |
| 4        | 4-bit quantized                                                                                  |
| 8        | 8-bit quantized                                                                                  |
| dx       | Per dimension quantized                                                                          |
| cx       | Per channel quantized                                                                            |
| c32      | Per block quantization, with block length multiple of 32                                         |
| scalef16 | Scale factors stored as floating-point 16-bit                                                    |
| p        | Indicates data is packed                                                                         |
| s16s0    | Packing order of data is interleaved                                                             |
| s1s0     | Packing order of data is sequential                                                              |
| s        | Scale factors are packed into buffer                                                             |
| b        | Bias values are packed into buffer                                                               |

Example: `qsi4cxp` which means quantized symmetric (`qs`) signed integer 4-bit data (`i4`) with per channel quantization (`cx`) that has been packed (`p`).

Input buffer descriptors **must** also include information about how the data has been packed to more easily identify the required packing microkernels. In matmul routines this is done by appending `mrxkr` or `nrxkr` to the descriptor where `mr` represents the number of rows of LHS that are packed together, `nr` the number of columns of RHS that are packed together, and `kr` the number of columns of LHS or rows of RHS that are packed together.

## Known naming issues

There are several kernels that unfortunately use the incorrect name. For now we don't change the name as that would break API.

| Kernel                                                           | Correct name                                                     | Comment                     |
| ---------------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------- |
| `imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa`        | `imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme2_mopa`       | Missing bias `b`            |
| `imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa` | `imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme2_mopa` | Misplaced scaling+bias `sb` |
| `lhs_pack_bf16p2vlx2_f32_sme`                                    | `lhs_pack_bf16p2vlx2_f32_sme2`                                   | Incorrectly indicating SME  |
| `lhs_pack_f32p2vlx1_f32_sme`                                     | `lhs_pack_x32p2vlx1_x32_sme`                                     | Legacy naming               |
| `matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa`         | `kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme2_mopa`    | Missing bias `b`            |
| `matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa`       | `matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2b_2vlx2vl_sme2_mopa`      | Also placed in incorrect directory (`fp32_...` should be `f32_...`) |
| `matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa`          | `matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa`        | Legacy naming               |
| `matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa`  | `matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme2_mopa`  | Misplaced scaling+bias `sb` |
| `rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme`                           | `rhs_pack_kxn_bf16p2vlx2b_f32_f32_sme2`                          | Incorrectly indicating SME  |
| `rhs_pack_kxn_f32p16vlx1b_f32_f32_sme`                           | `rhs_pack_kxn_x32p16vlx1b_x32_x32_sme`                           | Legacy naming               |
| `rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme`                      | `rhs_pack_kxn_x32p2vlx1b_x32_x32_sme`                            | Legacy naming               |
| `rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme`                      | `rhs_pack_nxk_x32p2vlx1b_x32_x32_sme`                            | Legacy naming               |

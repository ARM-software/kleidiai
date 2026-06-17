//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// To date, only AArch64 has these 8-bit kernels.
#ifdef __aarch64__

#include "gemm-wrappers.hpp"

#include "kai_ops_kernels.hpp"
#include "gemm_transpose.hpp"

// gemm_u8s8_quant - u8 activations with s8 weights
template void benchmark<gemm_u8s8_quant, false, QuantizationType::INTEGER, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test_quantized<sgemm_new, gemm_u8s8_quant>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_u8s8_quant, false>(GemmProblem *, unsigned int);

// gemm_u8s8fp32 - dequantize result to FP32
template void benchmark<gemm_u8s8fp32, false, QuantizationType::FLOAT, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test_dequantized<sgemm_new, gemm_u8s8fp32>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_u8s8fp32, false>(GemmProblem *, unsigned int);

#endif // aarch64

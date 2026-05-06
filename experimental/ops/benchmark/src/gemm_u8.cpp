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

// gemm_u8_new - GEMM only
template void benchmark<gemm_u8_new, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<uint8_t, uint32_t>, gemm_u8_new, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_u8_new, false>(GemmProblem *, unsigned int);

// gemm_u8_quant - requantizing (per layer) GEMM
template void benchmark<gemm_u8_quant, false, QuantizationType::INTEGER, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test_quantized<sgemm_new, gemm_u8_quant>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_u8_quant, false>(GemmProblem *, unsigned int);

template void benchmark<depthwise_u8q, false, QuantizationType::INTEGER, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void benchmark<depthwise_u8q, false, QuantizationType::INTEGER, true>(GemmProblem *, int, int, int, mapfn, const char *);
template void test_quantized<depthwise_fp32, depthwise_u8q>(GemmProblem *, int, int, const char *, FILE *);
template void test_quantized<depthwise_fp32, depthwise_u8q, true>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<depthwise_u8q, false>(GemmProblem *, unsigned int);

#endif // aarch64

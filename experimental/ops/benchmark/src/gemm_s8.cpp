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

// gemm_s8_new - GEMM only
template void benchmark<gemm_s8_new, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<int8_t, int32_t>, gemm_s8_new, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_s8_new, false>(GemmProblem *, unsigned int);

// gemm_s8fp32 - dequantize result to FP32
template void benchmark<gemm_s8fp32, false, QuantizationType::FLOAT, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test_dequantized<sgemm_new, gemm_s8fp32>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_s8fp32, false>(GemmProblem *, unsigned int);

// gemm_s8fp16 - dequantize result to FP16
template void benchmark<gemm_s8fp16, false, QuantizationType::FLOAT, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test_dequantized<hgemm_new, gemm_s8fp16>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_s8fp16, false>(GemmProblem *, unsigned int);

// gemm_s8_quant - requantizing (per layer) GEMM
template void benchmark<gemm_s8_quant, false, QuantizationType::INTEGER, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test_quantized<sgemm_new, gemm_s8_quant>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_s8_quant, false>(GemmProblem *, unsigned int);

template void benchmark<depthwise_s8q, false, QuantizationType::INTEGER, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void benchmark<depthwise_s8q, false, QuantizationType::INTEGER, true>(GemmProblem *, int, int, int, mapfn, const char *);
template void test_quantized<depthwise_fp32, depthwise_s8q>(GemmProblem *, int, int, const char *, FILE *);
template void test_quantized<depthwise_fp32, depthwise_s8q, true>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<depthwise_s8q, false>(GemmProblem *, unsigned int);

// gemm_s8_quant_perchannel - requantizing (per channel) GEMM
template void benchmark<gemm_s8_quant, false, QuantizationType::INTEGER, true>(GemmProblem *, int, int, int, mapfn, const char *);
template void test_quantized<sgemm_new, gemm_s8_quant, true>(GemmProblem *, int, int, const char *, FILE *);

#endif  // aarch64

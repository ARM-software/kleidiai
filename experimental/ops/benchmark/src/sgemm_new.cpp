//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm-wrappers.hpp"

#include "kai_ops_kernels.hpp"
#include "gemm_transpose.hpp"

template void benchmark<sgemm_new, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<float, float>, sgemm_new, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<sgemm_new, false>(GemmProblem *, unsigned int);

template void benchmark<winograd_fp32, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<float, float>, winograd_fp32, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<winograd_fp32, false>(GemmProblem *, unsigned int);

#ifdef __aarch64__
template void benchmark<depthwise_fp32, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<float, float>, depthwise_fp32, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<depthwise_fp32, false>(GemmProblem *, unsigned int);
#endif // AARCH64

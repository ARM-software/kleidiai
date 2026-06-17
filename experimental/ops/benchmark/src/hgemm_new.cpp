//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm-wrappers.hpp"

#include "kai_ops_kernels.hpp"
#include "gemm_transpose.hpp"

#ifdef __aarch64__
template void benchmark<hgemm_new, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<__fp16, __fp16>, hgemm_new, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<hgemm_new, false>(GemmProblem *, unsigned int);

template void benchmark<gemm_fp16fp32, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<__fp16, float>, gemm_fp16fp32, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_fp16fp32, false>(GemmProblem *, unsigned int);

template void benchmark<winograd_fp16, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<__fp16, __fp16>, winograd_fp16, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<winograd_fp16, false>(GemmProblem *, unsigned int);

template void benchmark<depthwise_fp16, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<__fp16, __fp16>, depthwise_fp16, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<depthwise_fp16, false>(GemmProblem *, unsigned int);
#endif // AArch64

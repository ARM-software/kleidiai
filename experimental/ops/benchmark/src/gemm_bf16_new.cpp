//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm-wrappers.hpp"

#include "kai_ops_kernels.hpp"
#include "gemm_transpose.hpp"

template void benchmark<gemm_bf16_new, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<bfloat16, float>, gemm_bf16_new, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_bf16_new, false>(GemmProblem *, unsigned int);

template void benchmark<gemm_bf16bf16, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<bfloat16, bfloat16>, gemm_bf16bf16, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_bf16bf16, false>(GemmProblem *, unsigned int);

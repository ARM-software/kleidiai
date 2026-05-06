//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// To date, only AArch64 has these 16-bit kernels.
#ifdef __aarch64__

#include "gemm-wrappers.hpp"

#include "kai_ops_kernels.hpp"
#include "gemm_transpose.hpp"

// gemm_s16 - GEMM only
template void benchmark<gemm_s16, false, QuantizationType::NONE, false>(GemmProblem *, int, int, int, mapfn, const char *);
template void test<gemm_transposeB<int16_t, int32_t>, gemm_s16, false>(GemmProblem *, int, int, const char *, FILE *);
template void print_kernels<gemm_s16, false>(GemmProblem *, unsigned int);

#endif // aarch64

//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfloat>
#include <cstddef>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"

namespace kai::benchmark {

inline void kai_run_lhs_quant_pack_qai8dxp_f32_as_void(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
    void* lhs_packed) {
    kai_run_lhs_quant_pack_qai8dxp_f32(
        m, k, mr, kr, sr, m_idx_start, static_cast<const float*>(lhs), lhs_stride, lhs_packed);
}

template <auto RunPack, auto RunMatMul>
void run_pack_matmul_base(
    size_t m, size_t n, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs,
    size_t lhs_stride_row, const void* rhs_packed, void* lhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col) {
    KAI_UNUSED(bl);

    RunPack(m, k, mr, kr, sr, m_idx_start, lhs, lhs_stride_row, lhs_packed);
    RunMatMul(m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row, dst_stride_col, -FLT_MAX, FLT_MAX);
}

template <auto RunPack, auto RunMatMul>
void run_pack_matmul_float(
    size_t m, size_t n, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs,
    size_t lhs_stride_row, const void* rhs_packed, void* lhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col) {
    KAI_UNUSED(bl);

    RunPack(m, k, mr, kr, sr, m_idx_start, lhs, lhs_stride_row, lhs_packed);
    RunMatMul(
        m, n, k, lhs_packed, rhs_packed, static_cast<float*>(dst), dst_stride_row, dst_stride_col, -FLT_MAX, FLT_MAX);
}

template <auto RunPack, auto RunMatMul>
void run_pack_matmul_blockwise_dynamic_quant(
    size_t m, size_t n, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs,
    size_t lhs_stride_row, const void* rhs_packed, void* lhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col) {
    RunPack(m, k, mr, kr, sr, m_idx_start, lhs, lhs_stride_row, lhs_packed);
    RunMatMul(
        m, n, k, bl, lhs_packed, rhs_packed, static_cast<float*>(dst), dst_stride_row, dst_stride_col, -FLT_MAX,
        FLT_MAX);
}

template <auto RunPack, auto RunMatMul>
void run_pack_matmul_blockwise_dynamic_quant_generic_dst(
    size_t m, size_t n, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs,
    size_t lhs_stride_row, const void* rhs_packed, void* lhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col) {
    RunPack(m, k, mr, kr, sr, m_idx_start, lhs, lhs_stride_row, lhs_packed);
    RunMatMul(m, n, k, bl, lhs_packed, rhs_packed, dst, dst_stride_row, dst_stride_col, -FLT_MAX, FLT_MAX);
}

}  // namespace kai::benchmark

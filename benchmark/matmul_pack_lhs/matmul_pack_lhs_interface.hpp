//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "kai/ukernels/matmul/kai_matmul_pack_lhs_types.h"

namespace kai::benchmark {

/// Interface for an LHS packing micro-kernel using the base function-based API.
struct MatMulPackLhsBaseInterface {
    size_t (*get_mr)(void);
    size_t (*get_kr)(void);
    size_t (*get_sr)(void);
    size_t (*get_lhs_packed_size)(size_t m, size_t k, size_t mr, size_t kr, size_t sr);
    void (*run)(
        size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
        void* lhs_packed);
    bool (*is_valid)(size_t m, size_t k, size_t bl) = nullptr;
};

/// Interface for an LHS packing micro-kernel with a floating-point input matrix.
struct MatMulPackLhsFloatInterface {
    size_t (*get_mr)(void);
    size_t (*get_kr)(void);
    size_t (*get_sr)(void);
    size_t (*get_lhs_packed_size)(size_t m, size_t k, size_t mr, size_t kr, size_t sr);
    void (*run)(
        size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs, size_t lhs_stride,
        void* lhs_packed);
};

/// Interface for a blockwise LHS packing micro-kernel.
struct MatMulPackLhsBlockwiseInterface {
    size_t (*get_mr)(void);
    size_t (*get_kr)(void);
    size_t (*get_sr)(void);
    size_t (*get_lhs_packed_size)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);
    void (*run)(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs,
        size_t lhs_stride, void* lhs_packed);
    bool (*is_valid)(size_t m, size_t k, size_t bl) = nullptr;
};

/// Interface for a blockwise LHS packing micro-kernel with a floating-point input matrix.
struct MatMulPackLhsBlockwiseFloatInterface {
    size_t (*get_mr)(void);
    size_t (*get_kr)(void);
    size_t (*get_sr)(void);
    size_t (*get_lhs_packed_size)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);
    void (*run)(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs,
        size_t lhs_stride, void* lhs_packed);
};

/// Interface for an LHS packing micro-kernel using the struct-based API.
struct MatMulPackLhsUkernelApiInterface {
    kai_matmul_pack_lhs_uker_config (*get_config)(void);
    kai_matmul_pack_lhs_uker_api (*get_api)(void);
    bool (*is_valid)(size_t m, size_t k, size_t bl) = nullptr;
};

}  // namespace kai::benchmark

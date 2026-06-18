//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "test/common/data_type.hpp"

namespace kai::benchmark {
using test::DataType;

/// High level description of the matrix multiplication operation.
enum class PackMatMulOp : uint8_t {
    GEMM,
    GEMV,
};

using PackMatMulRunner = void (*)(
    size_t m, size_t n, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs,
    size_t lhs_stride_row, const void* rhs_packed, void* lhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col);

/// Pack-matmul benchmark entry with an associated LHS packing micro-kernel.
struct PackMatMulEntry {
    std::string_view benchmark_name;
    std::string_view matmul_name;

    DataType lhs_type;
    DataType dst_type;
    PackMatMulOp matmul_op;
    bool needs_block_size;
    bool (*is_cpu_supported)(void);

    size_t (*get_mr)(void);
    size_t (*get_kr)(void);
    size_t (*get_sr)(void);
    size_t (*get_lhs_offset)(size_t m_idx, size_t lhs_stride_row);
    size_t (*get_lhs_packed_offset)(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr);
    size_t (*get_lhs_packed_size)(size_t m, size_t k, size_t mr, size_t kr, size_t sr);
    size_t (*get_matmul_lhs_packed_offset)(size_t m_idx, size_t k);
    PackMatMulRunner run_pack_matmul;
};

}  // namespace kai::benchmark

//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace kai::test {

/// Matrix multiplication tensor slots.
enum class MatMulSlot : uint32_t {
    CONFIG,       ///< Matrix multiplication operator configuration.
    PACK_ARGS,    ///< Packing arguments.
    MATMUL_ARGS,  ///< Matrix multiplication micro-kernel parameters.

    LHS_RAW,         ///< LHS data in F32.
    LHS_DATA,        ///< LHS data after conversion.
    LHS_QDATA,       ///< LHS data after quantization.
    LHS_QSCALE,      ///< LHS quantization scale.
    LHS_QZP,         ///< LHS quantization zero-point.
    LHS_QZP_NEG,     ///< Negative LHS quantization zero-point.
    REF_LHS_PACKED,  ///< Reference packed LHS.
    IMP_LHS_PACKED,  ///< Packed LHS from micro-kernel.

    RHS_RAW,               ///< RHS data in F32.
    RHS_T_RAW,             ///< Transposed RHS data in F32.
    RHS_T_DATA,            ///< Transposed RHS data after conversion.
    RHS_T_QDATA,           ///< Transposed RHS data after quantization.
    RHS_T_QDATA_SIGN,      ///< Transposed RHS data after quantization with opposite signedness.
    RHS_T_QDATA_SIGN_SUM,  ///< Row sum of transposed RHS after quantization with opposite signedness.
    RHS_T_QSCALE,          ///< Transposed RHS quantization scale.
    RHS_T_QZP,             ///< Transposed RHS quantization zero-point.
    REF_RHS_PACKED,        ///< Reference packed RHS.
    IMP_RHS_PACKED,        ///< Packed RHS from micro-kernel.

    BIAS_RAW,
    BIAS_DATA,
    BIAS_SCALE,
    BIAS_ZP,
    BIAS_PACKED,

    DST_RAW,
    REF_DST_DATA,
    DST_SCALE,
    DST_ZP,

    IMP_DST_DATA,

    LAST,  ///< Sentinel value equal to the number of slots.
};

template <typename T>
constexpr std::underlying_type_t<T> as_idx(T val) noexcept {
    return static_cast<std::underlying_type_t<T>>(val);
}

template <typename T>
constexpr size_t n_entries() noexcept {
    return static_cast<size_t>(as_idx(T::LAST));
}

}  // namespace kai::test

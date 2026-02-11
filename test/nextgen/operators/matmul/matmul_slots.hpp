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

    // If the input is floating-point:
    //   * DATA contains the input data.
    //   * CVT_DATA contains the converted input data if the operator computes
    //     in different data type (lower precision for better performance).
    //   * QDATA, QSCALE, QZP are unused.
    //
    // If the input is quantized:
    //   * DATA contains the source floating-point data in F32 which later
    //     will be used to determine the quantization parameters and then quantized.
    //   * CVT_DATA is unused.
    //   * QDATA, QSCALE, QZP contain the quantized data and quantization parameters
    //     calculated from the source floating-point data.

    LHS_DATA,        ///< LHS data.
    LHS_CVT_DATA,    ///< LHS data after conversion.
    LHS_QDATA,       ///< LHS data after quantization.
    LHS_QSCALE,      ///< LHS quantization scale.
    LHS_QZP,         ///< LHS quantization zero-point.
    LHS_QZP_NEG,     ///< Negative LHS quantization zero-point.
    LHS_PACKED,      ///< Packed LHS.
    LHS_PACKED_IMP,  ///< Packed LHS from micro-kernel.

    RHS_DATA,              ///< RHS data.
    RHS_CVT_DATA,          ///< RHS data after conversion.
    RHS_T_DATA,            ///< Transposed RHS data.
    RHS_T_CVT_DATA,        ///< Transposed RHS data after conversion.
    RHS_T_QDATA,           ///< Transposed RHS data after quantization.
    RHS_T_QDATA_SIGN,      ///< Transposed RHS data after quantization with opposite signedness.
    RHS_T_QDATA_SIGN_SUM,  ///< Row sum of transposed RHS after quantization with opposite signedness.
    RHS_T_QSCALE,          ///< Transposed RHS quantization scale.
    RHS_T_QZP,             ///< Transposed RHS quantization zero-point.
    RHS_PACKED,            ///< Packed RHS.
    RHS_PACKED_IMP,        ///< Packed RHS from micro-kernel.

    BIAS_DATA,      ///< Bias data.
    BIAS_CVT_DATA,  ///< Bias data after conversion.
    BIAS_SCALE,     ///< Bias quantization scale.
    BIAS_ZP,        ///< Bias quantization zero-point.
    BIAS_PACKED,    ///< Packed bias.

    DST_DATA,      ///< Output data (can be floating-point or quantized).
    DST_QSCALE,    ///< Output quantization scale.
    DST_QZP,       ///< Output quantization zero-point.
    DST_DATA_IMP,  ///< Output data from micro-kernel.

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

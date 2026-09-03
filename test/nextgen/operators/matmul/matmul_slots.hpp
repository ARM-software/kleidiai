//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::test {

/// Matrix multiplication tensor slots.
enum class MatMulSlot : size_t {
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

    LHS_DATA,                      ///< LHS data.
    LHS_CVT_DATA,                  ///< LHS data after conversion.
    LHS_QDATA,                     ///< LHS data after quantization.
    LHS_QDATA_SUM,                 ///< Per-block sum of quantized LHS data.
    LHS_QSCALE,                    ///< LHS quantization scale.
    LHS_QSCALE_CVT,                ///< Converted LHS quantization scale.
    LHS_QSCALE_MUL_LHS_QDATA_SUM,  ///< Scaled per-block sum of quantized LHS data.
    LHS_QSCALE_DIV_DST_QSCALE,     ///< LHS quantization scale divided by DST quantization scale.
    LHS_QZP,                       ///< LHS quantization zero-point.
    LHS_QZP_NEG,                   ///< Negative LHS quantization zero-point.
    LHS_PACKED,                    ///< Packed LHS.
    LHS_PACKED_IMP,                ///< Packed LHS from micro-kernel.

    RHS_DATA,        ///< RHS data.
    RHS_CVT_DATA,    ///< RHS data after conversion.
    RHS_QDATA,       ///< RHS data after quantization in non-transposed layout.
    RHS_QDATA_SIGN,  ///< Non-transposed RHS data after quantization with opposite signedness.
    RHS_PACKED,      ///< Packed RHS.
    RHS_PACKED_IMP,  ///< Packed RHS from micro-kernel.

    RHS_T_DATA,            ///< Transposed RHS data.
    RHS_T_CVT_DATA,        ///< Transposed RHS data after conversion.
    RHS_T_QDATA,           ///< Transposed RHS data after quantization.
    RHS_T_QDATA_SIGN,      ///< Transposed RHS data after quantization with opposite signedness.
    RHS_T_QDATA_SIGN_SUM,  ///< Row sum of transposed RHS after quantization with opposite signedness.
    RHS_T_QSCALE,          ///< Transposed RHS quantization scale.
    RHS_T_QSCALE_MUL_LHS_QSCALE_DIV_DST_QSCALE,  ///< Static Int8 RHS packed scale component.
    RHS_T_QZP,                                   ///< Transposed RHS quantization zero-point.

    ACC_BIAS_GLOBAL_DATA,                                    ///< Accumulator bias data, global.
    ACC_BIAS_M_DATA,                                         ///< Accumulator bias data per M.
    ACC_BIAS_N_DATA,                                         ///< Accumulator bias data per N.
    ACC_BIAS_N_QDATA,                                        ///< Quantized accumulator bias data per N.
    ACC_BIAS_N_QSCALE,                                       ///< Accumulator bias quantization scale per N.
    ACC_BIAS_N_QDATA_MINUS_LHS_QZP_MUL_RHS_T_QDATA_ROW_SUM,  ///< Static Int8 RHS packed bias component.

    SCALE_BIAS_GLOBAL_DATA,  ///< Scaled accumulator bias data, global.
    SCALE_BIAS_M_DATA,       ///< Scaled accumulator bias data per M.
    SCALE_BIAS_N_DATA,       ///< Scaled accumulator bias data per N.

    REF_ACC_MATMUL_DATA,  ///< Matrix multiplication accumulator without pre or post processing.

    ACC_SCALE_GLOBAL_DATA,  ///< Accumulator scale data, global.
    ACC_SCALE_M_DATA,       ///< Accumulator scale data per M.
    ACC_SCALE_N_DATA,       ///< Accumulator scale data per N.

    DST_DATA,      ///< Output data (can be floating-point or quantized).
    DST_QSCALE,    ///< Output quantization scale.
    DST_QZP,       ///< Output quantization zero-point.
    DST_DATA_IMP,  ///< Output data from micro-kernel.

    LAST,  ///< Sentinel value equal to the number of slots.
};

}  // namespace kai::test

//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "kai/ukernels/dwconv/kai_dwconv_types.h"
#include "test/common/enum_utils.hpp"
#include "test/nextgen/common/clamp_config.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/operators/dwconv/dwconv_bias_mode.hpp"
#include "test/nextgen/operators/dwconv/kernel_types.hpp"

namespace kai::test {

/// Stage parameter layout for a depthwise convolution micro-kernel using the struct-based API.
enum class DwConvUkerStageParameterLayout : uint8_t {
    GLOBAL,  ///< Scalar stage parameter.
    PER_H,   ///< Per-output-row stage parameters.
    PER_W,   ///< Per-output-column stage parameters.
    PER_C,   ///< Per-channel stage parameters.
};

/// Set of stage parameter layouts.
using DwConvUkerStageParameterLayoutSet = FlagSet<DwConvUkerStageParameterLayout>;

/// Output stage configuration for a depthwise convolution micro-kernel using the struct-based API.
struct DwConvUkerOutputStageConfig {
    DwConvUkerStageParameterLayoutSet acc_scale;   ///< Accumulator scaling parameter layouts.
    DwConvUkerStageParameterLayoutSet scale_bias;  ///< Scaled accumulator bias parameter layouts.

    DwConvSlot acc_scale_global_slot = DwConvSlot::ACC_SCALE_GLOBAL_DATA;
    DwConvSlot acc_scale_h_slot = DwConvSlot::ACC_SCALE_H_DATA;
    DwConvSlot acc_scale_w_slot = DwConvSlot::ACC_SCALE_W_DATA;
    DwConvSlot acc_scale_c_slot = DwConvSlot::ACC_SCALE_C_DATA;
    DwConvSlot scale_bias_global_slot = DwConvSlot::SCALE_BIAS_GLOBAL_DATA;

    std::vector<DwConvSlot> extra_ref_inputs;  ///< Extra tensors needed only for reference generation.
};

/// Wraps a depthwise convolution micro-kernel using the struct-based API.
/// Supports unit convolution stride and dilation, matching the testbench reference.
class DwConvUkerApiWrapper : public DwConvKernel {
public:
    /// Creates a wrapper with the micro-kernel's formats, clamp, bias delivery, and output stage.
    DwConvUkerApiWrapper(
        std::string_view name, kai_dwconv_uker_api api,                  // Micro-kernel identity and API
        const Poly<Format>& lhs_format, const Poly<Format>& rhs_format,  // Input formats
        const Poly<Format>& dst_format,                                  // Output format
        ClampConfig clamp_config,                                        // Clamping
        DwConvUkerApiBiasDeliveryStage bias_delivery_stage,              // Bias delivery
        DwConvUkerOutputStageConfig output_stage_config = {},            // Output stage
        std::optional<DwConvSlot> padding_value_slot = std::nullopt,     // Padding tensor
        DwConvSlot rhs_input_slot = DwConvSlot::RHS_PACKED,              // RHS tensor
        DwConvSlot lhs_input_slot = DwConvSlot::LHS_DATA,                // LHS tensor
        DwConvSlot acc_bias_c_slot = DwConvSlot::ACC_BIAS_C_DATA,        // Per-channel accumulator bias tensor
        kai_dwconv_uker_config uker_config = {}                          // Micro-kernel configuration
        ) :
        m_name(name),
        m_ukernel(api),
        m_uker_config(uker_config),
        m_lhs_format(lhs_format),
        m_rhs_format(rhs_format),
        m_dst_format(dst_format),
        m_clamp_config(clamp_config),
        m_bias_delivery_stage(bias_delivery_stage),
        m_output_stage_config(output_stage_config),
        m_padding_value_slot(padding_value_slot),
        m_rhs_input_slot(rhs_input_slot),
        m_lhs_input_slot(lhs_input_slot),
        m_acc_bias_c_slot(acc_bias_c_slot) {
    }

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::vector<DwConvSlot> run_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<DwConvSlot> ref_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<size_t> steps(DwConvShape shape, ConstTensorSet tensors) const override;
    void populate_constant_info(TensorSet tensors) const override;
    void run(DwConvShape full_shape, Span<const size_t> tile_coords, DwConvShape tile_shape, TensorSet tensors)
        const override;
    void compute_reference(DwConvShape shape, TensorSet tensors) const override;

private:
    std::string m_name;
    kai_dwconv_uker_api m_ukernel;
    kai_dwconv_uker_config m_uker_config;
    Poly<Format> m_lhs_format;
    Poly<Format> m_rhs_format;
    Poly<Format> m_dst_format;
    ClampConfig m_clamp_config;
    DwConvUkerApiBiasDeliveryStage m_bias_delivery_stage;
    DwConvUkerOutputStageConfig m_output_stage_config;
    std::optional<DwConvSlot> m_padding_value_slot;
    DwConvSlot m_rhs_input_slot;
    DwConvSlot m_lhs_input_slot;
    DwConvSlot m_acc_bias_c_slot;
};

}  // namespace kai::test

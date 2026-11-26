//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kai/ukernels/dwconv/kai_dwconv_pack_rhs_types.h"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/operators/dwconv/dwconv_bias_mode.hpp"
#include "test/nextgen/operators/dwconv/kernel_types.hpp"

namespace kai::test {

/// Tensor slots for optional depthwise RHS packing operands.
struct DwConvPackRhsOperandSlots {
    std::optional<DwConvSlot> bias_c;  ///< Per-channel accumulator bias.
};

/// Wraps a depthwise RHS packing micro-kernel using the struct-based API.
class DwConvPackRhsUkerApiWrapper : public DwConvPackKernel {
public:
    /// Creates a wrapper with the micro-kernel's API, formats, and tensor slots.
    /// The optional configuration must remain valid for the lifetime of the wrapper.
    DwConvPackRhsUkerApiWrapper(
        std::string_view name,                               // Name
        kai_dwconv_pack_rhs_uker_api api,                    // API
        const Poly<Format>& src_data_format,                 // Filter format
        const Poly<Format>& src_bias_format,                 // Bias format
        const Poly<Format>& dst_format,                      // Packed format
        DwConvUkerApiBiasDeliveryStage,                      // Matches the matmul wrapper interface
        DwConvSlot run_rhs_slot = DwConvSlot::RHS_DATA,      // Filter slot
        DwConvPackRhsOperandSlots operand_slots = {},        // Optional slots
        std::vector<DwConvSlot> reference_component_slots =  // Packed components
        {DwConvSlot::RHS_DATA},
        const kai_dwconv_pack_rhs_uker_config* config = nullptr  // Configuration
        ) :
        m_name(name),
        m_run_rhs_slot(run_rhs_slot),
        m_uker_config(config),
        m_api(api),
        m_src_data_format(src_data_format),
        m_src_bias_format(src_bias_format),
        m_dst_format(dst_format),
        m_operand_slots(operand_slots),
        m_reference_component_slots(std::move(reference_component_slots)) {
    }

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::vector<DwConvSlot> run_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<DwConvSlot> ref_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<size_t> steps(DwConvPackShape shape, ConstTensorSet) const override;
    void populate_constant_info(TensorSet tensors) const override;
    void run(
        DwConvPackShape full_shape, Span<const size_t> tile_coords,  // Full shape and tile origin
        DwConvPackShape tile_shape, TensorSet tensors                // Tile extent and tensor pool
    ) const override;
    void compute_reference(DwConvPackShape shape, TensorSet tensors) const override;

private:
    std::string m_name;
    DwConvSlot m_run_rhs_slot;
    const kai_dwconv_pack_rhs_uker_config* m_uker_config;
    kai_dwconv_pack_rhs_uker_api m_api;

    Poly<Format> m_src_data_format;
    Poly<Format> m_src_bias_format;
    Poly<Format> m_dst_format;

    DwConvPackRhsOperandSlots m_operand_slots;
    std::vector<DwConvSlot> m_reference_component_slots;
};

}  // namespace kai::test

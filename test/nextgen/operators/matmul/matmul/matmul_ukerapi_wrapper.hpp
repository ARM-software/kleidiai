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

#include "kai/ukernels/matmul/kai_matmul_types.h"
#include "test/common/data_type.hpp"
#include "test/common/enum_utils.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/matmul_dims.hpp"

namespace kai::test {

/// Clamping configuration for a matrix multiplication micro-kernel using the ukernel API.
class MatMulUkerClampConfig {
public:
    /// Clamping support.
    enum class Support : uint8_t {
        UNSUPPORTED,  ///< Clamping is not supported.
        OPTIONAL,     ///< Clamping parameters are supported but not required.
        REQUIRED,     ///< Clamping parameters are required.
    };

    /// Creates a configuration for a micro-kernel that does not support clamping.
    [[nodiscard]] static MatMulUkerClampConfig unsupported() {
        return MatMulUkerClampConfig(Support::UNSUPPORTED, std::nullopt);
    }

    /// Creates a configuration for a micro-kernel with optional clamping arguments.
    [[nodiscard]] static MatMulUkerClampConfig optional(DataType data_type) {
        return MatMulUkerClampConfig(Support::OPTIONAL, data_type);
    }

    /// Creates a configuration for a micro-kernel with required clamping arguments.
    [[nodiscard]] static MatMulUkerClampConfig required(DataType data_type) {
        return MatMulUkerClampConfig(Support::REQUIRED, data_type);
    }

    /// Gets the clamping support.
    [[nodiscard]] Support support() const {
        return m_support;
    }

    /// Gets the clamping argument data type, if clamping is supported.
    [[nodiscard]] std::optional<DataType> data_type() const {
        return m_data_type;
    }

private:
    MatMulUkerClampConfig(Support support, std::optional<DataType> data_type) :
        m_support(support), m_data_type(data_type) {
    }

    Support m_support;                    ///< Clamping support.
    std::optional<DataType> m_data_type;  ///< Clamping argument data type.
};

/// Stage parameter layout for a matrix multiplication micro-kernel using the ukernel API.
enum class MatMulUkerStageParameterLayout : uint8_t {
    GLOBAL,  ///< Scalar stage parameter.
    PER_M,   ///< Per-row stage parameters.
    PER_N,   ///< Per-column stage parameters.
};

/// Set of stage parameter layouts.
using MatMulUkerStageParameterLayoutSet = FlagSet<MatMulUkerStageParameterLayout>;

/// Output stage configuration for a matrix multiplication micro-kernel using the ukernel API.
struct MatMulUkerOutputStageConfig {
    MatMulUkerStageParameterLayoutSet acc_scale;   ///< Accumulator scaling parameter layouts.
    MatMulUkerStageParameterLayoutSet scale_bias;  ///< Scaled accumulator bias parameter layouts.
};

/// Wrapper for uker-api matrix multiplication micro-kernel.
class MatMulUkerApiWrapper : public KernelWrapper<MatMulShape> {
public:
    /// Creates a new wrapper.
    MatMulUkerApiWrapper(
        std::string_view name, kai_matmul_uker_api api, const Poly<Format>& lhs_format, const Poly<Format>& rhs_format,
        const Poly<Format>& dst_format, DataType acc_dtype, MatMulUkerClampConfig clamp_config,
        MatMulUkerApiBiasDeliveryStage bias_delivery_stage, MatMulUkerOutputStageConfig output_stage_config = {}) :
        m_name(name),
        m_uker_config(),
        m_ukernel(api),
        m_lhs_format(lhs_format),
        m_rhs_format(rhs_format),
        m_dst_format(dst_format),
        m_acc_dtype(acc_dtype),
        m_clamp_config(clamp_config),
        m_bias_delivery_stage(bias_delivery_stage),
        m_output_stage_config(output_stage_config) {
    }

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::vector<MatMulSlot> run_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<MatMulSlot> ref_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<size_t> steps(MatMulShape shape, ConstTensorSet tensors) const override;
    void populate_constant_info(TensorSet tensors) const override;
    void run(MatMulShape full_shape, Span<const size_t> tile_coords, MatMulShape tile_shape, TensorSet tensors)
        const override;
    void compute_reference(MatMulShape shape, TensorSet tensors) const override;

private:
    std::string m_name;                                    ///< Name of the matrix multiplication micro-kernel.
    kai_matmul_uker_config m_uker_config;                  ///< Micro-kernel configuration.
    kai_matmul_uker_api m_ukernel;                         ///< Micro-kernel API function table.
    Poly<Format> m_lhs_format;                             ///< LHS packed data format consumed by the micro-kernel.
    Poly<Format> m_rhs_format;                             ///< RHS packed data format consumed by the micro-kernel.
    Poly<Format> m_dst_format;                             ///< Destination data format produced by the micro-kernel.
    DataType m_acc_dtype;                                  ///< Accumulation data type.
    MatMulUkerClampConfig m_clamp_config;                  ///< Clamp argument configuration.
    MatMulUkerApiBiasDeliveryStage m_bias_delivery_stage;  ///< Stage where bias is delivered to the micro-kernel.
    MatMulUkerOutputStageConfig m_output_stage_config;     ///< Output stage configuration.
};

}  // namespace kai::test

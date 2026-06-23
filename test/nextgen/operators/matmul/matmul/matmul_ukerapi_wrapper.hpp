//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_interface.hpp"
#include "test/nextgen/operators/matmul/matmul_dims.hpp"

namespace kai::test {

/// Wrapper for floating-point matrix multiplication ukernel API.
class MatMulUkerApiWrapper : public KernelWrapper<MatMulShape> {
public:
    // ------------------------------------------------------------
    // 1️⃣ Explicit ukernel constructor (qmx / SME2 / future)
    // ------------------------------------------------------------
    MatMulUkerApiWrapper(
        std::string_view name,
        kai_matmul_uker_api ukernel,
        const Poly<Format>& lhs_format,
        const Poly<Format>& rhs_format,
        const Poly<Format>& dst_format)
        : m_name(name),
          m_uker_config(),
          m_ukernel(ukernel),
          m_lhs_format(lhs_format),
          m_rhs_format(rhs_format),
          m_dst_format(dst_format) {}

    // ------------------------------------------------------------
    // 2️⃣ Backward-compatible constructor (DEFAULT = qmx)
    // ✅ This FIXES your build error
    // ------------------------------------------------------------
    MatMulUkerApiWrapper(
        std::string_view name,
        const Poly<Format>& lhs_format,
        const Poly<Format>& rhs_format,
        const Poly<Format>& dst_format)
        : MatMulUkerApiWrapper(
              name,
              kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_qmx_mopa(),
              lhs_format,
              rhs_format,
              dst_format) {}

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::vector<MatMulSlot> run_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<MatMulSlot> ref_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<size_t> steps(
        MatMulShape shape,
        ConstTensorSet tensors) const override;

    void populate_constant_info(TensorSet tensors) const override;

    void run(
        MatMulShape full_shape,
        Span<const size_t> tile_coords,
        MatMulShape tile_shape,
        TensorSet tensors) const override;

    void compute_reference(
        MatMulShape shape,
        TensorSet tensors) const override;

private:
    std::string m_name;
    kai_matmul_uker_config m_uker_config;
    kai_matmul_uker_api m_ukernel;

    Poly<Format> m_lhs_format;
    Poly<Format> m_rhs_format;
    Poly<Format> m_dst_format;
};

// -----------------------------------------------------------------------------
// ✅ Optional explicit factories (recommended for new code)
// -----------------------------------------------------------------------------

inline MatMulUkerApiWrapper make_matmul_uker_qmx(
    std::string_view name,
    const Poly<Format>& lhs_format,
    const Poly<Format>& rhs_format,
    const Poly<Format>& dst_format) {
    return MatMulUkerApiWrapper(
        name,
        kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_qmx_mopa(),
        lhs_format,
        rhs_format,
        dst_format);
}

inline MatMulUkerApiWrapper make_matmul_uker_sme2(
    std::string_view name,
    const Poly<Format>& lhs_format,
    const Poly<Format>& rhs_format,
    const Poly<Format>& dst_format) {
    return MatMulUkerApiWrapper(
        name,
        kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(),
        lhs_format,
        rhs_format,
        dst_format);
}

}  // namespace kai::test
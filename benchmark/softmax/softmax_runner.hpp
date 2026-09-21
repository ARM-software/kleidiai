//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "softmax_interface.hpp"

namespace kai::benchmark {

/// Prepares and invokes a softmax micro-kernel.
class SoftmaxRunner {
public:
    /// Creates a softmax micro-kernel runner.
    ///
    /// @param[in] softmax_interface Softmax micro-kernel interface.
    explicit SoftmaxRunner(const SoftmaxInterface& softmax_interface) : m_api(softmax_interface.get_api()) {
    }

    /// Sets the size of dimension 0.
    ///
    /// @param[in] dim_0 Size of dimension 0.
    void set_dim_0(size_t dim_0) {
        m_dim_0 = dim_0;
    }

    /// Gets the required destination buffer size.
    ///
    /// @return Destination buffer size in bytes.
    [[nodiscard]] size_t get_dst_size() const {
        const kai_softmax_uker_dst_dim_args dst_shape = {m_dim_0};
        const kai_softmax_uker_dst_stride_args dst_stride = m_api.get_dst_stride(&m_config, &dst_shape);
        return m_api.get_dst_size(&m_config, &dst_shape, &dst_stride);
    }

    /// Prepares the softmax micro-kernel arguments.
    ///
    /// @param[in] src Source buffer.
    /// @param[out] dst Destination buffer.
    void prepare(const void* src, void* dst) {
        const kai_softmax_uker_src_dim_args src_shape = {m_dim_0};
        const kai_softmax_uker_src_stride_args src_stride = m_api.get_src_stride(&m_config, &src_shape);
        const kai_softmax_uker_dst_dim_args dst_shape = {m_dim_0};
        const kai_softmax_uker_dst_stride_args dst_stride = m_api.get_dst_stride(&m_config, &dst_shape);

        m_args.flags = 0;
        m_args.shape.dim_0 = m_dim_0;
        m_args.operand.src.ptr = src;
        m_args.operand.src.stride = src_stride;
        m_args.operand.dst.ptr = dst;
        m_args.operand.dst.stride = dst_stride;
    }

    /// Runs the softmax micro-kernel.
    void run() const {
        m_api.run(&m_config, &m_args);
    }

private:
    kai_softmax_uker_api m_api;
    kai_softmax_uker_config m_config{};
    size_t m_dim_0 = 0;
    kai_softmax_uker_args m_args{};
};

}  // namespace kai::benchmark

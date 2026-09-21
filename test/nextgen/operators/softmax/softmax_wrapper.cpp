//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/softmax/softmax_wrapper.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>

#include "kai/ukernels/softmax/kai_softmax_types.h"
#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/nextgen/harness/tensor.hpp"

namespace kai::test {

SoftmaxWrapper::SoftmaxWrapper(std::string_view name, kai_softmax_uker_api api) : m_name(name), m_api(api) {
}

std::string_view SoftmaxWrapper::name() const {
    return m_name;
}

void SoftmaxWrapper::run(const Tensor& src, Tensor& dst) const {
    KAI_TEST_ASSERT_MSG(m_api.run != nullptr, "Softmax: Missing run API function.");
    KAI_TEST_ASSERT_MSG(m_api.get_step != nullptr, "Softmax: Missing get_step API function.");
    KAI_TEST_ASSERT_MSG(m_api.get_src_stride != nullptr, "Softmax: Missing get_src_stride API function.");
    KAI_TEST_ASSERT_MSG(m_api.get_src_offset != nullptr, "Softmax: Missing get_src_offset API function.");
    KAI_TEST_ASSERT_MSG(m_api.get_dst_stride != nullptr, "Softmax: Missing get_dst_stride API function.");
    KAI_TEST_ASSERT_MSG(m_api.get_dst_offset != nullptr, "Softmax: Missing get_dst_offset API function.");
    KAI_TEST_ASSERT_MSG(m_api.get_dst_size != nullptr, "Softmax: Missing get_dst_size API function.");

    const Shape src_tensor_shape = src.shape();
    const Shape dst_tensor_shape = dst.shape();
    KAI_TEST_ASSERT_MSG(src_tensor_shape.size() == 1, "Softmax: The source tensor must be one-dimensional.");
    KAI_TEST_ASSERT_MSG(
        dst_tensor_shape.size() == src_tensor_shape.size() &&
            std::equal(dst_tensor_shape.begin(), dst_tensor_shape.end(), src_tensor_shape.begin()),
        "Softmax: Source and destination shapes must match.");

    const size_t length = src_tensor_shape.at(0);
    KAI_TEST_ASSERT_MSG(length > 0, "Softmax: The input length must be non-zero.");
    KAI_TEST_ASSERT_MSG(
        src.data().size() == src.format()->compute_size(src_tensor_shape), "Softmax: Unexpected source buffer size.");
    KAI_TEST_ASSERT_MSG(
        dst.data().size() == dst.format()->compute_size(dst_tensor_shape),
        "Softmax: Unexpected destination buffer size.");

    const kai_softmax_uker_dim_args step = m_api.get_step(&m_config);
    KAI_TEST_ASSERT_MSG(step.dim_0 == 0, "Softmax: The wrapper requires the full softmax dimension.");

    const kai_softmax_uker_src_dim_args src_shape = {length};
    const kai_softmax_uker_src_stride_args src_stride = m_api.get_src_stride(&m_config, &src_shape);
    const std::array unit_shape{static_cast<size_t>(1)};
    KAI_TEST_ASSERT_MSG(
        src_stride.dim_0 == src.format()->compute_size(unit_shape), "Softmax: Source stride helper mismatch.");
    const kai_softmax_uker_src_dim_args src_index = {0};
    const size_t src_offset = m_api.get_src_offset(&m_config, &src_index, &src_stride);
    KAI_TEST_ASSERT_MSG(src_offset == 0, "Softmax: Source offset helper mismatch.");

    const kai_softmax_uker_dst_dim_args dst_shape = {length};
    const kai_softmax_uker_dst_stride_args dst_stride = m_api.get_dst_stride(&m_config, &dst_shape);
    KAI_TEST_ASSERT_MSG(
        dst_stride.dim_0 == dst.format()->compute_size(unit_shape), "Softmax: Destination stride helper mismatch.");
    const kai_softmax_uker_dst_dim_args dst_index = {0};
    const size_t dst_offset = m_api.get_dst_offset(&m_config, &dst_index, &dst_stride);
    KAI_TEST_ASSERT_MSG(dst_offset == 0, "Softmax: Destination offset helper mismatch.");
    KAI_TEST_ASSERT_MSG(
        m_api.get_dst_size(&m_config, &dst_shape, &dst_stride) == dst.data().size(),
        "Softmax: Destination size helper mismatch.");

    kai_softmax_uker_args args{};
    args.flags = 0;
    args.shape.dim_0 = length;
    args.operand.src.ptr = src.data_ptr() + src_offset;
    args.operand.src.stride = src_stride;
    args.operand.dst.ptr = dst.data_ptr() + dst_offset;
    args.operand.dst.stride = dst_stride;

    abi_check([&] { m_api.run(&m_config, &args); });
}

}  // namespace kai::test

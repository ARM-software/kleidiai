//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/softmax/softmax_tb.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include "test/common/assert.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/common/enum_utils.hpp"
#include "test/common/memory.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/common/shape.hpp"
#include "test/nextgen/format/fill.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/softmax/softmax_operator.hpp"
#include "test/nextgen/operators/softmax/softmax_slots.hpp"
#include "test/nextgen/reference/softmax.hpp"

namespace kai::test {

SoftmaxTb::SoftmaxTb(size_t length, const SoftmaxOperator* op) : m_length(length), m_op(op) {
    KAI_TEST_ASSERT_MSG(m_length > 0, "Softmax: The input length must be non-zero.");
    KAI_TEST_ASSERT_MSG(m_op != nullptr, "Softmax: Operator must not be null.");
}

void SoftmaxTb::generate_test_data(Rng& rng, Range<double> input_range) {
    Tensor& src_data = get_tensor(SoftmaxSlot::SRC_DATA);
    Tensor& ref_dst_data = get_tensor(SoftmaxSlot::DST_DATA);

    const std::array shape{m_length};
    const Poly<Format> src_format(std::in_place_type<PlainFormat>, m_op->src_dtype);
    const uint32_t seed = rng();
    Rng data_rng(seed);
    const std::string src_id = "fill_random(" + src_format->uid() + "," + std::to_string(seed) + ",{" +
        std::to_string(m_length) + "},{" + std::to_string(input_range.min) + "," + std::to_string(input_range.max) +
        "})";

    src_data.set_shape(shape)
        .set_format(src_format)
        .set_data(
            src_format->generate(
                shape,
                [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
                    fill_random(gen_shape, dtype, output, data_rng, input_range);
                }),
            src_id);

    const SoftmaxFn reference_fn = make_softmax(m_op->src_dtype, m_op->dst_dtype);
    ref_dst_data.set_shape(shape)
        .set_format(make_poly<PlainFormat>(m_op->dst_dtype))
        .set_data(reference_fn(shape, src_data.data()));
}

void SoftmaxTb::test_output() {
    const std::array full_shape{m_length};

    const Tensor& src_data = get_tensor(SoftmaxSlot::SRC_DATA);
    const Tensor& ref_dst_data = get_tensor(SoftmaxSlot::DST_DATA);
    Tensor& imp_dst_data = get_tensor(SoftmaxSlot::DST_DATA_IMP);

    imp_dst_data.set_shape(full_shape).set_format(ref_dst_data.format()).allocate();
    m_op->softmax.run(src_data, imp_dst_data);

    const Format& format = *ref_dst_data.format();
    const std::array compare_shape{size_t{1}, m_length};
    const std::array compare_coords{size_t{0}, size_t{0}};
    constexpr float abs_error_threshold = 1.0e-6F;
    DefaultMismatchHandler handler(abs_error_threshold, 1.0e-3F, 0, 0.0F);
    const bool ok =
        format.compare(compare_shape, compare_coords, compare_shape, imp_dst_data.data(), ref_dst_data.data(), handler);
    KAI_TEST_ASSERT(ok);
}

Tensor& SoftmaxTb::get_tensor(SoftmaxSlot slot) {
    return m_tensors.at(as_idx(slot));
}

}  // namespace kai::test

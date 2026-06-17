//
// SPDX-FileCopyrightText: Copyright 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "kai/ops/conv/depthwise_common.hpp"

#include "common_internal/utils.hpp"

using kai::ops::iceildiv;

namespace kai {
namespace ops {
namespace depthwise {

std::tuple<size_t, size_t, size_t, size_t, size_t>
get_reduced_view_for_dilation(size_t out_size, size_t in_size, const size_t d,
                              const size_t dilation_factor,
                              const size_t kernel_size, const size_t stride,
                              const size_t orig_pad_before) {
    // Get the valid output range
    out_size = iceildiv(out_size - d, dilation_factor);

    // Compute the start offset and the amount of padding which applies to this
    // portion of the work.
    size_t start_pos = d * stride, pad_before = 0;
    if (start_pos < orig_pad_before) {
        pad_before = iceildiv(orig_pad_before - start_pos, dilation_factor);
    }
    start_pos += pad_before * dilation_factor - orig_pad_before;

    // Hence compute the valid input range
    in_size = start_pos < in_size
                  ? iceildiv(in_size - start_pos, dilation_factor)
                  : 0;

    // Finally, compute the "after" padding
    const size_t reqd_input = (out_size - 1) * stride + kernel_size;
    size_t pad_after = 0;
    if (reqd_input > (pad_before + in_size)) {
        pad_after = reqd_input - (pad_before + in_size);
    }

    return std::make_tuple(out_size, in_size, start_pos, pad_before, pad_after);
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

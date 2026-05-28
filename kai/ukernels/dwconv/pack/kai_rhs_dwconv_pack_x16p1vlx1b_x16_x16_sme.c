//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "kai_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme.h"

#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

enum {
    ELEM_SIZE_BYTES = sizeof(uint16_t),
};

size_t kai_rhs_get_dst_size_dwconv_pack_x16p1vlx1b_x16_x16_sme(
    size_t filter_height, size_t filter_width, size_t num_channels) {
    const size_t depth_elements = kai_roundup(num_channels, kai_get_sme_vector_length_u16());
    return depth_elements * (filter_height * filter_width + 1) * sizeof(uint16_t);
}

void kai_run_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme(
    size_t filter_height, size_t filter_width, size_t height, size_t width, size_t num_channels, const void* rhs,
    const void* bias, void* rhs_packed) {
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(height == filter_height);
    KAI_ASSUME(width == filter_width);

    // Cast the pointers to byte sizes
    const uint8_t* src = rhs;
    const uint8_t* bias_ptr = bias;
    uint8_t* dst = rhs_packed;

    const size_t vl = kai_get_sme_vector_length_u16();

    for (size_t n = 0; n < num_channels; n += vl) {
        const size_t count = KAI_MIN(vl, num_channels - n);
        memcpy(dst, bias_ptr, count * ELEM_SIZE_BYTES);
        dst += (vl * ELEM_SIZE_BYTES);
        bias_ptr += (count * ELEM_SIZE_BYTES);

        for (size_t idx = 0; idx < filter_height * filter_width; idx++) {
            const uint8_t* src_ptr = src + ((idx * num_channels + n) * ELEM_SIZE_BYTES);
            memcpy(dst, src_ptr, count * ELEM_SIZE_BYTES);
            dst += (vl * ELEM_SIZE_BYTES);  // move ptr.
        }
    }
}

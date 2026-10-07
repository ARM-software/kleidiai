//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

enum {
    NR = 2,
    KR = 2,
    NUM_BYTES_DATA = sizeof(uint16_t),
    NUM_BYTES_BIAS = sizeof(uint16_t),
    MAX_N_STEP = NR * ((KAI_SME_VEC_LENGTH_MAX_BYTES / NUM_BYTES_DATA) / KR),
};

void kai_kernel_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(
    size_t active_n, size_t k_chunk_length, const void* rhs, void* rhs_packed, const void* bias, size_t k_chunk_count,
    size_t rhs_stride_row);

/// Gets the number of rows in one packed N tile.
static size_t get_n_step(void) {
    const size_t n_step = NR * kai_get_sme_vector_length_u16() / KR;
    KAI_ASSERT(n_step >= KAI_VSCALE_UNIT_BYTES / sizeof(uint16_t));
    KAI_ASSERT(n_step <= MAX_N_STEP);
    KAI_ASSERT((n_step & (n_step - 1)) == 0);
    return n_step;
}

size_t kai_get_n_step_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(void) {
    return get_n_step();
}

size_t kai_get_rhs_offset_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(size_t n_idx, size_t rhs_stride_row) {
    KAI_ASSUME(n_idx % get_n_step() == 0);
    KAI_ASSUME(rhs_stride_row > 0);
    KAI_ASSUME(n_idx <= SIZE_MAX / rhs_stride_row);

    return n_idx * rhs_stride_row;
}

size_t kai_get_bias_offset_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(size_t n_idx) {
    KAI_ASSUME(n_idx <= SIZE_MAX / NUM_BYTES_BIAS);

    return n_idx * NUM_BYTES_BIAS;
}

size_t kai_get_rhs_packed_stride_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(
    size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(k_chunk_count > 0);
    KAI_ASSUME(k_chunk_length > 0);

    const size_t n_step = get_n_step();
    KAI_ASSUME(k_chunk_length <= SIZE_MAX - (KR - 1));
    const size_t padded_k_chunk_length = kai_roundup(k_chunk_length, KR);
    KAI_ASSUME(padded_k_chunk_length <= (SIZE_MAX / n_step - NUM_BYTES_BIAS) / NUM_BYTES_DATA / k_chunk_count);

    return n_step * (NUM_BYTES_BIAS + k_chunk_count * padded_k_chunk_length * NUM_BYTES_DATA);
}

size_t kai_get_rhs_packed_offset_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length) {
    const size_t n_step = get_n_step();
    KAI_ASSUME(n_idx % n_step == 0);
    const size_t n_tile_idx = n_idx / n_step;
    const size_t packed_tile_bytes =
        kai_get_rhs_packed_stride_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(k_chunk_count, k_chunk_length);
    KAI_ASSUME(n_tile_idx <= SIZE_MAX / packed_tile_bytes);
    return n_tile_idx * packed_tile_bytes;
}

size_t kai_get_rhs_packed_size_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(
    size_t n, size_t k_chunk_count, size_t k_chunk_length) {
    const size_t n_step = get_n_step();
    KAI_ASSUME(n <= SIZE_MAX - (n_step - 1));
    const size_t num_n_tiles = kai_roundup(n, n_step) / n_step;
    const size_t packed_tile_bytes =
        kai_get_rhs_packed_stride_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(k_chunk_count, k_chunk_length);
    KAI_ASSUME(num_n_tiles <= SIZE_MAX / packed_tile_bytes);
    return num_n_tiles * packed_tile_bytes;
}

void kai_run_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(
    size_t n, size_t k_chunk_count, size_t k_chunk_length, size_t rhs_stride_row, const void* rhs, const void* bias,
    void* rhs_packed) {
    KAI_ASSUME(n > 0);
    KAI_ASSUME(k_chunk_count > 0);
    KAI_ASSUME(k_chunk_length > 0);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(rhs_packed != NULL);

    // Requires each source row to hold all K chunks.
    KAI_ASSUME(k_chunk_length <= rhs_stride_row / NUM_BYTES_DATA / k_chunk_count);
    KAI_ASSUME((n - 1) <= (SIZE_MAX - k_chunk_count * k_chunk_length * NUM_BYTES_DATA) / rhs_stride_row);

    const size_t n_step = get_n_step();
    KAI_ASSUME(n <= SIZE_MAX - (n_step - 1));
    const size_t packed_tile_bytes =
        kai_get_rhs_packed_stride_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(k_chunk_count, k_chunk_length);
    KAI_ASSUME(kai_roundup(n, n_step) / n_step <= SIZE_MAX / packed_tile_bytes);

    uint8_t* rhs_packed_ptr = rhs_packed;
    const uint8_t* rhs_ptr = rhs;
    const uint8_t* bias_ptr = bias;

    kai_commit_za();

    // Packs each N tile as one bias block followed by the K chunks.
    for (size_t n_start = 0; n_start < n; n_start += n_step) {
        const size_t active_n = KAI_MIN(n - n_start, n_step);
        // Calculates the destination of this N tile.
        uint8_t* tile_base = rhs_packed_ptr + (n_start / n_step) * packed_tile_bytes;

        // Passes the first source row; assembly derives row addresses from the byte stride.
        const void* tile_rhs = rhs_ptr + n_start * rhs_stride_row;

        // Assembly packs the bias at the destination and advances to the weights, once per N tile.
        const void* tile_bias = bias_ptr + n_start * NUM_BYTES_BIAS;
        kai_kernel_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(
            active_n, k_chunk_length, tile_rhs, tile_base, tile_bias, k_chunk_count, rhs_stride_row);
    }
}

#endif  // Architectural features check.

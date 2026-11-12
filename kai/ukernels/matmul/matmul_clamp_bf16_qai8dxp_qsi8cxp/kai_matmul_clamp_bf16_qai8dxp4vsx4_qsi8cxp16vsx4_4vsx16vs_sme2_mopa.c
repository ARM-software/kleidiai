//
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !(defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME2))) && !defined(_M_ARM64)
#error "This file must be compiled for AArch64, FEAT_SVE2 or FEAT_SME2"
#else  // Architectural features check.

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

enum {
    LHS_QVALUE_BYTES = 1,
    LHS_SCALE_BYTES = 4,
    LHS_ZERO_POINT_BYTES = 4,

    RHS_QVALUE_BYTES = 1,
    RHS_ROW_SUM_BYTES = 4,
    RHS_SCALE_BYTES = 4,
    RHS_BIAS_BYTES = 4,

    DST_BYTES = 2,
    K_MULTIPLE = 32,

    MR_VSCALE = 4,
    NR_VSCALE = 16,
    M_STEP_VSCALE = 4,
    N_STEP_VSCALE = 16,

    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

// Mirrors the fixed offsets read by the assembly kernel. Do not reorder.
struct kai_matmul_uker_args_internal {
    void* dst;               // 0x00: BF16 destination buffer
    const void* lhs_packed;  // 0x08: packed QAI8DXP LHS buffer
    const void* rhs_packed;  // 0x10: packed QSI8CXP RHS buffer
    size_t dst_stride_row;   // 0x18: destination row stride in bytes
    size_t m;                // 0x20: number of output rows
    size_t n;                // 0x28: number of output columns
    size_t lhs_stride;       // 0x30: bytes between packed LHS MR blocks
    size_t rhs_stride;       // 0x38: bytes between packed RHS NR blocks
    size_t rhs_row_bytes;    // 0x40: RHS quantized-value bytes per NR block
    size_t m_blk;            // 0x48: LHS quantized-value bytes per MR block
    size_t dst_inc;          // 0x50: destination increment per M tile
    float clamp_min;         // 0x58: FP32 lower clamp value
    float clamp_max;         // 0x5c: FP32 upper clamp value
};

void kai_kernel_matmul_clamp_bf16_qai8dxp4vsx4_qsi8cxp16vsx4_4vsx16vs_sme2_mopa(
    const struct kai_matmul_uker_args_internal* args_ptr);

static size_t get_k_padded(size_t k) {
    return kai_roundup(k, K_MULTIPLE);
}

static size_t get_mr(void) {
    return MR_VSCALE * kai_get_sme_vscale();
}

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static size_t get_m_step(void) {
    return M_STEP_VSCALE * kai_get_sme_vscale();
}

static size_t get_n_step(void) {
    return N_STEP_VSCALE * kai_get_sme_vscale();
}

static size_t get_lhs_packed_stride_bytes(size_t k) {
    return get_mr() * (get_k_padded(k) * LHS_QVALUE_BYTES + LHS_SCALE_BYTES + LHS_ZERO_POINT_BYTES);
}

static size_t get_rhs_packed_stride_bytes(size_t k) {
    return get_nr() * (get_k_padded(k) * RHS_QVALUE_BYTES + RHS_ROW_SUM_BYTES + RHS_SCALE_BYTES + RHS_BIAS_BYTES);
}

static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dim_args step = {
        .m = get_m_step(),
        .n = get_n_step(),
        .k = 0,
    };
    return step;
}

static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);

    const struct kai_matmul_uker_lhs_stride_args stride = {
        .m = get_lhs_packed_stride_bytes(shape->k),
    };
    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL && stride != NULL);
    KAI_ASSUME(index->m % get_mr() == 0);
    KAI_ASSUME(index->k == 0);

    return index->m / get_mr() * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);

    const struct kai_matmul_uker_rhs_stride_args stride = {
        .n = get_rhs_packed_stride_bytes(shape->k),
    };
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL && stride != NULL);
    KAI_ASSUME(index->n % get_nr() == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / get_nr() * stride->n;
}

static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);

    const struct kai_matmul_uker_dst_stride_args stride = {
        .m = shape->n * DST_BYTES,
    };
    return stride;
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL && stride != NULL);
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->n % get_n_step() == 0);

    return index->m * stride->m + index->n * DST_BYTES;
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL && stride != NULL);

    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* run_args) {
    KAI_UNUSED(config);
    KAI_ASSUME(run_args != NULL);
    KAI_ASSUME_MSG((run_args->flags & ~((uint64_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");
    KAI_ASSUME(run_args->shape.k > 0);

    if (run_args->shape.m == 0 || run_args->shape.n == 0) {
        return;
    }

    KAI_ASSUME(run_args->operand.lhs.ptr != NULL);
    KAI_ASSUME(run_args->operand.rhs.ptr != NULL);
    KAI_ASSUME(run_args->operand.dst.ptr != NULL);

    KAI_ASSUME(run_args->operand.lhs.stride.m == get_lhs_packed_stride_bytes(run_args->shape.k));
    KAI_ASSUME(run_args->operand.rhs.stride.n == get_rhs_packed_stride_bytes(run_args->shape.k));
    KAI_ASSUME(run_args->operand.dst.stride.m >= run_args->shape.n * DST_BYTES);

    const bool enable_clamp = (run_args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!enable_clamp || run_args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!enable_clamp || run_args->activation.clamp.max_ptr != NULL);

    const size_t k_padded = get_k_padded(run_args->shape.k);
    const size_t mr = get_mr();
    const size_t nr = get_nr();

    const struct kai_matmul_uker_args_internal internal_args = {
        .dst = run_args->operand.dst.ptr,
        .lhs_packed = run_args->operand.lhs.ptr,
        .rhs_packed = run_args->operand.rhs.ptr,
        .dst_stride_row = run_args->operand.dst.stride.m,
        .m = run_args->shape.m,
        .n = run_args->shape.n,
        .lhs_stride = run_args->operand.lhs.stride.m,
        .rhs_stride = run_args->operand.rhs.stride.n,
        .rhs_row_bytes = nr * k_padded * RHS_QVALUE_BYTES,
        .m_blk = mr * k_padded * LHS_QVALUE_BYTES,
        .dst_inc = mr * run_args->operand.dst.stride.m,
        .clamp_min = enable_clamp ? *(const float*)run_args->activation.clamp.min_ptr : -FLT_MAX,
        .clamp_max = enable_clamp ? *(const float*)run_args->activation.clamp.max_ptr : FLT_MAX,
    };

    kai_commit_za();

    kai_kernel_matmul_clamp_bf16_qai8dxp4vsx4_qsi8cxp16vsx4_4vsx16vs_sme2_mopa(&internal_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_bf16_qai8dxp4vsx4_qsi8cxp16vsx4_4vsx16vs_sme2_mopa(void) {
    const struct kai_matmul_uker_api api = {
        .run = run,
        .get_step = get_step,

        .get_lhs_stride = get_lhs_stride,
        .get_lhs_offset = get_lhs_offset,

        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,

        .get_dst_stride = get_dst_stride,
        .get_dst_offset = get_dst_offset,
        .get_dst_size = get_dst_size,
    };
    return api;
}

#endif  // Architectural features check.

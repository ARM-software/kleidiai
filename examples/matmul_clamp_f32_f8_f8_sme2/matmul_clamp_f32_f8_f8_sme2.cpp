//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <arm_neon.h>
#include <arm_sme.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f8p_f8p/kai_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f8dxp_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f8dxp_f32_f32_neon.h"

// ----------------------- Common helpers -----------------------

static float clamp_to_range(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

// ----------------------- F8 parameters -----------------------

#define F8_E4M3_MAX_MAG 448.0f    // max finite magnitude
#define F8_E5M2_MAX_MAG 57344.0f  // max finite magnitude

// ----------------------- FPMR helpers -----------------------

#define FPMR_F8_MODE_MASK UINT64_C(0xFFFF)

static void set_fpmr_f8_mode(uint64_t mode_bits) {
    const uint64_t current = kai_read_fpmr_raw();
    const uint64_t next = (current & ~FPMR_F8_MODE_MASK) | (mode_bits & FPMR_F8_MODE_MASK);
    kai_write_fpmr_raw(next);
}

static void check_fpmr(uint64_t fpmr_original) {
    const uint64_t fpmr_current = kai_read_fpmr_raw();
    if (fpmr_current != fpmr_original) {
        printf(
            "Error: FPMR mismatch! Current FPMR: 0x%" PRIx64 ", expected 0x%" PRIx64 "\n", fpmr_current, fpmr_original);
    }
}

// ----------------------- Quantization helpers -----------------------
//
// LHS:  MxK, row-major, per-row scales (M scales)
// RHS:  KxN, row-major, per-column scales (N scales)
//
// Values are scaled so that the largest magnitude in each row/column
// maps (approximately) to the max representable F8 magnitude.
//
// For an input value x:
//   x_scaled = x / scale
//   x_f8     = f8(x_scaled)
//   x_hat    = F32(x_f8) * scale

static bool is_f8_e4m3(kai_f8_mode f8_mode) {
    return f8_mode == kai_f8_mode::KAI_F8_MODE_E4M3_SAT || f8_mode == kai_f8_mode::KAI_F8_MODE_E4M3_INF;
}

static bool is_f8_e5m2(kai_f8_mode f8_mode) {
    return f8_mode == kai_f8_mode::KAI_F8_MODE_E5M2_SAT || f8_mode == kai_f8_mode::KAI_F8_MODE_E5M2_INF;
}

/// Quantize an F32 LHS matrix into f8 with per-row scales.
///
/// @param[in]  lhs Input matrix with shape MxK in row-major layout.
/// @param[in]  m Number of rows (M).
/// @param[in]  k Number of columns (K).
/// @param[out] lhs_f8 Output matrix with shape MxK in row-major layout.
/// @param[out] lhs_scales Output vector of length M (one scale per row).
/// @param[in]  f8_mode F8 format/overflow mode for conversion.
static void kai_lhs_quant_f8_x32(
    const float* lhs, int m, int k, uint8_t* lhs_f8, float* lhs_scales, kai_f8_mode f8_mode) {
    uint64_t fpmr = kai_f8_mode_to_reg(f8_mode);

    float f8_max_abs = 0.0f;
    if (is_f8_e4m3(f8_mode)) {
        f8_max_abs = F8_E4M3_MAX_MAG;
    } else if (is_f8_e5m2(f8_mode)) {
        f8_max_abs = F8_E5M2_MAX_MAG;
    } else {
        abort();
    }

    const uint64_t fpmr_original = kai_read_fpmr_raw();

    for (int i = 0; i < m; ++i) {
        float max_abs = 0.0f;

        // Find max absolute value in row i
        for (int kk = 0; kk < k; ++kk) {
            float v = lhs[i * k + kk];
            float a = fabsf(v);
            if (a > max_abs) {
                max_abs = a;
            }
        }

        if (max_abs == 0.0f) {
            // All zeros: choose scale=1 and quantize to zero
            lhs_scales[i] = 1.0f;
            for (int kk = 0; kk < k; ++kk) {
                lhs_f8[i * k + kk] = (uint8_t)0;
            }
        } else {
            float scale = max_abs / f8_max_abs;
            lhs_scales[i] = scale;
            float inv_scale = 1.0f / scale;

            kai_convert_f8_f32_neon(&lhs[i * k], &lhs_f8[i * k], fpmr, k, inv_scale);
        }
    }
    kai_write_fpmr_raw(fpmr_original);
}

/// Quantize an F32 RHS matrix into f8 with per-column scales.
///
/// @param[in]  rhs Input matrix with shape KxN in row-major layout.
/// @param[in]  k Number of rows (K).
/// @param[in]  n Number of columns (N).
/// @param[out] rhs_f8_kxn Output matrix with shape KxN in row-major layout.
/// @param[out] rhs_scales Output vector of length N (one scale per column).
/// @param[in]  f8_mode F8 format/overflow mode for conversion.
static void kai_rhs_quant_f8_x32(
    const float* rhs, int k, int n, uint8_t* rhs_f8_kxn, float* rhs_scales, kai_f8_mode f8_mode) {
    uint64_t fpmr = kai_f8_mode_to_reg(f8_mode);

    float f8_max_abs = 0.0f;
    if (is_f8_e4m3(f8_mode)) {
        f8_max_abs = F8_E4M3_MAX_MAG;
    } else if (is_f8_e5m2(f8_mode)) {
        f8_max_abs = F8_E5M2_MAX_MAG;
    } else {
        abort();
    }

    uint64_t fpmr_original = kai_read_fpmr_raw();

    for (int j = 0; j < n; ++j) {
        float max_abs = 0.0f;

        // Find max absolute value in column j
        for (int kk = 0; kk < k; ++kk) {
            float v = rhs[kk * n + j];
            float a = fabsf(v);
            if (a > max_abs) {
                max_abs = a;
            }
        }

        if (max_abs == 0.0f) {
            rhs_scales[j] = 1.0f;
            for (int kk = 0; kk < k; ++kk) {
                rhs_f8_kxn[kk * n + j] = (uint8_t)0;
            }
        } else {
            float scale = max_abs / f8_max_abs;
            rhs_scales[j] = scale;
            float inv_scale = 1.0f / scale;

            for (int kk = 0; kk < k; ++kk) {
                kai_convert_f8_f32_neon(&rhs[kk * n + j], &rhs_f8_kxn[kk * n + j], fpmr, 1, inv_scale);
            }
        }
    }
    kai_write_fpmr_raw(fpmr_original);
}

// ----------------------- Reference GEMM -----------------------
//
// Computes: C[MxN] = A[MxK] * B[KxN] + bias[N]
//
// Inputs:
//   - lhs_f8:  MxK f8 (row-major)
//   - lhs_scales: M row scales (LHS), one per row of A
//   - rhs_f8_kxn:  KxN f8 (row-major)
//   - rhs_scales: N column scales (RHS), one per column of B
//   - bias: optional F32 bias of length N (per output column). May be NULL.
//   - min_value / max_value: clamp range for output
// Output:
//   - C: MxN F32 (row-major), clamped to [min_value, max_value]
//
// This is a deliberately simple reference implementation used for comparison,
// not a performance reference.

static void gemm_f8_reference(
    int m, int n, int k, const uint8_t* lhs_f8, const float* lhs_scales, const uint8_t* rhs_f8_kxn,
    const float* rhs_scales, const float* biases, float min_value, float max_value, float* dst, kai_f8_mode f8_mode) {
    uint64_t fpmr_original = kai_read_fpmr_raw();
    uint64_t fpmr = kai_f8_mode_to_reg(f8_mode);
    kai_write_fpmr_raw(fpmr);

    for (int i = 0; i < m; ++i) {
        float scale_a = lhs_scales[i];

        for (int j = 0; j < n; ++j) {
            float scale_b = rhs_scales[j];
            float combined_scale = scale_a * scale_b;
            float acc = 0.0f;
            float a;
            float b;

            for (int kk = 0; kk < k; ++kk) {
                uint8_t a8 = lhs_f8[i * k + kk];
                uint8_t b8 = rhs_f8_kxn[kk * n + j];

                kai_convert_f32_f8_neon(&a8, &a, 1);
                kai_convert_f32_f8_neon(&b8, &b, 1);

                acc += a * b;
            }

            float value = combined_scale * acc;
            if (biases != NULL) {
                value += biases[j];  // bias is per-output-column
            }

            dst[i * n + j] = clamp_to_range(value, min_value, max_value);
        }
    }
    kai_write_fpmr_raw(fpmr_original);
}

// ----------------------- Simple F32 GEMM (for comparison) -----------------------

static void gemm_f32_reference(
    int m, int n, int k, const float* lhs, const float* rhs, const float* biases, float min_value, float max_value,
    float* dst) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += lhs[i * k + kk] * rhs[kk * n + j];
            }
            if (biases != NULL) {
                acc += biases[j];
            }
            dst[i * n + j] = clamp_to_range(acc, min_value, max_value);
        }
    }
}

// ----------------------- Debug/printing -----------------------

static void print_matrix_f32(const char* name, const float* M, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        printf("  ");
        for (int j = 0; j < cols; ++j) {
            printf("%10.5f ", M[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

static void print_matrix_f8(const char* name, const uint8_t* M, int rows, int cols) {
    printf("%s (f8 E4M3, %dx%d, hex):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        printf("  ");
        for (int j = 0; j < cols; ++j) {
            printf("0x%02X ", (unsigned)M[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

static void fill_f32(const int rows, const int cols, float* M, float start_val, float step) {
    for (int i = 0; i < rows; ++i) {
        float v = start_val;
        for (int j = 0; j < cols; ++j) {
            M[i * cols + j] = v;
            v += step;
        }
    }
}

void fill_bias(float* bias, size_t k) {
    // By doing this it is easier to read the hex value.
    // 8 bit will read as 2.
    uint32_t bits = 0x02020202;
    for (size_t i = 0; i < k; ++i) {
        memcpy(&bias[i], &bits, sizeof(float));
    }
}

void print_hex_bytes(
    const std::string& name, const void* data, size_t size_bytes, size_t mr = 0, size_t kr = 0, size_t k_value = 0,
    size_t max_m_blocks = 1, size_t max_k_blocks = 0, size_t max_rows_per_block = 0) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    std::cout << name << " (" << size_bytes << " bytes)";

    for (size_t i = 0; i < size_bytes; i += 16) {
        std::cout << "  " << std::setw(4) << std::setfill('0') << i << ": ";
        const size_t line_end = std::min(size_bytes, i + 16);
        for (size_t j = i; j < line_end; ++j) {
            std::cout << std::dec << std::setw(3) << std::setfill(' ') << static_cast<unsigned>(bytes[j]) << " ";
        }
        std::cout << "\n";
    }
}

void transpose_matrix(const float* input, float* output, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// ----------------------- Demo / main -----------------------
int main(void) {
    // ------------------------ CONSTANTS ------------------------
    const size_t mr = kai_get_mr_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa();
    const size_t nr = kai_get_nr_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa();
    const size_t kr = kai_get_kr_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa();
    const size_t sr = kai_get_sr_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa();

    // Small example shapes
    const int m = 128;
    const int k = 32;
    const int n = 128;

    static const kai_f8_mode f8_modes[4] = {
        KAI_F8_MODE_E4M3_INF, KAI_F8_MODE_E4M3_SAT, KAI_F8_MODE_E5M2_INF, KAI_F8_MODE_E5M2_SAT};

    // Set a different mode here to check that KleidiAI does not trash FPMR
    const kai_f8_mode f8_mode_check = KAI_F8_MODE_E5M2_SAT;
    set_fpmr_f8_mode(kai_f8_mode_to_reg(f8_mode_check));
    const uint64_t fpmr_base = kai_read_fpmr_raw();

    // Example F32 LHS (MxK) and RHS (KxN), row-major
    float lhs[m * k];
    float rhs[k * n];
    float rhs_nxk[n * k];
    float biases[n];
    std::vector<float> dst_target(m * n);

    fill_f32(m, k, lhs, 1.0f, 1.0f);
    fill_f32(k, n, rhs, 1.0f, 1.0f);
    fill_f32(m, n, dst_target.data(), 1.0f, 1.0f);
    fill_bias(biases, n);
    transpose_matrix(rhs, rhs_nxk, k, n);

    const float min_value = -FLT_MAX;
    const float max_value = FLT_MAX;

    // Quantized buffers and scales
    uint8_t lhs_f8[m * k];
    uint8_t rhs_f8_kxn[k * n];
    float lhs_scales[m];
    float rhs_scales[n];

    float dst_f8[m * n];
    float dst_f32[m * n];

    // Quantize inputs
    // Save original FPMR to restore after. This will be in the LHS pack quant and pack function.
    // For Adv SIMD, the store/restore can be done in the function where F8 instructions are used, but
    // that can't be done for streaming functions which sets the FPMR to zero on entry and exit. To have
    // the same behaviour across both implementations, we do the save/restore at a top level function.
    const uint64_t fpmr_original = kai_read_fpmr_raw();

    const float abs_tolerance = 0.0001f;
    size_t tests_passed = 0;

    // Cycle through f8 modes and run reference and kernels with both.
    // The integrator is expected to provide the f8 format
    for (auto& f8_mode : f8_modes) {
        kai_lhs_quant_f8_x32(lhs, m, k, lhs_f8, lhs_scales, f8_mode);
        check_fpmr(fpmr_original);
        kai_rhs_quant_f8_x32(rhs, k, n, rhs_f8_kxn, rhs_scales, f8_mode);
        check_fpmr(fpmr_original);

        // GEMM using F8 + scales -> F32 output
        gemm_f8_reference(
            m, n, k, lhs_f8, lhs_scales, rhs_f8_kxn, rhs_scales, biases, min_value, max_value, dst_f8, f8_mode);

        // Full-precision GEMM for comparison
        gemm_f32_reference(m, n, k, lhs, rhs, biases, min_value, max_value, dst_f32);

        // ------------------------------- LHS PACKING ----------------------------------------
        const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_f8dxp_f32_neon(m, k, mr, kr, sr);
        const size_t lhs_packed_offset = kai_get_lhs_packed_offset_lhs_pack_f8dxp_f32_neon(0, k, mr, kr, sr);
        std::vector<uint8_t> lhs_packed_neon(lhs_packed_size);

        const size_t lhs_row_stride_bytes = k * sizeof(float);

        // Run Advanced SIMD Kernel
        kai_run_lhs_pack_f8dxp_f32_neon(
            m, k, mr, kr, sr, 0, lhs, lhs_row_stride_bytes, lhs_packed_neon.data() + lhs_packed_offset, f8_mode);

        check_fpmr(fpmr_original);

        // --------------------------------- RHS PACKING -------------------------------------
        const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_nxk_f8dxp_f32_f32_neon(n, k, nr, kr, sr);
        const size_t rhs_packed_offset = kai_get_rhs_packed_offset_rhs_pack_nxk_f8dxp_f32_f32_neon(0, k, nr, kr, sr);
        std::vector<uint8_t> rhs_packed_neon(rhs_packed_size);
        const size_t rhs_row_stride_bytes = k * sizeof(float);

        // Advanced SIMD Kernel
        kai_run_rhs_pack_nxk_f8dxp_f32_f32_neon(
            n, k, nr, kr, sr, 0, rhs_nxk, biases, rhs_row_stride_bytes, rhs_packed_neon.data() + rhs_packed_offset,
            f8_mode);

        check_fpmr(fpmr_original);

        // --------------------------------- RUN MatMul --------------------------------------
        const size_t dst_out_size = kai_get_dst_size_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(m, n);
        const size_t dst_out_offset =
            kai_get_dst_offset_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(0, 0, n * sizeof(float));

        kai_run_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(
            m, n, k, lhs_packed_neon.data(), rhs_packed_neon.data(), dst_target.data(), n * sizeof(float),
            sizeof(float), min_value, max_value, f8_mode);

        check_fpmr(fpmr_original);

        // ------------------------------------------------------------------------------------

#ifdef KAI_DEBUG
        // Print hex values, packed.
        print_hex_bytes(
            "LHS packed hex values (f8dxp) after pack (Advanced SIMD)", lhs_packed_neon.data(), lhs_packed_neon.size());

        print_hex_bytes(
            "RHS packed hex values (f8dxp) after pack (Advanced SIMD)", rhs_packed_neon.data(), rhs_packed_neon.size());

        // Print Input Data.
        print_matrix_f32("lhs (F32)", lhs, m, k);
        print_matrix_f8("lhs quantized", lhs_f8, m, k);
        print_matrix_f32("rhs nxk (F32)", rhs_nxk, n, k);
        print_matrix_f32("rhs kxn (F32)", rhs, k, n);
        print_matrix_f8("rhs quantized ref kxn", rhs_f8_kxn, k, n);

        printf("LHS scales (per row):\n  ");
        for (int i = 0; i < m; ++i) {
            printf("%10.6f ", lhs_scales[i]);
        }
        printf("\n\nRHS scales (per column):\n  ");
        for (int j = 0; j < n; ++j) {
            printf("%10.6f ", rhs_scales[j]);
        }
        printf("\n\n");

        print_matrix_f32("dst_target (F8 GEMM dequantized to F32)", dst_target.data(), m, n);
        print_matrix_f32("dst_f8_ref (F8 GEMM dequantized to F32)", dst_f8, m, n);
#endif  // KAI_DEBUG

        // Print per-element absolute error between F8 GEMM Reference and F8 SME2 Matmul kernels.
        size_t mismatches = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float diff = fabsf(dst_f8[i * n + j] - dst_target[i * n + j]);
                if (diff > abs_tolerance) {
                    mismatches++;
                }
            }
        }
        if (mismatches == 0) tests_passed++;

            // Probably not needed, remove later.
#ifdef KAI_DEBUG
        // Print per-element absolute error between F8 GEMM Reference and F32 existing.
        printf("Per-element absolute error for reference vs F32 (|dst_f8_ref - dst_f32|):\n");
        for (int i = 0; i < m; ++i) {
            printf("  ");
            for (int j = 0; j < n; ++j) {
                float diff = fabsf(dst_f8[i * n + j] - dst_f32[i * n + j]);
                printf("%10.5f ", diff);
            }
            printf("\n");
        }
#endif  // KAI_DEBUG

        if (kai_read_fpmr_raw() != fpmr_base) {
            printf(
                "Error: FPMR was not restored correctly! Final FPMR: 0x%" PRIx64 ", expected 0x%" PRIx64 "\n",
                kai_read_fpmr_raw(), fpmr_base);
        }
    }

    std::cout << "\n\nTests Passed : " << tests_passed << std::endl;
    return 0;
}

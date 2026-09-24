//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_pack_lhs_registry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_bf16p_bf16p/kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f16p_qsi4c32p/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme2.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p8x4_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f16pmrx2_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p1x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_bf16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.h"
#include "matmul_pack_lhs_benchmark_logic.hpp"
#include "matmul_pack_lhs_interface.hpp"
#include "test/common/cpu_info.hpp"

namespace kai::benchmark {
namespace {

inline constexpr MatMulPackLhsBaseInterface kai_lhs_pack_bf16p2vlx2_f32_sme_interface{
    .get_mr = kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme_mopa,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme,
    .run = kai_run_lhs_pack_bf16p2vlx2_f32_sme,
};

inline constexpr MatMulPackLhsBaseInterface kai_lhs_pack_bf16p2vlx2_f32_sme2_interface{
    .get_mr = kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme2,
    .run = kai_run_lhs_pack_bf16p2vlx2_f32_sme2,
};

inline constexpr MatMulPackLhsBaseInterface kai_lhs_pack_bf16p8x4_f16_neon_interface{
    .get_mr = kai_get_mr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_kr = kai_get_kr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_sr = kai_get_sr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_bf16p8x4_f16_neon,
    .run = kai_run_lhs_pack_bf16p8x4_f16_neon,
};

inline constexpr MatMulPackLhsBlockwiseInterface kai_lhs_pack_f16pmrx2_f32_neon_interface{
    .get_mr = kai_get_mr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_f16pmrx2_f32_neon,
    .run = kai_run_lhs_pack_f16pmrx2_f32_neon,
    .is_valid = [](size_t, size_t k, size_t) { return k % 16 == 0; },
};

inline constexpr MatMulPackLhsBaseInterface kai_lhs_pack_f32p2vlx1_f32_sme_interface{
    .get_mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme,
    .run = kai_run_lhs_pack_f32p2vlx1_f32_sme,
};

inline constexpr MatMulPackLhsBaseInterface kai_lhs_pack_x16p2vlx2_x16_sme_interface{
    .get_mr = kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme,
    .run = kai_run_lhs_pack_x16p2vlx2_x16_sme,
};

inline constexpr MatMulPackLhsBaseInterface kai_lhs_pack_x8p2vlx4_x8_sme_interface{
    .get_mr = kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa,
    .get_kr = kai_get_kr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa,
    .get_sr = kai_get_sr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme,
    .run = kai_run_lhs_pack_x8p2vlx4_x8_sme,
};

inline constexpr MatMulPackLhsBaseInterface kai_lhs_quant_pack_bf16p1x4_f32_neon_interface{
    .get_mr = kai_get_mr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
    .get_kr = kai_get_kr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
    .get_sr = kai_get_sr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p1x4_f32_neon,
    .run = kai_run_lhs_quant_pack_bf16p1x4_f32_neon,
    .is_valid = [](size_t m, size_t, size_t) { return m == 1; },
};

inline constexpr MatMulPackLhsBaseInterface kai_lhs_quant_pack_bf16p8x4_f32_neon_interface{
    .get_mr = kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_kr = kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_sr = kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon,
    .run = kai_run_lhs_quant_pack_bf16p8x4_f32_neon,
};

inline constexpr MatMulPackLhsBaseInterface kai_lhs_quant_pack_qai8dxp_bf16_neon_interface{
    .get_mr = kai_get_mr_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    .get_kr = kai_get_kr_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    .get_sr = kai_get_sr_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon,
    .run = kai_run_lhs_quant_pack_qai8dxp_bf16_neon,
    .is_valid = [](size_t, size_t k, size_t) { return k % 8 == 0; },
};

inline constexpr MatMulPackLhsBaseInterface kai_lhs_quant_pack_qai8dxp_f16_neon_interface{
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon,
    .run = kai_run_lhs_quant_pack_qai8dxp_f16_neon,
    .is_valid = [](size_t, size_t k, size_t) { return k % 8 == 0; },
};

inline constexpr MatMulPackLhsFloatInterface kai_lhs_quant_pack_qai8dxp_f32_interface{
    .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
    .run = kai_run_lhs_quant_pack_qai8dxp_f32,
};

inline constexpr MatMulPackLhsBlockwiseFloatInterface kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon_interface{
    .get_mr = kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
    .get_kr = kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
    .get_sr = kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
    .run = kai_run_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
};

inline constexpr MatMulPackLhsBlockwiseFloatInterface kai_lhs_quant_pack_qsi8d32p_f32_interface{
    .get_mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
    .get_kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
    .get_sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32,
    .run = kai_run_lhs_quant_pack_qsi8d32p_f32,
};

inline constexpr MatMulPackLhsBlockwiseFloatInterface kai_lhs_quant_pack_qsi8d32p_f32_neon_interface{
    .get_mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
    .get_kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
    .get_sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon,
    .run = kai_run_lhs_quant_pack_qsi8d32p_f32_neon,
};

inline constexpr MatMulPackLhsBlockwiseInterface kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon_interface{
    .get_mr = kai_get_mr_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f16_neon,
    .run = kai_run_lhs_quant_pack_qsi8d32pscalef32_f16_neon,
    .is_valid = [](size_t, size_t, size_t bl) { return bl % 32 == 0; },
};

inline constexpr MatMulPackLhsBlockwiseFloatInterface kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon_interface{
    .get_mr = kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon,
    .run = kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon,
};

inline constexpr MatMulPackLhsUkernelApiInterface kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon_interface{
    .get_config = [] { return kai_matmul_pack_lhs_uker_config{}; },
    .get_api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon,
    .is_valid = [](size_t, size_t k, size_t) { return k % 32 == 0; },
};

inline constexpr MatMulPackLhsUkernelApiInterface kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme_interface{
    .get_config = [] { return kai_matmul_pack_lhs_uker_config{}; },
    .get_api = kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme,
};

inline constexpr MatMulPackLhsUkernelApiInterface kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2_interface{
    .get_config = [] { return kai_matmul_pack_lhs_uker_config{}; },
    .get_api = kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2,
};

inline constexpr MatMulPackLhsUkernelApiInterface kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme_interface{
    .get_config = [] { return kai_matmul_pack_lhs_uker_config{}; },
    .get_api = kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme,
};

inline constexpr MatMulPackLhsUkernelApiInterface kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme_interface{
    .get_config = [] { return kai_matmul_pack_lhs_uker_config{}; },
    .get_api = kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme,
};

/// Returns the lazily registered matmul LHS packing benchmarks.
///
/// @return Registered matmul LHS packing benchmarks.
const auto& get_matmul_pack_lhs_benchmarks() {
    static const std::array matmul_pack_lhs_benchmarks{
        // Legacy API
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_pack_bf16p2vlx2_f32_sme",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBaseInterface>, kai_lhs_pack_bf16p2vlx2_f32_sme_interface,
            DataType::FP32, test::cpu_has_sme),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_pack_bf16p2vlx2_f32_sme2",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBaseInterface>, kai_lhs_pack_bf16p2vlx2_f32_sme2_interface,
            DataType::FP32, test::cpu_has_sme2),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_pack_bf16p8x4_f16_neon",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBaseInterface>, kai_lhs_pack_bf16p8x4_f16_neon_interface,
            DataType::FP16, test::cpu_check<test::cpu_has_bf16, test::cpu_has_fp16>),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_pack_f16pmrx2_f32_neon",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBlockwiseInterface>, kai_lhs_pack_f16pmrx2_f32_neon_interface,
            DataType::FP32, test::cpu_check<test::cpu_has_sme, test::cpu_has_fp16>),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_pack_f32p2vlx1_f32_sme",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBaseInterface>, kai_lhs_pack_f32p2vlx1_f32_sme_interface,
            DataType::FP32, test::cpu_has_sme),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_pack_x16p2vlx2_x16_sme",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBaseInterface>, kai_lhs_pack_x16p2vlx2_x16_sme_interface,
            DataType::FP16, test::cpu_has_sme),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_pack_x8p2vlx4_x8_sme",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBaseInterface>, kai_lhs_pack_x8p2vlx4_x8_sme_interface,
            DataType::I8, test::cpu_has_sme),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_quant_pack_bf16p1x4_f32_neon",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBaseInterface>, kai_lhs_quant_pack_bf16p1x4_f32_neon_interface,
            DataType::FP32, test::cpu_has_bf16),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_quant_pack_bf16p8x4_f32_neon",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBaseInterface>, kai_lhs_quant_pack_bf16p8x4_f32_neon_interface,
            DataType::FP32, test::cpu_has_bf16),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_quant_pack_qai8dxp_bf16_neon",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBaseInterface>, kai_lhs_quant_pack_qai8dxp_bf16_neon_interface,
            DataType::BF16, test::cpu_has_advsimd),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_quant_pack_qai8dxp_f16_neon",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBaseInterface>, kai_lhs_quant_pack_qai8dxp_f16_neon_interface,
            DataType::FP16, test::cpu_has_fp16),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_quant_pack_qai8dxp_f32",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsFloatInterface>, kai_lhs_quant_pack_qai8dxp_f32_interface,
            DataType::FP32, test::cpu_has_advsimd),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBlockwiseFloatInterface>,
            kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon_interface, DataType::FP32, test::cpu_has_advsimd),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_quant_pack_qsi8d32p_f32",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBlockwiseFloatInterface>,
            kai_lhs_quant_pack_qsi8d32p_f32_interface, DataType::FP32, [] { return true; }),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_quant_pack_qsi8d32p_f32_neon",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBlockwiseFloatInterface>,
            kai_lhs_quant_pack_qsi8d32p_f32_neon_interface, DataType::FP32, test::cpu_has_advsimd),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBlockwiseInterface>,
            kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon_interface, DataType::FP16, test::cpu_has_fp16),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsBlockwiseFloatInterface>,
            kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon_interface, DataType::FP32, test::cpu_has_advsimd),

        // Ukernel API
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsUkernelApiInterface>,
            kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon_interface, DataType::FP32, test::cpu_has_advsimd),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsUkernelApiInterface>,
            kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme_interface, DataType::FP16, test::cpu_has_sme),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsUkernelApiInterface>,
            kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2_interface, DataType::FP16, test::cpu_has_sme2),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsUkernelApiInterface>,
            kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme_interface, DataType::FP32, test::cpu_has_sme),
        RegisterBenchmark(
            "kai_matmul_pack_lhs/kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme",
            kai_benchmark_matmul_pack_lhs<MatMulPackLhsUkernelApiInterface>,
            kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme_interface, DataType::I8, test::cpu_has_sme),
    };

    return matmul_pack_lhs_benchmarks;
}

}  // namespace

void RegisterMatMulPackLhsBenchmarks(size_t m, size_t k, size_t bl) {
    for (const auto& benchmark : get_matmul_pack_lhs_benchmarks()) {
        benchmark->Args({static_cast<int64_t>(m), static_cast<int64_t>(k), static_cast<int64_t>(bl)})
            ->ArgNames({"m", "k", "bl"});
    }
}

}  // namespace kai::benchmark

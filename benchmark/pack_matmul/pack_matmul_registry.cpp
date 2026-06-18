//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "pack_matmul_registry.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <test/common/cpu_info.hpp>
#include <test/common/data_type.hpp>

#include "pack_matmul_benchmark_logic.hpp"
#include "pack_matmul_runner.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

// Matrix multiplication micro-kernels with associated LHS packing micro-kernels.
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_bf16p_bf16p/kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p8x4_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p1x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_bf16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"

namespace kai::benchmark {
namespace {
using DataType = test::DataType;

struct PackMatMulRegistryEntry {
    ::benchmark::internal::Benchmark* benchmark;
    const PackMatMulEntry* entry;
};

inline const PackMatMulEntry kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod/"
        "kai_lhs_quant_pack_qai8dxp_bf16_neon",
    .matmul_name = "kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod",
    .lhs_type = DataType::BF16,
    .dst_type = DataType::BF16,
    .matmul_op = PackMatMulOp::GEMV,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_dotprod_and_bf16,
    .get_mr = kai_get_mr_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_bf16_neon, kai_run_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod>,
};

inline const PackMatMulEntry kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm/"
        "kai_lhs_quant_pack_qai8dxp_bf16_neon",
    .matmul_name = "kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm",
    .lhs_type = DataType::BF16,
    .dst_type = DataType::BF16,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_i8mm_and_bf16,
    .get_mr = kai_get_mr_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    .get_kr = kai_get_kr_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    .get_sr = kai_get_sr_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_bf16_neon, kai_run_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla/"
        "kai_lhs_pack_bf16p8x4_f16_neon",
    .matmul_name = "kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_bf16,
    .get_mr = kai_get_mr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_kr = kai_get_kr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_sr = kai_get_sr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_lhs_offset = kai_get_lhs_offset_lhs_pack_bf16p8x4_f16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_pack_bf16p8x4_f16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_bf16p8x4_f16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_pack_bf16p8x4_f16_neon, kai_run_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa/"
        "kai_lhs_pack_x16p2vlx2_x16_sme",
    .matmul_name = "kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_sme2,
    .get_mr = kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa,
    .get_lhs_offset = kai_get_lhs_offset_lhs_pack_x16p2vlx2_x16_sme,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_pack_x16p2vlx2_x16_sme,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_pack_x16p2vlx2_x16_sme, kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa/"
        "kai_lhs_pack_x16p2vlx2_x16_sme",
    .matmul_name = "kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_sme,
    .get_mr = kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa,
    .get_lhs_offset = kai_get_lhs_offset_lhs_pack_x16p2vlx2_x16_sme,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_pack_x16p2vlx2_x16_sme,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_pack_x16p2vlx2_x16_sme, kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod/"
        "kai_lhs_quant_pack_qai8dxp_f16_neon",
    .matmul_name = "kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMV,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_dotprod_and_fp16,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_f16_neon, kai_run_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod/"
        "kai_lhs_quant_pack_qai8dxp_f16_neon",
    .matmul_name = "kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMV,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_dotprod_and_fp16,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_f16_neon, kai_run_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod/"
        "kai_lhs_quant_pack_qai8dxp_f16_neon",
    .matmul_name = "kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_dotprod_and_fp16,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_f16_neon, kai_run_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm/"
        "kai_lhs_quant_pack_qai8dxp_f16_neon",
    .matmul_name = "kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_i8mm_and_fp16,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_f16_neon, kai_run_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa/"
        "kai_lhs_quant_pack_qai8dxp_f16_neon",
    .matmul_name = "kai_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_sme2,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon,
    .get_matmul_lhs_packed_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_f16_neon, kai_run_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot/"
        "kai_lhs_quant_pack_qai8dxp_f16_neon",
    .matmul_name = "kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMV,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_sme2,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_f16_neon, kai_run_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod/"
        "kai_lhs_quant_pack_qai8dxp_f16_neon",
    .matmul_name = "kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMV,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_dotprod_and_fp16,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_f16_neon, kai_run_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod/"
        "kai_lhs_quant_pack_qai8dxp_f16_neon",
    .matmul_name = "kai_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMV,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_dotprod_and_fp16,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_f16_neon, kai_run_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod/"
        "kai_lhs_quant_pack_qai8dxp_f16_neon",
    .matmul_name = "kai_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_dotprod_and_fp16,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_f16_neon, kai_run_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
};

inline const PackMatMulEntry kai_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm/"
        "kai_lhs_quant_pack_qai8dxp_f16_neon",
    .matmul_name = "kai_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm",
    .lhs_type = DataType::FP16,
    .dst_type = DataType::FP16,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_i8mm_and_fp16,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_qai8dxp_f16_neon, kai_run_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
};

inline const PackMatMulEntry kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot/"
        "kai_lhs_quant_pack_bf16p1x4_f32_neon",
    .matmul_name = "kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot",
    .lhs_type = DataType::FP32,
    .dst_type = DataType::FP32,
    .matmul_op = PackMatMulOp::GEMV,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_dotprod,
    .get_mr = kai_get_mr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
    .get_kr = kai_get_kr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
    .get_sr = kai_get_sr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_bf16p1x4_f32_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_bf16p1x4_f32_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p1x4_f32_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_bf16p1x4_f32_neon, kai_run_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot>,
};

inline const PackMatMulEntry kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla/"
        "kai_lhs_quant_pack_bf16p8x4_f32_neon",
    .matmul_name = "kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla",
    .lhs_type = DataType::FP32,
    .dst_type = DataType::FP32,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_i8mm,
    .get_mr = kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_kr = kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_sr = kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_bf16p8x4_f32_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_bf16p8x4_f32_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_quant_pack_bf16p8x4_f32_neon, kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla>,
};

inline const PackMatMulEntry kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa/"
        "kai_lhs_pack_f32p2vlx1_f32_sme",
    .matmul_name = "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa",
    .lhs_type = DataType::FP32,
    .dst_type = DataType::FP32,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_sme2,
    .get_mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    .get_lhs_offset = kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_pack_f32p2vlx1_f32_sme,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_pack_f32p2vlx1_f32_sme, kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
};

inline const PackMatMulEntry kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa/"
        "kai_lhs_pack_f32p2vlx1_f32_sme",
    .matmul_name = "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa",
    .lhs_type = DataType::FP32,
    .dst_type = DataType::FP32,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_sme,
    .get_mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    .get_lhs_offset = kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_pack_f32p2vlx1_f32_sme,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_pack_f32p2vlx1_f32_sme, kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
};

inline const PackMatMulEntry kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa/"
        "kai_lhs_pack_bf16p2vlx2_f32_sme",
    .matmul_name = "kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa",
    .lhs_type = DataType::FP32,
    .dst_type = DataType::FP32,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_sme2,
    .get_mr = kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
    .get_lhs_offset = kai_get_lhs_offset_lhs_pack_bf16p2vlx2_f32_sme,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_pack_bf16p2vlx2_f32_sme,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
    .run_pack_matmul = run_pack_matmul_base<
        kai_run_lhs_pack_bf16p2vlx2_f32_sme, kai_run_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
};

inline const PackMatMulEntry kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa/"
        "kai_lhs_quant_pack_qai8dxp_f32",
    .matmul_name = "kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa",
    .lhs_type = DataType::FP32,
    .dst_type = DataType::FP32,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_sme2,
    .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
    .get_matmul_lhs_packed_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
    .run_pack_matmul = run_pack_matmul_float<
        kai_run_lhs_quant_pack_qai8dxp_f32_as_void,
        kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa>,
};

inline const PackMatMulEntry kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot/"
        "kai_lhs_quant_pack_qai8dxp_f32",
    .matmul_name = "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot",
    .lhs_type = DataType::FP32,
    .dst_type = DataType::FP32,
    .matmul_op = PackMatMulOp::GEMV,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_sme2,
    .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
    .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
    .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
    .run_pack_matmul = run_pack_matmul_float<
        kai_run_lhs_quant_pack_qai8dxp_f32_as_void, kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot>,
};

inline const PackMatMulEntry kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa/"
        "kai_lhs_quant_pack_qai8dxp_f32",
    .matmul_name = "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa",
    .lhs_type = DataType::FP32,
    .dst_type = DataType::FP32,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_sme2,
    .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
    .get_matmul_lhs_packed_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    .run_pack_matmul = run_pack_matmul_float<
        kai_run_lhs_quant_pack_qai8dxp_f32_as_void,
        kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
};

inline const PackMatMulEntry kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot/"
        "kai_lhs_quant_pack_qai8dxp_f32",
    .matmul_name = "kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot",
    .lhs_type = DataType::FP32,
    .dst_type = DataType::FP32,
    .matmul_op = PackMatMulOp::GEMV,
    .needs_block_size = false,
    .is_cpu_supported = test::cpu_has_sme2,
    .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    .run_pack_matmul = run_pack_matmul_float<
        kai_run_lhs_quant_pack_qai8dxp_f32_as_void, kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
};

inline const PackMatMulEntry kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa/"
        "kai_lhs_quant_pack_qai8dxp_f32",
    .matmul_name = "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa",
    .lhs_type = DataType::FP32,
    .dst_type = DataType::FP32,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = true,
    .is_cpu_supported = test::cpu_has_sme2,
    .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
    .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
    .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
    .get_matmul_lhs_packed_offset =
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
    .run_pack_matmul = run_pack_matmul_blockwise_dynamic_quant<
        kai_run_lhs_quant_pack_qai8dxp_f32_as_void,
        kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa>,
};

inline const PackMatMulEntry kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot/"
        "kai_lhs_quant_pack_qai8dxp_f32",
    .matmul_name = "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot",
    .lhs_type = DataType::FP32,
    .dst_type = DataType::FP32,
    .matmul_op = PackMatMulOp::GEMV,
    .needs_block_size = true,
    .is_cpu_supported = test::cpu_has_sme2,
    .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot,
    .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot,
    .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot,
    .run_pack_matmul = run_pack_matmul_blockwise_dynamic_quant<
        kai_run_lhs_quant_pack_qai8dxp_f32_as_void, kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot>,
};

inline const PackMatMulEntry kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod/"
        "kai_lhs_quant_pack_qai8dxp_bf16_neon",
    .matmul_name = "kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod",
    .lhs_type = DataType::BF16,
    .dst_type = DataType::BF16,
    .matmul_op = PackMatMulOp::GEMV,
    .needs_block_size = true,
    .is_cpu_supported = test::cpu_has_dotprod_and_bf16,
    .get_mr = kai_get_mr_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod,
    .run_pack_matmul = run_pack_matmul_blockwise_dynamic_quant_generic_dst<
        kai_run_lhs_quant_pack_qai8dxp_bf16_neon, kai_run_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod>,
};

inline const PackMatMulEntry kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm_lhs_pack_entry{
    .benchmark_name =
        "kai_pack_matmul/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm/"
        "kai_lhs_quant_pack_qai8dxp_bf16_neon",
    .matmul_name = "kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm",
    .lhs_type = DataType::BF16,
    .dst_type = DataType::BF16,
    .matmul_op = PackMatMulOp::GEMM,
    .needs_block_size = true,
    .is_cpu_supported = test::cpu_has_i8mm_and_bf16,
    .get_mr = kai_get_mr_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm,
    .get_kr = kai_get_kr_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm,
    .get_sr = kai_get_sr_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm,
    .get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon,
    .get_matmul_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm,
    .run_pack_matmul = run_pack_matmul_blockwise_dynamic_quant_generic_dst<
        kai_run_lhs_quant_pack_qai8dxp_bf16_neon, kai_run_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm>,
};

PackMatMulRegistryEntry RegisterPackMatMulBenchmarkEntry(const PackMatMulEntry& entry) {
    return {
        ::benchmark::RegisterBenchmark(std::string(entry.benchmark_name), kai_benchmark_pack_matmul, &entry),
        &entry,
    };
}

inline const std::array<PackMatMulRegistryEntry, 28> pack_matmul_entries{
    {
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(
            kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(
            kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(
            kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(
            kai_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod_lhs_pack_entry),
        RegisterPackMatMulBenchmarkEntry(kai_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm_lhs_pack_entry),
    },
};

void ConfigurePackMatMulBenchmark(const PackMatMulRegistryEntry& entry, const MatMulShape& shape, const size_t bl) {
    if (entry.entry->needs_block_size) {
        entry.benchmark
            ->Args(
                {static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n), static_cast<int64_t>(shape.k),
                 static_cast<int64_t>(bl)})
            ->ArgNames({"m", "n", "k", "bl"});
        return;
    }

    entry.benchmark->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n), static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

}  // namespace

void RegisterPackMatMulBenchmarks(const MatMulShape& shape, const size_t bl) {
    for (const auto& entry : pack_matmul_entries) {
        ConfigurePackMatMulBenchmark(entry, shape, bl);
    }
}

}  // namespace kai::benchmark

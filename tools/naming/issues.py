#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


@dataclass(frozen=True)
class KnownIssue:
    expected: str
    description: str


def _known_issues(
    description: str, issues: tuple[tuple[str, str], ...]
) -> dict[str, KnownIssue]:
    return {
        name: KnownIssue(expected=expected, description=description)
        for name, expected in issues
    }


KNOWN_UKERNEL_PROBLEMS: dict[str, KnownIssue] = {}

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Includes the K/block-depth component in the matmul tile field. "
            "The tile field names only the output block shape (`m x n`); "
            "packing depth belongs in the packed buffer descriptors."
        ),
        issues=(
            (
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm",
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm",
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4_neon_i8mm",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm",
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm",
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4_neon_i8mm",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm",
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4_neon_i8mm",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm",
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8_neon_i8mm",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm",
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm",
            ),
            (
                "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm",
                "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4_neon_i8mm",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Includes the K/block-depth component in the matmul tile field. "
            "The tile field names only the output block shape (`m x n`); "
            "packing depth belongs in the packed buffer descriptors. Uses "
            "deprecated `biasf*` spelling; packed bias should be encoded as "
            "`b<type>` after the packed shape."
        ),
        issues=(
            (
                "kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla",
                "kai_matmul_clamp_f16_f16_f16p16x1bf16_6x16_neon_mla",
            ),
            (
                "kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla",
                "kai_matmul_clamp_f32_f32_f32p8x1bf32_6x8_neon_mla",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "The packed descriptor is missing/reordering packed shape or puts "
            "pack order on an unpacked source buffer."
        ),
        issues=(
            (
                "kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0",
                "kai_rhs_pack_kxn_qsi4c32pnrx4_qsu4c32",
            ),
            (
                "kai_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon",
                "kai_rhs_pack_kxn_qsi4c32pnrx4s1s0_qsu4c32_neon",
            ),
            (
                "kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0",
                "kai_rhs_pack_kxn_qsi4cxpnrx4_qsi4cx",
            ),
            (
                "kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon",
                "kai_rhs_pack_kxn_qsi8cxpnrx4_qsi8cx_neon",
            ),
            (
                "kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon",
                "kai_rhs_pack_nxk_qai4c32pnrx4_qau4c32_f32_f32_f32_neon",
            ),
            (
                "kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon",
                "kai_rhs_pack_nxk_qai4c32pnrx4s1s0_qau4c32_f32_f32_f32_neon",
            ),
            (
                "kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s1s0_f32_f32_f32_neon",
                "kai_rhs_pack_nxk_qai4c32pnrx4s1s0_qau4c32_f32_f32_f32_neon",
            ),
            (
                "kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0",
                "kai_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32",
            ),
            (
                "kai_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon",
                "kai_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32_neon",
            ),
            (
                "kai_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon",
                "kai_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32_neon",
            ),
            (
                "kai_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon",
                "kai_rhs_pack_nxk_qsi4c32pnrx4s1s0_qsu4c32_neon",
            ),
            (
                "kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon",
                "kai_rhs_pack_nxk_qsi4c32pnrx4s4s0sf16_qsu4c32_neon",
            ),
            (
                "kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0",
                "kai_rhs_pack_nxk_qsi4cxpnrx4_qsi4cx",
            ),
            (
                "kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon",
                "kai_rhs_pack_nxk_qsi8cxpnrx4_qsi8cx_neon",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses `dotprod` as an instruction suffix; the grammar uses the "
            "`dot` instruction mnemonic. Includes the K/block-depth component "
            "in the matmul tile field. The tile field names only the output "
            "block shape (`m x n`); packing depth belongs in the packed "
            "buffer descriptors."
        ),
        issues=(
            (
                "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod",
                "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dot",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod",
                "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dot",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod",
                "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dot",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod",
                "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dot",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod",
                "kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8_neon_dot",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod",
                "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4_neon_dot",
            ),
            (
                "kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod",
                "kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4_neon_dot",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses `dotprod` as an instruction suffix; the grammar uses the "
            "`dot` instruction mnemonic. Uses untyped, misplaced, or reversed "
            "scale/bias suffixes; the grammar requires `s<type>` before "
            "`b<type>`."
        ),
        issues=(
            (
                "kai_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod",
                "kai_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4sf32bf32_1x4_neon_dot",
            ),
            (
                "kai_matmul_clamp_f32_qai8dxp4x4_qsu2cxp4x4bf32sf32_8x4_neon_dotprod",
                "kai_matmul_clamp_f32_qai8dxp4x4_qsu2cxp4x4sf32bf32_8x4_neon_dot",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses a bare `b` suffix; packed bias should be encoded as "
            "`b<type>` after the packed shape."
        ),
        issues=(
            (
                "kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla",
                "kai_dwconv_clamp_f16_f16_f16p1vlx1bf16_3x3_s1_4x4_sme2_mla",
            ),
            (
                "kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla",
                "kai_dwconv_clamp_f32_f32_f32p1vlx1bf32_3x3_s1_4xc_sme2_mla",
            ),
            (
                "kai_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme",
                "kai_rhs_dwconv_pack_x16p1vlx1bx16_x16_x16_sme",
            ),
            (
                "kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme",
                "kai_rhs_dwconv_pack_x32p1vlx1bx32_x32_x32_sme",
            ),
            (
                "kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa",
                "kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2bf16_2vlx2vl_sme_mopa",
            ),
            (
                "kai_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla",
                "kai_imatmul_clamp_f32_f32_f32p4vlx1bf32_6x4vl_sve_mla",
            ),
            (
                "kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa",
                "kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1bf32_2vlx2vl_sme2_mopa",
            ),
            (
                "kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa",
                "kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1bf32_2vlx2vl_sme_mopa",
            ),
            (
                "kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla",
                "kai_matmul_clamp_f16_bf16p8x4_bf16p12x4bf16_8x12_neon_mmla",
            ),
            (
                "kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot",
                "kai_matmul_clamp_f16_f16_f16p2vlx2bf16_1x16vl_sme2_dot",
            ),
            (
                "kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla",
                "kai_matmul_clamp_f16_f16_f16p2vlx2bf16_1x8vl_sme_mla",
            ),
            (
                "kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla",
                "kai_matmul_clamp_f16_f16_f16p32x1bf16_6x32_neon_mla",
            ),
            (
                "kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55",
                "kai_matmul_clamp_f16_f16_f16p32x1bf16_6x32_neon_mla_cortexa55",
            ),
            (
                "kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa",
                "kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2bf16_2vlx2vl_sme_mopa",
            ),
            (
                "kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot",
                "kai_matmul_clamp_f32_bf16p1x4_bf16p12x4bf32_1x36_neon_dot",
            ),
            (
                "kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla",
                "kai_matmul_clamp_f32_bf16p8x4_bf16p12x4bf32_8x12_neon_mmla",
            ),
            (
                "kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla",
                "kai_matmul_clamp_f32_f32_f32p16vlx1bf32_1x16vl_sme2_mla",
            ),
            (
                "kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla",
                "kai_matmul_clamp_f32_f32_f32p16x1bf32_6x16_neon_mla",
            ),
            (
                "kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55",
                "kai_matmul_clamp_f32_f32_f32p16x1bf32_6x16_neon_mla_cortexa55",
            ),
            (
                "kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla",
                "kai_matmul_clamp_f32_f32_f32p2vlx1bf32_1x16vl_sme2_mla",
            ),
            (
                "kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla",
                "kai_matmul_clamp_f32_f32_f32p2vlx1bf32_1x8vl_sme_mla",
            ),
            (
                "kai_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla",
                "kai_matmul_clamp_f32_f32_f32p4vlx1bf32_6x4vl_sve_mla",
            ),
            (
                "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa",
                "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1bf32_2vlx2vl_sme_mopa",
            ),
            (
                "kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme",
                "kai_rhs_pack_kxn_bf16p2vlx2bf32_f32_x32_sme",
            ),
            (
                "kai_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme",
                "kai_rhs_pack_kxn_f32p16vlx1bf32_f32_f32_sme",
            ),
            (
                "kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme",
                "kai_rhs_pack_kxn_x16p2vlx2bx16_x16_x16_sme",
            ),
            (
                "kai_rhs_pack_kxn_x16p32x1b_x16_x16_neon",
                "kai_rhs_pack_kxn_x16p32x1bx16_x16_x16_neon",
            ),
            (
                "kai_rhs_pack_kxn_x32p16x1b_x32_x32_neon",
                "kai_rhs_pack_kxn_x32p16x1bx32_x32_x32_neon",
            ),
            (
                "kai_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve",
                "kai_rhs_pack_kxn_x32p4vlx1bx32_x32_x32_sve",
            ),
            (
                "kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon",
                "kai_rhs_pack_nxk_qsi4cxpnrx4s1s0_qsu4cx_neon",
            ),
            (
                "kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme",
                "kai_rhs_pack_nxk_x16p2vlx2bx16_x16_x16_sme",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses deprecated `biasf*` spelling; packed bias should be encoded "
            "as `b<type>` after the packed shape."
        ),
        issues=(
            (
                "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa",
                "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1bf32_sme2_mopa",
            ),
            (
                "kai_rhs_pack_kxn_bf16p12x4biasf16_f16_neon",
                "kai_rhs_pack_kxn_bf16p12x4bf16_f16_neon",
            ),
            (
                "kai_rhs_pack_kxn_bf16p12x4biasf32_f16_neon",
                "kai_rhs_pack_kxn_bf16p12x4bf32_f16_neon",
            ),
            (
                "kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon",
                "kai_rhs_pack_kxn_f16p16x1bf16_f16_f16_neon",
            ),
            (
                "kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme",
                "kai_rhs_pack_kxn_f32p2vlx1bf32_f32_f32_sme",
            ),
            (
                "kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon",
                "kai_rhs_pack_kxn_f32p8x1bf32_f32_f32_neon",
            ),
            (
                "kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme",
                "kai_rhs_pack_nxk_f32p2vlx1bf32_f32_f32_sme",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses deprecated `scalef*` spelling; packed scale should be "
            "encoded as `s<type>` after the packed shape. The packed "
            "descriptor is missing/reordering packed shape or puts pack order "
            "on an unpacked source buffer."
        ),
        issues=(
            (
                "kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon",
                "kai_rhs_pack_nxk_qsi4c32pnrx4s1s0sf16_qsu4c32_neon",
            ),
            (
                "kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0",
                "kai_rhs_pack_nxk_qsi4c32pnrx4sf16_qsu4c32",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses the generic `matmul_pack_*` API prefix instead of the "
            "naming-rule pack operation."
        ),
        issues=(
            (
                "kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme",
                "kai_lhs_pack_x32p4vsx1_x32_sme",
            ),
            (
                "kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme",
                "kai_lhs_pack_x8p4vsx4_x8_sme",
            ),
            (
                "kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme",
                "kai_rhs_pack_kxn_x32p4vsx1bx32_x32_x32_sme",
            ),
            (
                "kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme",
                "kai_rhs_pack_kxn_x8p4vsx4_x8_sme",
            ),
            (
                "kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme",
                "kai_rhs_pack_nxk_x32p4vsx1bx32_x32_x32_sme",
            ),
            (
                "kai_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme",
                "kai_rhs_pack_nxk_x8p4vsx4_x8_sme",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses the generic `matmul_pack_*` API prefix instead of the "
            "naming-rule pack operation. Uses untyped, misplaced, or reversed "
            "scale/bias suffixes; the grammar requires `s<type>` before "
            "`b<type>`."
        ),
        issues=(
            (
                "kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme",
                "kai_rhs_pack_kxn_qsi8cxp4vsx4sf32bi32_qsi8_i32_f32_sme",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses the legacy `lhs_quant_pack` operation name; the output "
            "buffer descriptor should carry the quantized packed format "
            "instead."
        ),
        issues=(
            (
                "kai_lhs_quant_pack_bf16p1x4_f32_neon",
                "kai_lhs_pack_bf16p1x4_f32_neon",
            ),
            (
                "kai_lhs_quant_pack_bf16p8x4_f32_neon",
                "kai_lhs_pack_bf16p8x4_f32_neon",
            ),
            (
                "kai_lhs_quant_pack_qai8dxp_bf16_neon",
                "kai_lhs_pack_qai8dxpmrx4_bf16_neon",
            ),
            (
                "kai_lhs_quant_pack_qai8dxp_f16_neon",
                "kai_lhs_pack_qai8dxpmrx4_f16_neon",
            ),
            (
                "kai_lhs_quant_pack_qai8dxp_f32",
                "kai_lhs_pack_qai8dxpmrx4_f32",
            ),
            (
                "kai_lhs_quant_pack_qsi8d32p_f32",
                "kai_lhs_pack_qsi8d32pmrx4_f32",
            ),
            (
                "kai_lhs_quant_pack_qsi8d32p_f32_neon",
                "kai_lhs_pack_qsi8d32pmrx4_f32_neon",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses the legacy `lhs_quant_pack` operation name; the output "
            "buffer descriptor should carry the quantized packed format "
            "instead. Uses deprecated `scalef*` spelling; packed scale should "
            "be encoded as `s<type>` after the packed shape."
        ),
        issues=(
            (
                "kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon",
                "kai_lhs_pack_qsi8d32pmrx4sf32_f16_neon",
            ),
            (
                "kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon",
                "kai_lhs_pack_qsi8d32pmrx4sf32_f32_neon",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses the legacy `lhs_quant_pack` operation name; the output "
            "buffer descriptor should carry the quantized packed format "
            "instead. Uses untyped, misplaced, or reversed scale/bias "
            "suffixes; the grammar requires `s<type>` before `b<type>`."
        ),
        issues=(
            (
                "kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon",
                "kai_lhs_pack_qsi8d32p4x8sf16_f32_neon",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses the legacy `rhs_quant_pack` operation name; RHS packers "
            "should use `rhs_pack_kxn`. Uses deprecated `biasf*` spelling; "
            "packed bias should be encoded as `b<type>` after the packed shape."
        ),
        issues=(
            (
                "kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon",
                "kai_rhs_pack_kxn_bf16p12x4bf32_f32_neon",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses the legacy indirect-matmul pack operation name; pack "
            "micro-kernels should use `lhs_pack` or `rhs_pack_kxn`."
        ),
        issues=(
            (
                "kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme",
                "kai_lhs_pack_x16p2vlx2_x16_sme",
            ),
            (
                "kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme",
                "kai_lhs_pack_x32p2vlx1_x32_sme",
            ),
            (
                "kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme",
                "kai_lhs_pack_x8p2vlx4_x8_sme",
            ),
            (
                "kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme",
                "kai_rhs_pack_kxn_x16p2vlx2bx16_x16_x16_sme",
            ),
            (
                "kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme",
                "kai_rhs_pack_kxn_x32p2vlx1bx32_x32_x32_sme",
            ),
            (
                "kai_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve",
                "kai_rhs_pack_kxn_x32p4vlx1bx32_x32_x32_sve",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses the legacy indirect-matmul pack operation name; pack "
            "micro-kernels should use `lhs_pack` or `rhs_pack_kxn`. Uses "
            "untyped, misplaced, or reversed scale/bias suffixes; the grammar "
            "requires `s<type>` before `b<type>`."
        ),
        issues=(
            (
                "kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme",
                "kai_rhs_pack_kxn_qsi8cxp2vlx4sf32bi32_qsi8cx_f32_i32_sme",
            ),
        ),
    )
)

KNOWN_UKERNEL_PROBLEMS.update(
    _known_issues(
        description=(
            "Uses untyped, misplaced, or reversed scale/bias suffixes; the "
            "grammar requires `s<type>` before `b<type>`."
        ),
        issues=(
            (
                "kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa",
                "kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sf32bi32_2vlx2vl_sme_mopa",
            ),
            (
                "kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa",
                "kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sf32bi32_2vlx2vl_sme2_mopa",
            ),
            (
                "kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot",
                "kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sf32bi32_1x16vl_sme2_dot",
            ),
            (
                "kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_sme2_dot",
                "kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4sf32bi32_1x32vs_sme2_dot",
            ),
            (
                "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa",
                "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sf32bi32_2vlx2vl_sme_mopa",
            ),
            (
                "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa",
                "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sf32bi32_2vlx2vl_sme2_mopa",
            ),
            (
                "kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa",
                "kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2_mopa",
            ),
            (
                "kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme",
                "kai_rhs_pack_kxn_qsi8cxp2vlx4sf32bi32_qsi8cx_f32_i32_sme",
            ),
            (
                "kai_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon",
                "kai_rhs_pack_nxk_qsu2cxp4x4sf32bf32_qsu2cx_neon",
            ),
        ),
    )
)

KNOWN_DIRECTORY_PROBLEMS: dict[str, KnownIssue] = {
    "matmul_clamp_fp32_bf16p_bf16p": KnownIssue(
        expected="matmul_clamp_f32_bf16p_bf16p",
        description="There is an additional `p` in `fp32`",
    )
}

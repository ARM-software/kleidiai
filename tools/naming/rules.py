#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Describe the naming rules for micro-kernels
"""
from naming.grammar import Expr
from naming.grammar import Grammar
from naming.grammar import MatchResult
from naming.grammar import NaturalInt
from naming.grammar import OneOf
from naming.grammar import OneOrMore
from naming.grammar import OperandType
from naming.grammar import Optional
from naming.grammar import Seq


grammar = Grammar()


@grammar.rule("Quantization mode.")
def quantization() -> Expr:
    return OneOf("qa", "qs")


@grammar.rule("Quantization axis or block shape.")
def quantization_axis() -> Expr:
    return OneOf("cx", "dx", "d32", "c32")


@grammar.rule()
def size() -> Expr:
    return OneOf(
        NaturalInt(),
        Seq(NaturalInt(), "vl"),
        Seq(NaturalInt(), "vs"),
        "mr",
        "nr",
    )


@grammar.rule("Packed buffer width.")
def pack_width() -> Expr:
    return size


@grammar.rule("Packed buffer height.")
def pack_height() -> Expr:
    return size


@grammar.rule("Primary tile size.")
def tile_size() -> Expr:
    return Seq(size, "x", size)


@grammar.rule("Depthwise output block size.")
def dwconv_output_block_size() -> Expr:
    # `c` refers to full channel output
    return Seq(NaturalInt(), "x", OneOf(NaturalInt(), "c"))


@grammar.rule("Packed data order.")
def pack_order() -> Expr:
    return OneOf("s0s1", "s1s0", "s4s0", "s16s0")


@grammar.rule("Buffer descriptor.")
def buffer() -> Expr:
    return Seq(
        Optional(quantization),
        OperandType(),
        Optional(quantization_axis),
        Optional(
            Seq(
                "p",
                pack_width,
                "x",
                pack_height,
                Optional(pack_order),
                Optional(Seq("s", OperandType())),
                Optional(Seq("b", OperandType())),
            )
        ),
    )


@grammar.rule("Directory buffer descriptor.")
def simplified_buffer() -> Expr:
    return Seq(
        Optional(quantization),
        OperandType(),
        Optional(quantization_axis),
        Optional("p"),
    )


@grammar.rule("Fused operation descriptor.")
def matmul_fused_ops() -> Expr:
    return OneOf(
        Seq(Optional("i"), "matmul", Optional("_clamp")),
        "lhs_pack",
        "rhs_pack_kxn",
        "rhs_pack_nxk",
    )


@grammar.rule("Convolution filter size.")
def filter_size() -> Expr:
    return Seq(NaturalInt(), "x", NaturalInt())


@grammar.rule("Convolution stride.")
def dw_stride() -> Expr:
    return Seq("s", NaturalInt())


@grammar.rule("Primary instruction used by the micro-kernel.")
def instruction() -> Expr:
    return OneOf("dot", "i8mm", "mla", "mmla", "mopa", "sdot")


@grammar.rule("SIMD engine")
def tech() -> Expr:
    return OneOf("neon", "sve", "sve2", "sme", "sme2")


@grammar.rule("Target microarchitecture.")
def uarch() -> Expr:
    return OneOf("cortexa55")


@grammar.rule("Complete depthwise micro-kernel name.")
def dwconv_ukernel_name() -> Expr:
    return Seq(
        "kai_",
        OneOf(
            Seq(
                "dwconv_clamp",
                OneOrMore(Seq("_", buffer)),
                Seq("_", filter_size),
                Seq("_", dw_stride),
                Seq("_", dwconv_output_block_size),
                Seq(
                    "_",
                    tech,
                ),
                Optional(Seq("_", instruction)),
            ),
            Seq(
                "rhs_dwconv_pack",
                OneOrMore(Seq("_", buffer)),
                Seq(
                    "_",
                    tech,
                ),
            ),
        ),
    )


@grammar.rule("Complete matmul micro-kernel name.")
def matmul_ukernel_name() -> Expr:
    return Seq(
        "kai_",
        matmul_fused_ops,
        OneOrMore(Seq("_", buffer)),
        Optional(Seq("_", tile_size)),
        Optional(Seq("_", tech)),
        Optional(Seq("_", instruction)),
        Optional(Seq("_", uarch)),
    )


@grammar.rule("Complete micro-kernel name.")
def kernel_name() -> Expr:
    return OneOf(matmul_ukernel_name, dwconv_ukernel_name)


@grammar.rule("Complete micro-kernel directory name.")
def directory_name() -> Expr:
    return OneOf(
        "pack",
        Seq(
            matmul_fused_ops,
            "_",
            simplified_buffer,
            OneOrMore(Seq("_", simplified_buffer)),
        ),
        Seq(
            "dwconv",
            "_",
            simplified_buffer,
            OneOrMore(Seq("_", simplified_buffer)),
        ),
    )


def match_kernel_name(name: str) -> MatchResult:
    """Match the kernel name against the naming rules."""
    return kernel_name.match_full(name)


def match_directory_name(name: str) -> MatchResult:
    """Match the directory name against the naming rules."""
    return directory_name.match_full(name)

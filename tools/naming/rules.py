#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Describe the naming rules for micro-kernels
"""
from naming.grammar import Doc
from naming.grammar import Expr
from naming.grammar import Grammar
from naming.grammar import NaturalInt
from naming.grammar import OneOf
from naming.grammar import OneOrMore
from naming.grammar import OperandType
from naming.grammar import Optional
from naming.grammar import ParseResult
from naming.grammar import Seq


grammar = Grammar()


@grammar.rule(
    title="Quantization mode",
    description="Describes whether quantized data uses symmetric or asymmetric quantization.",
)
def quantization() -> Expr:
    return OneOf(
        Doc("qa", description="Asymmetrical quantization"),
        Doc("qs", description="Symmetrical quantization"),
    )


@grammar.rule(
    title="Quantization axis or block shape",
    description="Describes how quantization parameters are associated with data dimensions or blocks.",
)
def quantization_axis() -> Expr:
    return OneOf(
        Doc("cx", description="Per channel quantized"),
        Doc("dx", description="Per dimension quantized"),
        Doc(
            "d32",
            description="Per-dimension block quantization, with block length multiple of 32",
        ),
        Doc(
            "c32",
            description="Per block quantization, with block length multiple of 32",
        ),
    )


@grammar.rule(
    title="Dimension size",
    description=(
        "Describes a dimension size, which can be a positive integer, a "
        "vector-length multiple using `vl`, a vector-scale multiple using "
        "`vs`, or a parametric `mr` or `nr` value."
    ),
)
def size() -> Expr:
    return OneOf(
        Doc(NaturalInt(), description="Constant value"),
        Doc(
            Seq(NaturalInt(), "vl"),
            description=(
                "Accumulator vector length multiple. Assuming 32-bit accumulation "
                "on a 512-bit configuration, `4vl` means 64 elements"
            ),
        ),
        Doc(
            Seq(NaturalInt(), "vs"),
            description=(
                "Vector scale multiplier, assuming a vector length of 128 bits. "
                "For a 512-bit configuration, `4vs` means 16 elements"
            ),
        ),
        Doc("mr", description="Parametric size, given as argument to micro-kernel"),
        Doc("nr", description="Parametric size, given as argument to micro-kernel"),
    )


@grammar.rule(
    title="Primary tile size",
    description="Describes the primary row-by-column tile computed by a micro-kernel.",
)
def tile_size() -> Expr:
    return Seq(size, "x", size)


@grammar.rule(
    title="Depthwise output block size",
    description=(
        "Describes the output block shape produced by a depthwise convolution "
        "micro-kernel. The `c` column size means all channels."
    ),
)
def dwconv_output_block_size() -> Expr:
    # `c` refers to full channel output
    return Seq(NaturalInt(), "x", OneOf(NaturalInt(), "c"))


@grammar.rule(
    title="Packed data order",
    description="Describes the ordering of values inside a packed buffer.",
)
def pack_order() -> Expr:
    return OneOf(
        Doc("s1s0", description="Packing order of data is sequential"),
        Doc(
            "s4s0",
            description="Packing order of data is interleaved with nibble distance of 4",
        ),
        Doc("s16s0", description="Packing order of data is interleaved"),
    )


@grammar.rule(
    title="Packing layout",
    description="Describes the dimensions used when data is packed into a buffer.",
)
def packing_layout() -> Expr:
    return Seq(
        "p",
        Doc(size, description="Width component of a packed buffer layout"),
        "x",
        Doc(size, description="Height component of a packed buffer layout"),
    )


@grammar.rule(
    title="Packed scale type",
    description="Describes the operand type used for scale values stored in a packed buffer.",
)
def scale_type() -> Expr:
    return Seq("s", OperandType())


@grammar.rule(
    title="Packed bias type",
    description="Describes the operand type used for bias values stored in a packed buffer.",
)
def bias_type() -> Expr:
    return Seq("b", OperandType())


@grammar.rule(
    title="Packed buffer layout description",
    description="Describes the packed-buffer suffix of a full buffer descriptor.",
)
def pack_description() -> Expr:
    return Seq(
        packing_layout,
        Optional(pack_order),
        Optional(scale_type),
        Optional(bias_type),
    )


@grammar.rule(
    title="Buffer descriptor",
    description=(
        "Describes an input or output buffer in a full micro-kernel name. "
        "Packed buffers include their concrete packed layout."
    ),
)
def buffer() -> Expr:
    return Seq(
        Doc(Optional(quantization), description="Quantization indication"),
        Doc(OperandType(), description="The main data-type stored in buffer"),
        Doc(
            Optional(quantization_axis),
            description="If quantized, indicates quantization granularity",
        ),
        Doc(
            Optional(pack_description),
            description="If packed, indicates packing properties",
        ),
    )


@grammar.rule(
    title="Directory buffer descriptor",
    description=(
        "Describes an input or output buffer in a micro-kernel directory name. "
        "Directory names use abbreviated descriptors and omit concrete packed "
        "layout details."
    ),
)
def simplified_buffer() -> Expr:
    return Seq(
        Optional(quantization),
        OperandType(),
        Optional(quantization_axis),
        Optional("p"),
    )


@grammar.rule(
    title="Fused operation descriptor",
    description="Describes the primary matmul-family operation and any encoded fused operation.",
)
def matmul_fused_ops() -> Expr:
    return OneOf(
        Seq(Optional("i"), "matmul", Optional("_clamp")),
        "lhs_pack",
        "rhs_pack_kxn",
        "rhs_pack_nxk",
    )


@grammar.rule(
    title="Convolution filter size",
    description="Describes the height-by-width filter shape used by a depthwise convolution micro-kernel.",
)
def filter_size() -> Expr:
    return Seq(NaturalInt(), "x", NaturalInt())


@grammar.rule(
    title="Convolution stride",
    description="Describes the stride used by a depthwise convolution micro-kernel.",
)
def dw_stride() -> Expr:
    return Seq("s", NaturalInt())


@grammar.rule(
    title="Primary instruction family",
    description="Describes the predominant SIMD instruction family used by the implementation.",
)
def instruction() -> Expr:
    return OneOf("dot", "i8mm", "mla", "mmla", "mopa", "sdot")


@grammar.rule(
    title="Primary feature",
    description="Describes the predominant `FEAT_<feature>` used by the implementation.",
)
def feature() -> Expr:
    return OneOf("i8mm", "dotprod")


@grammar.rule(
    title="SIMD engine",
    description="Describes the SIMD engine targeted by the implementation.",
)
def tech() -> Expr:
    return OneOf("neon", "sve", "sve2", "sve2p1", "sme", "sme2", "sme2p1")


@grammar.rule(
    title="Target microarchitecture",
    description="Describes a target microarchitecture for which a micro-kernel is optimized.",
)
def uarch() -> Expr:
    return OneOf("cortexa55")


@grammar.rule(
    title="Depthwise micro-kernel name",
    description="Describes names for depthwise convolution and depthwise RHS packing micro-kernels.",
)
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


@grammar.rule(
    title="Matmul micro-kernel name",
    description="Describes names for matmul-family compute and packing micro-kernels.",
)
def matmul_ukernel_name() -> Expr:
    return Seq(
        "kai_",
        matmul_fused_ops,
        Doc(Seq("_", buffer), description="Destination buffer"),
        Doc(OneOrMore(Seq("_", buffer)), description="Input buffer(s)"),
        Doc(Optional(Seq("_", tile_size)), description="Tile size"),
        Doc(Optional(Seq("_", tech)), description="SIMD engine"),
        Doc(Optional(Seq("_", feature)), description="Primary feature"),
        Doc(Optional(Seq("_", instruction)), description="Primary instruction"),
        Doc(Optional(Seq("_", uarch)), description="Target micro-architecture"),
    )


@grammar.rule(
    title="Micro-kernel name",
    description="Describes every micro-kernel name accepted by the naming rules.",
)
def kernel_name() -> Expr:
    return OneOf(matmul_ukernel_name, dwconv_ukernel_name)


@grammar.rule(
    title="Micro-kernel directory name",
    description="Describes every micro-kernel directory name accepted by the naming rules.",
)
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


def parse_kernel_name(name: str) -> ParseResult:
    """Parse the kernel name against the naming rules."""
    return kernel_name.parse_full(name)


def parse_directory_name(name: str) -> ParseResult:
    """Parse the directory name against the naming rules."""
    return directory_name.parse_full(name)

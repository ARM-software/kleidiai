<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# The micro-kernel naming scheme

This document describes the KleidiAI micro-kernel naming scheme.

## Naming Structure

The high level view of micro-kernel naming is generated from the documented
grammar rules:

- Micro-kernel source files use names beginning with `kai_`.
- Matmul micro-kernels use the `matmul_ukernel_name` grammar.
- Depthwise convolution micro-kernels describe the operation, buffers, filter, stride, output block, SIMD engine, and optional instruction through the `dwconv_ukernel_name` grammar.
- Micro-kernel directories use the `directory_name` grammar.

### Common Name Shapes

- LHS packing micro-kernels are named `kai_lhs_pack_<output>_<input>_<description>`.
- RHS packing micro-kernels are named `kai_rhs_pack_<orientation>_<output>_<inputs>_<description>`.
- Matmul compute micro-kernels are named `kai_<operation>_<output>_<LHS input>_<RHS input>_<description>`.
- Depthwise convolution micro-kernels are named with the depthwise operation, buffers, filter, stride, output block, SIMD engine, and optional instruction.

For matmul-family compute micro-kernels, buffer descriptors appear in the order
destination, LHS input, then RHS input. The output descriptors of LHS and RHS
packing micro-kernels match the corresponding packed input descriptors of the
matmul micro-kernel.

### Packed Buffer Layouts

Packed buffers include a `p<width>x<height>` layout in full micro-kernel names.
For matmul-family names, the LHS packed width is normally the row blocking
dimension (`MR`), and the RHS packed width is normally the column blocking
dimension (`NR`). The packed height is the block depth (`BD`), which is derived
from the `KR` and `SR` values used by the micro-kernel as `KR / SR`.

Packed buffers can also encode data order, packed scale type, and packed bias
type. Scale values are encoded as `s<type>` and bias values as `b<type>`.
Legacy `scalef*` and `biasf*` spellings are not valid in the grammar.

### How to Read the Grammar

- **`rule_name`**: Reference to another grammar rule.
- **`"text"`**: Literal text that appears in the name.
- **`a b`**: Sequence: `a` followed by `b`.
- **`a | b`**: Choice: either `a` or `b`.
- **`[a]`**: Optional expression.
- **`a+`**: One or more repetitions of `a`.
- **`(a b)`**: Grouped expression.
- **`@natural_int`**: Positive integer literal.
- **`@pow2_int`**: Power-of-two integer literal.
- **`@operand_type`**: Data type descriptor.

## Naming Rules

The documented naming rules are listed below.

### Micro-kernel directory name

Describes every micro-kernel directory name accepted by the naming rules.

**`directory_name`** = `"pack" | matmul_fused_ops "_" simplified_buffer ("_" simplified_buffer)+ | "dwconv" "_" simplified_buffer ("_" simplified_buffer)+`

### Micro-kernel name

Describes every micro-kernel name accepted by the naming rules.

**`kernel_name`** = `matmul_ukernel_name | dwconv_ukernel_name`

### Matmul micro-kernel name

Describes names for matmul-family compute and packing micro-kernels.

**`matmul_ukernel_name`** = `"kai_" matmul_fused_ops ("_" buffer) ("_" buffer)+ ["_" tile_size] ["_" tech] ["_" feature] ["_" instruction] ["_" uarch]`

where:

- **`"_" buffer`**: Destination buffer
- **`("_" buffer)+`**: Input buffer(s)
- **`["_" tile_size]`**: Tile size
- **`["_" tech]`**: SIMD engine
- **`["_" feature]`**: Primary feature
- **`["_" instruction]`**: Primary instruction
- **`["_" uarch]`**: Target micro-architecture

### Depthwise micro-kernel name

Describes names for depthwise convolution and depthwise RHS packing micro-kernels.

**`dwconv_ukernel_name`** = `"kai_" ("dwconv_clamp" ("_" buffer)+ ("_" filter_size) ("_" dw_stride) ("_" dwconv_output_block_size) ("_" tech) ["_" instruction] | "rhs_dwconv_pack" ("_" buffer)+ ("_" tech))`

### Target microarchitecture

Describes a target microarchitecture for which a micro-kernel is optimized.

**`uarch`** = `"cortexa55"`

### SIMD engine

Describes the SIMD engine targeted by the implementation.

**`tech`** = `"neon" | "sve" | "sve2" | "sve2p1" | "sme" | "sme2" | "sme2p1"`

### Primary feature

Describes the predominant `FEAT_<feature>` used by the implementation.

**`feature`** = `"i8mm" | "dotprod"`

### Primary instruction family

Describes the predominant SIMD instruction family used by the implementation.

**`instruction`** = `"dot" | "i8mm" | "mla" | "mmla" | "mopa" | "sdot"`

### Convolution stride

Describes the stride used by a depthwise convolution micro-kernel.

**`dw_stride`** = `"s" @natural_int`

### Convolution filter size

Describes the height-by-width filter shape used by a depthwise convolution micro-kernel.

**`filter_size`** = `@natural_int "x" @natural_int`

### Fused operation descriptor

Describes the primary matmul-family operation and any encoded fused operation.

**`matmul_fused_ops`** = `["i"] "matmul" ["_clamp"] | "lhs_pack" | "rhs_pack_kxn" | "rhs_pack_nxk"`

### Directory buffer descriptor

Describes an input or output buffer in a micro-kernel directory name. Directory names use abbreviated descriptors and omit concrete packed layout details.

**`simplified_buffer`** = `[quantization] @operand_type [quantization_axis] ["p"]`

### Buffer descriptor

Describes an input or output buffer in a full micro-kernel name. Packed buffers include their concrete packed layout.

**`buffer`** = `[quantization] @operand_type [quantization_axis] [pack_description]`

where:

- **`[quantization]`**: Quantization indication
- **`@operand_type`**: The main data-type stored in buffer
- **`[quantization_axis]`**: If quantized, indicates quantization granularity
- **`[pack_description]`**: If packed, indicates packing properties

### Packed buffer layout description

Describes the packed-buffer suffix of a full buffer descriptor.

**`pack_description`** = `packing_layout [pack_order] [scale_type] [bias_type]`

### Packed bias type

Describes the operand type used for bias values stored in a packed buffer.

**`bias_type`** = `"b" @operand_type`

### Packed scale type

Describes the operand type used for scale values stored in a packed buffer.

**`scale_type`** = `"s" @operand_type`

### Packing layout

Describes the dimensions used when data is packed into a buffer.

**`packing_layout`** = `"p" size "x" size`

where:

- **`size`**: Width component of a packed buffer layout
- **`size`**: Height component of a packed buffer layout

### Packed data order

Describes the ordering of values inside a packed buffer.

**`pack_order`** = `"s1s0" | "s4s0" | "s16s0"`

where:

- **`"s1s0"`**: Packing order of data is sequential
- **`"s4s0"`**: Packing order of data is interleaved with nibble distance of 4
- **`"s16s0"`**: Packing order of data is interleaved

### Depthwise output block size

Describes the output block shape produced by a depthwise convolution micro-kernel. The `c` column size means all channels.

**`dwconv_output_block_size`** = `@natural_int "x" (@natural_int | "c")`

### Primary tile size

Describes the primary row-by-column tile computed by a micro-kernel.

**`tile_size`** = `size "x" size`

### Dimension size

Describes a dimension size, which can be a positive integer, a vector-length multiple using `vl`, a vector-scale multiple using `vs`, or a parametric `mr` or `nr` value.

**`size`** = `@natural_int | @natural_int "vl" | @natural_int "vs" | "mr" | "nr"`

where:

- **`@natural_int`**: Constant value
- **`@natural_int "vl"`**: Accumulator vector length multiple. Assuming 32-bit accumulation on a 512-bit configuration, `4vl` means 64 elements
- **`@natural_int "vs"`**: Vector scale multiplier, assuming a vector length of 128 bits. For a 512-bit configuration, `4vs` means 16 elements
- **`"mr"`**: Parametric size, given as argument to micro-kernel
- **`"nr"`**: Parametric size, given as argument to micro-kernel

### Quantization axis or block shape

Describes how quantization parameters are associated with data dimensions or blocks.

**`quantization_axis`** = `"cx" | "dx" | "d32" | "c32"`

where:

- **`"cx"`**: Per channel quantized
- **`"dx"`**: Per dimension quantized
- **`"d32"`**: Per-dimension block quantization, with block length multiple of 32
- **`"c32"`**: Per block quantization, with block length multiple of 32

### Quantization mode

Describes whether quantized data uses symmetric or asymmetric quantization.

**`quantization`** = `"qa" | "qs"`

where:

- **`"qa"`**: Asymmetrical quantization
- **`"qs"`**: Symmetrical quantization

## Rule Enforcement

The naming scheme is implemented in `tools/naming/rules.py`. Micro-kernel and
directory names are checked by `tools/check_microkernel_names.py`, and CI uses
the same checker to enforce the rules.

This document is generated from the naming rules. Regenerate it with
`tools/check_microkernel_names.py --generate-documentation` after updating the
naming scheme.

## Known naming issues

Some legacy micro-kernel and directory names do not match the current grammar,
but are retained to preserve the existing API. The known exceptions are listed
in `tools/naming/issues.py`. To include them in checker output, run
`tools/check_microkernel_names.py --report-known-issues`.

## Full naming grammar

The grammar below is generated from the naming rules.

```text
directory_name = "pack" | matmul_fused_ops "_" simplified_buffer ("_" simplified_buffer)+ | "dwconv" "_" simplified_buffer ("_" simplified_buffer)+
kernel_name = matmul_ukernel_name | dwconv_ukernel_name
matmul_ukernel_name = "kai_" matmul_fused_ops ("_" buffer) ("_" buffer)+ ["_" tile_size] ["_" tech] ["_" feature] ["_" instruction] ["_" uarch]
dwconv_ukernel_name = "kai_" ("dwconv_clamp" ("_" buffer)+ ("_" filter_size) ("_" dw_stride) ("_" dwconv_output_block_size) ("_" tech) ["_" instruction] | "rhs_dwconv_pack" ("_" buffer)+ ("_" tech))
uarch = "cortexa55"
tech = "neon" | "sve" | "sve2" | "sve2p1" | "sme" | "sme2" | "sme2p1"
feature = "i8mm" | "dotprod"
instruction = "dot" | "i8mm" | "mla" | "mmla" | "mopa" | "sdot"
dw_stride = "s" @natural_int
filter_size = @natural_int "x" @natural_int
matmul_fused_ops = ["i"] "matmul" ["_clamp"] | "lhs_pack" | "rhs_pack_kxn" | "rhs_pack_nxk"
simplified_buffer = [quantization] @operand_type [quantization_axis] ["p"]
buffer = [quantization] @operand_type [quantization_axis] [pack_description]
pack_description = packing_layout [pack_order] [scale_type] [bias_type]
bias_type = "b" @operand_type
scale_type = "s" @operand_type
packing_layout = "p" size "x" size
pack_order = "s1s0" | "s4s0" | "s16s0"
dwconv_output_block_size = @natural_int "x" (@natural_int | "c")
tile_size = size "x" size
size = @natural_int | @natural_int "vl" | @natural_int "vs" | "mr" | "nr"
quantization_axis = "cx" | "dx" | "d32" | "c32"
quantization = "qa" | "qs"
```

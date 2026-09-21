<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Softmax

Softmax micro-kernels currently compute normalized probabilities over a one-dimensional input.

The public softmax micro-kernel API is defined by [`kai_softmax_types.h`](kai_softmax_types.h). Each implementation
returns the API structure by value; callers use its step, stride, offset, size, and run callbacks instead of
implementation-specific entry points.

## Dimension convention

The API exposes a single `dim_0` dimension, along which softmax is computed. Shapes do not include a batch dimension;
callers handle batching by invoking the micro-kernel once for each batch element.

Source and destination operands each provide a stride in bytes for `dim_0`. The `get_src_stride` and `get_dst_stride`
callbacks return the default strides for a shape. Callers pass the selected strides to the offset and size callbacks
and to `run`.

Micro-kernel naming is described in
[docs/microkernel_names.md](../../../docs/microkernel_names.md).

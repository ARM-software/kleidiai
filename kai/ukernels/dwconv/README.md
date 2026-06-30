<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# About

This document contains information related to depthwise convolution (dwconv)
micro-kernels.

# Depthwise Conv

Dw conv micro-kernels operate directly on tensors stored in memory buffers. The RHS buffer is
normally pre-packed into a more efficient data layout taking into account vector length (or with interleaved bias).

Micro-kernel naming is described in
[docs/microkernel_names.md](../../../docs/microkernel_names.md).

<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# About

This document contains information related to matrix-multiplication (matmul)
micro-kernels. At the moment there are two main types of micro-kernels, matrix
multiplication and indirect matrix multiplication micro-kernels. The indirect
micro-kernels are denoted _imatmul_.

# Matmul

Matmul micro-kernels operate directly on matrices stored in memory buffers, where the
buffers are normally first packed into a more efficient layout.

# Indirect Matmul

The indirect matmul micro-kernels operate on indirection buffers, matrices of pointers
to actual data.

Micro-kernel naming is described in
[docs/microkernel_names.md](../../../docs/microkernel_names.md).

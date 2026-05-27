<!--
    SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Introduction

> **WARNING:** *The software in this directory is currently experimental.*

This directory contains an experimental open-source AI operator-level library providing highly
optimized General Matrix Multiply (GEMM) and convolution kernels for Arm® architectures. The
library includes specialized implementations for various data types (FP32, FP16, BF16, INT8,
INT16, quantized), with support for advanced architectural features such as SVE and SME.

**Note:** This library is not accepting external contributions at this time.

# Building

To build this library in the `build` directory, type

```sh
cmake -B build
cmake --build build
```

This will produce a static binary called `gemm` which can be used to test and benchmark all the
kernels.

There are a set of present toolchain configurations in `toolchains/`. E.g.

```sh
cmake -B build --toolchain toolchains/native/native-linux-aarch64-gnu.cmake
cmake --build build
```

Or you can manually set configuration options from `CMakeLists.txt`.

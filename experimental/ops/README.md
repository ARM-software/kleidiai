<!--
    SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Introduction

This directory contains an experimental open-source AI operator-level library providing highly optimized General Matrix Multiply (GEMM) and convolution kernels for Arm® architectures. The library includes specialized implementations for various data types (FP32, FP16, BF16, INT8, INT16, quantized), with support for advanced architectural features such as SVE and SME.

**Note:** This library is not accepting external contributions at this time.

# CONTRIBUTING

Make sure to install the pre-commit hooks before making any changes/contributions, like so

```bash
pip install pre-commit
pre-commit install
```

When you make a commit, the pre-commit hooks will run and lint your code. If any of the hooks fail, address the error messages, stage the modified files again, then try to commit the code again.

If for some reason, you are sure that the only remaining errors produced by the pre-commit hooks are not related to your changes or need to be addressed in a separate branch/PR, you can bypass the hooks with the `--no-verify` flag:

```bash
git commit [...] --no-verify
```

# BUILDING

To build this library in the `build` directory, type

```sh
cmake -B build
cmake --build build
```

This will produce a static binary called `gemm` which can be used to test and benchmark all the kernels.

There are a set of present toolchain configurations in `toolchains/`. E.g.

```sh
cmake -B build --toolchain toolchains/native/native-linux-aarch64-gnu.cmake
cmake --build build
```

Or you can manaully set configuration options from `CMakeLists.txt`.

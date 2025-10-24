<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

<h1><b>KleidiAI</b></h1>

KleidiAI is an open-source library that provides optimized performance-critical routines, also known as <strong>micro-kernels</strong>, for artificial intelligence (AI) workloads tailored for Arm® CPUs.

These routines are tuned to exploit the capabilities of specific Arm® hardware architectures, aiming to maximize performance.

The KleidiAI library has been designed for ease of adoption into C or C++ machine learning (ML) and AI frameworks. Specifically, developers looking to incorporate specific micro-kernels into their projects can only include the corresponding <strong>.c</strong> and <strong>.h</strong> files associated with those micro-kernels and a common header file.

<h1> Who is this library for? </h1>

KleidiAI is a library for AI/ML framework developers interested in accelerating the computation on Arm® CPUs.

<h1> What is a micro-kernel? </h1>

A micro-kernel, or <strong>ukernel</strong>, can be defined as a near-minimum amount of software to accelerate a given ML operator with high performance.

Following are examples of a micro-kernel

- Function to perform [packing](kai/ukernels/matmul/pack/README.md)
- Function to perform matrix multiplication

<em>However, why are the preceding operations not called kernels or functions instead?</em>

<b>This is because the micro-kernels are designed to give the flexibility to process also a portion of the output tensor</b>.

> ℹ️ The API of the micro-kernel is intended to provide the flexibility to dispatch the operation among different working threads or process only a section of the output tensor. Therefore, the caller can control what to process and how.

A micro-kernel exists for different Arm® architectures, technologies, and computational parameters (for example, different output tile sizes). These implementations are called <strong>micro-kernel variants</strong>. All micro-kernel variants of the same micro-kernel type perform the same operation and return the same output result.

<h1> Key features </h1>

Some of the key features of KleidiAI are the following:

- No dependencies on external libraries

- No dynamic memory allocation

- No memory management​

- No scheduling

- Stateless, stable, and consistent API​

- Performance-critical compute-bound and memory-bound micro-kernels

- Specialized micro-kernels utilizing different Arm® CPU architectural features (for example, <strong>FEAT_DotProd</strong> and <strong>FEAT_I8MM</strong>)

- Specialized micro-kernels for different fusion patterns

- Micro-kernel as a standalone library, consisting of only a <strong>.c</strong> and <strong>.h</strong> files

> ℹ️ The micro-kernel API is designed to be as generic as possible for integration into third-party runtimes.

<h1> Supported instructions and extensions </h1>

- Advanced SIMD instructions
- Scalable Matrix Extension(SME)
- Scalable Matrix Extension 2(SME2)

The SME and SME2 micro-kernels require compiler support to generate SME ABI-compliant code.
You can still use the micro-kernels without compiler support, but not within a call chain that already uses ZA register.
At the moment this is not automatically detected, and you need to build with `KLEIDIAI_INTERNAL_EXTRA_ARCH=+sme` to
enable this support.

<h1> Filename convention </h1>

The `kai/ukernels` directory is the home for all micro-kernels. The micro-kernels are grouped in separate directories based on the performed operation. For example, all the matrix-multiplication micro-kernels are held in the `matmul/` operator directory.

Inside the operator directory, you can find:

- *The common micro-kernels*, which are helper micro-kernels necessary for the correct functioning of the main ones. For example, some of these may be required for packing the input tensors and held in the `pack` subdirectory.
- *The micro-kernels* files, which are held in separate sub-directories.

The name of the micro-kernel folder provides the description of the operation performed and the data type of the destination and source tensors. The general syntax for the micro-kernel folder is as follows:

`<op>_<dst-data-type>_<src0-data-type>_<src1-data-type>_...`

All <strong>.c</strong> and <strong>.h</strong> pair files in that folder are micro-kernel variants. The variants are differentiated by specifying the computational paramaters (for example, the block size), the Arm® technology (for example, Arm® Neon™), and Arm® architecture feature exploited (for example, <strong>FEAT_DotProd</strong>). The general syntax for the micro-kernel variant is as follows:

`kai_<micro-kernel-folder>_<compute-params>_<technology>_<arch-feature>.c/.h`

> ℹ️ These files, only depend on the `kai_common.h` file.

All functions defined in the <strong>.h</strong> header file of the micro-kernel variant has the following syntax:

`kai_<op>_<micro-kernel-variant-filename>.c/.h`

<h1> Supported micro-kernels </h1>

For a list of supported micro-kernels refer to the [source](/kai/ukernels/) directory. The micro-kernels are grouped in separate directories based on the performed operation.
For example, all the matrix-multiplication micro-kernels are held in the `matmul/` directory. In there, the micro-kernels are grouped into folders whose name syntax describes the micro-kernel from a data type point of view of inputs and outputs.

<h1> How to build </h1>

<h2> Prerequisites </h2>

KleidiAI requires the following dependencies, obtainable via your preferred package manager, to be installed and available on your system to be able to build the project.

- `build-essential`
- `cmake >= 3.18`

In addition, you may choose to use the following toolchains:

- (Optional) `Arm GNU toolchain` available to download from the [Arm Developer](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) website.
- (Optional) `Android NDK` available to download from the [Android Developer](https://developer.android.com/ndk/downloads/index.html) website.

<h2> Compile natively on an Arm®-based system </h2>

You can quickly compile KleidiAI on your system with an Arm® processor by using the following commands:

```shell
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build/
cmake --build ./build
```

<h2> Cross-compile to Android™ </h2>

Cross-compiling for Android systems requires the Android NDK toolset. The downloaded NDK contains the CMake toolchain file necessary for cross-compiling the project and must be provided to CMake with the `-DCMAKE_TOOLCHAIN_FILE` option.

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -S . -B build/
cmake --build ./build
```

<h2> Cross-compile to Linux® </h2>

The Arm GNU toolchain can be used to cross-compile to a Linux system with an Arm® processor like a Raspberry Pi from an x86_64 Linux host machine. Ensure the toolchain is available on your PATH and provide to CMake the Arm GNU Toolchain CMakefile found in `cmake/toolchains` directory with the `-DCMAKE_TOOLCHAIN_FILE` option.

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-none-linux-gnu.toolchain.cmake -S . -B build/
cmake --build ./build
```

<h1> Release </h1>
<h2> Cadence </h2>

Two releases will be done per month. All releases can be found in the [release](https://gitlab.arm.com/kleidi/kleidiai/-/releases) section.

<h2> Version </h2>

The release version conforms to Semantic Versioning.

> ⚠️ Please note that API modifications, including function name changes, and feature enhancements may occur without advance notice.

<h1> Support </h1>

Please raise a [GitLab Issue](https://gitlab.arm.com/kleidi/kleidiai/-/issues/new) for technical support.

<h1> Frequently Asked Questions (FAQ) </h1>

<h2> What is the difference between the Compute Library for the Arm® Architecture (ACL) and KleidiAI? </h2>

This question will pop up naturally if you are familiar with the **[ACL](https://github.com/ARM-software/ComputeLibrary)**.

<em>ACL and KleidiAI differ with respect to the integration point into the AI/ML framework</em>.

ACL provides a complete suite of ML operators for Arm® CPUs and Arm Mali™ GPUs. It also provides a runtime with memory management, thread management, fusion capabilities, etc.

Therefore, <strong>ACL is a library suitable for frameworks that need to delegate the model inference computation entirely</strong>.

KleidiAI offers performance-critical operators for ML, like matrix multiplication, pooling, depthwise convolution, and so on. As such, <strong>KleidiAI is designed for frameworks where the runtime, memory manager, thread management, and fusion mechanisms are already available</strong>.

<h2> Can the micro-kernels be multi-threaded? </h2>

<strong>Yes, they can</strong>. The micro-kernel can be dispatched among different threads using the thread management available in the target AI/ML framework.

<em>The micro-kernel does not use any internal threading mechanism</em>. However, the micro-kernel's API is designed to allow the computation to be carried out only on specific areas of the output tensor. Therefore, this mechanism is sufficient to split the workload on parallel threads. More information on dispatching the micro-kernels among different threads will be available soon.

<h1> License </h1>

KleidiAI is distributed under the software licenses in LICENSES directory.

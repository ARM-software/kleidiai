<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# KleidiAI Benchmark Tool

KleidiAI provides a single benchmarking binary that runs multiple variants via subcommands:

- `kleidiai_benchmark matmul` for standard matrix multiplication (matmul)
- `kleidiai_benchmark imatmul` for indirect matrix multiplication (imatmul, chunked K)
- `kleidiai_benchmark dwconv` for depthwise convolution (dwconv)

The tool supports flexible argument parsing and Benchmark Framework options.
If no operator is specified, `matmul` will be used by default.

## Building

From the KleidiAI root directory:

### Build instructions

```
mkdir -p build && cd build
cmake -DKLEIDIAI_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release ../
make -j
```

### Linux®-target

```
$ mkdir -p build && cd build
$ cmake -DCMAKE_C_COMPILER=/path/to/aarch64-none-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/path/to/aarch64-none-linux-gnu-g++ -DKLEIDIAI_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release ../
```

### Android™-target

```
$ mkdir -p build && cd build
$ cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=30 -DKLEIDIAI_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release ../
```

## Usage

### Quick Examples

Run both matmul, imatmul and dwconv with example dimensions:

```sh
./kleidiai_benchmark matmul  -m 32 -n 32 -k 32
./kleidiai_benchmark imatmul -m 32 -n 32 -c 4 -l 8
./kleidiai_benchmark dwconv  --input_height 32 --input_width 32 --channels 64 --padding 1,1,1,1
```

### Matmul Benchmark

The dimensions of the LHS- and RHS-matrices needs to be specified with the `-m`, `-n` and `-k` options.
The shape of the LHS-matrix is MxK, and the shape of the RHS-matrix is KxN.
Run the matmul benchmark with matrix dimensions:

```
./kleidiai_benchmark matmul -m <M> -n <N> -k <K>
```

Example:

```
$ ./kleidiai_benchmark matmul -m 13 -n 17 -k 18
Run on (8 X 1800 MHz CPU s)
Load Average: 10.01, 10.06, 10.06
-----------------------------------------------------------------------------------------------------
Benchmark                                                           Time             CPU   Iterations
-----------------------------------------------------------------------------------------------------
matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod        123 ns          123 ns      1234567
matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod        123 ns          123 ns      1234567
matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm           123 ns          123 ns      1234567
matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm           123 ns          123 ns      1234567
matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm           123 ns          123 ns      1234567
matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm           123 ns          123 ns      1234567
```

### iMatmul Benchmark (chunked K)

Run the imatmul benchmark with matrix dimensions and chunking:

```
./kleidiai_benchmark imatmul -m <M> -n <N> -c <CHUNK_COUNT> -l <CHUNK_LENGTH>
```

Where:

- `-m`, `-n` are matrix dimensions (LHS: MxK, RHS: KxN)
- `-c` is the number of K chunks
- `-l` is the length of each K chunk

Example:

```
./kleidiai_benchmark imatmul -m 32 -n 32 -c 4 -l 16
Run on (12 X 24 MHz CPU s)
Load Average: 4.59, 3.95, 3.95
---------------------------------------------------------------------------------------------------------
Benchmark                                                               Time             CPU   Iterations
---------------------------------------------------------------------------------------------------------
imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa               123 ns          123 ns      1234567
imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa               123 ns          123 ns      1234567
imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa              123 ns          123 ns      1234567
imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa               123 ns          123 ns      1234567
imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa         123 ns          123 ns      1234567
imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa        123 ns          123 ns      1234567
```

### DWConv Benchmark (depthwise convolution)

Run the dwconv benchmark specifying the input shape, number of channels, and optional stride/padding/dilation.
The output dimensions are inferred automatically.

```
./kleidiai_benchmark dwconv --input_height <H> --input_width <W> --channels <C>
                            [--stride <S_h>,<S_w>] [--padding <P_top>,<P_bottom>,<P_left>,<P_right>]
                            [--dilation <D_h>,<D_w>]
```

- `--stride` takes two positive comma-separated integers `(stride_rows,stride_cols)` supplied as a single argument
  (for example, `--stride 1,1` or `--stride=1,2`).
- `--padding` takes four non-negative comma-separated integers `(top,bottom,left,right)` so you can control padding on
  both sides of each dimension.
- `--dilation` takes two positive comma-separated integers `(dilation_rows,dilation_cols)`.
  Defaults for stride and dilation are `1,1`, and padding defaults to `0,0,0,0`.
- **Note:** The currently registered DWConv micro-kernels only support `stride=1` and `dilation=1`.
  The benchmark driver will reject other values to avoid advertising unsupported functionality.

Example:

```
./kleidiai_benchmark dwconv --input_height 32 --input_width 32 --channels 64 --stride 1,1 --padding 1,1,1,1
Run on (12 X 24 MHz CPU s)
Load Average: 4.59, 3.95, 3.95
---------------------------------------------------------------------------------------------------------
Benchmark                                                               Time             CPU   Iterations
---------------------------------------------------------------------------------------------------------
kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla                123 ns          123 ns      1234567
```

### Filtering

Benchmarks can be filtered using the --benchmark_filter option, which accepts a regex. For example, to only run the sme2 microkernels:
(Note: The measurement results are placeholders)

```
./kleidiai_benchmark matmul  --benchmark_filter=sme2 -m 13 -n 17 -k 18
./kleidiai_benchmark imatmul --benchmark_filter=sme2 -m 13 -n 17 -c 1 -l 18
./kleidiai_benchmark dwconv  --benchmark_filter=sme2 --input_height 32 --input_width 32 --channels 64 --padding 1,1,1,1
Run on (8 X 1800 MHz CPU s)
Load Average: 10.09, 10.13, 10.09
-----------------------------------------------------------------------------------------------------
Benchmark                                                           Time             CPU   Iterations
-----------------------------------------------------------------------------------------------------
matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot        123 ns          123 ns      1234567
imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa           123 ns          123 ns      1234567
kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla           123 ns          123 ns      1234567
```

### Listing Available Benchmarks

To list all available benchmarks:

```
./kleidiai_benchmark  --benchmark_list_tests

```

Specify the micro-kernel operator to list all the benchmarks of a certain type.

```
./kleidiai_benchmark matmul  --benchmark_list_tests
./kleidiai_benchmark imatmul --benchmark_list_tests
./kleidiai_benchmark dwconv  --benchmark_list_tests
```

### Notes

This application uses [Google Benchmark](https://github.com/google/benchmark), so all options that Google Benchmark provides can be used.
To list the options provided use the `--help` flag or refer to the [user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md).

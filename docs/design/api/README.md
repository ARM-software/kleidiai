<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Micro-kernel API design

## Directory structure

The public API of each set of micro-kernels is organized in 2 header files:

- `kai/ukernels/<op>/kai_<uker>_types.h` contains the data structures that defines the micro-kernel API.
- `kai/ukernels/<op>/kai_<uker>.h` contains the list of micro-kernel creation functions.

where:

- `<op>` - the group of micro-kernels with closely related functionality. E.g. `matmul`, `dwconv`.
- `<uker>` - the micro-kernel with distinct functionality. E.g. `matmul`, `matmul_pack_lhs`, `matmul_pack_rhs`.

## API definition

The micro-kernel API definition is implemented in `kai/ukernels/<op>/kai_<uker>_types.h`. It consists of 3 main components:

- `kai_<uker>_api` structure containing all the function pointers.
- `kai_<uker>_config` structure containing the set of parameters to configure the functionality of the micro-kernel. All functions in the micro-kernel API take the config as an input.
- `kai_<uker>_args` structure containing the information to run the micro-kernel. It can include parameters that affects the functionality of the micro-kernel, but only for the `run` function.

### API structure

The API structure `kai_<uker>_api` contains the following function pointers:

| Name | Description |
|-|-|
| `run` | It runs the micro-kernel given the micro-kernel configuration and run arguments. |
| `get_<dim>_step` | These functions are used to get the alignment requirement of a given dimension `<dim>` so that the callers can split the workload into smaller chunks and invoke the micro-kernel in multiple threads. |
| `get_<operand>_stride_<dim>` | These functions calculate the default stride in dimension `<dim>` in bytes of operand `<operand>`. The stride is supplied to the run function so that the micro-kernel can access a portion of the input operands to compute a small portion of the output. |
| `get_<operand>_offset` | These functions calculate the pointer offset in bytes for operand `<operand>` given the start coordinate and the strides so that the micro-kernel can access the right portion of the operands. |
| `get_<operand>_size` | These functions are only provided for output operands so that the user can allocate sufficient memory to store the result. |

### Configuration structure

The configuration structure is organized in multiple level of structures, all of them has the type name with `kai_<uker>_` prefix and `_config` suffix. The specific design of the configuration structure is highly dependent on each micro-kernel family, but the top-level structure `kai_<uker>_config` always consists of the following fields:

| Name | Type | Description |
|-|-|-|
| `<info>` | `kai_<uker>_<info>_config` | The configuration of a certain aspect (`<info>`) of the micro-kernel. |

The configuration structure should be used only when there are certain configurable parameters that affect the functionality of the micro-kernel and all its supporting functions. If a parameter only affects the functionality of the `run` function, the run arguments structure is the prefered place to specify such parameter.

### Run arguments structure

The run argument structure is organized in multiple level of structures, all of them has the type name with `kai_<uker>_` prefix and `_args` suffix. The top-level structure `kai_<uker>_args` consists of the following fields:

| Name | Type | Description |
|-|-|-|
| `flags` | `uint64_t` | The bit flags that can be used to control the functionality of the micro-kernel. If no flag is used, **this field must be set to 0**. The possible flag values are provided in `kai_<uker>_args_flags` enum. |
| `shape` | `kai_<uker>_shape_args` | The problem shape. |
| `operands` | `kai_<uker>_operands_args` | The run-time information of each operand involved in the execution of the micro-kernel. |

#### Problem shape

Depending on the specific functionality of the micro-kernel family, the problem shape can be defined differently. For example:

- Matrix multiplication micro-kernel takes `m`, `n`, and `k` as the problem shape.
- LHS packing micro-kernel for matrix multiplication takes `m` and `k` as the problem shape.

#### Operands

The list of operands is highly dependent on the micro-kernel functionality and not all of them are used by a given micro-kernel. To allow future extension of the list of operands, as well as the future extension of the definition of the operand, each operand has its own distict structure.

The operands structure `kai_<uker>_operands_args` contains a list of fields named `<operand>` of type `kai_<uker>_<operand>_args` containing:

| Name | Type | Description |
|-|-|-|
| `ptr` | `const void*` or `void*` | The function pointer to the data buffer. Constantness is determined by whether this is input, output or scratch operands. Only `void` pointer is used since the same API is shared by micro-kernels operating on different data type. |
| `stride_<dim>` | `size_t` | The strides in bytes of dimension `<dim>`. Depending on the rank of the operand and the actual data format, different number of strides is needed. |

## API creation

The API structure is the primary way to access each individual function of a given micro-kernel. Each micro-kernel variant provides a function to return the API structure to the caller. The list of all micro-kernel variants of micro-kernel `<uker>` is provided in `kai/ukernels/<op>/kai_<uker>.h`.

Each micro-kernel variant `<variant>` provides the following functions:

| Name | Return type | Description |
|-|-|-|
| `kai_<uker>_<variant>` | `kai_<uker>_uker_api` | Returns the API structure containing the function pointers to the micro-kernel variant. |

Notes:

- The creation function always return the API structure **by value**.
- The creation function does not take any input. If the functionality of the micro-kernel is configurable, the caller needs to populate necessary information to the configuration structure and/or the run arguments structure.

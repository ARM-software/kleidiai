//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_dims.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

namespace kai::test {

using MatMulKernel = KernelWrapper<MatMulShape, MatMulSlot>;
using MatMulPackKernel = KernelWrapper<MatShape, MatMulSlot>;

using MatMulKernelPtr = std::unique_ptr<MatMulKernel>;
using MatMulPackKernelPtr = std::unique_ptr<MatMulPackKernel>;

}  // namespace kai::test

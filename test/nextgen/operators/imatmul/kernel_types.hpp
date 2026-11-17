//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/imatmul/imatmul_dims.hpp"
#include "test/nextgen/operators/imatmul/imatmul_slots.hpp"

namespace kai::test {

using IMatMulKernel = KernelWrapper<IMatMulShape, IMatMulSlot>;
using IMatMulPackKernel = KernelWrapper<IMatShape, IMatMulSlot>;

using IMatMulKernelPtr = std::unique_ptr<IMatMulKernel>;
using IMatMulPackKernelPtr = std::unique_ptr<IMatMulPackKernel>;

}  // namespace kai::test

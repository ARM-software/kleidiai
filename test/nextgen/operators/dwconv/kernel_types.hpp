//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/dwconv/dwconv_dims.hpp"
#include "test/nextgen/operators/dwconv/dwconv_slots.hpp"

namespace kai::test {

using DwConvKernel = KernelWrapper<DwConvShape, DwConvSlot>;
using DwConvPackKernel = KernelWrapper<DwConvPackShape, DwConvSlot>;

using DwConvKernelPtr = std::unique_ptr<DwConvKernel>;
using DwConvPackKernelPtr = std::unique_ptr<DwConvPackKernel>;

}  // namespace kai::test

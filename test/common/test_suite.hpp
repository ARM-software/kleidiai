//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <string_view>

// clang-format off
#define UKERNEL_MATMUL_VARIANT(name)          \
    {kai_get_m_step_matmul_## name,            \
     kai_get_n_step_matmul_## name,            \
     kai_get_mr_matmul_## name,                \
     kai_get_nr_matmul_## name,                \
     kai_get_kr_matmul_## name,                \
     kai_get_sr_matmul_## name,                \
     kai_get_lhs_packed_offset_matmul_## name, \
     kai_get_rhs_packed_offset_matmul_## name, \
     kai_get_dst_offset_matmul_## name,        \
     kai_get_dst_size_matmul_## name,          \
     kai_run_matmul_## name}

#define UKERNEL_MATMUL_PACK_VARIANT(name, features_check, lhs_pack, rhs_pack)                           \
    {                                                                                                   \
        {UKERNEL_MATMUL_VARIANT(name), "kai_matmul_" #name, features_check},                            \
        {                                                                                               \
            kai_get_lhs_packed_size_##lhs_pack,                                                         \
            kai_get_rhs_packed_size_##rhs_pack,                                                         \
            kai_get_lhs_packed_offset_##lhs_pack,                                                       \
            kai_get_rhs_packed_offset_##rhs_pack,                                                       \
            kai_get_lhs_offset_##lhs_pack,                                                              \
            kai_get_rhs_offset_##rhs_pack,                                                              \
            kai_run_##lhs_pack,                                                                         \
            kai_run_##rhs_pack                                                                          \
        }                                                                                               \
    }
// clang-format on

namespace kai::test {

template <typename T>
struct UkernelVariant {
    /// Interface for testing variant.
    T interface;

    /// Name of the test variant.
    std::string_view name{};

    /// Check if CPU supports required features.
    ///
    /// @return Supported (true) or not supported (false).
    std::function<bool(void)> fn_is_supported;

    UkernelVariant(T interface, const std::string_view name, const std::function<bool(void)>& fn_is_supported) :
        interface(interface), name(name), fn_is_supported(fn_is_supported) {
    }
};

template <typename T, typename P>
struct UkernelPackVariant {
    /// Interface for testing variant.
    UkernelVariant<T> ukernel;
    P pack_interface;
};

}  // namespace kai::test

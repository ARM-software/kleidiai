//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/matmul.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/functions/fused_mul_add.hpp"

namespace kai::test {

namespace {

template <typename Lhs, typename Rhs, typename Acc>
[[nodiscard]] Buffer matmul_nt_t(
    size_t shape_m, size_t shape_n, size_t shape_k, Span<const std::byte> lhs, Span<const std::byte> rhs) {
    Buffer dst(shape_m * round_up_division(shape_n * size_in_bits<Acc>, 8), 0);

    for (size_t row = 0; row < shape_m; ++row) {
        for (size_t col = 0; col < shape_n; ++col) {
            Acc acc = static_cast<Acc>(0);

            for (size_t depth = 0; depth < shape_k; ++depth) {
                const Acc lhs_value = static_cast<Acc>(read_2d<Lhs>(lhs, shape_k, row, depth));
                const Acc rhs_value = static_cast<Acc>(read_2d<Rhs>(rhs, shape_k, col, depth));

                if constexpr (std::is_floating_point_v<Acc>) {
                    acc = fused_mul_add<Acc>(lhs_value, rhs_value, acc);
                } else {
                    acc += lhs_value * rhs_value;
                }
            }

            write_2d<Acc>(dst.view(), shape_n, row, col, acc);
        }
    }

    return dst;
}

struct MatMulEntry {
    DataType lhs_dtype;
    DataType rhs_dtype;
    DataType acc_dtype;
    MatMulFn fn;
};

// Binds a runtime DataType value to its matching C++ storage type.
template <DataType DType, typename Type>
struct TypedData {
    static constexpr DataType dtype = DType;
    using type = Type;
};

using FP32 = TypedData<DataType::FP32, float>;
using U8 = TypedData<DataType::U8, uint8_t>;
using I32 = TypedData<DataType::I32, int32_t>;

// Creates one dispatch table entry mapping a DataType combination to its reference implementation.
template <typename Lhs, typename Rhs, typename Acc>
[[nodiscard]] constexpr MatMulEntry make_entry() {
    return {
        Lhs::dtype,
        Rhs::dtype,
        Acc::dtype,
        &matmul_nt_t<typename Lhs::type, typename Rhs::type, typename Acc::type>,
    };
}

}  // namespace

MatMulFn make_matmul_nt_t(DataType lhs_dtype, DataType rhs_dtype, DataType acc_dtype) {
    static constexpr std::array entries = {
        make_entry<FP32, FP32, FP32>(),
        make_entry<U8, U8, I32>(),
    };

    for (const auto& [e_lhs, e_rhs, e_acc, fn] : entries) {
        if (lhs_dtype == e_lhs && rhs_dtype == e_rhs && acc_dtype == e_acc) {
            return fn;
        }
    }

    KAI_TEST_ERROR("Matmul data type combination is not implemented.");
}

}  // namespace kai::test

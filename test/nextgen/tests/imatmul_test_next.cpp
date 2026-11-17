//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/enum_utils.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/round.hpp"
#include "test/common/seed.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/common/test_config.hpp"
#include "test/nextgen/common/test_registry.hpp"
#include "test/nextgen/operators/imatmul/imatmul_config.hpp"
#include "test/nextgen/operators/imatmul/imatmul_operator.hpp"
#include "test/nextgen/operators/imatmul/imatmul_tb.hpp"

namespace kai::test {

namespace {

struct IMatMulDistribution {
    /// Seeds fixture generation using the operator name.
    explicit IMatMulDistribution(std::string_view operator_name) :
        m_rng(seed_stream("IMatMulNext::setup:" + std::string(operator_name))()) {
    }

    Rng m_rng;

    /// Partitioned distributions for matrix dimensions.
    std::uniform_int_distribution<size_t> m_small_shape_dist{1, 3};
    std::uniform_int_distribution<size_t> m_medium_shape_dist{4, 24};
    std::uniform_int_distribution<size_t> m_large_shape_dist{25, 150};

    /// Partitioned distributions for the number of K chunks.
    std::uniform_int_distribution<size_t> m_small_k_chunk_count_dist{1, 3};
    std::uniform_int_distribution<size_t> m_medium_k_chunk_count_dist{4, 24};
    std::uniform_int_distribution<size_t> m_large_k_chunk_count_dist{25, 64};

    /// Partitioned distributions for K chunk lengths.
    std::uniform_int_distribution<size_t> m_small_k_chunk_length_dist{1, 3};
    std::uniform_int_distribution<size_t> m_medium_k_chunk_length_dist{4, 24};
    std::uniform_int_distribution<size_t> m_large_k_chunk_length_dist{25, 128};

    std::uniform_real_distribution<float> m_probability_dist{0.0F, 1.0F};

    /// Element offset sampling
    std::bernoulli_distribution m_has_source_offset_dist{0.9};
    std::uniform_int_distribution<size_t> m_source_offset_dist{1, 128};
};

enum class ShapeSize {
    SMALL,
    MEDIUM,
    LARGE,
};

/// Samples one valid enum value with equal probability.
template <typename T>
T sample_mode(Rng& rng) {
    static constexpr auto modes = enum_values<T>();
    static_assert(!modes.empty(), "Cannot sample an empty enum.");
    std::uniform_int_distribution<size_t> mode_dist(0, modes.size() - 1);
    return modes[mode_dist(rng)];
}

struct IMatMulFixtureParams {
    size_t iteration_no;

    size_t shape_m;
    size_t shape_n;
    size_t shape_k_chunk_count;
    size_t shape_k_chunk_length;
    IMatMulInputMode input_mode;
    size_t source_offset;

    const IMatMulOperator* op;

    /// Builds a descriptive name for these fixture parameters.
    ///
    /// @return A unique name string used for test registration and caching.
    [[nodiscard]] std::string name() const {
        return std::string(op->name) +                                     //
            ",m=" + std::to_string(shape_m) +                              //
            ",n=" + std::to_string(shape_n) +                              //
            ",k_chunk_count=" + std::to_string(shape_k_chunk_count) +      //
            ",k_chunk_length=" + std::to_string(shape_k_chunk_length) +    //
            ",input=" + std::to_string(static_cast<size_t>(input_mode)) +  //
            ",offset=" + std::to_string(source_offset) +                   //
            ",iteration=" + std::to_string(iteration_no);
    }
};

struct IMatMulTestParams {
    MatrixPortion portion;

    /// Builds a descriptive name for the selected matrix portion.
    ///
    /// @return A name string encoding the portion location and size.
    [[nodiscard]] std::string name() const {
        return "start_row=" + std::to_string(portion.start_row()) + ",size_row=" + std::to_string(portion.height()) +
            ",start_col=" + std::to_string(portion.start_col()) + ",size_col=" + std::to_string(portion.width());
    }
};

class IMatMulFixture : public testing::Test {
public:
    /// Creates a fixture bound to a specific parameter set.
    ///
    /// @param[in] params The fixture parameters used for this instance.
    explicit IMatMulFixture(const IMatMulFixtureParams& params) : m_fixture_params(params) {
    }

protected:
    /// Returns the parameters for this fixture instance.
    ///
    /// @return A constant reference to the fixture parameters.
    [[nodiscard]] const IMatMulFixtureParams& fixture_params() const {
        return m_fixture_params;
    }

    /// Returns the cached test bench for this fixture, creating it if needed.
    ///
    /// The test bench is keyed by the fixture name, and test data is generated
    /// once per unique fixture to keep results deterministic.
    ///
    /// @return A reference to the cached test bench.
    [[nodiscard]] IMatMulTb& test_bench() {
        const std::string name = m_fixture_params.name();

        // If the test has already been created, uses it.
        const auto it = test_benches.find(name);

        if (it != test_benches.end()) {
            return it->second;
        }

        // Creates a new test if it hasn't been created.
        IMatMulTb test(
            m_fixture_params.shape_m, m_fixture_params.shape_n, m_fixture_params.shape_k_chunk_count,
            m_fixture_params.shape_k_chunk_length, m_fixture_params.input_mode, m_fixture_params.source_offset,
            m_fixture_params.op);

        const std::string seed_key = "IMatMulNext::testbench:" + name;
        auto& feed = seed_stream(seed_key);
        Rng rng(feed());

        test.generate_test_data(rng);

        return test_benches[name] = std::move(test);
    }

private:
    static std::unordered_map<std::string, IMatMulTb> test_benches;

    IMatMulFixtureParams m_fixture_params;
};

std::unordered_map<std::string, IMatMulTb> IMatMulFixture::test_benches;

class IMatMulPackLhsTest : public IMatMulFixture {
public:
    /// Creates an LHS packing test instance.
    ///
    /// @param[in] fixture_params The fixture parameters shared across tests.
    /// @param[in] test_params The parameters describing the matrix portion.
    explicit IMatMulPackLhsTest(const IMatMulFixtureParams& fixture_params, const IMatMulTestParams& test_params) :
        IMatMulFixture(fixture_params), m_test_params(test_params) {
    }

private:
    /// Runs the LHS packing validation for the selected portion.
    void TestBody() override {
        IMatMulTb& test = test_bench();
        const IMatMulFixtureParams& params = fixture_params();
        const MatrixPortion& portion = m_test_params.portion;

        const auto [step_m, step_k_chunk_count, step_k_chunk_length] = test.lhs_packing_steps();
        KAI_TEST_ASSERT(step_k_chunk_length == params.shape_k_chunk_length);
        const Rect rect = portion.compute_portion(
            params.shape_m, params.shape_k_chunk_count * params.shape_k_chunk_length, step_m,
            step_k_chunk_count * step_k_chunk_length);

        const size_t start_m = rect.start_row();
        const size_t start_k_chunk_count = rect.start_col() / params.shape_k_chunk_length;
        const size_t start_k_chunk_length = 0;
        const size_t size_m = rect.height();
        const size_t size_k_chunk_count = rect.width() / params.shape_k_chunk_length;
        const size_t size_k_chunk_length = params.shape_k_chunk_length;

        test.test_lhs_packing(
            start_m, start_k_chunk_count, start_k_chunk_length, size_m, size_k_chunk_count, size_k_chunk_length);
    }

private:
    IMatMulTestParams m_test_params;
};

class IMatMulPackRhsTest : public IMatMulFixture {
public:
    using IMatMulFixture::IMatMulFixture;

private:
    /// Reports that RHS packing validation is not implemented yet.
    void TestBody() override {
        FAIL() << "Indirect matmul RHS packing validation is not implemented yet.";
    }
};

class IMatMulComputeTest : public IMatMulFixture {
public:
    using IMatMulFixture::IMatMulFixture;

private:
    /// Reports that indirect matmul validation is not implemented yet.
    void TestBody() override {
        FAIL() << "Indirect matmul compute validation is not implemented yet.";
    }
};

/// Returns the matrix portions used in tests.
///
/// @return An array containing full and corner portions.
auto make_test_portions() {
    return std::array{
        MatrixPortion(0, 0, 1, 1),             // Full matrix.
        MatrixPortion(0, 0, 0.25, 0.25),       // Top-left corner.
        MatrixPortion(0.75, 0.75, 0.25, 0.25)  // Bottom-right corner.
    };
}

/// Picks a shape size range for the fixture.
ShapeSize pick_shape_size(IMatMulDistribution& dist) {
    const float probability = dist.m_probability_dist(dist.m_rng);
    if (probability < 0.1F) {
        return ShapeSize::SMALL;
    }
    if (probability >= 0.95F) {
        return ShapeSize::LARGE;
    }
    return ShapeSize::MEDIUM;
}

/// Picks the matrix dimensions and K chunk geometry from the selected size range.
///
/// @param[in] shape_size The shape size range.
/// @param[in,out] dist The shared distribution context.
///
/// @return M, N, K chunk count, and K chunk length, in that order.
std::tuple<size_t, size_t, size_t, size_t> pick_shape(ShapeSize shape_size, IMatMulDistribution& dist) {
    auto* shape_dist = &dist.m_medium_shape_dist;
    auto* k_chunk_count_dist = &dist.m_medium_k_chunk_count_dist;
    auto* k_chunk_length_dist = &dist.m_medium_k_chunk_length_dist;
    switch (shape_size) {
        case ShapeSize::SMALL:
            shape_dist = &dist.m_small_shape_dist;
            k_chunk_count_dist = &dist.m_small_k_chunk_count_dist;
            k_chunk_length_dist = &dist.m_small_k_chunk_length_dist;
            break;
        case ShapeSize::MEDIUM:
            break;
        case ShapeSize::LARGE:
            shape_dist = &dist.m_large_shape_dist;
            k_chunk_count_dist = &dist.m_large_k_chunk_count_dist;
            k_chunk_length_dist = &dist.m_large_k_chunk_length_dist;
            break;
        default:
            KAI_TEST_ASSERT_MSG(false, "Unexpected indirect matmul shape size.");
            break;
    }
    const size_t shape_m = (*shape_dist)(dist.m_rng);
    const size_t shape_n = (*shape_dist)(dist.m_rng);
    const size_t k_chunk_count = (*k_chunk_count_dist)(dist.m_rng);
    const size_t k_chunk_length = (*k_chunk_length_dist)(dist.m_rng);
    return {shape_m, shape_n, k_chunk_count, k_chunk_length};
}

/// Selects a valid fixture configuration for an operator and portion.
///
/// @param[in] shape_no The shape iteration index.
/// @param[in] op The operator to test.
/// @param[in] portion The output matrix portion to validate.
/// @param[in,out] dist_ctx The shared distribution context.
///
/// @return A populated fixture parameter set.
IMatMulFixtureParams pick_fixture(
    size_t shape_no,               //
    const IMatMulOperator& op,     //
    const MatrixPortion& portion,  //
    IMatMulDistribution& dist_ctx) {
    size_t shape_m = 0;
    size_t shape_n = 0;
    size_t shape_k_chunk_count = 0;
    size_t shape_k_chunk_length = 0;

    const ShapeSize shape_size = pick_shape_size(dist_ctx);

    static constexpr uint32_t max_attempts = 500'000;
    uint32_t attempts = 0;
    while (true) {
        KAI_TEST_ASSERT_MSG(attempts <= max_attempts, "Unable to find matching shape after _many_ tries");
        ++attempts;
        std::tie(shape_m, shape_n, shape_k_chunk_count, shape_k_chunk_length) = pick_shape(shape_size, dist_ctx);

        if (op.is_shape_suitable(shape_m, shape_n, shape_k_chunk_count, shape_k_chunk_length, portion)) {
            break;
        }
    }

    const auto input_mode = sample_mode<IMatMulInputMode>(dist_ctx.m_rng);
    const bool has_source_offset = dist_ctx.m_has_source_offset_dist(dist_ctx.m_rng);
    const size_t source_offset_elements = has_source_offset ? dist_ctx.m_source_offset_dist(dist_ctx.m_rng) : 0;
    const size_t element_bytes = round_up_division(data_type_size_in_bits(op.lhs_dtype), 8);
    const size_t source_offset = source_offset_elements * element_bytes;

    return {
        shape_no, shape_m, shape_n, shape_k_chunk_count, shape_k_chunk_length, input_mode, source_offset, &op,
    };
}

/// Registers packing and compute tests for a given operator and portion.
///
/// @param[in] test_suite_name The test suite name to register with.
/// @param[in] op The operator under test.
/// @param[in] fixture_params The fixture parameter set.
/// @param[in] portion The output matrix portion for this test instance.
void register_operator_test(
    const std::string& test_suite_name,          //
    const IMatMulOperator& op,                   //
    const IMatMulFixtureParams& fixture_params,  //
    const MatrixPortion& portion) {
    const IMatMulTestParams test_params{
        portion,
    };

    const std::string desc = fixture_params.name() + "," + test_params.name();

    if (op.pack_lhs.has_value()) {
        KAI_REGISTER_TEST(
            IMatMulFixture, IMatMulPackLhsTest, test_suite_name.c_str(), ("PackLhs/" + desc).c_str(), fixture_params,
            test_params);
    }

    if (op.pack_rhs.has_value()) {
        KAI_REGISTER_TEST(
            IMatMulFixture, IMatMulPackRhsTest, test_suite_name.c_str(), ("PackRhs/" + desc).c_str(), fixture_params);
    }

    if (op.imatmul.has_value()) {
        KAI_REGISTER_TEST(
            IMatMulFixture, IMatMulComputeTest, test_suite_name.c_str(), ("IMatMul/" + desc).c_str(), fixture_params);
    }
}

/// Registers all IMatMulNext tests with the test registry.
const auto imatmul_tests_setup = TestRegistry::register_setup([]() {
    const size_t num_shapes_per_op = TestConfig::Get().num_shapes();
    const auto test_portions = make_test_portions();

    const Span<const IMatMulOperator> available_operators = get_available_imatmul_operators();

    for (const IMatMulOperator& op : available_operators) {
        if (!op.is_cpu_supported()) {
            continue;
        }

        const std::string test_suite_name = "IMatMulNext";
        // Each operator has its own stream, independent of registration order.
        IMatMulDistribution dist_ctx(op.name);

        for (size_t shape_no = 0; shape_no < num_shapes_per_op; ++shape_no) {
            for (const MatrixPortion& portion : test_portions) {
                const IMatMulFixtureParams fixture = pick_fixture(shape_no, op, portion, dist_ctx);
                register_operator_test(test_suite_name, op, fixture, portion);
            }
        }
    }
});

}  // namespace

}  // namespace kai::test

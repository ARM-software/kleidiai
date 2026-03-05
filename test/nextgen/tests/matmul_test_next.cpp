//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/seed.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/common/test_registry.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/matmul_operator.hpp"
#include "test/nextgen/operators/matmul/matmul_tb.hpp"

namespace kai::test {

namespace {

struct MatMulDistribution {
    Rng m_rng{seed_stream("MatMulNext::setup")()};
    std::uniform_int_distribution<size_t> m_shape_dist{1, 150};
    std::uniform_real_distribution<float> m_probability_dist{0.0F, 1.0F};
    std::uniform_real_distribution<float> m_dist_70_to_100{0.7F, 1.0F};
    std::uniform_real_distribution<float> m_dist_0_to_70{0.0F, 0.7F};
};

struct MatMulFixtureParams {
    size_t iteration_no;

    size_t shape_m;
    size_t shape_n;
    size_t shape_k;
    MatMulBiasMode bias_mode;
    float clamp_ratio;

    const MatMulOperator* op;

    /// Builds a descriptive name for these fixture parameters.
    ///
    /// @return A unique name string used for test registration and caching.
    [[nodiscard]] std::string name() const {
        return std::string(op->name) + ",m=" + std::to_string(shape_m) + ",n=" + std::to_string(shape_n) +
            ",k=" + std::to_string(shape_k) + ",bias=" + matmul_bias_mode_name(bias_mode) +
            ",clamp_ratio=" + std::to_string(clamp_ratio) + ",iteration=" + std::to_string(iteration_no);
    }
};

struct MatMulTestParams {
    MatrixPortion portion;

    /// Builds a descriptive name for the selected matrix portion.
    ///
    /// @return A name string encoding the portion location and size.
    [[nodiscard]] std::string name() const {
        return "start_m=" + std::to_string(portion.start_row()) + ",size_m=" + std::to_string(portion.height()) +
            ",start_n=" + std::to_string(portion.start_col()) + ",size_n=" + std::to_string(portion.width());
    }
};

class MatMulFixture : public testing::Test {
public:
    /// Creates a fixture bound to a specific parameter set.
    ///
    /// @param[in] params The fixture parameters used for this instance.
    explicit MatMulFixture(const MatMulFixtureParams& params) : m_fixture_params(params) {
    }

protected:
    /// Returns the parameters for this fixture instance.
    ///
    /// @return A constant reference to the fixture parameters.
    [[nodiscard]] const MatMulFixtureParams& fixture_params() const {
        return m_fixture_params;
    }

    /// Returns the cached test bench for this fixture, creating it if needed.
    ///
    /// The test bench is keyed by the fixture name, and test data is generated
    /// once per unique fixture to keep results deterministic.
    ///
    /// @return A reference to the cached test bench.
    [[nodiscard]] MatMulTb& test_bench() {
        const std::string name = m_fixture_params.name();

        // If the test has already been created, uses it.
        const auto it = test_benches.find(name);

        if (it != test_benches.end()) {
            return it->second;
        }

        // Creates a new test if it hasn't been created.
        MatMulTb test(
            m_fixture_params.shape_m, m_fixture_params.shape_n, m_fixture_params.shape_k, m_fixture_params.bias_mode,
            m_fixture_params.clamp_ratio, m_fixture_params.op);

        const std::string seed_key = "MatMulNext::testbench:" + name;
        auto& feed = seed_stream(seed_key);
        Rng rng(feed());

        test.generate_test_data(rng);

        return test_benches[name] = std::move(test);
    }

private:
    static std::unordered_map<std::string, MatMulTb> test_benches;

    MatMulFixtureParams m_fixture_params;
};

std::unordered_map<std::string, MatMulTb> MatMulFixture::test_benches;

class MatMulPackLhsTest : public MatMulFixture {
public:
    /// Creates an LHS packing test instance.
    ///
    /// @param[in] fixture_params The fixture parameters shared across tests.
    /// @param[in] test_params The parameters describing the matrix portion.
    explicit MatMulPackLhsTest(const MatMulFixtureParams& fixture_params, const MatMulTestParams& test_params) :
        MatMulFixture(fixture_params), m_test_params(test_params) {
    }

    /// Runs the LHS packing validation for the selected portion.
    void TestBody() override {
        MatMulTb& test = test_bench();
        const MatMulFixtureParams& params = fixture_params();
        const MatrixPortion& portion = m_test_params.portion;

        const auto [step_m, step_k] = test.lhs_packing_steps();
        const Rect rect = portion.compute_portion(params.shape_m, params.shape_k, step_m, step_k);

        const size_t start_m = rect.start_row();
        const size_t start_k = rect.start_col();
        const size_t size_m = rect.height();
        const size_t size_k = rect.width();

        test.test_lhs_packing(start_m, start_k, size_m, size_k);
    }

private:
    MatMulTestParams m_test_params;
};

class MatMulPackRhsTest : public MatMulFixture {
public:
    /// Creates an RHS packing test instance.
    ///
    /// @param[in] fixture_params The fixture parameters shared across tests.
    /// @param[in] test_params The parameters describing the matrix portion.
    explicit MatMulPackRhsTest(const MatMulFixtureParams& fixture_params, const MatMulTestParams& test_params) :
        MatMulFixture(fixture_params), m_test_params(test_params) {
    }

    /// Runs the RHS packing validation for the selected portion.
    void TestBody() override {
        MatMulTb& test = test_bench();
        const MatMulFixtureParams& params = fixture_params();
        const MatrixPortion& portion = m_test_params.portion;

        const auto [step_n, step_k] = test.rhs_packing_steps();
        const Rect rect = portion.compute_portion(params.shape_n, params.shape_k, step_n, step_k);

        const size_t start_n = rect.start_row();
        const size_t start_k = rect.start_col();
        const size_t size_n = rect.height();
        const size_t size_k = rect.width();

        test.test_rhs_packing(start_n, start_k, size_n, size_k);
    }

private:
    MatMulTestParams m_test_params;
};

class MatMulMatMulTest : public MatMulFixture {
public:
    /// Creates a matrix multiplication test instance.
    ///
    /// @param[in] fixture_params The fixture parameters shared across tests.
    /// @param[in] test_params The parameters describing the matrix portion.
    explicit MatMulMatMulTest(const MatMulFixtureParams& fixture_params, const MatMulTestParams& test_params) :
        MatMulFixture(fixture_params), m_test_params(test_params) {
    }

    /// Runs the matrix multiplication validation for the selected portion.
    void TestBody() override {
        MatMulTb& test = test_bench();
        const MatMulFixtureParams& params = fixture_params();
        const MatrixPortion& portion = m_test_params.portion;

        const auto [step_m, step_n] = test.matmul_steps();
        const Rect rect = portion.compute_portion(params.shape_m, params.shape_n, step_m, step_n);

        const size_t start_m = rect.start_row();
        const size_t start_n = rect.start_col();
        const size_t size_m = rect.height();
        const size_t size_n = rect.width();

        test.test_matmul(start_m, start_n, size_m, size_n);
    }

private:
    MatMulTestParams m_test_params;
};

struct BiasSelector {
public:
    /// Builds a bias-mode selector for a given operator.
    ///
    /// @param[in] op The operator providing supported bias modes.
    explicit BiasSelector(const MatMulOperator& op) {
        // Separates no-bias mode from the list of supported bias modes.
        //
        // Reason: no-bias and with-bias cases are chosen by a distribution,
        // and if the with-bias case is chosen, each with-bias modes will be chosen
        // by another distribution.
        has_no_bias_mode =
            std::find(op.supported_bias_modes.begin(), op.supported_bias_modes.end(), MatMulBiasMode::NO_BIAS) !=
            op.supported_bias_modes.end();

        std::copy_if(
            op.supported_bias_modes.begin(), op.supported_bias_modes.end(), std::back_inserter(with_bias_modes),
            [](MatMulBiasMode bias_mode) { return bias_mode != MatMulBiasMode::NO_BIAS; });
        has_with_bias_mode = !with_bias_modes.empty();
        if (has_with_bias_mode) {
            with_bias_dist = std::uniform_int_distribution<size_t>(0, with_bias_modes.size() - 1);
        }

        KAI_TEST_ASSERT_MSG(has_no_bias_mode || has_with_bias_mode, "At least one bias mode is needed!");
    }

    /// Picks a bias mode based on the configured distributions.
    ///
    /// @param[in,out] dist_ctx The shared distribution context.
    ///
    /// @return The selected bias mode.
    MatMulBiasMode pick(MatMulDistribution& dist_ctx) {
        // Bias mode:
        //   * 70% of tests have bias.
        //   * 30% of tests have no bias.
        //
        // If there is no bias mode, with bias mode is always chosen.
        // If with bias mode is chosen, each bias modes (except no bias mode)
        // will be chosen uniformly.
        const float bias_prob = dist_ctx.m_probability_dist(dist_ctx.m_rng);
        const bool with_bias = !has_no_bias_mode || (has_with_bias_mode && bias_prob < 0.7F);

        if (with_bias) {
            const size_t bias_mode_idx = with_bias_dist(dist_ctx.m_rng);
            return with_bias_modes.at(bias_mode_idx);
        }

        return MatMulBiasMode::NO_BIAS;
    }

private:
    bool has_no_bias_mode = false;
    bool has_with_bias_mode = false;
    std::vector<MatMulBiasMode> with_bias_modes;
    std::uniform_int_distribution<size_t> with_bias_dist;
};

/// Returns the output matrix portions used in tests.
///
/// @return An array containing full and corner portions.
std::array<MatrixPortion, 3> make_output_portions() {
    return {
        MatrixPortion(0, 0, 1, 1),        // Full matrix.
        MatrixPortion(0, 0, 0.25, 0.25),  // Top-left corner.
        MatrixPortion(0.75, 0.75, 1, 1)   // Bottom-right corner.
    };
}

/// Picks a clamp ratio following the configured probabilities.
///
/// @param[in,out] dist_ctx The shared distribution context.
///
/// @return A clamp ratio in the range [0, 1], where 1.0 means no clamping.
float pick_clamp_ratio(MatMulDistribution& dist_ctx) {
    // Clamping range:
    //   * 20% of tests have no clamping.
    //   * 40% of tests have clamping range between 70% to 100% the output range.
    //   * 40% of tests have clamping range between 0% to 70% the output range.
    const float clamp_prob = dist_ctx.m_probability_dist(dist_ctx.m_rng);

    const bool no_clamp = clamp_prob < 0.2F;
    const bool clamp_70_to_100 = clamp_prob >= 0.2F && clamp_prob < 0.6F;
    const bool clamp_0_to_70 = clamp_prob >= 0.6F;

    if (no_clamp) {
        return 1.0F;
    }

    if (clamp_70_to_100) {
        return dist_ctx.m_dist_70_to_100(dist_ctx.m_rng);
    }

    KAI_TEST_ASSERT(clamp_0_to_70);
    return dist_ctx.m_dist_0_to_70(dist_ctx.m_rng);
}

/// Selects a valid fixture configuration for an operator and portion.
///
/// @param[in] shape_no The shape iteration index.
/// @param[in] op The operator to test.
/// @param[in] portion The output matrix portion to validate.
/// @param[in,out] dist_ctx The shared distribution context.
/// @param[in,out] bias_selector The bias selection helper.
///
/// @return A populated fixture parameter set.
MatMulFixtureParams pick_fixture(
    size_t shape_no,               //
    const MatMulOperator& op,      //
    const MatrixPortion& portion,  //
    MatMulDistribution& dist_ctx,  //
    BiasSelector& bias_selector) {
    size_t shape_m = 0;
    size_t shape_n = 0;
    size_t shape_k = 0;
    MatMulBiasMode bias_mode = MatMulBiasMode::NO_BIAS;
    float clamp_ratio = 0.0F;

    static constexpr uint32_t max_attempts = 10'000;
    uint32_t attempts = 0;
    while (true) {
        KAI_TEST_ASSERT_MSG(attempts++ <= max_attempts, "Unable to find matching shape after _many_ tries");
        shape_m = dist_ctx.m_shape_dist(dist_ctx.m_rng);
        shape_n = dist_ctx.m_shape_dist(dist_ctx.m_rng);
        shape_k = dist_ctx.m_shape_dist(dist_ctx.m_rng);

        if (!op.is_shape_suitable(shape_m, shape_n, shape_k, portion)) {
            continue;
        }

        bias_mode = bias_selector.pick(dist_ctx);
        clamp_ratio = pick_clamp_ratio(dist_ctx);

        break;
    }

    return {
        shape_no, shape_m, shape_n, shape_k, bias_mode, clamp_ratio, &op,
    };
}

/// Registers pack and matmul tests for a given operator and portion.
///
/// @param[in] test_suite_name The test suite name to register with.
/// @param[in] op The operator under test.
/// @param[in] fixture_params The fixture parameter set.
/// @param[in] portion The output matrix portion for this test instance.
void register_operator_test(
    const std::string& test_suite_name,         //
    const MatMulOperator& op,                   //
    const MatMulFixtureParams& fixture_params,  //
    const MatrixPortion& portion) {
    const MatMulTestParams test_params{
        portion,
    };

    const std::string desc = fixture_params.name() + "," + test_params.name();

    if (op.pack_lhs.has_value()) {
        KAI_REGISTER_TEST(
            MatMulFixture, MatMulPackLhsTest, test_suite_name.c_str(), ("PackLhs/" + desc).c_str(), fixture_params,
            test_params);
    }

    if (op.pack_rhs.has_value()) {
        KAI_REGISTER_TEST(
            MatMulFixture, MatMulPackRhsTest, test_suite_name.c_str(), ("PackRhs/" + desc).c_str(), fixture_params,
            test_params);
    }

    if (op.matmul != nullptr) {
        KAI_REGISTER_TEST(
            MatMulFixture, MatMulMatMulTest, test_suite_name.c_str(), ("MatMul/" + desc).c_str(), fixture_params,
            test_params);
    }
}

/// Registers all MatMulNext tests with the test registry.
const auto matmul_tests_setup = TestRegistry::register_setup([]() {
    const size_t num_shapes_per_op = 100;
    const auto output_portions = make_output_portions();

    /* NOTE: that `dist_cxt` must only be constructed once here, as
     * it contains the per test suite seed stream */
    const Span<const MatMulOperator> available_operators = get_available_matmul_operators();
    MatMulDistribution dist_ctx;

    for (const MatMulOperator& op : available_operators) {
        if (!op.is_cpu_supported()) {
            continue;
        }

        const std::string test_suite_name = "MatMulNext";
        BiasSelector bias_selector(op);

        for (size_t shape_no = 0; shape_no < num_shapes_per_op; ++shape_no) {
            for (const MatrixPortion& portion : output_portions) {
                const MatMulFixtureParams fixture = pick_fixture(shape_no, op, portion, dist_ctx, bias_selector);
                register_operator_test(test_suite_name, op, fixture, portion);
            }
        }
    }
});

}  // namespace

}  // namespace kai::test

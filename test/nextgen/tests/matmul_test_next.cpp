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

struct MatMulFixtureParams {
    size_t iteration_no;

    size_t shape_m;
    size_t shape_n;
    size_t shape_k;
    MatMulBiasMode bias_mode;
    float clamp_ratio;

    const MatMulOperator* op;

    [[nodiscard]] std::string name() const {
        return std::string(op->name) + ",m=" + std::to_string(shape_m) + ",n=" + std::to_string(shape_n) +
            ",k=" + std::to_string(shape_k) + ",bias=" + matmul_bias_mode_name(bias_mode) +
            ",clamp_ratio=" + std::to_string(clamp_ratio) + ",iteration=" + std::to_string(iteration_no);
    }
};

struct MatMulTestParams {
    MatrixPortion portion;

    [[nodiscard]] std::string name() const {
        return "start_m=" + std::to_string(portion.start_row()) + ",size_m=" + std::to_string(portion.height()) +
            ",start_n=" + std::to_string(portion.start_col()) + ",size_n=" + std::to_string(portion.width());
    }
};

class MatMulFixture : public testing::Test {
public:
    explicit MatMulFixture(const MatMulFixtureParams& params) : m_fixture_params(params) {
    }

protected:
    [[nodiscard]] const MatMulFixtureParams& fixture_params() const {
        return m_fixture_params;
    }

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
    explicit MatMulPackLhsTest(const MatMulFixtureParams& fixture_params, const MatMulTestParams& test_params) :
        MatMulFixture(fixture_params), m_test_params(test_params) {
    }

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
    explicit MatMulPackRhsTest(const MatMulFixtureParams& fixture_params, const MatMulTestParams& test_params) :
        MatMulFixture(fixture_params), m_test_params(test_params) {
    }

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
    explicit MatMulMatMulTest(const MatMulFixtureParams& fixture_params, const MatMulTestParams& test_params) :
        MatMulFixture(fixture_params), m_test_params(test_params) {
    }

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

const auto matmul_tests_setup = TestRegistry::register_setup([]() {
    const size_t num_shapes_per_op = 100;
    const std::array output_portions = {
        MatrixPortion(0, 0, 1, 1),        // Full matrix.
        MatrixPortion(0, 0, 0.25, 0.25),  // Top-left corner.
        MatrixPortion(0.75, 0.75, 1, 1)   // Bottom-right corner.
    };

    const Span<const MatMulOperator> available_operators = get_available_matmul_operators();
    auto& seed_feed = seed_stream("MatMulNext::setup");
    Rng rng(seed_feed());

    std::uniform_int_distribution<size_t> shape_dist(1, 150);
    std::uniform_real_distribution<float> probability_dist(0.0F, 1.0F);
    std::uniform_real_distribution<float> dist_70_to_100(0.7F, 1.0F);
    std::uniform_real_distribution<float> dist_0_to_70(0.0F, 0.7F);

    for (const MatMulOperator& op : available_operators) {
        if (!op.is_cpu_supported()) {
            continue;
        }

        const bool test_pack_lhs = op.pack_lhs.has_value();
        const bool test_pack_rhs = op.pack_rhs.has_value();
        const bool test_matmul = op.matmul != nullptr;

        const char* test_suite_name = "MatMulNext";

        // Separates no-bias mode from the list of supported bias modes.
        //
        // Reason: no-bias and with-bias cases are chosen by a distribution,
        // and if the with-bias case is chosen, each with-bias modes will be chosen
        // by another distribution.
        const bool has_no_bias_mode =
            std::find(op.supported_bias_modes.begin(), op.supported_bias_modes.end(), MatMulBiasMode::NO_BIAS) !=
            op.supported_bias_modes.end();

        std::vector<MatMulBiasMode> with_bias_modes;
        std::copy_if(
            op.supported_bias_modes.begin(), op.supported_bias_modes.end(), std::back_inserter(with_bias_modes),
            [](MatMulBiasMode bias_mode) { return bias_mode != MatMulBiasMode::NO_BIAS; });
        const bool has_with_bias_mode = !with_bias_modes.empty();
        std::uniform_int_distribution<size_t> with_bias_dist(0, with_bias_modes.size() - 1);

        KAI_TEST_ASSERT_MSG(has_no_bias_mode || has_with_bias_mode, "At least one bias mode is needed!");

        for (size_t shape_no = 0; shape_no < num_shapes_per_op; ++shape_no) {
            size_t shape_m = 0;
            size_t shape_n = 0;
            size_t shape_k = 0;
            MatMulBiasMode bias_mode = MatMulBiasMode::NO_BIAS;
            float clamp_ratio = 0;

            while (true) {
                shape_m = shape_dist(rng);
                shape_n = shape_dist(rng);
                shape_k = shape_dist(rng);

                if (!op.is_shape_suitable(shape_m, shape_n, shape_k)) {
                    continue;
                }

                // Bias mode:
                //   * 70% of tests have bias.
                //   * 30% of tests have no bias.
                //
                // If there is no bias mode, with bias mode is always chosen.
                // If with bias mode is chosen, each bias modes (except no bias mode)
                // with be chosen uniformly.
                const float bias_prob = probability_dist(rng);
                const bool with_bias = !has_no_bias_mode || (has_with_bias_mode && bias_prob < 0.7F);

                if (with_bias) {
                    const size_t bias_mode_idx = with_bias_dist(rng);
                    bias_mode = with_bias_modes.at(bias_mode_idx);
                } else {
                    bias_mode = MatMulBiasMode::NO_BIAS;
                }

                // Clamping range:
                //   * 20% of tests have no clamping.
                //   * 40% of tests have clamping range between 70% to 100% the output range.
                //   * 40% of tests have clamping range between 0% to 70% the output range.
                const float clamp_prob = probability_dist(rng);

                const bool no_clamp = clamp_prob < 0.2F;
                const bool clamp_70_to_100 = clamp_prob >= 0.2F && clamp_prob < 0.6F;
                const bool clamp_0_to_70 = clamp_prob >= 0.6F;

                if (no_clamp) {
                    clamp_ratio = 1.0F;
                } else if (clamp_70_to_100) {
                    clamp_ratio = dist_70_to_100(rng);
                } else {
                    KAI_TEST_ASSERT(clamp_0_to_70);
                    clamp_ratio = dist_0_to_70(rng);
                }

                break;
            }

            const MatMulFixtureParams fixture_params{
                shape_no, shape_m, shape_n, shape_k, bias_mode, clamp_ratio, &op,
            };

            for (const MatrixPortion& portion : output_portions) {
                const MatMulTestParams test_params{
                    portion,
                };

                const std::string params_name = fixture_params.name() + "," + test_params.name();

                if (test_pack_lhs) {
                    const std::string test_name = "PackLhs/" + params_name;
                    KAI_REGISTER_TEST(
                        MatMulFixture, MatMulPackLhsTest, test_suite_name, test_name.c_str(), fixture_params,
                        test_params);
                }

                if (test_pack_rhs) {
                    const std::string test_name = "PackRhs/" + params_name;
                    KAI_REGISTER_TEST(
                        MatMulFixture, MatMulPackRhsTest, test_suite_name, test_name.c_str(), fixture_params,
                        test_params);
                }

                if (test_matmul) {
                    const std::string test_name = "MatMul/" + params_name;
                    KAI_REGISTER_TEST(
                        MatMulFixture, MatMulMatMulTest, test_suite_name, test_name.c_str(), fixture_params,
                        test_params);
                }
            }
        }
    }
});

}  // namespace

}  // namespace kai::test

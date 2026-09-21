//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <string>

#include "test/common/range.hpp"
#include "test/common/seed.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/common/test_config.hpp"
#include "test/nextgen/common/test_registry.hpp"
#include "test/nextgen/operators/softmax/softmax_operator.hpp"
#include "test/nextgen/operators/softmax/softmax_tb.hpp"

namespace kai::test {
namespace {

struct SoftmaxFixtureParams {
    size_t iteration_no;
    size_t length;
    const SoftmaxOperator* op;
    Range<double> input_range{0.0, 1.0};

    /// Builds a unique, descriptive test name.
    [[nodiscard]] std::string name() const {
        return std::string(op->name) +                         //
            ",length=" + std::to_string(length) +              //
            ",iteration=" + std::to_string(iteration_no) +     //
            ",input_min=" + std::to_string(input_range.min) +  //
            ",input_max=" + std::to_string(input_range.max);
    }
};

class SoftmaxFixture : public testing::Test {
public:
    explicit SoftmaxFixture(const SoftmaxFixtureParams& params) : m_params(params) {
    }

protected:
    /// Gets the fixture parameters.
    [[nodiscard]] const SoftmaxFixtureParams& fixture_params() const {
        return m_params;
    }

private:
    SoftmaxFixtureParams m_params;
};

class SoftmaxOutputTest : public SoftmaxFixture {
public:
    explicit SoftmaxOutputTest(const SoftmaxFixtureParams& params) : SoftmaxFixture(params) {
    }

    void TestBody() override {
        const SoftmaxFixtureParams& params = fixture_params();
        SoftmaxTb test(params.length, params.op);
        auto& feed = seed_stream("SoftmaxNext::testbench:" + params.name());
        Rng rng(feed());
        test.generate_test_data(rng, params.input_range);

        test.test_output();
    }
};

struct SoftmaxDistribution {
    Rng rng{seed_stream("SoftmaxNext::setup")()};
    std::uniform_int_distribution<size_t> length_dist{1, 1024};
};

SoftmaxFixtureParams pick_fixture(size_t iteration_no, const SoftmaxOperator& op, SoftmaxDistribution& dist) {
    return {iteration_no, dist.length_dist(dist.rng), &op};
}

const auto softmax_tests_setup = TestRegistry::register_setup([]() {
    SoftmaxDistribution dist;

    for (const SoftmaxOperator& op : get_available_softmax_operators()) {
        if (!op.is_cpu_supported()) {
            continue;
        }

        for (size_t iteration_no = 0; iteration_no < TestConfig::Get().num_shapes(); ++iteration_no) {
            const SoftmaxFixtureParams params = pick_fixture(iteration_no, op, dist);
            const std::string test_name = "Output/" + params.name();
            KAI_REGISTER_TEST(SoftmaxFixture, SoftmaxOutputTest, "SoftmaxNext", test_name.c_str(), params);
        }

        // Large positive logits overflow FP32 exp without maximum subtraction.
        const SoftmaxFixtureParams overflow_params{0, 17, &op, {100.0, 101.0}};
        const std::string overflow_name = "Overflow/" + overflow_params.name();
        KAI_REGISTER_TEST(SoftmaxFixture, SoftmaxOutputTest, "SoftmaxNext", overflow_name.c_str(), overflow_params);

        // Large negative logits underflow FP32 exp to zero without maximum subtraction.
        const SoftmaxFixtureParams underflow_params{0, 17, &op, {-110.0, -109.0}};
        const std::string underflow_name = "Underflow/" + underflow_params.name();
        KAI_REGISTER_TEST(SoftmaxFixture, SoftmaxOutputTest, "SoftmaxNext", underflow_name.c_str(), underflow_params);
    }
});

}  // namespace
}  // namespace kai::test

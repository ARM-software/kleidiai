//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/enum_utils.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/seed.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/common/test_config.hpp"
#include "test/nextgen/common/test_registry.hpp"
#include "test/nextgen/operators/dwconv/dwconv_operator.hpp"
#include "test/nextgen/operators/dwconv/dwconv_tb.hpp"

namespace kai::test {

namespace {

struct DwConvDistribution {
    /// Seeds fixture generation for the operator.
    explicit DwConvDistribution(std::string_view operator_name) :
        m_rng(seed_stream("DwConvNext::setup:" + std::string(operator_name))()) {
    }

    Rng m_rng;

    /// Partitioned distribution for spatial width.
    std::uniform_int_distribution<size_t> m_small_input_shape_dist{1, 3};
    std::uniform_int_distribution<size_t> m_medium_input_shape_dist{4, 24};
    std::uniform_int_distribution<size_t> m_large_input_shape_dist{25, 64};

    /// Partitioned distribution for channels
    std::uniform_int_distribution<size_t> m_narrow_channels_dist{1, 24};
    std::uniform_int_distribution<size_t> m_wide_channels_dist{25, 512};
    std::uniform_int_distribution<size_t> m_large_channels_dist{513, 1024};

    /// Distribution for kernels with variable filter sizes
    std::uniform_int_distribution<size_t> m_filter_shape_dist{1, 24};

    // percentage distribution
    std::uniform_real_distribution<float> m_probability_dist{0.0F, 1.0F};

    // Distrubutions for asymmetric partitioning
    std::uniform_real_distribution<float> m_dist_70_to_100{0.7F, 1.0F};
    std::uniform_real_distribution<float> m_dist_10_to_70{0.1F, 0.7F};
};

enum class ShapeSize {
    SMALL,
    MEDIUM,
    LARGE,
};

enum class PaddingMode {
    VALID,
    SAME,
    ASYMMETRIC,
    LARGE_ASYMMETRIC,
    LAST,
};

/// Samples one valid enum value with equal probability.
template <typename T>
T sample_mode(Rng& rng) {
    static constexpr auto modes = enum_values<T>();
    static_assert(!modes.empty(), "Cannot sample an empty enum.");
    std::uniform_int_distribution<size_t> mode_dist(0, modes.size() - 1);
    return modes[mode_dist(rng)];
}

/// Converts a padding mode into per-edge padding for the selected filter.
Padding2D pick_padding(PaddingMode mode, size_t filter_height, size_t filter_width) {
    static constexpr std::array asymmetric_paddings{
        // A unit pad on right and bottom isolates one-sided padding on each axis.
        Padding2D{0, 1, 0, 1},
        // Distinct odd pads above one cover multi-element asymmetry on all edges.
        // Their exact magnitudes are arbitrary and assume no filter or tile size.
        Padding2D{5, 11, 7, 3},
    };

    switch (mode) {
        case PaddingMode::VALID:
            return {0, 0, 0, 0};
        case PaddingMode::SAME: {
            const size_t total_height = filter_height - 1;
            const size_t total_width = filter_width - 1;
            const size_t top = total_height / 2;
            const size_t left = total_width / 2;
            return {left, total_width - left, top, total_height - top};
        }
        case PaddingMode::ASYMMETRIC:
            return asymmetric_paddings[0];
        case PaddingMode::LARGE_ASYMMETRIC:
            return asymmetric_paddings[1];
        default:
            KAI_TEST_ERROR("Unexpected depthwise padding mode.");
    }
}

struct DwConvFixtureParams {
    size_t iteration_no;

    size_t in_height;
    size_t in_width;
    size_t filter_height;
    size_t filter_width;
    size_t channels;
    Padding2D padding;
    std::optional<float> clamp_keep_ratio;

    const DwConvOperator* op;

    /// Gets the output height for this input and filter.
    ///
    /// @return The spatial output height.
    [[nodiscard]] size_t out_height() const {
        return in_height + padding.top + padding.bottom + 1 - filter_height;
    }

    /// Gets the output width for this input and filter.
    ///
    /// @return The spatial output width.
    [[nodiscard]] size_t out_width() const {
        return in_width + padding.left + padding.right + 1 - filter_width;
    }

    /// Builds a descriptive name for these fixture parameters.
    ///
    /// @return A unique name string used for test registration and caching.
    [[nodiscard]] std::string name() const {
        std::string clamp_keep_ratio_name = "noclamp";
        if (clamp_keep_ratio.has_value()) {
            clamp_keep_ratio_name = std::to_string(clamp_keep_ratio.value());
        }

        std::string result(op->name);
        result += ",h=" + std::to_string(in_height);
        result += ",w=" + std::to_string(in_width);
        result += ",fh=" + std::to_string(filter_height);
        result += ",fw=" + std::to_string(filter_width);
        result += ",c=" + std::to_string(channels);
        result += ",pad_left=" + std::to_string(padding.left);
        result += ",pad_right=" + std::to_string(padding.right);
        result += ",pad_top=" + std::to_string(padding.top);
        result += ",pad_bottom=" + std::to_string(padding.bottom);
        result += ",clamp_keep_ratio=" + clamp_keep_ratio_name;
        result += ",iteration=" + std::to_string(iteration_no);
        return result;
    }
};

struct DwConvTestParams {
    MatrixPortion portion;

    /// Builds a descriptive name for the selected spatial portion.
    ///
    /// @return A name string encoding the portion location and size.
    [[nodiscard]] std::string name() const {
        std::string result = "start_h=" + std::to_string(portion.start_row());
        result += ",size_h=" + std::to_string(portion.height());
        result += ",start_w=" + std::to_string(portion.start_col());
        result += ",size_w=" + std::to_string(portion.width());
        return result;
    }
};

class DwConvFixture : public testing::Test {
public:
    /// Creates a fixture bound to a specific parameter set.
    ///
    /// @param[in] params The fixture parameters used for this instance.
    explicit DwConvFixture(const DwConvFixtureParams& params) : m_fixture_params(params) {
    }

protected:
    /// Returns the parameters for this fixture instance.
    ///
    /// @return A constant reference to the fixture parameters.
    [[nodiscard]] const DwConvFixtureParams& fixture_params() const {
        return m_fixture_params;
    }

    /// Returns the cached test bench for this fixture, creating it if needed.
    ///
    /// The test bench is keyed by the fixture name, and test data is generated
    /// once per unique fixture to keep results deterministic.
    ///
    /// @return A reference to the cached test bench.
    [[nodiscard]] DwConvTb& test_bench() {
        const std::string name = m_fixture_params.name();

        // If the test has already been created, uses it.
        const auto it = test_benches.find(name);

        if (it != test_benches.end()) {
            return it->second;
        }

        // Creates a new test if it hasn't been created.
        const size_t padded_height =
            m_fixture_params.in_height + m_fixture_params.padding.top + m_fixture_params.padding.bottom;
        const size_t padded_width =
            m_fixture_params.in_width + m_fixture_params.padding.left + m_fixture_params.padding.right;
        size_t out_height = 0;
        size_t out_width = 0;
        if (padded_height >= m_fixture_params.filter_height && padded_width >= m_fixture_params.filter_width) {
            out_height = m_fixture_params.out_height();
            out_width = m_fixture_params.out_width();
        }
        DwConvTb test(
            m_fixture_params.in_height, m_fixture_params.in_width, out_height, out_width,
            m_fixture_params.filter_height, m_fixture_params.filter_width, m_fixture_params.channels,
            m_fixture_params.padding, m_fixture_params.clamp_keep_ratio, m_fixture_params.op);

        const std::string seed_key = "DwConvNext::testbench:" + name;
        auto& feed = seed_stream(seed_key);
        Rng rng(feed());

        test.generate_test_data(rng);

        return test_benches[name] = std::move(test);
    }

private:
    static std::unordered_map<std::string, DwConvTb> test_benches;

    DwConvFixtureParams m_fixture_params;
};

std::unordered_map<std::string, DwConvTb> DwConvFixture::test_benches;

class DwConvPackRhsTest : public DwConvFixture {
public:
    /// Creates an RHS packing test instance.
    ///
    /// @param[in] params The fixture parameters shared across tests.
    explicit DwConvPackRhsTest(const DwConvFixtureParams& params) : DwConvFixture(params) {
    }

private:
    /// Runs the RHS packing validation for the complete filter.
    void TestBody() override {
        DwConvTb& test = test_bench();
        const DwConvFixtureParams& params = fixture_params();

        const auto [step_h, step_w, step_c] = test.rhs_packing_steps();
        KAI_TEST_ASSERT(step_h == params.filter_height && step_w == params.filter_width && step_c == params.channels);

        test.test_rhs_packing(0, 0, 0, step_h, step_w, step_c);
    }
};

class DwConvComputeTest : public DwConvFixture {
public:
    /// Creates a depthwise convolution test instance.
    ///
    /// @param[in] params The fixture parameters shared across tests.
    /// @param[in] test_params The parameters describing the output portion.
    DwConvComputeTest(const DwConvFixtureParams& params, const DwConvTestParams& test_params) :
        DwConvFixture(params), m_test_params(test_params) {
    }

private:
    /// Runs depthwise convolution validation for the selected portion.
    void TestBody() override {
        DwConvTb& test = test_bench();
        const DwConvFixtureParams& params = fixture_params();
        const MatrixPortion& portion = m_test_params.portion;

        const auto [step_h, step_w] = test.dwconv_steps();
        const Rect rect = portion.compute_portion(params.out_height(), params.out_width(), step_h, step_w);

        const size_t start_h = rect.start_row();
        const size_t start_w = rect.start_col();
        const size_t size_h = rect.height();
        const size_t size_w = rect.width();

        KAI_TEST_ASSERT(size_h > 0 && size_w > 0);
        test.test_dwconv(start_h, start_w, size_h, size_w);
    }

    DwConvTestParams m_test_params;
};

/// Returns the spatial portions supported by the operator.
///
/// @param[in] op The operator under test.
///
/// @return Full and partial output portions supported by the kernel.
std::vector<MatrixPortion> make_output_portions(const DwConvOperator& op) {
    if (op.planar) {
        return {
            MatrixPortion(0, 0, 1, 1),        // Full output.
            MatrixPortion(0, 0, 0.25, 1),     // Top rows.
            MatrixPortion(0.75, 0, 0.25, 1),  // Bottom rows.
        };
    }
    return {
        MatrixPortion(0, 0, 1, 1),              // Full output.
        MatrixPortion(0, 0, 0.25, 0.25),        // Top-left corner.
        MatrixPortion(0.75, 0.75, 0.25, 0.25),  // Bottom-right corner.
        MatrixPortion(0.25, 0.25, 0.5, 0.5),    // Center.
    };
}

/// Picks a clamp keep ratio for the operator clamp mode.
///
/// @param[in] op The operator to test.
/// @param[in,out] dist_ctx The shared distribution context.
///
/// @return A clamp keep ratio if clamping arguments should be provided.
std::optional<float> pick_clamp_keep_ratio(const DwConvOperator& op, DwConvDistribution& dist_ctx) {
    if (op.clamp_mode == DwConvClampMode::UNSUPPORTED) {
        return std::nullopt;
    }

    KAI_TEST_ASSERT(op.clamp_mode == DwConvClampMode::OPTIONAL || op.clamp_mode == DwConvClampMode::REQUIRED);

    // Clamp keep ratio:
    //   * 50% of tests have no clamping.
    //   * 10% of tests keep the full output range.
    //   * 20% of tests keep between 70% to 100% of the output range.
    //   * 20% of tests keep between 10% to 70% of the output range.
    const float clamp_prob = dist_ctx.m_probability_dist(dist_ctx.m_rng);

    const bool no_clamp = clamp_prob < 0.5F;
    const bool clamp_100 = clamp_prob >= 0.5F && clamp_prob < 0.6F;
    const bool clamp_70_to_100 = clamp_prob >= 0.6F && clamp_prob < 0.8F;
    const bool clamp_10_to_70 = clamp_prob >= 0.8F;

    if (no_clamp) {
        return std::nullopt;
    }

    if (clamp_100) {
        return 1.0F;
    }

    if (clamp_70_to_100) {
        return dist_ctx.m_dist_70_to_100(dist_ctx.m_rng);
    }

    KAI_TEST_ASSERT(clamp_10_to_70);
    return dist_ctx.m_dist_10_to_70(dist_ctx.m_rng);
}

/// Picks the filter dimensions from the operator constraints or the distribution.
///
/// @param[in] op The operator under test.
/// @param[in, out] dist The shared distribution context.
///
/// @return The filter width and height, in that order.
std::pair<size_t, size_t> pick_filter_shape(const DwConvOperator& op, DwConvDistribution& dist) {
    const size_t height = op.filter_height ? op.filter_height.value() : dist.m_filter_shape_dist(dist.m_rng);
    const size_t width = op.filter_width ? op.filter_width.value() : dist.m_filter_shape_dist(dist.m_rng);
    return {width, height};
}

/// Picks a spatial input size range for the fixture.
ShapeSize pick_spatial_size(DwConvDistribution& dist) {
    const float probability = dist.m_probability_dist(dist.m_rng);
    if (probability < 0.1F) {
        return ShapeSize::SMALL;
    }
    if (probability >= 0.95F) {
        return ShapeSize::LARGE;
    }
    return ShapeSize::MEDIUM;
}

/// Picks a channel size range for the fixture.
ShapeSize pick_channels_size(DwConvDistribution& dist) {
    const float probability = dist.m_probability_dist(dist.m_rng);
    if (probability < 0.2F) {
        return ShapeSize::MEDIUM;
    }
    if (probability >= 0.95F) {
        return ShapeSize::LARGE;
    }
    return ShapeSize::SMALL;
}

/// Picks the input dimensions from the selected size ranges.
///
/// @param[in] spatial_size The spatial input size range.
/// @param[in] channels_size The channel size range.
/// @param[in, out] dist The shared distribution context.
///
/// @return The input width, height, and channels, in that order.
std::tuple<size_t, size_t, size_t> pick_input_shape(
    ShapeSize spatial_size, ShapeSize channels_size, DwConvDistribution& dist) {
    auto* input_shape_dist = &dist.m_medium_input_shape_dist;
    switch (spatial_size) {
        case ShapeSize::SMALL:
            input_shape_dist = &dist.m_small_input_shape_dist;
            break;
        case ShapeSize::MEDIUM:
            break;
        case ShapeSize::LARGE:
            input_shape_dist = &dist.m_large_input_shape_dist;
            break;
        default:
            KAI_TEST_ASSERT_MSG(false, "Unexpected spatial size.");
            break;
    }
    const size_t height = (*input_shape_dist)(dist.m_rng);
    const size_t width = (*input_shape_dist)(dist.m_rng);

    auto* channels_dist = &dist.m_narrow_channels_dist;
    switch (channels_size) {
        case ShapeSize::SMALL:
            break;
        case ShapeSize::MEDIUM:
            channels_dist = &dist.m_wide_channels_dist;
            break;
        case ShapeSize::LARGE:
            channels_dist = &dist.m_large_channels_dist;
            break;
        default:
            KAI_TEST_ASSERT_MSG(false, "Unexpected channel size.");
            break;
    }
    const size_t channels = (*channels_dist)(dist.m_rng);
    return {width, height, channels};
}

/// Selects a valid fixture configuration for an operator and output portion.
///
/// @param[in] iteration_no The shape iteration index.
/// @param[in] op The operator under test.
/// @param[in] portion The output portion to validate.
/// @param[in, out] dist The shared distribution context.
///
/// @return A populated fixture parameter set.
DwConvFixtureParams pick_fixture(
    size_t iteration_no, const DwConvOperator& op, const MatrixPortion& portion, DwConvDistribution& dist) {
    size_t in_height = 0;
    size_t in_width = 0;
    size_t filter_height = 0;
    size_t filter_width = 0;
    size_t channels = 0;
    Padding2D padding{};

    const ShapeSize spatial_size = pick_spatial_size(dist);
    const ShapeSize channels_size = pick_channels_size(dist);

    static constexpr size_t max_attempts = 500'000;
    size_t attempts = 0;
    while (true) {
        KAI_TEST_ASSERT_MSG(attempts < max_attempts, "Unable to find a suitable depthwise shape.");
        ++attempts;
        std::tie(filter_width, filter_height) = pick_filter_shape(op, dist);
        std::tie(in_width, in_height, channels) = pick_input_shape(spatial_size, channels_size, dist);
        in_height = std::max(in_height, filter_height);
        in_width = std::max(in_width, filter_width);
        const PaddingMode padding_mode = sample_mode<PaddingMode>(dist.m_rng);
        padding = pick_padding(padding_mode, filter_height, filter_width);
        if (op.is_shape_suitable(in_height, in_width, filter_height, filter_width, channels, padding, portion)) {
            break;
        }
    }

    const std::optional<float> clamp_keep_ratio = pick_clamp_keep_ratio(op, dist);
    return {iteration_no, in_height, in_width, filter_height, filter_width, channels, padding, clamp_keep_ratio, &op};
}

/// Registers packing and compute tests for one fixture.
///
/// @param[in] test_suite_name The test suite name to register with.
/// @param[in] op The operator under test.
/// @param[in] fixture_params The fixture parameter set.
/// @param[in] portions The output portions to validate.
void register_operator_test(
    const std::string& test_suite_name,         //
    const DwConvOperator& op,                   //
    const DwConvFixtureParams& fixture_params,  //
    Span<const MatrixPortion> portions) {
    const std::string fixture_desc = fixture_params.name();
    const char* suite = test_suite_name.c_str();

    // RHS packing covers the full filter independently of the output portion.
    if (op.pack_rhs.has_value()) {
        const std::string case_name = "PackRhs/" + fixture_desc;
        KAI_REGISTER_TEST(DwConvFixture, DwConvPackRhsTest, suite, case_name.c_str(), fixture_params);
    }

    if (!op.dwconv.has_value()) {
        return;
    }

    const size_t in_h = fixture_params.in_height;
    const size_t in_w = fixture_params.in_width;
    const size_t filter_h = fixture_params.filter_height;
    const size_t filter_w = fixture_params.filter_width;
    const size_t channels = fixture_params.channels;
    const Padding2D& padding = fixture_params.padding;

    for (const MatrixPortion& portion : portions) {
        KAI_TEST_ASSERT(op.is_shape_suitable(in_h, in_w, filter_h, filter_w, channels, padding, portion));

        const DwConvTestParams test_params{portion};
        const std::string desc = fixture_desc + "," + test_params.name();
        const std::string case_name = "DwConv/" + desc;
        KAI_REGISTER_TEST(DwConvFixture, DwConvComputeTest, suite, case_name.c_str(), fixture_params, test_params);
    }
}

/// Registers all nextgen depthwise tests.
const auto dwconv_tests_setup = TestRegistry::register_setup([]() {
    const size_t num_shapes_per_op = TestConfig::Get().num_shapes();
    const Span<const DwConvOperator> available_operators = get_available_dwconv_operators();

    for (const DwConvOperator& op : available_operators) {
        if (!op.is_cpu_supported()) {
            continue;
        }

        const std::string test_suite_name = "DwConvNext";
        const std::vector<MatrixPortion> portions = make_output_portions(op);
        // The generated partial portions are suitable whenever the full output is suitable.
        const MatrixPortion& full_portion = portions.front();

        // Each operator has its own stream, independent of registration order.
        DwConvDistribution dist(op.name);
        for (size_t iteration_no = 0; iteration_no < num_shapes_per_op; ++iteration_no) {
            const DwConvFixtureParams fixture = pick_fixture(iteration_no, op, full_portion, dist);
            register_operator_test(test_suite_name, op, fixture, portions);
        }
    }
});

}  // namespace

}  // namespace kai::test

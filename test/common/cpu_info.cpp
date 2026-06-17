//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/cpu_info.hpp"

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "kai/kai_common.h"
#include "test/common/enum_utils.hpp"

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif  // defined(__aarch64__) && defined(__linux__)

#if defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif  // defined(__aarch64__) && defined(__APPLE__)

#if (defined(__aarch64__) && defined(_WIN64)) || defined(_M_ARM64)
#include <Windows.h>
#include <processthreadsapi.h>
#include <sysinfoapi.h>
#include <winnt.h>
#endif  // (defined(__aarch64__) && defined(_WIN64)) || defined(_M_ARM64)

namespace kai::test {

namespace {

enum class CpuFeature : size_t {
    ADVSIMD = 0,  //
    DOTPROD,      //
    I8MM,         //
    FP16,         //
    BF16,         //
    SVE,          //
    SVE2,         //
    SME,          //
    SME2,         //
    LAST          // This should be last element, please add new CPU capabilities before it
};

constexpr std::array<std::tuple<CpuFeature, std::string_view>, n_elements<CpuFeature>()> cpu_features{{
    {CpuFeature::ADVSIMD, "ADVSIMD"},  //
    {CpuFeature::DOTPROD, "DOTPROD"},  //
    {CpuFeature::I8MM, "I8MM"},        //
    {CpuFeature::FP16, "FP16"},        //
    {CpuFeature::BF16, "BF16"},        //
    {CpuFeature::SVE, "SVE"},          //
    {CpuFeature::SVE2, "SVE2"},        //
    {CpuFeature::SME, "SME"},          //
    {CpuFeature::SME2, "SME2"},        //
}};

constexpr const char* forced_cpu_features_env_name = "KAI_TEST_FORCE_CPU_FEATURES";

/// Return substring with trimmed leading and trailing whitespaces from input.
std::string_view trim_whitespace(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.remove_suffix(1);
    }
    return value;
}

/// Get description of available features
std::string available_features() {
    std::string names;

    for (size_t i = 0; i < cpu_features.size(); ++i) {
        if (i != 0) {
            names += ", ";
        }
        names += std::get<1>(cpu_features[i]);
    }

    return names;
}

/// Read and tokenize the forced features
std::optional<std::array<bool, n_elements<CpuFeature>()>> parse_forced_cpu_features() {
    const char* env = std::getenv(forced_cpu_features_env_name);
    if (env == nullptr) {
        return std::nullopt;
    }

    puts("WARNING: Reading enabled CPU features from KAI_TEST_FORCE_CPU_FEATURES");
    std::array<bool, n_elements<CpuFeature>()> features{};
    std::string_view env_view{env};
    size_t token_start = 0;

    while (token_start <= env_view.size()) {
        size_t token_end = env_view.find(',', token_start);
        if (token_end == std::string_view::npos) {
            token_end = env_view.size();
        }

        const std::string_view token = trim_whitespace(env_view.substr(token_start, token_end - token_start));
        if (!token.empty()) {
            bool matched = false;

            for (const auto& [feature, name] : cpu_features) {
                if (token == name) {
                    features[as_idx(feature)] = true;
                    matched = true;
                    break;
                }
            }

            // There was an invalid token, report error
            if (!matched) {
                std::string error_message = "Unrecognized CPU feature provided by ";
                error_message += forced_cpu_features_env_name;
                error_message += ": ";
                error_message += token;
                error_message += ". Supported features: ";
                error_message += available_features();
                KAI_ERROR(error_message.c_str());
            }
        }

        if (token_end == env_view.size()) {
            break;
        }
        token_start = token_end + 1;
    }

    return features;
}

const std::optional<std::array<bool, n_elements<CpuFeature>()>>& forced_cpu_features() {
    static const std::optional<std::array<bool, n_elements<CpuFeature>()>> forced = parse_forced_cpu_features();
    return forced;
}

#if defined(__aarch64__) && defined(__linux__)
/// Define CPU capabilities not available in toolchain definitions
#ifndef HWCAP_ASIMD
constexpr uint64_t HWCAP_ASIMD = 1UL << 1;
#endif
#ifndef HWCAP_FPHP
constexpr uint64_t HWCAP_FPHP = 1UL << 9;
#endif
#ifndef HWCAP_ASIMDHP
constexpr uint64_t HWCAP_ASIMDHP = 1UL << 10;
#endif
#ifndef HWCAP_ASIMDDP
constexpr uint64_t HWCAP_ASIMDDP = 1UL << 20;
#endif
#ifndef HWCAP_SVE
constexpr uint64_t HWCAP_SVE = 1UL << 22;
#endif
#ifndef HWCAP2_SVE2
constexpr uint64_t HWCAP2_SVE2 = 1UL << 1;
#endif
#ifndef HWCAP2_I8MM
constexpr uint64_t HWCAP2_I8MM = 1UL << 13;
#endif
#ifndef HWCAP2_BF16
constexpr uint64_t HWCAP2_BF16 = 1UL << 14;
#endif
#ifndef HWCAP2_SME
constexpr uint64_t HWCAP2_SME = 1UL << 23;
#endif
#ifndef HWCAP2_SME2
constexpr uint64_t HWCAP2_SME2 = 1UL << 37;
#endif

const std::array<std::tuple<CpuFeature, uint64_t, uint64_t>, n_elements<CpuFeature>()> cpu_caps{{
    {CpuFeature::ADVSIMD, AT_HWCAP, HWCAP_ASIMD},              //
    {CpuFeature::DOTPROD, AT_HWCAP, HWCAP_ASIMDDP},            //
    {CpuFeature::I8MM, AT_HWCAP2, HWCAP2_I8MM},                //
    {CpuFeature::FP16, AT_HWCAP, HWCAP_FPHP | HWCAP_ASIMDHP},  //
    {CpuFeature::BF16, AT_HWCAP2, HWCAP2_BF16},                //
    {CpuFeature::SVE, AT_HWCAP, HWCAP_SVE},                    //
    {CpuFeature::SVE2, AT_HWCAP2, HWCAP2_SVE2},                //
    {CpuFeature::SME, AT_HWCAP2, HWCAP2_SME},                  //
    {CpuFeature::SME2, AT_HWCAP2, HWCAP2_SME2},                //
}};

bool get_cap_support(CpuFeature feature) {
    const size_t feature_idx = static_cast<size_t>(as_idx(feature));
    KAI_ASSUME_ALWAYS(feature_idx < cpu_caps.size());

    if (const auto& forced = forced_cpu_features(); forced.has_value()) {
        return (*forced)[feature_idx];
    }

    auto [cpu_feature, cap_id, cap_bits] = cpu_caps[feature_idx];
    // Make sure CPU feature is correctly initialized
    KAI_ASSERT_ALWAYS(feature == cpu_feature);

    const uint64_t hwcaps = getauxval(cap_id);

    return (hwcaps & cap_bits) == cap_bits;
}
#elif defined(__aarch64__) && defined(__APPLE__)
const std::array<std::tuple<CpuFeature, std::string_view>, n_elements<CpuFeature>()> cpu_caps{{
    {CpuFeature::ADVSIMD, "hw.optional.arm64"},  // Advanced SIMD is always present on arm64
    {CpuFeature::DOTPROD, "hw.optional.arm.FEAT_DotProd"},
    {CpuFeature::I8MM, "hw.optional.arm.FEAT_I8MM"},
    {CpuFeature::FP16, "hw.optional.arm.FEAT_FP16"},
    {CpuFeature::BF16, "hw.optional.arm.FEAT_BF16"},
    {CpuFeature::SVE, ""},   // not supported
    {CpuFeature::SVE2, ""},  // not supported
    {CpuFeature::SME, "hw.optional.arm.FEAT_SME"},
    {CpuFeature::SME2, "hw.optional.arm.FEAT_SME2"},
}};

bool get_cap_support(CpuFeature feature) {
    const size_t feature_idx = as_idx(feature);
    KAI_ASSUME_ALWAYS(feature_idx < cpu_caps.size());

    if (const auto& forced = forced_cpu_features(); forced.has_value()) {
        return (*forced)[feature_idx];
    }

    auto [cpu_feature, cap_name] = cpu_caps[feature_idx];
    KAI_ASSERT_ALWAYS(feature == cpu_feature);

    uint32_t value{};

    if (cap_name.length() > 0) {
        size_t size = sizeof(value);

        KAI_ASSERT_ALWAYS(sysctlbyname(cap_name.data(), nullptr, &size, nullptr, 0) == 0);
        KAI_ASSERT_ALWAYS(size == sizeof(value));

        [[maybe_unused]] int status = sysctlbyname(cap_name.data(), &value, &size, nullptr, 0);
        KAI_ASSERT_ALWAYS(status == 0);
    }

    return value == 1;
}
#elif (defined(__aarch64__) && defined(_WIN64)) || defined(_M_ARM64)
// Some system registers are provided in HARDWARE\DESCRIPTION\System\CentralProcessor\* registry.
//
// The registry name is encoded as
//   CP {op0 & 1, op1, CRn, CRm, op2}
//
// These can be used to detect architectural features that are unable to detect reliably
// using IsProcessorFeaturePresent. It must not be used to detect architectural features
// that require operating system support such as SVE and SME.
const char* ID_AA64PFR0_EL1 = "CP 4020";
const char* ID_AA64ISAR1_EL1 = "CP 4031";

const std::array<std::tuple<CpuFeature, DWORD, const char*, uint64_t>, n_elements<CpuFeature>()> cpu_caps{{
    {CpuFeature::ADVSIMD, PF_ARM_NEON_INSTRUCTIONS_AVAILABLE, nullptr, 0},
    {CpuFeature::DOTPROD, PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE, nullptr, 0},
    {CpuFeature::I8MM, 0, ID_AA64ISAR1_EL1, 0x00f0000000000000ULL},
    {CpuFeature::FP16, 0, ID_AA64PFR0_EL1, 0x00000000000f0000ULL},
    {CpuFeature::BF16, 0, ID_AA64ISAR1_EL1, 0x0000f00000000000ULL},
    {CpuFeature::SVE, 46, nullptr, 0},
    {CpuFeature::SVE2, 47, nullptr, 0},
    {CpuFeature::SME, 0, nullptr, 0},
    {CpuFeature::SME2, 0, nullptr, 0},
}};

uint64_t read_sysreg(const char* name) {
    uint64_t value = 0;
    DWORD size = sizeof(value);

    const LSTATUS status = RegGetValueA(
        HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", name, RRF_RT_REG_QWORD, nullptr,
        &value, &size);

    KAI_ASSERT_ALWAYS(status == ERROR_SUCCESS);

    return value;
}

bool get_cap_support(CpuFeature feature) {
    const size_t feature_idx = as_idx(feature);
    KAI_ASSUME_ALWAYS(feature_idx < cpu_caps.size());

    if (const auto& forced = forced_cpu_features(); forced.has_value()) {
        return (*forced)[feature_idx];
    }
    auto [cpu_feature, cap_id, reg_name, reg_mask] = cpu_caps[feature_idx];

    if (cap_id != 0) {
        return IsProcessorFeaturePresent(cap_id);
    }

    if (reg_name != nullptr) {
        const uint64_t value = read_sysreg(reg_name);
        const bool is_aarch64 = IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE);
        const bool has_feature = (value & reg_mask) != 0;

        return is_aarch64 && has_feature;
    }

    return false;
}
#elif defined(__aarch64__)
#error Please add a way how to check implemented CPU features
#else
bool get_cap_support(CpuFeature feature) {
    const size_t feature_idx = as_idx(feature);

    if (const auto& forced = forced_cpu_features(); forced.has_value()) {
        return (*forced)[feature_idx];
    }

    KAI_UNUSED(feature);
    return false;
}
#endif

/// Information about the CPU that is executing the program.
struct CpuInfo {
    CpuInfo() :
        has_advsimd(get_cap_support(CpuFeature::ADVSIMD)),
        has_dotprod(get_cap_support(CpuFeature::DOTPROD)),
        has_i8mm(get_cap_support(CpuFeature::I8MM)),
        has_fp16(get_cap_support(CpuFeature::FP16)),
        has_bf16(get_cap_support(CpuFeature::BF16)),
        has_sve(get_cap_support(CpuFeature::SVE)),
        has_sve2(get_cap_support(CpuFeature::SVE2)),
        has_sme(get_cap_support(CpuFeature::SME)),
        has_sme2(get_cap_support(CpuFeature::SME2)) {
    }

    /// Gets the singleton @ref CpuInfo object.
    static const CpuInfo& current() {
        static const CpuInfo cpu_info{};
        return cpu_info;
    }

    const bool has_advsimd{};  ///< AdvSIMD is supported.
    const bool has_dotprod{};  ///< DotProd is supported.
    const bool has_i8mm{};     ///< I8MM is supported.
    const bool has_fp16{};     ///< FP16 is supported.
    const bool has_bf16{};     ///< B16 is supported.
    const bool has_sve{};      ///< SVE is supported.
    const bool has_sve2{};     ///< SVE2 is supported.
    const bool has_sme{};      ///< SME is supported.
    const bool has_sme2{};     ///< SME2 is supported.
};

}  // namespace

/// Helper functions
bool cpu_has_advsimd() {
    return CpuInfo::current().has_advsimd;
}

bool cpu_has_dotprod() {
    return CpuInfo::current().has_dotprod;
}

bool cpu_has_dotprod_and_fp16() {
    return cpu_has_dotprod() && cpu_has_fp16();
}

bool cpu_has_i8mm() {
    return CpuInfo::current().has_i8mm;
}

bool cpu_has_i8mm_and_fp16() {
    return cpu_has_i8mm() && cpu_has_fp16();
}

bool cpu_has_fp16() {
    return CpuInfo::current().has_fp16;
}

bool cpu_has_bf16() {
    return CpuInfo::current().has_bf16;
}

bool cpu_has_sve() {
    return CpuInfo::current().has_sve;
}

bool cpu_has_sve_vl256() {
    if (CpuInfo::current().has_sve) {
        return (kai_get_sve_vector_length_u8() == 32);
    }
    return false;
}

bool cpu_has_sve2() {
    return CpuInfo::current().has_sve2;
}

bool cpu_has_sme() {
    return CpuInfo::current().has_sme;
}

bool cpu_has_sme2() {
    return CpuInfo::current().has_sme2;
}

bool cpu_has_dotprod_and_bf16() {
    return cpu_has_dotprod() && cpu_has_bf16();
}

bool cpu_has_i8mm_and_bf16() {
    return cpu_has_i8mm() && cpu_has_bf16();
}

}  // namespace kai::test

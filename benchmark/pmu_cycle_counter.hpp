//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace kai::benchmark {

/// Architectural PMU cycle counter for a single-core AArch64 EL3 environment.
struct PmuCycleCounter {
    /// Configures the PMU and verifies that the cycle counter advances.
    static bool init() noexcept {
        uint64_t el = 0;
        __asm__ volatile("mrs %0, CurrentEL" : "=r"(el));
        if (el != 12) {
            return false;
        }
        uint64_t features = 0;
        __asm__ volatile("mrs %0, ID_AA64DFR0_EL1" : "=r"(features));
        const uint64_t pmu_version = (features >> 8) & 15;
        if (pmu_version == 0 || pmu_version == 15) {
            return false;
        }

        // Disables counters and interrupts and permits counting at EL3.
        const uint64_t all = 0xffffffff;
        __asm__ volatile(
            "msr pmcntenclr_el0, %0\n"
            "msr pmintenclr_el1, %0\n"
            "msr pmovsclr_el0, %0\n"
            "msr pmccfiltr_el0, xzr\n"
            "isb\n"
            :
            : "r"(all)
            : "memory");
        start();
        const uint64_t begin = read();
        const uint64_t end = read();
        stop();
        return end != begin;
    }

    /// Resets and enables the undivided 64-bit cycle counter.
    static void start() noexcept {
        // PMCR: LC=1, C=1, E=1, D=0, DP=0. PMCNTENSET: C=1.
        const uint64_t control = (1U << 6) | (1U << 2) | 1U;
        const uint64_t enable = uint64_t{1} << 31;
        __asm__ volatile(
            "dsb sy\n"
            "msr pmcr_el0, %0\n"
            "msr pmcntenset_el0, %1\n"
            "isb\n"
            :
            : "r"(control), "r"(enable)
            : "memory");
    }

    /// Reads a counter sample after completing preceding memory operations.
    static uint64_t read() noexcept {
        uint64_t value = 0;
        __asm__ volatile("dsb sy\nisb\nmrs %0, pmccntr_el0\nisb" : "=r"(value) : : "memory");
        return value;
    }

    /// Disables cycle counting before reporting results.
    static void stop() noexcept {
        const uint64_t disable = uint64_t{1} << 31;
        __asm__ volatile("msr pmcntenclr_el0, %0\nisb" : : "r"(disable) : "memory");
    }
};

}  // namespace kai::benchmark

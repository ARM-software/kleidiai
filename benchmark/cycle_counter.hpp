//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace kai::benchmark {

#if defined(__APPLE__) && defined(KAI_BENCHMARK_KPERF) && KAI_BENCHMARK_KPERF

extern "C" {
int kpc_force_all_ctrs_set(int enable);
int kpc_force_all_ctrs_get(int* enable);
int kpc_set_config(uint32_t classes, uint64_t* config);
int kpc_set_counting(uint32_t classes);
int kpc_set_thread_counting(uint32_t classes);
uint32_t kpc_get_counter_count(uint32_t classes);
int kpc_get_thread_counters(uint32_t tid, uint32_t buf_count, uint64_t* buf);
}

inline void log_kpc_error(const char* op) {
    std::cerr << "KPC call failed: " << op << " (errno=" << errno << ": " << std::strerror(errno) << ")\n";
}
#endif

struct CycleCounterState {
    bool available = false;
    bool initialized = false;
#if defined(__APPLE__) && defined(KAI_BENCHMARK_KPERF) && KAI_BENCHMARK_KPERF
    uint32_t classes = 0;
    uint32_t counter_count = 0;
#endif
};

inline CycleCounterState& cycle_counter_state() {
    static CycleCounterState state{};
    return state;
}

inline bool cycle_counter_init() {
    CycleCounterState& state = cycle_counter_state();
    if (state.initialized) {
        return state.available;
    }
    state.initialized = true;

#if defined(__APPLE__) && defined(KAI_BENCHMARK_KPERF) && KAI_BENCHMARK_KPERF
    const auto init_kperf = [&]() -> bool {
        int force_counters = 0;
        errno = 0;
        int rc = kpc_force_all_ctrs_get(&force_counters);
        if (rc != 0) {
            log_kpc_error("kpc_force_all_ctrs_get");
            return false;
        }
        (void)force_counters;
        static constexpr uint32_t KPC_CLASS_FIXED = 0;
        static constexpr uint32_t KPC_CLASS_FIXED_MASK = 1u << KPC_CLASS_FIXED;

        state.classes = KPC_CLASS_FIXED_MASK;

        errno = 0;
        rc = kpc_set_counting(state.classes);
        if (rc != 0) {
            log_kpc_error("kpc_set_counting");
            return false;
        }

        errno = 0;
        rc = kpc_set_thread_counting(state.classes);
        if (rc != 0) {
            log_kpc_error("kpc_set_thread_counting");
            return false;
        }

        errno = 0;
        const uint32_t nctrs = kpc_get_counter_count(state.classes);
        if (nctrs == 0 || nctrs > 64) {
            std::cerr << "KPC counter count invalid: " << nctrs << "\n";
            return false;
        }

        state.counter_count = nctrs;
        state.available = true;
        return true;
    };

    (void)init_kperf();
    if (!state.available) {
        std::cerr << "Unable to enable KPC counters; CPU cycle counter data will be omitted.\n";
    }
#endif

    return state.available;
}

inline void cycle_counter_start() {
#if defined(__APPLE__) && defined(KAI_BENCHMARK_KPERF) && KAI_BENCHMARK_KPERF
    CycleCounterState& state = cycle_counter_state();
    if (state.available) {
        if (kpc_force_all_ctrs_set(1) != 0 || kpc_set_counting(state.classes) != 0 ||
            kpc_set_thread_counting(state.classes) != 0) {
            state.available = false;
        }
    }

#endif
}

inline void cycle_counter_stop() {
#if defined(__APPLE__) && defined(KAI_BENCHMARK_KPERF) && KAI_BENCHMARK_KPERF
    kpc_set_thread_counting(0);
    kpc_set_counting(0);
    kpc_force_all_ctrs_set(0);
#endif
}

inline bool cycle_counter_read(uint64_t& value) {
#if defined(__APPLE__) && defined(KAI_BENCHMARK_KPERF) && KAI_BENCHMARK_KPERF
    CycleCounterState& state = cycle_counter_state();
    if (!state.available) {
        return false;
    }

    uint64_t counters[64] = {};
    errno = 0;
    if (kpc_get_thread_counters(0, state.counter_count, counters) != 0) {
        log_kpc_error("kpc_get_thread_counters");
        state.available = false;
        return false;
    }
    value = counters[0];
    return true;
#else
    (void)value;
    return false;
#endif
}

inline void cycle_counter_shutdown() {
    CycleCounterState& state = cycle_counter_state();
    state.available = false;
    state.initialized = false;
}

}  // namespace kai::benchmark

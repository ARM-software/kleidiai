//
// SPDX-FileCopyrightText: Copyright 2019-2020, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifndef NO_MULTI_THREADING

#include <atomic>

namespace kai {
namespace ops {

class barrier {
private:
    unsigned int        m_threads;

    std::atomic<unsigned int> m_waiters;
    std::atomic<unsigned int> m_leavers;

public:
    barrier(unsigned int threads) : m_threads(threads), m_waiters(0), m_leavers(0) { }

    // Add a move constructor because these objects might be moved around at setup time.
    // Moving while the barrier is active won't work.
    barrier(barrier &&other) : m_threads(other.m_threads), m_waiters(0), m_leavers(0) {
        // This doesn't make it safe, but will have a chance of firing if something odd is occurring.
        assert(other.m_waiters==0);
        assert(other.m_leavers==0);
    }

    /* This isn't safe if any thread is waiting... */
    void set_nthreads(unsigned int nthreads) {
        m_threads = nthreads;
    }

    void arrive_and_wait() {
        m_waiters++;

        while (m_waiters != m_threads) {
            ; /* spin */
        }

        unsigned int v = m_leavers.fetch_add(1);

        if (v == (m_threads - 1)) {
            m_waiters -= m_threads;
            m_leavers = 0;
        } else {
            while (m_leavers > 0) {
                ; /* spin */
            }
        }
    }
};

}  // namespace ops
}  // namespace kai

#else

namespace kai {
namespace ops {

class barrier {
public:
    barrier(unsigned int) { }

    void arrive_and_wait() { }
    void set_nthreads(unsigned int ) { }
};

}  // namespace ops
}  // namespace kai

#endif

//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <functional>

#include "kai/ops/gemm/kai_ops.hpp"

#include "kai/ops/gemm/ndrange.hpp"

/*
 * Class to handle (1D or) 2D splits for kai_ops kernels.
 *
 * This is used for static scheduling (to divide into threads) and dynamic
 * (to divide into granules).
 */
class granulator {
private:
    kai::ops::IGemmCommon * const m_gemm;
    unsigned int                  m_nthreads;
    unsigned int                  m_dim0_parts;
    unsigned int                  m_dim1_parts;

    static unsigned int split_2d(unsigned int max_threads, const kai::ops::ndrange_t &window_size, const unsigned int dim0_override);

public:
    granulator(kai::ops::IGemmCommon *gemm, const unsigned int nthreads, const unsigned int dim0_override=0) :
        m_gemm(gemm),
        m_nthreads(nthreads),
        m_dim0_parts(split_2d(nthreads, gemm->get_window_size(), dim0_override)),
        m_dim1_parts(nthreads / m_dim0_parts)
    {
        auto range = m_gemm->get_window_size();
        unsigned int max_threads = std::min<unsigned int>(m_dim0_parts,range.get_size(0)) * std::min<unsigned int>(m_dim1_parts, range.get_size(1));
#ifndef SILENT
        unsigned int orig_threads = m_nthreads;
#endif

        if (max_threads < m_nthreads) {
            m_dim0_parts = std::min(m_dim0_parts, range.get_size(0));
            m_dim1_parts = std::min(m_dim1_parts, range.get_size(1));

            m_nthreads = max_threads;
            gemm->set_nthreads(max_threads);
        }

#ifndef SILENT
        if(orig_threads > m_nthreads) {
            printf("Granulator: Clamping threads to %u (was %u).\n",m_nthreads,orig_threads);
        }
        printf("Set up scheduling: %ux%u granules (%ux%u total size).\n",m_dim1_parts,m_dim0_parts,range.get_size(1),range.get_size(0));
#endif
    }

    void run_granule(unsigned int granule_id, unsigned int threadid);
};

/*
 * Semi-abstract 'scheduler' class.
 *
 * All implementations will want to capture the gemm pointer and the thread
 * count, so store those in here.
 */
class scheduler {
protected:
    kai::ops::IGemmCommon * const m_gemm;
    const unsigned int            m_nthreads;

public:
    scheduler(kai::ops::IGemmCommon *gemm, const unsigned int nthreads) :
        m_gemm(gemm), m_nthreads(nthreads) { }

    virtual void execute(unsigned int threadid) = 0;

    virtual ~scheduler() { }
};

/* Static scheduler just divides work across threads. */
class static_scheduler : public scheduler {
private:
    granulator m_granulator;

public:
    static_scheduler(kai::ops::IGemmCommon *gemm, const unsigned int nthreads, const unsigned int dim0_override=0) :
        scheduler(gemm, nthreads), m_granulator(m_gemm, m_nthreads, dim0_override) { }

    virtual void execute(unsigned int threadid) override;
};

#ifndef NO_MULTI_THREADING

#include <atomic>
#include <mutex>

/*
 * Dynamic scheduler also needs to track scheduling granules: _pos indicates
 * the next granule to be processed, and _granule_count the total number of
 * granules in the workload.
 */
class dynamic_scheduler : public scheduler {
private:
    std::atomic<unsigned int>  m_pos;
    unsigned int               m_granule_count;
    granulator                 m_granulator;

    static unsigned int compute_granule_count(kai::ops::IGemmCommon *, unsigned int);

public:
    /* Non-trivial constructor to sort of the granule count. */
    dynamic_scheduler(kai::ops::IGemmCommon *gemm, const unsigned int nthreads, const unsigned int granule_count=0);

    virtual void execute(unsigned int threadid) override;
};

class taskmaster_scheduler : public scheduler {
private:
    class thread_queue;

    bool populate_queues();

    void                        *m_threadmem;
    std::vector<thread_queue *>  m_queues;
    std::mutex                   m_lock;
    volatile unsigned int        m_pos=0;
    unsigned int                 m_granule_count;
    unsigned int                 m_completed_count=0;
    granulator                   m_granulator;

    static unsigned int compute_granule_count(kai::ops::IGemmCommon *, unsigned int);

public:
    taskmaster_scheduler(kai::ops::IGemmCommon *gemm, const unsigned int nthreads, const unsigned int granule_count=0);
    ~taskmaster_scheduler() override;

    void execute(unsigned int threadid) override;
};

#endif

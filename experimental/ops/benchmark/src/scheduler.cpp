//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>
#include <cmath>

#include "scheduler.hpp"
#include "gemm_lib.hpp"

/*
 * Helper function to handle 2D work splitting.  This takes two parameters - a thread count and an ndrange_t.
 *
 * The return value is the number of pieces to divide the first dimension into.
 */
unsigned int granulator::split_2d(const unsigned int max_threads, const kai::ops::ndrange_t &range, const unsigned int dim0_override) {
    auto m = range.get_size(0);
    auto n = range.get_size(1);

    // Save some time if it's actually a 1D range.  This avoids the need to
    // prove that the code below will (eventually) do this.
    if (n==1) {
        return max_threads;
    }

    // If we have a preconfigured shape, check it is legal before returning it.
    if (dim0_override > 0) {
        if ((dim0_override <= max_threads) &&
            (dim0_override <= m) &&
            (max_threads % dim0_override == 0)) {
            return dim0_override;
        }
    }

    // We assume that we want to divide the work into approximately square blocks.
    //
    // Therefore we want to find m, n such that m/n ~= aspect ratio of work area and m*n == total threads.
    //
    // So n = m / aspect ratio, substituting into the other equation we get m^2/ratio == total threads
    // Therefore m = sqrt(total threads * ratio)
    float ratio = m / static_cast<float>(n);

    unsigned int ideal_height = ::round(std::sqrt(max_threads * ratio));

    // If it's so incredibly wide that it rounds down to zero, return 1.
    if (ideal_height == 0) {
        return 1;
    }

    // Search up and down from the "ideal" number until we find a ratio that works.
    for (unsigned int adj=0; adj<ideal_height; adj++) {
        const unsigned int round_down = ideal_height - adj;

        if (max_threads % round_down == 0) {
            return round_down;
        }

        const unsigned int round_up = ideal_height + adj;

        if (max_threads % round_up == 0) {
            return round_up;
        }
    }

    // We can't get here - 'round_down' will become 1 before the loop terminates.
    assert(false);
    __builtin_unreachable();
}

void granulator::run_granule(unsigned int granule_id, unsigned int threadid) {
    // Ignore out of range granules.
    if (granule_id >= this->m_nthreads) {
        return;
    }

    auto winsize = this->m_gemm->get_window_size();

    // Dimension 0 is M, and M is rows.  So "X" is columns which is dimension 1.
    auto thread_y = granule_id % m_dim0_parts;
    auto thread_x = granule_id / m_dim0_parts;

    auto size_y = winsize.get_size(0);
    auto size_x = winsize.get_size(1);

    auto y_start = (thread_y * size_y) / m_dim0_parts;
    auto y_end = ((thread_y+1) * size_y) / m_dim0_parts;

    auto x_start = (thread_x * size_x) / m_dim1_parts;
    auto x_end = ((thread_x+1) * size_x) / m_dim1_parts;

    kai::ops::ndcoord_t pos = { { y_start, y_end - y_start }, { x_start, x_end - x_start} };
    kai::ops::ndcoord_t thr = { { thread_y, m_dim0_parts }, { thread_x, m_dim1_parts } };

    if (!(x_end > x_start && y_end > y_start)) {
        printf("Warning: no work for thread %u\n", threadid);
        return;
    }

//    printf("Thread %u, running granule %u x=%u-%u (of %u) y=%u-%u (of %u)\n",threadid, granule_id,x_start,x_end,size_x,y_start,y_end,size_y);

    this->m_gemm->execute(pos, thr, threadid);
}

void static_scheduler::execute(unsigned int threadid) {
    m_granulator.run_granule(threadid, threadid);
}

#ifndef NO_MULTI_THREADING

unsigned int dynamic_scheduler::compute_granule_count(kai::ops::IGemmCommon *gemm, unsigned int granule_count) {
    // The actual granule count can never exceed the window size of the GEMM.
    // We also set these equal if the supplied granule count is 'zero'.  Fix that up here.
    unsigned int winsize = gemm->get_window_size().total_size();

    if (granule_count==0 || granule_count > winsize) {
        granule_count = winsize;
    }

    return granule_count;
}

dynamic_scheduler::dynamic_scheduler(kai::ops::IGemmCommon *gemm, const unsigned int nthreads, const unsigned int granule_count)
        : scheduler(gemm, nthreads), m_pos(0),
          m_granule_count(compute_granule_count(gemm, granule_count)),
          m_granulator(m_gemm, m_granule_count)
{
}

void dynamic_scheduler::execute(unsigned int threadid) {
    for (;;) {
        unsigned int granule = m_pos.fetch_add(1);

        if (granule >= m_granule_count) {
            // Every thread will obtain one last granule which is out of
            // bounds (indicating that it should exit).  This is true even
            // if there are races where the other threads complete all the
            // work before the last threads start up (in that case the last
            // threads will still request one granule, which will be out of
            // bounds).
            //
            // As IDs start from zero, this means the last thread to exit
            // will get a granule ID equal to (granule count + thread count)
            // - 1.
            //
            // If that's us, reset the position to zero for the next
            // iteration.  This is safe, because we get that granule ID iff
            // no other threads are going to request one until after we have
            // exited (we assume that the GEMM will not be executed again
            // until all threads have finished it this time).
            if (granule == m_granule_count + this->m_nthreads - 1) {
                m_pos=0;
            }

            break;
        }

        m_granulator.run_granule(granule, threadid);
    }
}

#define FIFO_SIZE 16

class taskmaster_scheduler::thread_queue {
private:
    // Class to maintain a head/tail pointer that should occupy its own
    // cache line (assumes 64 byte cache lines)
    class ht_ptr {
    private:
        volatile uint64_t  m_value;
        uint64_t           padding[7] = {};

    public:
        ht_ptr() : m_value(0) { }

        uint64_t get() const {
            return m_value;
        }

        uint64_t get_and_increment() {
            return m_value++;
        }
    };

    ht_ptr        m_head;
    ht_ptr        m_tail;
    unsigned int  m_fifo[FIFO_SIZE] = {};

public:
    thread_queue() : m_head(), m_tail() { }

    unsigned int size() {
        return m_head.get() - m_tail.get();
    }

    void additem(unsigned int v) {
        m_fifo[m_head.get_and_increment() % FIFO_SIZE] = v;
    }

    unsigned int popitem() {
        assert(size() > 0);
        return m_fifo[m_tail.get_and_increment() % FIFO_SIZE];
    }
};

bool taskmaster_scheduler::populate_queues() {
    if (m_pos >= m_granule_count) {
        // Reset logic: Every thread will call into this function with the
        // lock held, as a "true" return from here is the only way out of
        // the main loop.
        //
        // So just count the threads in, when the last one completes reset
        // the completion count and the position.
        m_completed_count++;

        if (m_completed_count == this->m_nthreads) {
            m_completed_count = 0;
            m_pos = 0;
        }

        return true;
    }

    // Cache the sizes here.
    std::vector<unsigned int> sizes(this->m_nthreads);

    for (unsigned int t=0; t<this->m_nthreads; t++) {
        sizes[t] = m_queues[t]->size();
    }

    // "Deal" out work to the shortest queues first.
    for (unsigned int s=0; s<(FIFO_SIZE-1); s++) {
        for (unsigned int t=0; t<this->m_nthreads; t++) {
            if (sizes[t] <= s) {
                m_queues[t]->additem(m_pos++);

                if (m_pos >= m_granule_count) {
                    return false;
                }
            }
        }
    }

    return false;
}

unsigned int taskmaster_scheduler::compute_granule_count(kai::ops::IGemmCommon *gemm, unsigned int granule_count) {
    unsigned int winsize = gemm->get_window_size().total_size();

    if (granule_count == 0 || granule_count > winsize) {
        granule_count = winsize;
    }

    return granule_count;
}

taskmaster_scheduler::taskmaster_scheduler(kai::ops::IGemmCommon *gemm, const unsigned int nthreads, const unsigned int granule_count)
        : scheduler(gemm, nthreads), m_pos(0), m_granule_count(compute_granule_count(gemm, granule_count)),
          m_granulator(m_gemm, m_granule_count) {
    printf("Taskmaster scheduler initialized; %u granules.\n", m_granule_count);

    m_queues.resize(nthreads);

    m_threadmem = allocate_aligned_memory( 64, sizeof(thread_queue) * nthreads );
    for (unsigned int i=0; i<nthreads; i++) {
        m_queues[i] = new (static_cast<thread_queue *>(m_threadmem) + i) thread_queue();
    }
}

taskmaster_scheduler::~taskmaster_scheduler() {
    for(unsigned int i=0; i<this->m_nthreads; i++) {
        m_queues[i]->~thread_queue();
    }

    free_aligned_memory(this->m_threadmem);
}

void taskmaster_scheduler::execute(unsigned int threadid) {
    for(;;) {
        if (m_queues[threadid]->size() > 0) {
            unsigned int v = m_queues[threadid]->popitem();

            m_granulator.run_granule(v, threadid);

            continue;
        }

        if (m_lock.try_lock()) {
            bool abort = false;

            if (m_queues[threadid]->size() == 0) {
                abort = populate_queues();
            }

            m_lock.unlock();

            if (abort)
                break;
        }
    }
}

#endif

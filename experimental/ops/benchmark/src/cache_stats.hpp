//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Class to collect cache statistics using perf events
//
// It would be possible to create a bare metal version of this that hits the PMU registers directly, but for now
// this is Linux only, using the perf interface.

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <unistd.h>

#define NUM_EVENTS 6

// PMU event IDs for L1D fill, L1D access, L2 fill, L2 access, LLC fill, LLC access.
// This relationship is relied on in the statistics output function below.
static const uint32_t event_ids[] = { 0x3, 0x4, 0x17, 0x16, 0x37, 0x36 };

#ifdef __linux__

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <syscall.h>

namespace {

static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags)
{
    int ret;

    ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                  group_fd, flags);
    return ret;
}

} // anonymous namespace

#endif

class cache_stats {
    // Cache statistics need to be gathered per thread - one of these needs to be created per active thread (this
    // construction can occur anywhere), then init() is called to set up the counter (this must be done on the
    // thread in question), then start_count() and stop_count() (which probably work from anywhere but at present
    // are also called on the thread in question).
    //
    // stop_count() collects the stats into the m_counters[] array for each thread, then "report_summary()" in the
    // main class (below) aggregates across the threads.
    class cache_stats_thread {
        uint64_t m_counters[NUM_EVENTS];
        uint64_t m_ids[NUM_EVENTS];
        int m_fds[NUM_EVENTS];
        bool m_valid;

        struct read_format {
            uint64_t nr;
            uint64_t time_enabled;
            uint64_t time_running;
            struct {
                uint64_t value;
                uint64_t id;
            } values[NUM_EVENTS];
        };

public:
        cache_stats_thread() : m_valid(false) {
            // Initialize the 'fds' array so we can tidy up properly later.
            for (unsigned int i=0; i<NUM_EVENTS; i++) {
                m_fds[i] = -1;
            }
        }

        ~cache_stats_thread() {
            for (unsigned int i=0; i<NUM_EVENTS; i++) {
                if (m_fds[i] >= 0) {
                    close(m_fds[i]);
                }
            }
        }

        // Initialise counters: this is not included in the constructor to allow construction en-masse and
        // initialization per-thread later.
        void init() {
#ifdef __linux__
            // Bare metal code would need to do this directly.

            for (unsigned int i=0; i<NUM_EVENTS; i++) {
                struct perf_event_attr pea;

                memset(&pea, 0, sizeof(struct perf_event_attr));
                pea.type = PERF_TYPE_RAW;
                pea.size = sizeof(struct perf_event_attr);
                pea.config = event_ids[i];
                pea.disabled = 1;
                pea.exclude_kernel = 1;
                pea.exclude_hv = 1;
                pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

                m_fds[i] = perf_event_open(&pea, 0, -1, (i == 0) ? -1 : m_fds[0], 0);
                if (m_fds[i] == -1) {
                    printf("Can't set up perf event %u!\n", i);
                    perror("perf_event_open");
                    return;
                }
                ioctl(m_fds[i], PERF_EVENT_IOC_ID, &m_ids[i]);
            }

            m_valid = true;
#endif
        }

        void start_count() {
            if (!m_valid) {
                return;
            }
#ifdef __linux__
            ioctl(m_fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
            ioctl(m_fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
#endif
        }

        void stop_count() {
            if (!m_valid) {
                for (unsigned int j=0; j<NUM_EVENTS; j++) {
                    m_counters[j] = 0;
                }
                return;
            }

#ifdef __linux__
            read_format res;

            ioctl(m_fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

            ssize_t r = read(m_fds[0], &res, sizeof(res));

            if (r != sizeof(res)) {
                printf("perf: bad read length %zd, expected %zu\n", r, sizeof(res));
            }

            if (res.nr != NUM_EVENTS) {
                printf("perf: number of events mismatch!\n");
                return;
            }

            for (unsigned int i=0; i<NUM_EVENTS; i++) {
                for (unsigned int j=0; j<NUM_EVENTS; j++) {
                    if (res.values[i].id == m_ids[j]) {
                        m_counters[j] = res.values[i].value;
                        break;
                    }
                }
            }
#endif
        }

        const uint64_t *get_counters() const {
            return m_counters;
        }

        bool is_valid() const {
            return m_valid;
        }
    };

    std::vector<cache_stats_thread> m_thread_data;

public:
    cache_stats(unsigned int num_threads) : m_thread_data(num_threads) { }

    cache_stats_thread *get_thread_entry(unsigned int threadid) {
        return &(m_thread_data[threadid]);
    }

    void report_summary() {
        uint64_t totals[NUM_EVENTS];

        for (unsigned int j=0; j<NUM_EVENTS; j++) {
            totals[j] = 0;
        }

        for (unsigned int i=0; i<m_thread_data.size(); i++) {
            if (!m_thread_data[i].is_valid()) {
                printf("At least some performance counters failed to initialise - not reporting cache data.\n");
                return;
            }

            const uint64_t *thread_counters = m_thread_data[i].get_counters();

            for (unsigned int j=0; j<NUM_EVENTS; j++) {
                totals[j] += thread_counters[j];
            }
        }

        const char *descs[] = { "L1", "L2", "LL" };

        for (unsigned int j=0; j<NUM_EVENTS/2; j++) {
            uint64_t accesses = totals[j*2+1];
            uint64_t fills = totals[j*2];

            printf("%s cache: %" PRIu64 " accesses, %" PRIu64 " fills, %.1f%% miss rate.\n", descs[j], accesses, fills, (float)fills*100/accesses);
        }
    }
};

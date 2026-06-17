//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>

#include <time.h>

#include "cache_stats.hpp"

#include "gemm_lib.hpp"
#include "kai/ops/bfloat.hpp"

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif

#ifdef BARE_METAL
#ifndef __aarch64__
static unsigned int s;
#endif // __aarch64__

static void enable_cyclecounter() {
#ifdef __aarch64__
    uint64_t tmp;

    __asm __volatile (
        "mrs	%[tmp], pmcr_el0\n"
        "orr	%[tmp], %[tmp], #1\n"
        "msr	pmcr_el0, %[tmp]\n"
        "mrs	%[tmp], pmcntenset_el0\n"
        "orr	%[tmp], %[tmp], #1<<31\n"
        "msr    pmcntenset_el0, %[tmp]\n"
        : [tmp] "=r" (tmp)
    );
#else
    s = clock();
#endif
}

static void disable_cyclecounter() {
#ifdef __aarch64__
    uint64_t tmp;

    __asm __volatile (
            "mov   %[tmp], #0x3f\n"
            "orr   %[tmp], %[tmp], #1<<31\n"
            "msr    pmcntenclr_el0, %[tmp]\n"
            : [tmp] "=r" (tmp)
            );
#endif
}

static inline uint64_t get_cyclecounter() {
#ifdef __aarch64__
    uint64_t retval;

    __asm __volatile (
        "mrs	%[retval], pmccntr_el0\n"
    : [retval] "=r" (retval));

    return retval;
#else
    return clock()-s;
#endif
}
#else

/* Prototypes from perf.cpp */
void start_counter(int fd);
long long get_counter(int fd);
long long stop_counter(int fd);
int open_instruction_counter(void);
int open_cycle_counter(void);

#endif

class run_timer {
private:
    uint64_t total=0;
    uint64_t first=0;
    uint64_t best=0;
    unsigned int runs=0;
    unsigned int run_delay=0;
    unsigned int min_ms=0;

    uint64_t macs;

    uint64_t grand_total=0;
    unsigned int grand_total_iters=0;

    uint64_t best_batch_total;
    unsigned int best_batch_iters=0;

#ifdef CYCLE_TIMING
    const char *time_units = "cycles";
    const char *rate_units = "MACs/cycle";
#else
    const char *time_units = "ns";
    const char *rate_units = "GMAC/sec";
#endif

public:
    run_timer(unsigned int run_delay, unsigned int min_ms) : run_delay(run_delay), min_ms(min_ms) { }

    ~run_timer() {
#ifndef SILENT
        if (grand_total_iters <= 1) {
            return;
        }

        if (min_ms==0) {
            if (grand_total_iters > 1) {
                printf("%u total runs, %" PRId64 " total %s, %.2f average %s, %" PRId64 " lowest %s.\n",grand_total_iters,grand_total,time_units,(float)grand_total/grand_total_iters,time_units,best,time_units);
            }
            if (grand_total_iters > 2) {
                printf("   %.2f average %s (excluding first run)\n",(float)(grand_total-first)/(grand_total_iters-1),time_units);
            }
            return;
        }

        printf("Grand total: %u iterations, %" PRId64 " %s, %.1f average %s, %.2f %s.\n",grand_total_iters,grand_total,time_units,(float)grand_total/grand_total_iters,time_units,(float)(macs * grand_total_iters) / grand_total,rate_units);
        printf("Best batch: %u iterations, %" PRId64" %s, %.1f average %s, %.2f %s.\n",best_batch_iters,best_batch_total,time_units,(float)best_batch_total/best_batch_iters,time_units,(float)(macs * best_batch_iters) / best_batch_total,rate_units);
        printf("Best single iteration: %" PRId64" %s, %.2f %s\n",best,time_units,(float)macs/best,rate_units);
#endif
    }

    // Do we need more iterations in this run?
    // "Total" will actually measure either ns or cycles, so in cycle mode "ms" means "millions of cycles".
    bool need_more() {
        if (runs==0 || total < (min_ms * 1000000)) {
            return true;
        }

        grand_total += total;
        grand_total_iters += runs;

        // In "single run" mode, produce the same output as before
        if (min_ms==0) {
#ifndef SILENT
            printf("Benchmark: %" PRId64 " %s, %f %s.\n",total,time_units,(float)macs/total,rate_units);
#endif // !SILENT
            total=0;
            runs=0;
            return false;
        }

        // In "multi run" mode, produce a summary for this batch of runs.
#ifndef SILENT
        printf("%u runs, %f avg %s, %f avg %s.\n",runs,(float)total/runs,time_units,(float)(macs * runs)/total,rate_units);
#endif // !SILENT

	if (best_batch_iters == 0 || ((float)total / runs) < ((float)best_batch_total / best_batch_iters)) {
	    best_batch_total = total;
	    best_batch_iters = runs;
        }

        total=0;
        runs=0;

        return false;
    }

    template <typename T>
    void operator()(const char *name, uint64_t macs, T func) {
        this->macs = macs;

        uint64_t res;
#ifdef CYCLE_TIMING
#ifdef BARE_METAL
        uint64_t start, end;

        enable_cyclecounter();

        start = get_cyclecounter();
        func();
        end = get_cyclecounter();
        disable_cyclecounter();

        res=end-start;
#else
        int cyclefd = open_cycle_counter();

        start_counter(cyclefd);
        func();
        res=stop_counter(cyclefd);

        // If run_delay is defined, wait for a bit here.
        sleep(run_delay);

        close(cyclefd);
#endif
#else
        struct timespec ts1, ts2;

#ifdef __APPLE__
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        func();
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts2);
#else
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        func();
        clock_gettime(CLOCK_MONOTONIC, &ts2);
#endif // __APPLE__

        res=(ts2.tv_sec - ts1.tv_sec) * 1000000000 + (ts2.tv_nsec - ts1.tv_nsec);
        // If run_delay is defined, wait for a bit here.
        sleep(run_delay);
#endif
        total += res;
        if (first==0) {
            first = res;
        }
        if (best==0 or res < best) {
            best = res;
        }
        runs++;
    }
};

// Need "cache stats" class similar to run timer
//
// - Core class with aggregated stats, averages, etc.
// - Per thread subclass that manages events, reads each iteration and combines on to aggregated stats
// - For each iteration

/*
 * Wrapper to allow "legacy" or "new" kernels to be used by the test/benchmark functions.
 *
 * This is the "legacy" version - extract M, N, K from the GemmProblem and pass to kernel constructor.
 */
template<bool legacy, QuantizationType quantized, bool perchannel, typename T>
struct TestKernel {
    template <typename To, typename Tr>
    static T get(To A, To B, Tr C, Tr bias, GemmProblem *p, int nthreads, bool do_init) {
        return T(A,B,C,p->output_width,p->output_channels,p->input_channels,nthreads,do_init);
    }

    /* This method makes no sense for legacy kernels (there is only one
     * kernel which is directly implemented.  */
    static void print_kernels(GemmProblem *p, unsigned int nthreads) {
        printf("Legacy kernel - can't list kernels.\n");
    }
};

/*
 * This is the "new" version - pass GemmProblem directly to the kernel constructor.
 */
template<bool perchannel, typename T>
struct TestKernel<false, QuantizationType::NONE, perchannel, T> {
    template <typename To, typename Tr>
    static T get(To A, To B, Tr C, Tr bias, GemmProblem *p, int nthreads, bool do_init) {
        return T(A,B,C,bias,p,nthreads,do_init);
    }

    static void print_kernels(GemmProblem *p, unsigned int nthreads) {
        T::print_kernels(p, nthreads);
    }
};

template<bool perchannel, typename T>
struct TestKernel<false, QuantizationType::FLOAT, perchannel, T> {
    template <typename To, typename Tw, typename Tr>
    static T get(std::shared_ptr<Matrix<To>> &A, std::shared_ptr<Matrix<Tw>> &B, std::shared_ptr<Matrix<Tr>> &C,
                 std::shared_ptr<Matrix<Tr>> &bias, GemmProblem *p, int nthreads, bool do_init) {
        QuantizeParameters<To> qp_A = QuantizeParameters<To>(0, std::pair<float, float>(p->after_relu ? 0.0f : -1.0f, 6.0f));
        QuantizeParameters<To> qp_B = QuantizeParameters<To>(0, std::pair<float, float>(-1.0f, 1.0f));
        kai::ops::DequantizeFloat scale(qp_A.m_scale * qp_B.m_scale);
        return T(A,B,C,bias,scale,p,nthreads,do_init);
    }

    static void print_kernels(GemmProblem *p, unsigned int nthreads) {
        T::print_kernels(p, nthreads);
    }
};

/* This is the quantized version. */
template<bool perchannel, typename T>
struct TestKernel<false, QuantizationType::INTEGER, perchannel, T> {
    template <typename To, typename Tw, typename Tr>
    static T get(std::shared_ptr<Matrix<To>> &A, std::shared_ptr<Matrix<Tw>> &B, std::shared_ptr<Matrix<Tr>> &C,
                 std::shared_ptr<Matrix<Tr>> &bias, GemmProblem *p, int nthreads, bool do_init) {
        unsigned int quantize_sets = perchannel ? p->output_channels : 1;

        /* Invent some dummy quantize parameters */
        bool do_relu = (p->act.type == Activation::Type::ReLU || p->act.type == Activation::Type::BoundedReLU);

	auto bias_real = std::make_shared<Matrix<int32_t>>(1, p->output_channels, p->output_channels, 1, p->multis);

	if (bias) {
	    bias_real->Copy(bias);
        } else {
            bias_real.reset();
        }

        QuantizeParameters<To> qp_A = QuantizeParameters<To>(std::pair<float, float>(p->after_relu ? 0.0f : -1.0f, 6.0f));
        std::vector<QuantizeParameters<Tw> > qp_B;
        for (unsigned int i=0; i<quantize_sets; i++) {
            if (perchannel) {
                qp_B.emplace_back(QuantizeParameters<Tw>(0, std::pair<float, float>(-1.0f, 1.0f)));
            } else {
                qp_B.emplace_back(QuantizeParameters<Tw>(std::pair<float, float>(-1.0f, 1.0f)));
            }
        }
        QuantizeParameters<Tr> qp_C = QuantizeParameters<Tr>(std::pair<float, float>(do_relu ? 0.0f : -6.0f, 6.0f));

        return T(A,B,C,bias_real,qp_A,qp_B,qp_C,p,nthreads,do_init);
    }

    static void print_kernels(GemmProblem *p, unsigned int nthreads) {
        T::print_kernels(p, nthreads);
    }
};

/*
 * This function benchmarks a kernel class, as included above.
 *
 * The kernel to be benchmarked is templated on the kernel class.
 *
 * The output is ignored, so for safety the kernel will need some mechanism
 * such as a volatile asm statement to prevent the compiler optimizing it
 * out.
 */
template <typename T, bool legacy, QuantizationType quantized, bool perchannel>
void benchmark(GemmProblem *p, int iterations, int nthreads, int run_delay, mapfn cpuid_map, const char *kernel_name) {
    const int total_threads = nthreads * p->n_teams;
    uint64_t mac_count = (int64_t)p->output_height*p->output_width*p->kernel_height*p->kernel_width*p->output_channels*(p->input_channels/p->groups)*p->batches*p->multis*p->n_teams;
    cache_stats cs(total_threads);

    if (p->add_problem) {
        mac_count /= p->input_channels;
    }

    // Each team has its own set of operand matrices, and it's own instance
    // of the test kernel.  Deal with that in this class.
    struct team_data {
        GemmProblem * &p;
        int nthreads;

        std::shared_ptr<Matrix <typename T::lhs_operand_type> > A;
        std::shared_ptr<Matrix <typename T::rhs_operand_type> > B;
        std::shared_ptr<Matrix <typename T::result_type> > C;
        std::shared_ptr<Matrix <typename T::result_type> > bias;

        T kern;

        static std::shared_ptr<Matrix <typename T::lhs_operand_type> > get_a_matrix(const GemmProblem *p, std::shared_ptr<Matrix <typename T::lhs_operand_type> > from = nullptr) {
            auto A = std::make_shared<Matrix <typename T::lhs_operand_type> >((p->input_height * p->input_width), p->input_channels, std::max(p->input_channels, p->a_stride), p->batches, p->multis);

#ifndef BARE_METAL
            if (from) {
                A->Copy(from);
            } else {
                A->Randomize();
            }
#endif

            return A;
        }

        static std::shared_ptr<Matrix <typename T::rhs_operand_type> > get_b_matrix(const GemmProblem *p, std::shared_ptr<Matrix <typename T::rhs_operand_type> > from = nullptr) {
            auto gemm_k = (p->input_channels / p->groups) * p->kernel_height * p->kernel_width;
            auto B = std::make_shared<Matrix <typename T::rhs_operand_type> >(gemm_k, p->output_channels, std::max(p->output_channels, p->b_stride), 1, p->multis);

#ifndef BARE_METAL
            if (from) {
                B->Copy(from);
            } else {
                B->Randomize();
            }
#endif

            return B;
        }

        static std::shared_ptr<Matrix <typename T::result_type> > get_bias(const GemmProblem *p, std::shared_ptr<Matrix <typename T::result_type> > from = nullptr) {
            if (!p->use_bias) {
                return nullptr;
            }

            auto bias = std::make_shared<Matrix <typename T::result_type> >(1, p->output_channels, p->output_channels, 1, p->multis);

#ifndef BARE_METAL
            if (from) {
                bias->Copy(from);
            } else {
                bias->Randomize();
            }
#endif
            return bias;
        }

        static bool init_needed() {
#ifdef BARE_METAL
            return false;
#else
            return true;
#endif
        }

        team_data(GemmProblem *&p, int nthreads) :
            p(p), nthreads(nthreads),
            A(get_a_matrix(p)), B(get_b_matrix(p)),
            C(std::make_shared<Matrix <typename T::result_type> >((p->output_height * p->output_width), p->output_channels, p->output_channels, p->batches, p->multis)),
            bias(get_bias(p)),
            kern(TestKernel<legacy, quantized, perchannel, T>::get(A, B, C, bias, p, nthreads, init_needed()))
        {
        }

        team_data(GemmProblem *&p, int nthreads, team_data &from) :
            p(p), nthreads(nthreads),
            A(get_a_matrix(p, from.A)), B(get_b_matrix(p, from.B)),
            C(std::make_shared<Matrix <typename T::result_type> >((p->output_height * p->output_width), p->output_channels, p->output_channels, p->batches, p->multis)),
            bias(get_bias(p, from.bias)),
            kern(TestKernel<legacy, quantized, perchannel, T>::get(A, B, C, bias, p, nthreads, init_needed()))
        {
        }
    };

    int m_round = T::get_m_block();
    int n_round = T::get_n_block();
    int k_round = T::get_k_block();

    // Round requested sizes to something the kernel can do.
    if (m_round > 1 || n_round > 1 || k_round > 1) {
        assert(p->is_basic_gemm());

        p->output_width=p->input_width = round_up_to_nearest_multiple(p->input_width, m_round);
        p->output_channels = round_up_to_nearest_multiple(p->output_channels, n_round);
        p->input_channels = round_up_to_nearest_multiple(p->input_channels, k_round);
    }

    std::vector<team_data> teams;
    run_timer thetimer(run_delay, p->min_ms);

#ifdef TEST_TEAMS
    // In team test mode, copy the data for subsequent teams from team 0.
    // That way we can check that the results at the end are the same.
    teams.emplace_back(p, nthreads);
    for (int i=1; i<p->n_teams; i++) {
        teams.emplace_back(p, nthreads, teams[0]);
    }
#else
    // In normal mode, each team generates its own data.
    for (int i=0; i<p->n_teams; i++) {
        teams.emplace_back(p, nthreads);
    }
#endif

    // For the benchmark, print something out to confirm sizes.  This can be
    // checked against "golden" output to make sure we're not running
    // unexpected sizes.
#ifndef SILENT
    if (p->is_basic_gemm()) {
        printf("Starting MM benchmark: threads=%d teams=%d iterations=%d batches=%lld multis=%lld M=%lld N=%lld K=%lld\n", nthreads, p->n_teams, iterations, (long long)p->batches, (long long)p->multis, (long long)p->output_width, (long long)p->output_channels, (long long)p->input_channels);
    } else {
        printf("Starting Conv2D benchmark: threads=%d teams=%d iterations=%d batches=%lld multis=%lld input=%lldx%lld, input channels=%lld, "
               "kernel=%lldx%lld, stride=%lldx%lld, dilation=%lldx%lld, output=%lldx%lld, output channels=%lld\n",
                   nthreads, p->n_teams, iterations, (long long)p->batches, (long long)p->multis, (long long)p->input_height, (long long)p->input_width,
                    (long long)p->input_channels,
                    (long long)p->kernel_height, (long long)p->kernel_width,
                    (long long)p->out_stride_h, (long long)p->out_stride_w,
                    (long long)p->in_stride_h, (long long)p->in_stride_w,
                    (long long)p->output_height, (long long)p->output_width,
                    (long long)p->output_channels);
    }
#endif


#ifdef THREADS
#define COMPLETER_STRIDE 1
    if (total_threads > 1) {
        // In multithreaded mode, we start up the extra threads and have them indicate they are waiting first.
        // The actual timed operation consists of: Releasing the waiters, doing the thread 0 work and joining the worker threads.
        std::atomic_int waiters(0);
        volatile int *completed = (volatile int *)malloc(COMPLETER_STRIDE * total_threads * sizeof(int));
        std::vector<std::thread> thethreads;
        volatile int go=0;
        unsigned int maxcpus = std::thread::hardware_concurrency();
        volatile int reset=0;
        volatile int finished=0;


        for (int i=1; i<total_threads; i++) {
            thethreads.emplace_back( [&](int myid) {
#ifdef BIND_THREADS
                if (cpuid_map) {
                    // Bind thread to relevant CPU.
                    cpu_set_t set;

                    CPU_ZERO(&set);
                    CPU_SET(cpuid_map(myid, maxcpus), &set);

                    if (sched_setaffinity(0, sizeof(set), &set)) {
                        perror("Error setting worker thread affinity");
                    }
                }
#endif
                int myteam = myid / nthreads;
                int mythread = myid % nthreads;

                auto csd = cs.get_thread_entry(myid);
                if (p->cache_stats) {
                    csd->init();
                }

                // Worker threads will loop forever until "finished" gets set.
                for (;;) {
                    completed[myid * COMPLETER_STRIDE] = 0;

                    __asm __volatile ("dsb sy");

                    waiters++;


                    while (go==0) ;

                    if (finished) {
                        break;
                    }

                    if (p->cache_stats) {
                        csd->start_count();
                        teams[myteam].kern.Run(mythread);
                        csd->stop_count();
                    } else {
                        teams[myteam].kern.Run(mythread);
                    }

                    completed[myid * COMPLETER_STRIDE] = 1;


                    while (reset==0) ;
                }
            }, i);
        }

#ifdef BIND_THREADS
        if (cpuid_map) {
            // Bind main thread
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpuid_map(0, maxcpus), &set);
            if(sched_setaffinity(0, sizeof(set), &set)) {
                perror("Error setting main thread affinity");
            }
        }
#endif

        auto csd = cs.get_thread_entry(0);
        if (p->cache_stats) {
            csd->init();
        }

        for (int it=0; it<iterations; it++) {
            while (thetimer.need_more()) {
                while (waiters < (total_threads-1) ) {

                }

                reset=0;
                waiters=0;
                __asm __volatile ("dsb sy");


                thetimer("Benchmark", mac_count, [&](void) {
                    go=1;

                    if (p->cache_stats) {
                        csd->start_count();
                        teams[0].kern.Run(0);
                        csd->stop_count();
                    } else {
                        teams[0].kern.Run(0);
                    }


                    for (int i=1; i<total_threads; i++) {
                        while (!completed[i * COMPLETER_STRIDE]) {
                            /* spin until thread complete */
                        }
                    }
                });

                csd->stop_count();

                go=0;
                __asm __volatile ("dsb sy");

                reset=1;
            }
            if (p->cache_stats) {
                cs.report_summary();
            }
        }

        /* Make the worked threads exit */
        {
            finished=1;
            __asm __volatile ("dsb sy");
            go=1;
        }

        /* Clear up worker threads. */
        for (int i=1; i<total_threads; i++) {
            thethreads[i-1].join();
        }

        free((void *)completed);
    } else {
#else /* THREADS */
    if (1) {
#endif /* THREADS */
#ifdef BIND_THREADS
        if (cpuid_map) {
            // If bind is selected, bind even in single thread mode.
            unsigned int maxcpus = std::thread::hardware_concurrency();
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpuid_map(0, maxcpus), &set);
            if(sched_setaffinity(0, sizeof(set), &set)) {
                perror("Error setting main thread affinity");
            }
        }
#endif
        auto csd = cs.get_thread_entry(0);
        if (p->cache_stats) {
            csd->init();
        }

        for (int it=0; it<iterations; it++) {
            while(thetimer.need_more()) {
                thetimer("Benchmark", mac_count, [&](void) {
                    if (p->cache_stats) {
                        csd->start_count();
                        teams[0].kern.Run(0);
                        csd->stop_count();
                    } else {
                        teams[0].kern.Run(0);
                    }
                });
            }
            if (p->cache_stats) {
                cs.report_summary();
            }
        }
    }

#ifdef TEST_TEAMS
    // Make sure all the teams got the same result.
    if (p->n_teams > 1) {
        for (int t=1; t<p->n_teams; t++) {
            teams[0].C->Compare(teams[t].C);
        }
    }
#endif
}

// Helper class to generate test data such that no rounding needs to occur.
// This means results should be the same regardless of accumulation order
// (which is not guaranteed and varies by kernel and blocking strategy).
//
// For the most part this uses integers as it is easier to reason about.
//
// Work out how many useful mantissa bits can be stored in the result (i.e.
// what size integer the result type can reliably represent exactly - for FP
// types this includes the implicit 1 bit before the binary point), and how
// many results will be accumulated.  This gives a number of bits that each
// result can have, which can then be divided between the LHS and RHS
// operands.
//
// When the reference result is computed, the positive and negative
// contributions to each result are accumulated separately and checked that
// they are smaller than the maximum representable integer mentioned above.
// This means that no precision can be lost by rounding regardless of how
// the sum is reassociated.
//
// Some configurations require more precision in accumulation than the final
// result (e.g.  FP16 operands and result, FP32 accumulation).  For these,
// use fractional bits in the operands to make sure the accumulation is
// being done properly.
template<typename Tres>
class test_data_helper {
private:
    int acc_operand_bits;
    int res_operand_bits;

    int lhs_acc_bits;
    int lhs_res_bits;

    int rhs_acc_bits;
    int rhs_res_bits;

    int acc_mantissa_bits;
    int res_mantissa_bits;

public:
    test_data_helper(GemmProblem *p) {
        const int64_t input_hw = p->input_height * p->input_width;
        const int64_t kernel_hwi = p->kernel_height * p->kernel_width * (p->input_channels / p->groups);
        const int64_t output_hw = p->output_height * p->output_width;

        if (std::is_same<Tres, bfloat16>::value) {
            // BF16 result implies BF16 operands
            // Fast mode implies BF16 accumulation, otherwise it's FP32
            res_mantissa_bits = 6;
            if (p->fast_mode) {
                acc_mantissa_bits = 6;
            } else {
                acc_mantissa_bits = 24;
            }
        } else if (std::is_same<Tres, __fp16>::value) {
            // FP16 result - in fast mode accumulation can also be FP16,
            // otherwise it's required to be FP32.
            res_mantissa_bits = 11;
            if (p->fast_mode) {
                acc_mantissa_bits = 11;
            } else {
                acc_mantissa_bits = 24;
            }
        } else if (std::is_same<Tres, float>::value) {
            // FP32 result implies FP32 accumulation.
            acc_mantissa_bits = 24;
            res_mantissa_bits = 24;
        } else if (std::is_same<Tres, double>::value) {
            acc_mantissa_bits = 53;
            res_mantissa_bits = 53;
        } else { // Assume these are (U)INT32
            // Default case - mostly integer types.
            acc_mantissa_bits = 30;
            res_mantissa_bits = 30;
        }

        // If bias is to be used, it counts as one additional result so
        // account for that here.
        const int64_t acc_depth = kernel_hwi + ((p->use_bias || p->accumulate) ? 1 : 0);

        // How many of those bits will be used up by accumulation?
        int accu_bits = 1;
        while ((1 << accu_bits) < acc_depth) {
            accu_bits++;
        }

        // How many bits are left over for operands?  Must have at least
        // one.
        //
        // Note that in extreme cases where operands end up being 1 bit
        // wide, most of the results end up being zero so significantly more
        // accumulations can be made safely.  There's no attempt to model
        // this - there is the safety check later that the total
        // accumulation isn't too large.
        acc_operand_bits = std::max(1, acc_mantissa_bits - accu_bits);
        res_operand_bits = std::max(1, res_mantissa_bits - accu_bits);

        // Divide this amongst LHS and RHS
        lhs_acc_bits = std::max(1, (acc_operand_bits+1) / 2);
        rhs_acc_bits = std::max(1, (acc_operand_bits - lhs_acc_bits));

        lhs_res_bits = std::max(1, (res_operand_bits+1) / 2);
        rhs_res_bits = std::max(1, (res_operand_bits - lhs_res_bits));

#ifndef SILENT
        printf("Generating test data: %d accumulator mantissa bits, %d result mantissa bits, %ld total accumulations, %d accumulation bits\n", acc_mantissa_bits, res_mantissa_bits, acc_depth, accu_bits);
        printf("Accumulation: %d bits available, %d lhs, %d rhs\n", acc_operand_bits, lhs_acc_bits, rhs_acc_bits);
        printf("Result: %d bits available, %d lhs, %d rhs\n", res_operand_bits, lhs_res_bits, rhs_res_bits);
#endif
    }

    float lhs_quant_scale() {
        float max_v = float(1 << lhs_res_bits);
        return max_v / 128.0f;
    }

    float rhs_quant_scale() {
        float max_v = float(1 << rhs_res_bits);
        return max_v / 128.0f;
    }

    template<typename T1>
    void populate_lhs(std::shared_ptr<Matrix <T1>> lhs) {
        lhs->RandomizeBits(lhs_acc_bits, lhs_res_bits);
    }

    template<typename T1>
    void populate_rhs(std::shared_ptr<Matrix <T1>> rhs) {
        rhs->RandomizeBits(rhs_acc_bits, rhs_res_bits);
    }

    template <typename T1>
    void populate_bias(std::shared_ptr<Matrix <T1>> bias) {
        bias->RandomizeBits(acc_operand_bits, res_operand_bits);
    }

    void check_max_val(uint64_t max_val) {
        if (max_val > (1LL << res_mantissa_bits)) {
            printf("Accumulated value too large - test data not reassociation safe.\n");
            printf("Aborting.");
            exit(1);
        }
    }
};


template<typename T_ref, typename T_test>
void test_dequantized(GemmProblem *p, int iterations, int nthreads, const char *kernel_name, FILE *dump_in) {
    const int64_t input_hw = p->input_height * p->input_width;
    const int64_t output_hw = p->output_height * p->output_width;
    const int64_t kernel_hwi = p->kernel_height * p->kernel_width * (p->input_channels / p->groups);

    // Define quantize parameter types
    using QType_ol = QuantizeParameters<typename T_test::lhs_operand_type, typename T_ref::lhs_operand_type>;
    using QType_or = QuantizeParameters<typename T_test::rhs_operand_type, typename T_ref::rhs_operand_type>;

    QType_ol A_qp;
    QType_or B_qp;

    auto A_q = std::make_shared<Matrix <typename T_test::lhs_operand_type> >(input_hw, p->input_channels, p->input_channels, p->batches, p->multis);
    auto B_q = std::make_shared<Matrix <typename T_test::rhs_operand_type> >(kernel_hwi, p->output_channels, p->output_channels, 1, p->multis);

    auto C_in = std::make_shared<Matrix <typename T_ref::result_type> >(output_hw, p->output_channels, p->output_channels, p->batches, p->multis);
    auto C_r = std::make_shared<Matrix <typename T_test::result_type> >(output_hw, p->output_channels, p->output_channels, p->batches, p->multis);
    auto C_fr = std::make_shared<Matrix <typename T_ref::result_type> >(output_hw, p->output_channels, p->output_channels, p->batches, p->multis);


    std::shared_ptr<Matrix <typename T_ref::result_type> > bias_f = nullptr;

    if (dump_in) {
        A_q->read_dump(dump_in);
        B_q->read_dump(dump_in);

        if(p->use_bias) {
            bias_f = std::make_shared<Matrix <typename T_ref::result_type> >(1, p->output_channels, p->output_channels, 1, p->multis);
            bias_f->read_dump(dump_in);
        }

        C_fr->read_dump(dump_in);

        fread(&A_qp, sizeof(QType_ol), 1, dump_in);
        fread(&B_qp, sizeof(QType_or), 1, dump_in);
    } else {
        auto A_f = std::make_shared<Matrix <typename T_ref::lhs_operand_type> >(input_hw, p->input_channels, p->input_channels, p->batches, p->multis);
        auto B_f = std::make_shared<Matrix <typename T_ref::rhs_operand_type> >(kernel_hwi, p->output_channels, p->output_channels, 1, p->multis);

        if (p->use_bias) {
            bias_f = std::make_shared<Matrix <typename T_ref::result_type> >(1, p->output_channels, p->output_channels, 1, p->multis);
        }

        test_data_helper<typename T_ref::result_type> test_data(p);

        test_data.populate_lhs(A_f);

        if (p->after_relu) {
            // Activate the incoming data
            auto ptr_multi = A_f->data;
            for (int m = 0; m < A_f->multis; m++) {
                auto ptr_batch = ptr_multi;
                ptr_multi += A_f->multi_stride;

                for (int b = 0; b < A_f->batches; b++) {
                    auto ptr_row = ptr_batch;
                    ptr_batch += A_f->batch_stride;

                    for (int y = 0; y < A_f->M; y++) {
                        for (int x = 0; x < A_f->N; x++) {
                            ptr_row[x] = std::max<typename T_ref::lhs_operand_type>(0, ptr_row[x]);
                        }
                        ptr_row += A_f->stride;
                    }
                }
            }
        }

        test_data.populate_rhs(B_f);

        if (p->use_bias) {
            test_data.populate_bias(bias_f);
        }

        if (p->accumulate) {
            test_data.populate_bias(C_in);
            C_fr->Copy(C_in);
        }

        // Use case for testing here is dynamic quantization
        // where scaling factor is multiplication of scaling
        // factors for activations and weights. Activations
        // can have non-zero zero point but that is not handled
        // by gemm-linux so we set it to zero here
        A_qp = QType_ol(test_data.lhs_quant_scale(), 0);
        B_qp = QType_or(test_data.rhs_quant_scale(), 0);

        A_f->QuantizeRound(A_qp);
        B_f->QuantizeRound(B_qp);

        GemmProblem p2 = *p;
        p2.kernel_filter="";
        p2.fast_mode=0;

        T_ref kern_ref(A_f, B_f, C_fr, bias_f, &p2, nthreads, true);
#ifdef THREADS
        if (nthreads > 1) {
            std::vector<std::thread> thethreads;
            for (int i=1; i<nthreads; i++) {
                thethreads.emplace_back( [&kern_ref](int myid) { kern_ref.Run(myid); }, i );
            }
            kern_ref.Run(0);
            for (int i=1; i<nthreads; i++) {
                thethreads[i-1].join();
            }
        } else {
#else
        if (1) {
#endif
            kern_ref.Run(0);
        }

        A_q->Quantize(A_f, A_qp);
        B_q->Quantize(B_f, B_qp);

    }

    kai::ops::DequantizeFloat scale(A_qp.m_scale * B_qp.m_scale);
    std::cout << "Scaling factor is: " << scale.scale << std::endl;
    T_test kern_test(A_q, B_q, C_r, bias_f, scale, p, nthreads, true);
    bool compare_ok = false;
    for(int it=0; it<iterations;it++) {
        if (p->accumulate) {
            C_r->Copy(C_in);
        }
#ifdef THREADS
        if (nthreads > 1) {
            std::vector<std::thread> thethreads;
            for (int i=1; i<nthreads; i++) {
                thethreads.emplace_back( [&kern_test](int myid) { kern_test.Run(myid); }, i );
            }
            kern_test.Run(0);
            for (int i=1; i<nthreads; i++) {
                thethreads[i-1].join();
            }
        } else {
#else
        if (1) {
#endif
            kern_test.Run(0);
        }
        if(dump_in) {
            C_fr->Compare(C_r, 0);
        } else {
            compare_ok = C_fr->Compare(C_r);
        }
    }

    if(compare_ok && p->dump_file != "") {
        FILE *fp = write_dump(p, p->dump_file.c_str());

        A_q->dump_out(fp);
        B_q->dump_out(fp);
        if(p->use_bias) {
            bias_f->dump_out(fp);
        }
        C_r->dump_out(fp);

        fwrite(&A_qp, sizeof(QType_ol), 1, fp);
        fwrite(&B_qp, sizeof(QType_or), 1, fp);

        fclose(fp);
    }


}

template<typename T_ref, typename T_test, bool per_channel=false>
void test_quantized(GemmProblem *p, int iterations, int nthreads, const char *kernel_name, FILE *dump_in) {

    const int64_t input_hw = p->input_height * p->input_width;
    const int64_t output_hw = p->output_height * p->output_width;
    const int64_t kernel_hwi = p->kernel_height * p->kernel_width * (p->input_channels / p->groups);

    // Define quantize parameter types
    using QType_o = QuantizeParameters<typename T_test::lhs_operand_type, typename T_ref::lhs_operand_type>;
    using QType_w = QuantizeParameters<typename T_test::rhs_operand_type, typename T_ref::rhs_operand_type>;
    using QType_r = QuantizeParameters<typename T_test::result_type, typename T_ref::result_type>;
    using QType_i = QuantizeParameters<int32_t, typename T_ref::result_type>;

    QType_o A_qp;
    std::vector<QType_w> B_qp;
    QType_r C_qp;

    // Define quantized matrices
    auto A_q = std::make_shared<Matrix <typename T_test::lhs_operand_type> >(input_hw, p->input_channels, p->input_channels, p->batches, p->multis);
    auto B_q = std::make_shared<Matrix <typename T_test::rhs_operand_type> >(kernel_hwi, p->output_channels, p->output_channels, 1, p->multis);
    auto Cin_q = std::make_shared<Matrix <typename T_test::result_type> >(output_hw, p->output_channels, p->output_channels, p->batches, p->multis); // For ADD problems, "Cin" is quantized to operand type.
    auto C_q  = std::make_shared<Matrix <typename T_test::result_type> >(output_hw, p->output_channels, p->output_channels, p->batches, p->multis);
    auto C_qr = std::make_shared<Matrix <typename T_test::result_type> >(output_hw, p->output_channels, p->output_channels, p->batches, p->multis);

    std::shared_ptr<Matrix <int32_t> > bias_q = nullptr;

    if (dump_in) {
        A_q->read_dump(dump_in);
        B_q->read_dump(dump_in);

        if (p->use_bias) {
            bias_q = std::make_shared<Matrix <int32_t> >(1, p->output_channels, p->output_channels, 1, p->multis);
            bias_q->read_dump(dump_in);
        }

        C_qr->read_dump(dump_in);

        fread(&A_qp, sizeof(QType_o), 1, dump_in);

        A_qp.print("LHS");

        const int64_t l = per_channel ? p->output_channels : 1;

        for (int64_t i=0; i<l; i++) {
            QType_w q;
            fread(&q, sizeof(QType_w), 1, dump_in);

            q.print("RHS");

            B_qp.push_back(q);
        }

        fread(&C_qp, sizeof(QType_r), 1, dump_in);

        C_qp.print("Result");
    } else {
        auto A_f = std::make_shared<Matrix <typename T_ref::lhs_operand_type> >(input_hw, p->input_channels, p->input_channels, p->batches, p->multis);
        auto B_f = std::make_shared<Matrix <typename T_ref::rhs_operand_type> >(kernel_hwi, p->output_channels, p->output_channels, 1, p->multis);
        auto Cin_f = std::make_shared<Matrix <typename T_ref::result_type> >(output_hw, p->output_channels, p->output_channels, p->batches, p->multis);
        auto Cr_f = std::make_shared<Matrix <typename T_ref::result_type> >(output_hw, p->output_channels, p->output_channels, p->batches, p->multis);

        std::shared_ptr<Matrix <typename T_ref::result_type> > bias_f = nullptr;

        if (p->use_bias) {
            bias_f = std::make_shared<Matrix <typename T_ref::result_type> >(1, p->output_channels, p->output_channels, 1, p->multis);
            bias_q = std::make_shared<Matrix <int32_t> >(1, p->output_channels, p->output_channels, 1, p->multis);
        }

        /* Use reference kernel to generate (generally FP) reference results. */
        A_f->Randomize();

        if (p->after_relu) {
            // Activate the incoming data
            auto ptr_multi = A_f->data;
            for (int m = 0; m < A_f->multis; m++) {
                auto ptr_batch = ptr_multi;
                ptr_multi += A_f->multi_stride;

                for (int b = 0; b < A_f->batches; b++) {
                    auto ptr_row = ptr_batch;
                    ptr_batch += A_f->batch_stride;

                    for (int y = 0; y < A_f->M; y++) {
                        for (int x = 0; x < A_f->N; x++) {
                            ptr_row[x] = std::max<typename T_ref::lhs_operand_type>(0, ptr_row[x]);
                        }
                        ptr_row += A_f->stride;
                    }
                }
            }
        }

        if (p->add_problem) {
            if (p->use_bias) {
                B_f->Randomize();
            } else {
                B_f->AllOne();
            }
            Cin_f->Randomize();
        } else {
            B_f->Randomize();
        }

        if (p->use_bias) {
            bias_f->Randomize();
        }

        /* Now work out quantization parameters */
        /* A is always per layer regardless. */
        A_qp = QType_o(A_f->get_range()[0]);

        /* For add problems - "Cin" quantization is also always per layer. */
        QType_r Cin_qp(Cin_f->get_range()[0]);

        /* B might be per channel or per layer - B_ranges will be returned at the appropriate length (either 1 or #channels). */
        auto B_ranges = B_f->template get_range<per_channel>();

        /* Compute a set of quantize parameters for each range supplied */
        for (auto &&i : B_ranges) {
            if (per_channel) {
                int zero_pt = 0;
                if (std::is_same<typename T_test::rhs_operand_type, uint8_t>::value) {
                    zero_pt = 128;
                }
                B_qp.emplace_back(zero_pt, i, std::is_same<typename T_test::rhs_operand_type, int8_t>::value ? -127 : std::numeric_limits<typename T_test::rhs_operand_type>::min());
            } else {
                B_qp.emplace_back(i, std::is_same<typename T_test::rhs_operand_type, int8_t>::value ? -127 : std::numeric_limits<typename T_test::rhs_operand_type>::min());
            }
        }

        /* Same for the bias */
        std::vector<QType_i> bias_qp;
        for (auto &&i : B_qp) {
            bias_qp.emplace_back(A_qp, i);
        }

        A_f->QuantizeRound(A_qp);
        B_f->QuantizeRound(B_qp);

        if (p->add_problem) {
            Cin_f->QuantizeRound(Cin_qp);
            Cr_f->Copy(Cin_f);
        }

        if (p->use_bias) {
            bias_f->QuantizeRound(bias_qp);
        }

        GemmProblem p2 = *p;
        p2.kernel_filter="";
        p2.fast_mode=0;

        T_ref kern_ref(A_f, B_f, Cr_f, bias_f, &p2, nthreads, true);

        /* Run multithreaded if requested */
#ifdef THREADS
        if (nthreads > 1) {
            std::vector<std::thread> thethreads;
            for (int i=1; i<nthreads; i++) {
                thethreads.emplace_back( [&kern_ref](int myid) { kern_ref.Run(myid); }, i );
            }
            kern_ref.Run(0);
            for (int i=1; i<nthreads; i++) {
                thethreads[i-1].join();
            }
        } else {
#else
        if (1) {
#endif
            kern_ref.Run(0);
        }

        C_qp = QType_r(Cr_f->get_range()[0]);

        // Generate reference quantized output
        C_qr->Quantize(Cr_f, C_qp);

        // Prepare to run the test kernel - quantize the inputs and bias.
        A_q->Quantize(A_f, A_qp);
        B_q->Quantize(B_f, B_qp);

        if (p->add_problem) {
            Cin_q->Quantize(Cin_f, Cin_qp);
            // For add problems, we (ab)use the array of quantize parameters for B to store the C input quantize parameters.
            // This is OK as per-channel add quantization is not supported.
            B_qp.emplace_back(*reinterpret_cast<QType_w *>(&Cin_qp));
        }

        if (p->use_bias) {
            bias_q->Quantize(bias_f, bias_qp);
        }
    }

    T_test kern_test(A_q, B_q, C_q, bias_q, A_qp, B_qp, C_qp, p, nthreads, true);
    bool compare_ok = false;

    for (int it=0; it<iterations; it++) {
        C_q->Copy(Cin_q);
#ifdef THREADS
        if (nthreads > 1) {
            std::vector<std::thread> thethreads;
            for (int i=1; i<nthreads; i++) {
                thethreads.emplace_back( [&kern_test](int myid) { kern_test.Run(myid); }, i );
            }
            kern_test.Run(0);
            for (int i=1; i<nthreads; i++) {
                thethreads[i-1].join();
            }
        } else {
#else
        if (1) {
#endif
            kern_test.Run(0);
        }

        // Compare the result to the quantized reference.  Tolerate off-by-one errors, unless we are reading a dump.
        // Note this deliberately doesn't set compare_ok so you can't read and create a dump at the same time.
        if (dump_in) {
            C_qr->Compare(C_q, 0);
        } else {
            compare_ok = C_qr->Compare(C_q, 1);
        }
    }

    if (compare_ok && p->dump_file != "") {
        FILE *fp = write_dump(p, p->dump_file.c_str());

        A_q->dump_out(fp);
        B_q->dump_out(fp);
        if (p->use_bias) {
            bias_q->dump_out(fp);
        }
        C_q->dump_out(fp);

        fwrite(&A_qp, sizeof(QType_o), 1, fp);

        const int64_t l = per_channel ? p->output_channels : 1;

        for (int64_t i=0; i<l; i++) {
            fwrite(&(B_qp[i]), sizeof(QType_w), 1, fp);
        }

        fwrite(&C_qp, sizeof(QType_r), 1, fp);

        fclose(fp);
    }
}

/*
 * This function runs two kernels and compares their output.
 *
 * Generally this is used to check that an optimized kernel produces the
 * same output as an unoptimized reference kernel.
 *
 * Templated on the reference kernel and test kernel.
 */
template<typename T_ref, typename T_test, bool legacy>
void test(GemmProblem *p, int iterations, int nthreads, const char *kernel_name, FILE *dump_in) {
    // Only test kernels with the same operand and result types.
    static_assert(std::is_same<typename T_ref::operand_type, typename T_test::lhs_operand_type>::value, "Kernels must use same operand type.");
    static_assert(std::is_same<typename T_ref::result_type, typename T_test::result_type>::value, "Kernels must use same result type.");

    // Assume reference kernel can deal with any size, just get rounding values from test kernel.
    int m_round = T_test::get_m_block();
    int n_round = T_test::get_n_block();
    int k_round = T_test::get_k_block();

    // Round requested sizes to something both kernels can do.
    if (m_round != 1 || n_round != 1 || k_round != 1) {
        assert(p->is_basic_gemm());

        p->output_width=p->input_width = round_up_to_nearest_multiple(p->output_width, m_round);
        p->output_channels = round_up_to_nearest_multiple(p->output_channels, n_round);
        p->input_channels = round_up_to_nearest_multiple(p->input_channels, k_round);
    }

    const int64_t input_hw = p->input_height * p->input_width;
    const int64_t kernel_hwi = p->kernel_height * p->kernel_width * (p->input_channels / p->groups);
    const int64_t output_hw = p->output_height * p->output_width;

    auto A = std::make_shared<Matrix <typename T_ref::operand_type> >(input_hw, p->input_channels, std::max(p->input_channels, p->a_stride), p->batches, p->multis);
    auto B = std::make_shared<Matrix <typename T_ref::operand_type> >(kernel_hwi, p->output_channels, std::max(p->output_channels, p->b_stride), 1, p->multis);
    auto C = std::make_shared<Matrix <typename T_ref::result_type> >(output_hw, p->output_channels, p->output_channels, p->batches, p->multis);
    auto C1 = std::make_shared<Matrix <typename T_ref::result_type> >(output_hw, p->output_channels, p->output_channels, p->batches, p->multis);
    auto C2 = std::make_shared<Matrix <typename T_test::result_type> >(output_hw, p->output_channels, p->output_channels, p->batches, p->multis);

    std::shared_ptr<Matrix <typename T_test::result_type> > bias;

    if (dump_in) {
        // If provided, read references from input dump file
        A->read_dump(dump_in);
        B->read_dump(dump_in);

        if (p->use_bias) {
            bias = std::make_shared<Matrix <typename T_test::result_type> >(1, p->output_channels, p->output_channels, 1, p->multis);
            bias->read_dump(dump_in);
        }

        C1->read_dump(dump_in);
    } else {
        test_data_helper<typename T_ref::result_type> test_data(p);

        if (p->use_bias) {
            bias = std::make_shared<Matrix <typename T_test::result_type> >(1, p->output_channels, p->output_channels, 1, p->multis);

            test_data.populate_bias(bias);
        }

        test_data.populate_lhs(A);

        if (p->add_problem) {
            if (p->use_bias) {
                B->Randomize();
            } else {
                B->AllOne();
            }
            C->Randomize();
            C1->Copy(C);
        } else {
            test_data.populate_rhs(B);
            if (p->accumulate) {
                test_data.populate_bias(C);
                C1->Copy(C);
            }
        }

        // FP32 fast mode means the operands can be cast to BF16 - mask them
        // off here so the reference will match.
        if (std::is_same<typename T_ref::operand_type, float>::value && p->fast_mode) {
            A->template Mask<uint32_t>(0xffff0000);
            B->template Mask<uint32_t>(0xffff0000);
        }

        T_ref kern_ref(A, B, C1, bias, p, true);

        // The reference kernel returns the largest absolute positive or
        // negative contribution to any individual result, which is used to
        // check that the accumulator can't possibly get out of range and
        // start rounding.  If the check fails, abort the test.
        int64_t max_val = kern_ref.Run();
        test_data.check_max_val(max_val);
    }

    T_test kern_test=TestKernel<legacy, QuantizationType::NONE, false, T_test>::get(A, B, C2, bias, p, nthreads, true);

    bool compare_ok=false;

    for (int it=0; it<iterations; it++) {
        C2->Copy(C);
#ifdef THREADS
        if (nthreads > 1) {
            std::vector<std::thread> thethreads;
            for (int i=1; i<nthreads; i++) {
                thethreads.emplace_back( [&kern_test](int myid) { kern_test.Run(myid); }, i );
            }
            kern_test.Run(0);
            for (int i=1; i<nthreads; i++) {
                thethreads[i-1].join();
            }
        } else {
#else /* THREADS */
        if(1) {
#endif
            kern_test.Run(0);
        }

        if (dump_in) {
            C1->Compare(C2, 0);
        } else {
            compare_ok = C1->Compare(C2);
        }
    }

    if (compare_ok && p->dump_file != "") {
        FILE *fp = write_dump(p, p->dump_file.c_str());

        A->dump_out(fp);
        B->dump_out(fp);
        if (p->use_bias) {
            bias->dump_out(fp);
        }
        C2->dump_out(fp);

        fclose(fp);
    }
}

template<typename T, bool legacy>
void print_kernels(GemmProblem *p, unsigned int nthreads) {
     TestKernel<legacy, QuantizationType::NONE, false, T>::print_kernels(p, nthreads);
}

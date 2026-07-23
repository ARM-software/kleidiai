//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#define NO_LONG_OPT
#endif

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifndef NO_LONG_OPT
#include <getopt.h>
#else
#include <unistd.h>  // for standard getopt()
#endif

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#include "kai/ops/bfloat.hpp"

#define restrict __restrict

#ifdef THREADS
#include <atomic>
#include <thread>
#ifdef BIND_THREADS
#include <sched.h>
#endif
#endif

#include "src/gemm_lib.hpp"

// Test mode for the teams code - not normally needed.
#undef TEST_TEAMS

#if defined(BARE_METAL)
// Make sure SVE is enabled on bare metal builds (needed on certain
// compilers but harmless even if unnecessary).
__attribute__((constructor)) void setup_cptr() {
    {
        uint64_t bit12 = 0x1100;
        uint64_t temp;
        __asm __volatile(
            "mrs        %[temp], CPTR_EL3\n"
            "orr        %[temp], %[temp], %[bit12]\n"
            "msr        CPTR_EL3, %[temp]\n"
            : [temp] "=&r"(temp)
            : [bit12] "r"(bit12)
            : "memory");
    }
}
#endif

// Default cache sizes, set here so they can be overridden on the command line.
int l1_cache_size = 32768;
int l2_cache_size = 131072;

int force_cpu = 0;

// Functions to benchmark and test kernels.  These are compiled externally.
template <typename T, bool legacy, enum QuantizationType quantized, bool perchannel>
void benchmark(GemmProblem* p, int iterations, int nthreads, int run_delay, mapfn cpuid_map, const char* kernel_name);

template <typename T_ref, typename T_test, bool per_channel = false>
void test_quantized(GemmProblem* p, int iterations, int nthreads, const char* kernel_name, FILE*);

template <typename T_ref, typename T_test>
void test_dequantized(GemmProblem* p, int iterations, int nthreads, const char* kernel_name, FILE*);

template <typename T_ref, typename T_test, bool legacy>
void test(GemmProblem* p, int iterations, int nthreads, const char* kernel_name, FILE*);

template <typename T, bool legacy>
void print_kernels(GemmProblem* p, unsigned int nthreads);

/*
 * Various type names are used in the macros below to construct the kernel list.
 *
 * All code actually depending on these types is compiled elsewhere, so we
 * just need to forward declare them all.
 */

// Reference GEMM class
template <typename To, typename Tr>
class gemm_transposeB;

// kai_ops wrapper classes
template <typename Tlop, typename Trop, typename Tret, enum QuantizationType quantized = QuantizationType::NONE>
class kai_ops_wrapper;
template <typename Tlop, typename Trop, typename Tret>
using kai_ops_quantized = kai_ops_wrapper<Tlop, Trop, Tret, QuantizationType::INTEGER>;

template <typename Ta, typename Tb, typename Tref, enum QuantizationType quantized = QuantizationType::NONE>
class depthwise_wrapper;
template <typename Top, typename Tret>
class winograd_wrapper;

// Instantiations of kai_ops wrapper classes.
// FP types.
typedef kai_ops_wrapper<float, float, float> sgemm_new;
typedef kai_ops_wrapper<__fp16, __fp16, __fp16> hgemm_new;
typedef kai_ops_wrapper<bfloat16, bfloat16, float> gemm_bf16_new;
typedef kai_ops_wrapper<bfloat16, bfloat16, bfloat16> gemm_bf16bf16;
typedef kai_ops_wrapper<__fp16, __fp16, float> gemm_fp16fp32;
typedef depthwise_wrapper<float, float, float> depthwise_fp32;
typedef depthwise_wrapper<__fp16, __fp16, __fp16> depthwise_fp16;
typedef winograd_wrapper<float, float> winograd_fp32;
typedef winograd_wrapper<__fp16, __fp16> winograd_fp16;

// int types.
typedef kai_ops_wrapper<uint16_t, uint16_t, uint32_t> gemm_u16;
typedef kai_ops_wrapper<int16_t, int16_t, int32_t> gemm_s16;
typedef kai_ops_wrapper<uint8_t, uint8_t, uint32_t> gemm_u8_new;
typedef kai_ops_wrapper<int8_t, int8_t, int32_t> gemm_s8_new;
typedef kai_ops_wrapper<int8_t, int8_t, float, QuantizationType::FLOAT> gemm_s8fp32;
typedef kai_ops_wrapper<int8_t, int8_t, __fp16, QuantizationType::FLOAT> gemm_s8fp16;
typedef kai_ops_wrapper<uint8_t, int8_t, float, QuantizationType::FLOAT> gemm_u8s8fp32;
typedef kai_ops_quantized<uint8_t, uint8_t, uint8_t> gemm_u8_quant;
typedef kai_ops_quantized<int8_t, int8_t, int8_t> gemm_s8_quant;
typedef kai_ops_quantized<uint8_t, int8_t, uint8_t> gemm_u8s8_quant;
typedef depthwise_wrapper<int8_t, int8_t, int8_t, QuantizationType::INTEGER> depthwise_s8q;
typedef depthwise_wrapper<uint8_t, uint8_t, uint8_t, QuantizationType::INTEGER> depthwise_u8q;

/*
 * Infrastructure to allow the kernel to be tested to be selected.
 *
 * Here are some function pointers to a benchmark function and a test
 * function (which should be specializations of the templated functions
 * defined above).
 */
typedef void (*benchfn)(GemmProblem*, int, int, int, mapfn, const char*);
typedef void (*testfn)(GemmProblem*, int, int, const char*, FILE*);
typedef void (*printfn)(GemmProblem*, unsigned int);

/*
 * This structure declares a list entry consisting of a name and the corresponding functions.
 */
struct kernentry {
    const char* name;
    benchfn bench;
    testfn test;
    printfn print;
    int flags;
};

#define KERN_GEMV 0x1        /* GEMV kernel (force M=1) */
#define KERN_LEGACY 0x2      /* Legacy kernel (M,N,K interface w/no transpose, alpha, beta) */
#define KERN_ACTOK 0x4       /* Kernel which supports activations */
#define KERN_WGOK 0x8        /* Kernel of a type that supports Winograd */
#define KERN_QUANTIZED 0x10  /* Quantized kernel */
#define KERN_CONV 0x20       /* Convolution supporting kernel */
#define KERN_GEMV_N 0x40     /* GEMV kernel with M/N swapped (N=1) */
#define KERN_CONV_ONLY 0x80  /* Convolution-only kernel (i.e. force -I conv2d) */
#define KERN_DEPTHWISE 0x100 /* Depthwise kernel */
#define KERN_ADD 0x200       /* Add kernel */
#define KERN_NOACC 0x400     /* No accumulate support */
#define KERN_FAST_ONLY 0x800 /* Force fast mode */

/*
 * This macro generates a structure declaration, based on the kernel name,
 * operand type and result type.
 *
 * It uses the type parameters to specialize test() with an appropriately
 * specialized reference implementation.
 */
#define KERN(name, optype, restype, flags)                                               \
    {#name, benchmark<name, !!(((flags) & KERN_LEGACY)), QuantizationType::NONE, false>, \
     test<gemm_transposeB<optype, restype>, name, !!(((flags) & KERN_LEGACY))>,          \
     print_kernels<name, !!(((flags) & KERN_LEGACY))>, flags},

#define KERN_DEQUANT(name, comp)                                                                  \
    {#name, benchmark<name, false, QuantizationType::FLOAT, false>, test_dequantized<comp, name>, \
     print_kernels<name, false>, KERN_ACTOK | KERN_CONV},

#define KERN_QUANT_COMP(name, comp, flags)                                                        \
    {#name, benchmark<name, false, QuantizationType::INTEGER, false>, test_quantized<comp, name>, \
     print_kernels<name, false>, KERN_ACTOK | KERN_QUANTIZED | flags},

#define KERN_QUANT(name, flags) KERN_QUANT_COMP(name, sgemm_new, flags)

#define KERN_QUANT_PERCHANNEL_COMP(name, comp, flags)                                                                \
    {#name "_perchannel", benchmark<name, false, QuantizationType::INTEGER, true>, test_quantized<comp, name, true>, \
     print_kernels<name, false>, KERN_ACTOK | KERN_QUANTIZED | flags},

#define KERN_QUANT_PERCHANNEL(name, flags) KERN_QUANT_PERCHANNEL_COMP(name, sgemm_new, flags)

// clang-format off
/*
 * The list of supported kernels.  This consists of a call to the
    KERN() macro for each kernel included above.
 */
kernentry thekerns[] = {
/* START OF KERNEL LIST */

KERN(sgemm_new, float, float, KERN_WGOK | KERN_ACTOK | KERN_CONV )
KERN(gemm_bf16_new, bfloat16, float, KERN_WGOK | KERN_ACTOK | KERN_CONV)
KERN(gemm_bf16bf16, bfloat16, bfloat16, KERN_WGOK | KERN_ACTOK | KERN_CONV)

KERN(winograd_fp32, float, float, KERN_CONV | KERN_CONV_ONLY | KERN_ACTOK | KERN_NOACC )

#ifdef __aarch64__
KERN(gemm_fp16fp32, __fp16, float, KERN_WGOK | KERN_ACTOK | KERN_CONV)

KERN(hgemm_new, __fp16, __fp16, KERN_ACTOK | KERN_CONV )

KERN(winograd_fp16, __fp16, __fp16, KERN_CONV | KERN_CONV_ONLY | KERN_ACTOK | KERN_NOACC )

KERN(depthwise_fp32, float, float, KERN_CONV | KERN_CONV_ONLY | KERN_ACTOK | KERN_DEPTHWISE)
KERN(depthwise_fp16, __fp16, __fp16, KERN_CONV | KERN_CONV_ONLY | KERN_ACTOK | KERN_DEPTHWISE | KERN_FAST_ONLY )

KERN(gemm_s16, int16_t, int32_t, KERN_CONV)
KERN(gemm_u16, uint16_t, uint32_t, KERN_CONV)
KERN(gemm_s8_new, int8_t, int32_t, KERN_CONV)
KERN(gemm_u8_new, uint8_t, uint32_t, KERN_CONV)
KERN_DEQUANT(gemm_s8fp32, sgemm_new)
KERN_DEQUANT(gemm_s8fp16, hgemm_new)
KERN_DEQUANT(gemm_u8s8fp32, sgemm_new)
KERN_QUANT(gemm_u8_quant,KERN_CONV)
KERN_QUANT(gemm_s8_quant,KERN_CONV)
KERN_QUANT(gemm_u8s8_quant,KERN_CONV)
KERN_QUANT_PERCHANNEL(gemm_s8_quant,KERN_CONV)

KERN_QUANT_COMP(depthwise_s8q, depthwise_fp32, KERN_CONV | KERN_CONV_ONLY | KERN_ACTOK | KERN_DEPTHWISE)
KERN_QUANT_PERCHANNEL_COMP(depthwise_s8q, depthwise_fp32, KERN_CONV | KERN_CONV_ONLY | KERN_ACTOK | KERN_DEPTHWISE)
KERN_QUANT_COMP(depthwise_u8q, depthwise_fp32, KERN_CONV | KERN_CONV_ONLY | KERN_ACTOK | KERN_DEPTHWISE)
KERN_QUANT_PERCHANNEL_COMP(depthwise_u8q, depthwise_fp32, KERN_CONV | KERN_CONV_ONLY | KERN_ACTOK | KERN_DEPTHWISE)
#endif
/* END OF KERNEL LIST */
{ "", nullptr, nullptr, nullptr, 0 } };

/*
 * Similar to above, infrastructure to allow selection of thread binding
 * strategies.
 */
struct threadmapentry {
    const char *name;
    mapfn fn;
};

#define THREADMAP(name) \
{ \
    #name, \
    threadmap_##name \
},

/* Here are the actual strategies.  Included inline as they are so short. */

/* Linear mapping, wrap at maxcpus */
int threadmap_linear(int i, int maxcpus) {
    return i % maxcpus;
}

/* Map to even CPUs, then odd CPUs, e.g. 0,2,4,6,1,3,5,7 */
int threadmap_evenodd(int i, int maxcpus) {
    i %= maxcpus; /* Restrict range to 0-(maxcpus-1) initially */

    return ((i % (maxcpus/2)) * 2) + ((i >= maxcpus/2) ? 1 : 0);
}

/* Cycle between 4 and 5 */
int threadmap_a57(int i, int maxcpus) {
    return (i % 2) + 4;
}

/* Cycle between 1 and 2 */
int threadmap_a72(int i, int maxcpus) {
    return (i % 2) + 1;
}

/* Cycle 0, 3, 4, 5 */
int threadmap_a53(int i, int maxcpus) {
    i %= 4;
    return i + (i ? 2 : 0);
}

/* Cycle between 4 and maxcpus */
int threadmap_start_at_4(int i, int maxcpus) {
    int numcpus = maxcpus - 4;
    return (i % numcpus) + 4;
}

/* Cycle between 6 and maxcpus */
int threadmap_start_at_6(int i, int maxcpus) {
    int numcpus = maxcpus - 6;
    return (i % numcpus) + 6;
}

/* Cycle through all but start at 4 */
int threadmap_loop_from_4(int i, int maxcpus) {
    return (i + 4) % maxcpus;
}

/* Reverse order */
int threadmap_reverse(int i, int maxcpus) {
    return maxcpus - (i % maxcpus) - 1;
}

threadmapentry themaps[] = {
/* START OF THREAD STRATEGY LIST */
THREADMAP(linear)
THREADMAP(evenodd)
THREADMAP(a57)
THREADMAP(a72)
THREADMAP(a53)
THREADMAP(start_at_4)
THREADMAP(start_at_6)
THREADMAP(loop_from_4)
THREADMAP(reverse)
/* END OF THREAD STRATEGY LIST */
{ "", NULL } };

// This is used to allow specification of a strategy on the command line.
strategy_name strategies_list[] = {
    { "fail", ConvStrategy::fail },
    { "im2row", ConvStrategy::im2row },
    { "indirect", ConvStrategy::indirect },
    { "convolution", ConvStrategy::convolution } };

ConvStrategy str2strat(std::string name) {
    for (auto &&i : strategies_list) {
        if (i.name == name) {
            return i.strat;
        }
    }

    return ConvStrategy::fail;
}

void printhelp(const char *progname) {
    printf("Usage: %s [-M Msize] [-N Nsize] [-K Ksize] [-A] [-a fn] [-b] [-c] [-t nthreads] [-f] [-m mapfn] [-i iterations] [-d delay] [-l L1_size] [-L L2_size] [-s a_stride] [-S b_stride] [-C cpu_code] [-p] [-n testID] [-q] [-F filter] [-u features] [-U features] [-o|-O dump_file] [-P] [-h] <kernel>\n"
           " where:\n"
           "  -M, -N, -K:  Set the problem size (M = height of output, N = width of output, K = depth of dot product).\n"
           "               Default size is M=N=K=768 in benchmark mode, or M=N=K=100 in check mode.\n"
           "  -a:          Specify activation function (relu or relu6).\n"
           "  -b:          Enable random bias.\n"
           "  -c:          Turns on checking functionality (default: benchmark mode).\n"
#ifdef THREADS
           "  -t:          Sets the number of execution threads to use.\n"
           "  -T:          Sets the number of thread teams (of size T) to create.\n"
           "  -G:          Enables dynamic scheduling and specifies maximum number of granules (0=unlimited).\n"
           "  -w:          For 2D scheduling, force a particular tile width.\n"
#ifdef BIND_THREADS
           "  -m:          Enables thread binding and specifies the mapping function to use.\n"
#else
           "  -m:          Configure thread binding (not enabled on this build).\n"
#endif  // BIND_THREADS
#else
           "  -t, -m, -G:  Configure threading (not enabled on this build).\n"
#endif
           "  -f:          Enable \"fast mode\" lower precision computation.\n"
           "  -i:          Configure how many iterations to run in benchmark mode.\n"
           "  -d:          Configures the delay (in seconds) between each iteration.\n"
           "  -l:          Configure L1 cache size (bytes).\n"
           "  -L:          Configure L2 cache size (bytes).\n"
           "  -s, -S:      Set the stride of the A and B matrix respectively.\n"
           "  -C:          Overrides the detected CPU type.\n"
           "  -y:          Specifies number of batches (same B, multiple A, C).\n"
           "  -z:          Specifies number of multis (multiple A, B, C).\n"
           "  -D:          Select test suite name (see list below).\n"
           "  -n:          Select test ID (overrides M,N,K,batch,multi settings).\n"
           "  -I:          Select test implementation strategy (see list below, default is suite dependent).\n"
           "  -q:          List valid kernel implementations for the given problem.\n"
           "  -F:          Substring filter to select kernel to use.\n"
           "  -j:          Override inner block size (kernel dependent).\n"
           "  -k:          Override outer block size (kernel dependent).\n"
           "  -V:          Specifies how to handle convolutions (im2row, indirect, convolution).\n"
           "  -r:          Minimum number of ms to run for each iteration.\n"
           "  -W:          Time weight transform as part of kernel run (for dynamic weights cases).\n"
           "  -A:          Accumulate - add to what is already in output matrix.\n"
           "  -B:          Supply RHS in transposed format (if supported).\n"
           "  -U:          Force enable named CPU architecture features (comma separated list).\n"
           "                (fp16, dotprod, sve, sve2, bf16, i8mm, svebf16, svei8mm, svef32mm,\n"
           "                 sme, sme_i8i32, sme_f16f32, sme_b16b32, sme_f32f32, sme2)\n"
           "  -u:          Force disable named CPU architecture features (comma separated list).\n"
           "  -p:          Specify convolution parameters (comma separated):\n"
           "               in_h,in_w,in_ch,kern_h,kern_w,out_ch,stride_h,stride_w,pad_top,pad_left,pad_bottom,pad_right,dilation_h,dilation_w\n"
           "  -o:          Valid only in conjunction with -c; specifies dump file to create if comparison is valid.\n"
           "  -O:          Specifies dump file to process; this overrides the problem specification from the dump file.\n"
           "  -Q:          Gather cache statistics: Note that this option impacts performance.\n"
           "  -X:          Specify additional option:\n"
           "                * after-relu: Activate the activations before running the layer.\n"
           "  -P:          Request fixed format kernel.\n"
#ifdef BARE_METAL
           "  -Z:          Flush transformed weight data from cache after setup.\n"
#endif
           "  -h:          Prints this help and exits.\n"
           "  kernel:      Specifies the kernel to run.\n", progname);

    printf("\nList of available kernels:\n");
    for (int i=0; thekerns[i].name[0]; i++) {
        printf("\t%s\n", thekerns[i].name);
    }

#ifdef BIND_THREADS
    printf("\nList of available thread binding maps:\n");
    for (int i=0; themaps[i].name[0]; i++) {
        printf("\t%s\n",themaps[i].name);
    }
#endif

    print_test_suites();
}

extern const char *git_revision;
extern const char *git_branch;

int main(int argc, char **argv) {
    /* Container for the various options */
    GemmProblem p;

    /* Options that don't go in GemmProblem */
    bool        docheck            = false;
    bool        doquery            = false;
    bool        manual_spec        = false;
    int         nthreads           = 1;
    int         iterations         = 1;
    int         run_delay          = 0;
    mapfn       mapfn              = NULL;
    int         test_id            = 0;
    std::string test_suite         = "";
    std::string test_impl          = "";
    std::string read_dump_fname    = "";

    FILE *read_dump_file;

#ifndef SILENT
    printf("gemm-linux: %s (%s)\n", git_branch, git_revision);
#endif

#ifndef NO_LONG_OPT
    struct option longopts[] = {
      { "winograd-output-rows", 1, nullptr, 128 },
      { "winograd-output-cols", 1, nullptr, 129 },
      { "winograd-input-transform", 1, nullptr, 130 },
      { "winograd-weight-transform", 1, nullptr, 131 },
      { "winograd-output-transform", 1, nullptr, 132 },
      { "weight-format", 1, nullptr, 133 },
      { nullptr, 0, nullptr, 0 },  // Last element sentinel
    };
#endif

    int opt;
#ifndef NO_LONG_OPT
    while ((opt = getopt_long(argc, argv, "M:N:K:a:bcd:t:T:m:i:s:S:l:L:C:y:z:n:D:I:fhF:j:k:G:w:V:r:WABu:U:p:X:PqO:o:QZ", longopts, nullptr)) >= 0) {
#else
    while ((opt = getopt(argc, argv, "M:N:K:a:bcd:t:T:m:i:s:S:l:L:C:y:z:n:D:I:fhF:j:k:G:w:V:r:WABu:U:p:X:PqO:o:QZ")) >= 0) {
#endif
        switch(opt) {
            case 'M':
                p.input_width=p.output_width=strtol(optarg, NULL, 0);
                break;

            case 'N':
                p.output_channels=strtol(optarg, NULL, 0);
                break;

            case 'K':
                p.input_channels=strtol(optarg, NULL, 0);
                break;

            case 'a':
                if (!strcmp(optarg, "relu")) {
                    p.act=Activation(Activation::Type::ReLU);
                } else if (!strcmp(optarg, "relu6")) {
                    p.act=Activation(Activation::Type::BoundedReLU, 6.0f);
                } else if (!strcmp(optarg, "relu7")) {
                    p.act=Activation(Activation::Type::BoundedReLU, 7.0f);
                } else {
                    printf("Unknown activation function: %s\n",optarg);
                    return 0;
                }
                break;

            case 'b':
                p.use_bias=true;
                break;

            case 'c':
                docheck=true;
                break;

            case 'd':
                run_delay=strtol(optarg, NULL, 0);
                break;

#ifdef THREADS
            case 't':
                nthreads=strtol(optarg, NULL, 0);
                break;

            case 'T':
                p.n_teams = strtol(optarg, NULL, 0);
                break;

            case 'w':
                p.schedule_shape_override = strtol(optarg, NULL, 0);
                break;
            case 'm':
#ifdef BIND_THREADS
                for (int i=0; themaps[i].name[0]; i++) {
                    if (!strcmp(themaps[i].name, optarg)) {
                        mapfn = themaps[i].fn;
                        break;
                    }
                }

                if (!mapfn) {
                    printf("Unknown thread ID map '%s'.  Type %s -h for a list.\n", optarg, argv[0]);
                    return 1;
                }
#else
                printf("Error: Thread binding not enabled on this build.\n");
                return 1;
#endif // BIND_THREADS
                break;
#else
            case 't':
            case 'T':
            case 'w':
            case 'm':
                printf("Error: Threading not enabled on this build.\n");
                return 1;
#endif // THREADS

            case 'i':
                iterations=strtol(optarg, NULL, 0);
                break;

            case 's':
                p.a_stride = strtol(optarg, NULL, 0);
                break;

            case 'S':
                p.b_stride = strtol(optarg, NULL, 0);
                break;

            case 'l':
                {
                    CPUInfo *ci = get_CPUInfo();
                    ci->set_L1_cache_size(strtol(optarg, NULL, 0));
                    break;
                }

            case 'L':
                {
                    CPUInfo *ci = get_CPUInfo();
                    ci->set_L2_cache_size(strtol(optarg, NULL, 0));
                    break;
                }

            case 'C':
                {
                    CPUInfo *ci = get_CPUInfo();
                    ci->midr_override(strtol(optarg, NULL, 0));
                    break;
                }

            case 'y':
                p.batches = strtol(optarg, NULL, 0);
                break;

            case 'z':
                p.multis = strtol(optarg, NULL, 0);
                break;

            case 'D':
                test_suite = optarg;
                break;

            case 'n':
                test_id=strtol(optarg, NULL, 0);
                if (test_id <= 0) {
                    printf("Test ID must be at least 1.\n");
                    return 1;
                }
                break;

            case 'I':
                test_impl = optarg;
                break;

            case 'q':
                doquery = true;
                break;

            case 'F':
                p.kernel_filter = optarg;
                break;

            case 'G':
                p.dynamic_granule_count=strtol(optarg, NULL, 0);
                p.dynamic_scheduling=true;
                break;

            case 'j':
                p.inner_block_size=strtol(optarg, NULL, 0);
                break;

            case 'k':
                p.outer_block_size=strtol(optarg, NULL, 0);
                break;

            case 'V':
                p.strategy = str2strat(optarg);
                if (p.strategy == ConvStrategy::fail) {
                    printf("Unknown convolution strategy: %s\n",optarg);
                    return 1;
                }
                break;

            case 'r':
                p.min_ms=strtol(optarg, NULL, 0);
                break;

            case 'W':
                p.time_weight_transform=true;
                break;

            case 'A':
                p.accumulate = true;
                break;

            case 'B':
                p.transposed_b = true;
                break;

            case 'f':
                p.fast_mode = true;
                break;

            case 'u':
            case 'U':
                {
                    bool setv = false;
                    if (opt == 'U') {
                        setv = true;
                    }
                    CPUInfo *ci = get_CPUInfo();
                    char *p=optarg;

                    // Handle comma separation - look for next comma and blonk it out.
                    while (p) {
                        char *np = strchr(p, ',');
                        if (np) {
                            *np = '\0';
                            np++;
                        }

                        bool ret = ci->force_feature(p, setv);

                        if (!ret) {
                            printf("Warning: Unknown CPU feature %s.\n", p);
                        }

                        // Now process next feature.
                        p=np;
                    }
                }
                break;

            case 'p':
           //"  -p:          Specify convolution parameters (comma separated):\n"
           //"               in_h,in_w,in_ch,kern_h,kern_w,out_ch,stride_h,stride_w,pad_top,pad_left,pad_bottom,pad_right,dilation_h,dilation_w\n"
                {
                    int64_t pad_right=0, pad_bottom=0;
                    int64_t *arg_ptrs[] = { &p.input_height, &p.input_width, &p.input_channels, &p.kernel_height, &p.kernel_width,
                                            &p.output_channels, &p.out_stride_h, &p.out_stride_w, &p.padding_top, &p.padding_left, &pad_bottom, &pad_right,
                                            &p.in_stride_h, &p.in_stride_w, nullptr };

                    char *cp = optarg;
                    int argidx=0;

                    while(cp) {
                        char *ncp = strchr(cp, ',');
                        if (ncp) {
                            *ncp = '\0';
                            ncp++;
                        }

                        if (arg_ptrs[argidx] == nullptr) {
                            printf("Warning: Too many parameters to -p\n");
                            break;
                        }

                        *arg_ptrs[argidx] = strtoull(cp, nullptr, 0);

                        argidx++;
                        cp=ncp;
                    }

                    // Re-derive output width/height
                    auto eff_kernel_w = (p.kernel_width - 1)*p.in_stride_w;
                    auto eff_kernel_h = (p.kernel_height - 1)*p.in_stride_h;
                    p.output_width  = (p.input_width + p.padding_left + pad_right - eff_kernel_w + (p.out_stride_w-1)) / p.out_stride_w;
                    p.output_height = (p.input_height + p.padding_top + pad_bottom - eff_kernel_h + (p.out_stride_h-1)) / p.out_stride_h;

                    manual_spec = true;
                }
                break;

            case 'o':
                p.dump_file = optarg;
                break;

            case 'O':
                read_dump_fname = optarg;
                break;

            case 'Q':
                p.cache_stats = true;
                break;

            case 'X':
                // Switch on the following argument
                if (!strcmp(optarg, "after-relu")) {
                    // Indicate that the incoming activations should be passed
                    // through a ReLU before being passed into the
                    // GEMM/Depthwise/Winograd layer. This can enable some
                    // optimisations for some layers.
                    p.after_relu = true;
                } else {
                    printf("Unknown -X argument '%s'\n", optarg);
                    return -1;
                }
                break;

            case 128:
                p.winograd_args.output_tile_rows = strtoul(optarg, nullptr, 0);
                break;

            case 129:
                p.winograd_args.output_tile_cols = strtoul(optarg, nullptr, 0);
                break;

            case 130:
                p.winograd_args.input_transform_filter = optarg;
                break;

            case 131:
                p.winograd_args.weight_transform_filter = optarg;
                break;

            case 132:
                p.winograd_args.output_transform_filter = optarg;
                break;

            case 133:
                p.weight_format=static_cast<kai::ops::WeightFormat>(strtoul(optarg, nullptr, 0));
                break;

            case 'P':
                p.fixed_format = true;
                break;

            case 'Z':
#ifdef BARE_METAL
                p.cache_flush = true;
#else
                printf("Warning: -Z option ignored for non-bare metal builds.");
#endif
                break;

            case 'h':
                printhelp(argv[0]);
                return 0;

            case '?':
                printhelp(argv[0]);
                return 1;

            default:
                fprintf(stderr,
                        "Internal error: getopt returned unhandled option %d. "
                        "The getopt option list and switch cases are out of sync.\n",
                        opt);
                return 1;
        }
    }

    kernentry *mykern = NULL;

    if (optind==argc) {
        /* No kernel name specified. */
        printf("Error: No kernel name specified.\n");
    } else {
        for (int i=0; thekerns[i].name[0]; i++) {
            if (!strcmp(thekerns[i].name, argv[optind])) {
                mykern = &(thekerns[i]);
                break;
            }
        }

        if (!mykern) {
            printf("Unknown kernel: %s\n", argv[optind]);
        }
    }

    /* Print help and exit if kernel name was not specified or not found. */
    if (!mykern) {
        printhelp(argv[0]);
        return 0;
    }

    /* Set default sizes if needed, based on check mode */
    if (docheck) {
        if (!p.input_width)      p.input_width=p.output_width=100;
        if (!p.output_channels)  p.output_channels=100;
        if (!p.input_channels)   p.input_channels=100;
    } else {
        if (!p.input_width)      p.input_width=p.output_width=768;
        if (!p.output_channels)  p.output_channels=768;
        if (!p.input_channels)   p.input_channels=768;
    }

    if (mykern->flags & KERN_CONV_ONLY) {
        test_impl = "conv2d";
    }

    if (mykern->flags & KERN_FAST_ONLY) {
        p.fast_mode = true;
    }

    if (mykern->flags & KERN_ADD) {
        p.add_problem = true;
        p.accumulate = true;
        p.input_channels = p.output_channels;
    }

    /* Set test config */
    if (test_id) {
        if (!get_test_config(&p, test_suite, test_id, test_impl, mykern->flags & KERN_WGOK, mykern->flags & KERN_CONV)) {
            printf("Error selecting test config, aborting.\n");
            exit(1);
        }
        manual_spec = false;
    }

    if (read_dump_fname != "") {
        read_dump_file = read_dump(&p, read_dump_fname.c_str());
        manual_spec = false;
    } else {
        read_dump_file = nullptr;
    }

    /* Override groups for manual depthwise problems. */
    if (mykern->flags & KERN_DEPTHWISE && manual_spec) {
        p.groups = p.input_channels;
    }

    /* Override M size for gemv kernels. */
    if (mykern->flags & KERN_GEMV) {
        p.input_width=p.output_width=1;
        if (mykern->flags & KERN_GEMV_N) {
            p.input_width=p.output_width=p.output_channels;
            p.output_channels=1;
        }
    }

    /* Warn about unsupported parameters for legacy kernels. */
    if (mykern->flags & KERN_LEGACY) {
        // Legacy kernels also don't support convolution - but that will be sorted out by the convolution check
        // below.  Here we just test for parameters that are legal in "pure GEMMs" but not supported by the legacy
        // kernels.

        if (p.use_bias) {
            printf("Warning: Kernel %s does not support bias, ignoring -b\n", mykern->name);
            p.use_bias=false;
        }

        if (p.batches != 1) {
            printf("Warning: Kernel %s does not support batches, ignoring -y\n", mykern->name);
            p.batches=1;
        }

        if (p.multis != 1) {
            printf("Warning: Kernel %s does not support multis, ignoring -z\n", mykern->name);
            p.multis=1;
        }
    }

    if (!(mykern->flags & KERN_ACTOK)) {
        if (p.act.type != Activation::Type::None) {
            printf("Warning: Kernel %s does not support activation, ignoring -a\n", mykern->name);
            p.act=Activation();
        }
    }

    if (mykern->flags & (KERN_QUANTIZED | KERN_NOACC)) {
        if (!(mykern->flags & KERN_ADD) && p.accumulate) {
            printf("Warning: accumulate not supported by %s, ignoring -A.\n", mykern->name);
            p.accumulate=false;
        }
    }

    if (!p.is_basic_gemm() && !(mykern->flags & KERN_CONV)) {
        printf("Error: Kernel %s does not support convolution problems, aborting.\n", mykern->name);
        return 0;
    }

    if (doquery) {
        mykern->print(&p, nthreads);
        return 0;
    }

    // Delegate some checks to GemmProblem.
    if (!p.validate()) {
        return 0;
    }

    /* TODO: Add something to force it to error out rather than rounding up
     * the sizes, to make efficiency calculations based on the size/runtime
     * reliable.
     */
    if (docheck) {
        mykern->test(&p, iterations, nthreads, mykern->name, read_dump_file);
    } else {
        mykern->bench(&p, iterations, nthreads, run_delay, mapfn, mykern->name);
        printf("Benchmark complete (no compare).\n");
    }

    return 0;
}

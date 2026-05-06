//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>

#include "kai/ops/gemm/kai_ops.hpp"
#include "kai/ops/bfloat.hpp"

// Various other useful functions

// Compute greatest common divisor using Euclid's algorithm
static inline int gcd(int a, int b) {
    for (;;) {
        if (a == 0) return b;
        b %= a;
        if (b == 0) return a;
        a %= b;
    }
}

// Compute lowest common multiple using GCD defined above.
static inline int lcm(int a, int b) {
    int temp = gcd(a, b);

    return temp ? (a / temp * b) : 0;
}

// Round up to the nearest multiple
static inline int round_up_to_nearest_multiple(int val, int multiple) {
    val = val + (multiple - 1);
    val /= multiple;
    return (val * multiple);
}

/* Memory allocation function */
void *allocate_aligned_memory(size_t alignment, size_t size);
void free_aligned_memory(void *ptr);

/* Thread mapping function.  This function takes two parameters: our thread
 * ID and the number of threads detected in the system.  It returns a CPU ID
 * to bind this thread to.
 *
 * Declare this type even in non-threading scenarios so that we don't have
 * to #ifdef function signatures.
 */
typedef int (*mapfn)(int, int);

class GemmRandom {
//    std::random_device rd;
    std::mt19937 eng;
    std::uniform_real_distribution<> distr;

public:
    GemmRandom() : eng(0xf00df00d), distr(-0.9, 1.0) {
        eng.seed(1);
    }

    double operator()(void) {
        return distr(eng);
    }

    uint32_t get_int(void) {
        return eng();
    }
};

// Generate a random number between -50.00 and +49.99...
template <typename T>
static inline T get_random_number(void) {
    static GemmRandom gr;

    return static_cast<T>( gr() );
//    return (T) (((double)(rand() % 100000) / 100.0) - 50.0);
}

template <>
inline uint8_t get_random_number<>(void) {
    return rand() % 256;
}

template <>
inline int8_t get_random_number<>(void) {
    return (rand() % 255) - 127;
}

template <>
inline uint16_t get_random_number<>(void) {
    return rand() % 65535;
}

template <>
inline int16_t get_random_number<>(void) {
    return (rand() % 65535) - 32768;
}

// Return a NaN for floating point types, or fall back on the above routines for integer types.
template <typename T>
static inline T get_random_number_or_NaN(void) {
#ifdef __ANDROID__
    return get_random_number<T>();
#else
    if (std::numeric_limits<T>::has_quiet_NaN) {
        return std::numeric_limits<T>::quiet_NaN();
    } else {
        return get_random_number<T>();
    }
#endif
}

template<typename T>
inline T get_random_number_bits(int bits, bool pos_only=false) {
    static GemmRandom gr;

    uint32_t raw_value = gr.get_int();
    int32_t value;

    // Assuming int is 32 bit
    if (bits < 31) {
        int sign = (raw_value >> 31) && (pos_only == false);
        int masked = raw_value & ((1 << bits) - 1);

        if (std::is_unsigned<T>::value) {
            sign=0;
        }

        value = (sign ? -1 : 1) * masked;
    } else {
        value = (int32_t) raw_value;
    }

    return (T)value;
}

template <typename T>
static inline T get_compare_threshold(void) {
    if (std::is_integral<T>::value) {
        return 0;
    } else {
        return 0.0;
    }
}

/*
template<>
inline float get_compare_threshold<float>(void) {
    return 0;
}

template<>
inline __fp16 get_compare_threshold<__fp16>(void) {
    return 0;
}

template<>
inline bfloat16 get_compare_threshold<bfloat16>(void) {
    return 0;
}
*/

// Reference activation implementation
using kai::ops::Activation;

template<typename T>
inline T activate(Activation act, T value) {
    switch(act.type) {
        default:
        case Activation::Type::None:
            return value;

        case Activation::Type::ReLU:
            return std::max(value, static_cast<T>(0));

        case Activation::Type::BoundedReLU:
            return std::min(static_cast<T>(act.param1), std::max(value, static_cast<T>(0)));
    }
}

// Descriptor for quantization parameters, with methods to do appropriate conversions
template <typename intT, typename floatT = float>
struct QuantizeParameters {
    // Scale factor - how much FP32 range is represented by each increment of the integer type.
    floatT           m_scale;
    // Zero point - which integer value represents zero.
    intT             m_zeropt;
    // Minimum quantized value - might not be relevant type minimum (e.g. s8 excluding -128).
    intT             m_minval = std::numeric_limits<intT>::min();
    // Maximum quantized value - might not be relevant type maximum (e.g. 12 bit numbers in u16).
    intT             m_maxval = std::numeric_limits<intT>::max();

    // This looks a bit like an explosion in a cast factory, but we have to
    // keep everything in the floating point type to avoid overflow in the
    // integer type.
    intT quantize(const floatT v) {
        const floatT scaled_f = ::round(v / m_scale);
        const floatT moved_f  = scaled_f + static_cast<floatT>(m_zeropt);

        return static_cast<intT>(std::min(static_cast<floatT>(m_maxval), std::max(static_cast<floatT>(m_minval), moved_f)));
    }

    floatT dequantize(const intT v) {
        const floatT moved_f = static_cast<floatT>(v) - static_cast<floatT>(m_zeropt);

        return moved_f * m_scale;
    }

    QuantizeParameters() : m_scale(0.0f), m_zeropt(0) {}

    QuantizeParameters(floatT scale, intT zeropt) : m_scale(scale), m_zeropt(zeropt) {  }
    QuantizeParameters(floatT scale, intT zeropt, intT minval, intT maxval) :
        m_scale(scale), m_zeropt(zeropt), m_minval(minval), m_maxval(maxval) { }

    // Forced zero point and fp range
    QuantizeParameters(intT zeropt, std::pair<floatT, floatT> range,
                       intT minval = std::numeric_limits<intT>::min(),
                       intT maxval = std::numeric_limits<intT>::max()) : m_zeropt(zeropt), m_minval(minval), m_maxval(maxval) {
        // Figure out how many representable values appear each side of the forced zero point.
        const floatT steps_upper = static_cast<floatT>(maxval) - static_cast<floatT>(zeropt);
        const floatT steps_lower = static_cast<floatT>(zeropt) - static_cast<floatT>(minval);

        // Compute corresponding scale values for above and below zero.
        // Avoid cases that don't make sense, either because there are no
        // values to represent (e.g. minimum value above zero implies there
        // are no negative numbers to represent) or there is no space to
        // represent them (e.g. unsigned type with forced zero point of 0
        // means we can't represent any negative numbers).
        const floatT upper_scale = (range.second <= 0 || steps_upper==0) ? 0 : range.second / steps_upper;
        const floatT lower_scale = (range.first >= 0 || steps_lower==0) ? 0 : -range.first / steps_lower;

        // The scale is the highest of these two values.
        m_scale = std::max(upper_scale, lower_scale);
    }

    // FP range
    QuantizeParameters(std::pair<floatT, floatT> range,
                       intT minval = std::numeric_limits<intT>::min(),
                       intT maxval = std::numeric_limits<intT>::max()) : m_minval(minval), m_maxval(maxval) {
        // Figure out how many steps there are...
        const floatT output_steps = static_cast<floatT>(maxval) - static_cast<floatT>(minval);

        // We always have to be able to represent zero, so figure out what the actual minimum value is
        const floatT f_minval = std::min(range.first, static_cast<floatT>(0));

        // And what range of numbers we need to represent (expanded if needed to include 0).
        const floatT input_range = std::max(range.second, static_cast<floatT>(0)) - f_minval;

        // Scale is just one over the other...
        m_scale  = input_range / output_steps;

        // For the zero point, figure out how many steps the lowest output
        // value is below zero, and add that many steps on to the bottom of
        // the range.  Need to do in float type, because this number of
        // steps might not be representable in intT (for example, if
        // quantizing to a signed type and the range is entirely negative).
        m_zeropt = static_cast<intT>(::round(-f_minval / m_scale) + static_cast<floatT>(minval));
    }

    // In order to quantize the bias, we need to combine the scale factors of both the operand matrices.
    template<typename other_intT1, typename other_intT2, typename other_floatT>
    QuantizeParameters(QuantizeParameters<other_intT1, other_floatT> op1, QuantizeParameters<other_intT2, other_floatT> op2) {
        m_scale = op1.m_scale * op2.m_scale;
        m_zeropt = 0;
    }

    void print(const char *desc) {
#ifndef SILENT
        printf("%s quantization parameters: scale=%f, zero point=%d\n",desc,m_scale,m_zeropt);
#endif
    }
};


// GemmProblem struct: Stores the parameters for the GEMM problem.
//
// For maximum flexibility, problems are expressed as convolution problems.
//
// It is assumed that input and output activations are in (M)NHWC layout,
// with inputs on the GEMM LHS, while weights are (M)HWIO on the GEMM RHS.
// For now, it is further assumed that 'HW' for input/output, and 'HWI' for
// weights, are packed, i.e.  there is no explicit stride specified for
// these.
//
// GEMM problems are mapped to convolution problems as follows:
//
// input_width = output_width = M
// output_channels = N
// input_channels = K
// input_height = output_height = kernel_width = kernel_height = out_stride_h = out_stride_w = 1;
// padding = false (although this should not affect 1x1 kernels)

// This enum is for the gemm-linux classes - when presented with a convolution problem, what should the wrapper do?
//
// fail: abort
// im2row: allocate an im2row buffer, perform im2row, then perform a normal GEMM
// indirect: create an 'indirect' buffer, request indirect GEMM and pass it in
// convolution: request indirect GEMM, pass in convolution parameters

enum class ConvStrategy {
    fail,
    im2row,
    indirect,
    convolution
};

struct strategy_name {
    std::string name;
    ConvStrategy strat;
};

struct WinogradProblem {
  unsigned int output_tile_rows = 0u, output_tile_cols = 0u;
  std::string input_transform_filter = {};
  std::string weight_transform_filter = {};
  std::string output_transform_filter = {};
};

struct GemmProblem {
    // Default parameters are set such that you can get a pure "GEMM" problem by setting:
    //  input_width=output_width=M, input_channels=K, output_channels=N
    // These values therefore default to zero.
    // Strides are also set to zero - these get populated appropriately later.
    int64_t               input_height = 1;
    int64_t               input_width = 0;
    int64_t               input_channels = 0;
    int64_t               kernel_height = 1;
    int64_t               kernel_width = 1;
    int64_t               output_height = 1;
    int64_t               output_width = 0;
    int64_t               output_channels = 0;
    int64_t               padding_top = 0;
    int64_t               padding_left = 0;
    // "output stride" - how much we stride across the input array, per output point
    int64_t               out_stride_h = 1;
    int64_t               out_stride_w = 1;
    // "input stride" - how much we stride across the input array, per kernel point (sometimes called "dilation")
    int64_t               in_stride_h = 1;
    int64_t               in_stride_w = 1;
    int64_t               groups = 1;
    int64_t               batches = 1;
    int64_t               multis = 1;
    int64_t               a_stride = 0;
    int64_t               b_stride = 0;
    bool                  fast_mode = false;
    bool                  fixed_format = false;
    bool                  time_weight_transform = false;
    bool                  dynamic_scheduling = false;
    int64_t               dynamic_granule_count = 0;
    std::string           kernel_filter = {};
    kai::ops::WeightFormat weight_format = kai::ops::WeightFormat::ANY;
    int64_t               inner_block_size = 0;
    int64_t               outer_block_size = 0;
    bool                  accumulate = false;  // Add to what's already in C.
    bool                  add_problem = false; // Force an add problem.
    bool                  transposed_b = false; // Transpose B if supported
    // Activate the input before handing it to the layer: this can enable some
    // optimisations.
    bool                  after_relu = false;
    // bias note: this flag just tells the infrastructure whether to
    // generate the bias.  Kernels apply any bias they are given and ignore
    // this flag.
    bool                  use_bias = false;
    // n_teams note: this flag is just for the infrastructure.
    int                   n_teams = 1;
    // min_ms note: also just for the infrastructure.
    unsigned int          min_ms = 0;
    // cache_stats: also just for the infrastructure.
    bool                  cache_stats = false;
    bool                  cache_flush = false;
    unsigned int          schedule_shape_override = 0;
    Activation            act = {};
    ConvStrategy          strategy = ConvStrategy::fail;
    // Winograd specific configuration filters
    WinogradProblem       winograd_args;
    // File to dump problem output to
    std::string           dump_file = {};


    // Is this a valid mapping of a GEMM problem as per the criteria above?
    bool is_basic_gemm(void) {
        if (kernel_height != 1 || kernel_width != 1)
            return false;

        if (out_stride_h != 1 || out_stride_w != 1)
            return false;

        if (in_stride_h != 1 || in_stride_w != 1)
            return false;

        if (input_height != 1 || output_height != 1)
            return false;

        if (input_width != output_width)
            return false;

        if (groups != 1)
            return false;

        return true;
    }

    // Is this a valid combination of parameters?
    bool validate(void) {
        // Grouping must divide channels evenly.
        if (input_channels % groups) {
            printf("Error: groups (%lld) is not a foctor of input_channels (%lld).\n",(long long)groups, (long long)input_channels);
            return false;
        }

        if (output_channels % groups) {
            printf("Error: groups (%lld) is not a factor of output_channels (%lld).\n",(long long)groups, (long long)output_channels);
            return false;
        }

        return true;
    }
};

template<typename T>
struct DeltaType {
    typedef T t;
};

template<>
struct DeltaType<int8_t> {
    typedef int t;
};

template<>
struct DeltaType<int16_t> {
    typedef int t;
};

// Matrix class: Stores matrices for use by the GEMM kernels.
// Also includes various functions for randomization, setting test patterns, transposing, interleaving, etc.
template <typename T>
class Matrix {
public:
    T *data=nullptr;
    int M;
    int N;
    int batches;
    int multis;
    int stride;
    int batch_stride=0;
    int multi_stride=0;
    const bool is_clone=false; // clones don't own their data

    Matrix(Matrix &) = delete;
    Matrix operator=(Matrix &) = delete;

    // Constructor for clones / submatrices
    Matrix(T *data, int M, int N, int batches, int multis, int stride, int batch_stride, int multi_stride)
        : data(data), M(M), N(N), batches(batches), multis(multis), stride(stride),
          batch_stride(batch_stride), multi_stride(multi_stride), is_clone(true) { }

    Matrix(int M, int N, int stride, int batches=1, int multis=1) : M(M), N(N), batches(batches), multis(multis), stride(stride) {
        // Do some aligning of batches and multis, this will make sure the right strides are being used.
        int batch_size = M*N*sizeof(T);

        // Make batch size an exact number of cache lines.  If it already
        // is, add an empty line of padding (by adding 0x40 instead of
        // 0x3F).
        batch_size = (batch_size + 0x40) & (~0x3F);

        batch_stride = batch_size / sizeof(T);

        int multi_size = batch_size * batches;

        // Make multi size an exact number of pages.
        multi_size = (multi_size + 0xFFF) & (~0xFFF);

        multi_stride = multi_size / sizeof(T);

        data = (T *)allocate_aligned_memory(4096, (multi_size * (multis-1)) +
                                                  (batch_size * (batches-1)) +
                                                  (sizeof(T) * (stride * (M-1) + N)));
//        printf("New Matrix, size (%d, %d, %d, %d)\n", M, N, batches, multis);

        if (data == nullptr) {
            printf("Matrix: memory allocation error.\n");
            exit(1);
        }
    }

    Matrix(int M, int N) : M(M), N(N), batches(1), multis(1), stride(N), batch_stride(0), multi_stride(0) {
        data = (T *)allocate_aligned_memory(4096, M * N * sizeof(T));
    }

    ~Matrix() {
        if (!is_clone) {
            free_aligned_memory(data);
        }
//        printf("Destroyed Matrix, size (%d, %d)\n", M, N);
    }

    // For performance reasons it's expected that normal users will just access data directly (it's public).
    // Provide this helper function for non-performance critical stuff like transposes.
    T safe_read(int multi, int batch, int y, int x) const {
        if ((y>=0) && (x>=0) && (y < M) && (x < N) && (multi < multis) && (batch < batches)) {
            return data[multi*multi_stride + batch*batch_stride + y*stride + x];
        }

        return 0;
    }

    void safe_set(int multi, int batch, int y, int x, T v) {
        if ((y>=0) && (x>=0) && (y<M) && (x<N) && (multi<multis) && (batch<batches)) {
            data[multi*multi_stride + batch*batch_stride + y*stride + x] = v;
        } else {
            printf("Invalid write: %d, %d, %d, %d [ %d, %d, %d, %d ]\n",multi,batch,y,x,M,N,multis,batches);
        }
    }

    void dump_out(FILE *fp) {
        // Dump data size and matrix dimensions as a check.
        size_t data_size = sizeof(T);
        fwrite(&data_size, sizeof(size_t), 1, fp);
        fwrite(&M, sizeof(M), 1, fp);
        fwrite(&N, sizeof(N), 1, fp);
        fwrite(&batches, sizeof(batches), 1, fp);
        fwrite(&multis, sizeof(multis), 1, fp);

        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    fwrite(data + m*multi_stride + b*batch_stride + y*stride, sizeof(T), N, fp);
                }
            }
        }
    }

    void read_dump(FILE *fp) {
        size_t dump_size;

        fread(&dump_size, sizeof(size_t), 1, fp);

        if (dump_size != sizeof(T)) {
            printf("read_dump(): Data size mismatch.\n");
            exit(1);
        }

        int check_dims[] = { M, N, batches, multis };

        for (unsigned int i=0; i<4; i++) {
            int dim;
            fread(&dim, sizeof(dim), 1, fp);

            if (dim != check_dims[i]) {
                printf("read_dump(): Incorrect dimension %u.\n", i);
                exit(1);
            }
        }

        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    fread(data + m*multi_stride + b*batch_stride + y*stride, sizeof(T), N, fp);
                }
            }
        }
    }

    void RandomizeBits(int acc_bits, int res_bits=0, bool spam=false) {
        if (res_bits == 0) {
            res_bits = acc_bits;
        }

        for(int m=0; m<multis; m++) {
            for(int b=0; b<batches; b++) {
                for(int y=0; y<M; y++) {
                    for(int x=0; x<N; x++) {
                        // Random number generation outsourced to function above to
                        // allow it to be specialized for particular types.
                        double res = get_random_number_bits<double>(acc_bits, std::is_unsigned<T>::value);
                        if (res_bits < acc_bits) {
                            int divisor = 1 << (acc_bits - res_bits + 1);
                            res = res / (double)divisor;
                        }
                        data[(m*multi_stride) + (b * batch_stride) + (y*stride) + x] = res;
                        if (spam) {
                            printf("Random %d,%d: %f (%f)\n", acc_bits, res_bits, res, (double)data[(m*multi_stride) + (b * batch_stride) + (y*stride) + x]);
                        }
                    }
                }
            }
        }
    }

#ifdef SLOW_RANDOMIZE
    void Randomize(void) {
        for(int m=0; m<multis; m++) {
            for(int b=0; b<batches; b++) {
                for(int y=0; y<M; y++) {
                    for(int x=0; x<N; x++) {
                        // Random number generation outsourced to function above to
                        // allow it to be specialized for particular types.
                        data[(m*multi_stride) + (b * batch_stride) + (y*stride) + x] = get_random_number<T>();
                    }
                }
            }
        }
    }

    void RandomizeOrNaN(void) {
        for(int m=0; m<multis; m++) {
            for(int b=0; b<batches; b++) {
                for(int y=0; y<M; y++) {
                    for(int x=0; x<N; x++) {
                        // Random number generation outsourced to function above to
                        // allow it to be specialized for particular types.
                        data[(m*multi_stride) + (b * batch_stride) + (y*stride) + x] = get_random_number_or_NaN<T>();
                    }
                }
            }
        }
    }
#else
    // Faster randomize option: Get a buffer of prime size to minimize chance of resonance, blat across array.
    template <bool NaN>
    void Randomize_int(T multiplier) {
        const size_t buffer_size = 251;

        T random_buffer[buffer_size];

        for (unsigned int i=0; i<buffer_size; i++) {
            if (NaN) {
                random_buffer[i] = get_random_number_or_NaN<T>() * multiplier;
            } else {
                random_buffer[i] = get_random_number<T>() * multiplier;
            }
        }

        // This class always occupies a contiguous chunk of memory, regardless of strides.
        // So compute the total footprint here.
        size_t totalsize=(multis-1)*multi_stride + (batches-1)*batch_stride+(M-1)*stride+N;

        for (size_t i=0; i<totalsize; i+=buffer_size) {
            const int64_t num = std::min(buffer_size, (totalsize-i));

            memcpy(reinterpret_cast<void *>(data + i), reinterpret_cast<void *>(random_buffer), num * sizeof(T));
        }
    }

    void Randomize(T multiplier=static_cast<T>(1)) {
        Randomize_int<false>(multiplier);
    }

    void RandomizeOrNaN(T multiplier=static_cast<T>(1)) {
        Randomize_int<true>(multiplier);
    }
#endif

    template<typename T2>
    void Mask(T2 msk) {
        for(int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for(int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        union ugh {
                            T v1;
                            T2 v2;

                            ugh() : v1() { }
                        } u;

                        u.v1 = safe_read(m,b,y,x);
                        u.v2 &= msk;
                        safe_set(m,b,y,x,u.v1);
                    }
                }
            }
        }
    }

    // Set the matrix to contain the ID matrix
    void ID(void) {
        for(int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for(int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        T v=0;

                        if (y==x) {
                            v=1;
                        }

                        data[(m*multi_stride) + (b*batch_stride) + (y*stride)+x] = v;
                    }
                }
            }
        }
    }

    void AllOne(void) {
        for(int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for(int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        data[(m*multi_stride) + (b*batch_stride) + (y*stride)+x] = 1;
                    }
                }
            }
        }
    }

    // Set the matrix to contain a recognizable test pattern, here just
    // increasing numbers in the top row.
    void TestPattern(bool enumerate_columns=false,
                     bool enumerate_rows=false) {
        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for(int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        T v=0;
                        if (!enumerate_columns && !enumerate_rows) {
                            if (y == 0) {
                                v = (m*10000) + (b * 100) + x+1;
                            }
                        } else if (enumerate_columns && enumerate_rows) {
                            v = y * N + x + 1;
                        } else if (enumerate_columns) {
                            v = x + 1;
                        } else if (enumerate_rows) {
                            v = y + 1;
                        }

                        data[(m*multi_stride) + (b*batch_stride) + (y*stride)+x] = v;
                    }
                }
            }
        }
    }

    // Zero the matrix.
    void Zero() {
        memset((void *)data, 0, ((multis-1) * multi_stride + (batches-1) * batch_stride + (M-1) * stride + N) * sizeof(T));
    }


    // Copy some other matrix.
    template<typename T2>
    void Copy(std::shared_ptr<Matrix<T2>> from) {
        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        data[(m*multi_stride) + (b*batch_stride) + (y * stride) + x] = static_cast<T>(from->safe_read(m, b, y, x));
                    }
                }
            }
        }
    }


    void Transpose(std::shared_ptr<Matrix> from) {
        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        data[(m*multi_stride) + (b*batch_stride) + (y * stride) + x] = from->safe_read(m, b, x, y);
                    }
                }
            }
        }
    }

    // Set up an interleaved matrix such that each row contains several rows interleaved.
    // i.e. for a normal M*N matrix interleaved by 8, size should be (M/8)*(N*8)
    void Interleave(std::shared_ptr<Matrix> from, int int_by) {
        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    T *out_ptr = data + (y * stride) + (b * batch_stride) + (m * multi_stride);
                    for (int x=0; x<(N/int_by); x++) {
                        for (int z = 0; z<int_by; z++) {
                            *out_ptr++ = from->safe_read(m, b, (y * int_by) + z, x);
                        }
                    }
                }
            }
        }
    }

    // As above, but transposes at the same time.
    void Interleave_Transposed(std::shared_ptr<Matrix> from, int int_by) {
        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    T *out_ptr = data + (y * stride) + (b * batch_stride) + (m * multi_stride);
                    for (int x=0; x<(N/int_by); x++) {
                        for (int z = 0; z<int_by; z++) {
                            *out_ptr++ = from->safe_read(m, b, x, (y * int_by) + z);
                        }
                    }
                }
            }
        }
    }

    // Interleave_Blocked copies a block of values at a time instead of just
    // one.  The main use of this is the gemmlowp with the "dot product"
    // instruction, where each operation consumes 4 values, so we need to
    // copy blocks of 4 values.
    void Interleave_Blocked(std::shared_ptr<Matrix> from, int int_by, int block) {
        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    T *out_ptr = data + (y * stride) + (b * batch_stride) + (m * multi_stride);
                    for (int x=0; x<(N/int_by); x+=block) {
                        for (int z = 0; z<int_by; z++) {
                            for (int a = 0; a<block; a++) {
                                *out_ptr++ = from->safe_read(m, b, (y * int_by) + z, x+a);
                            }
                        }
                    }
                }
            }
        }
    }

    // As abov, but with a transpose at the same time.
    template<typename T2>
    void Interleave_Blocked_Transposed(std::shared_ptr<Matrix<T2>> from, int int_by, int block) {
        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    T *out_ptr = data + (y * stride) + (b * batch_stride) + (m * multi_stride);
                    for (int x=0; x<(N/int_by); x+=block) {
                        for (int z = 0; z<int_by; z++) {
                            for (int a = 0; a<block; a++) {
                                *out_ptr++ = from->safe_read(m, b, x+a, (y * int_by) + z);
                            }
                        }
                    }
                }
            }
        }
    }

    bool Compare(std::shared_ptr<Matrix> with, T threshold=get_compare_threshold<T>()) {
        int e=0;

        for(int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for(int y=0; y<M; y++) {
                    for(int x=0; x<N; x++) {
                        T golden = safe_read(m, b, y, x);
                        T test = with->safe_read(m, b, y, x);

                        // Make delta always positive so that things don't explode if T is an unsigned integer type.
                        typename DeltaType<T>::t delta = (golden >= test) ? (typename DeltaType<T>::t)(golden-test) : (typename DeltaType<T>::t)(test-golden);

                        int compare_ok = 1;

                        if (std::is_integral<T>::value) {
                            if (delta > threshold) {
                                compare_ok = 0;
                            }
                        } else {
                            if (delta > threshold || std::isnan(test)) {
                                compare_ok = 0;
                            }
                        }

                        if (!compare_ok) {
                            printf("Compare error: m=%d, b=%d, x=%d, y=%d, golden=%g, test=%g, delta=%g, test addr=%p\n",m,b,x,y,(double)golden,(double)test,(double)delta,(void *)(with->data + m*with->multi_stride + b*with->batch_stride + y*with->stride + x));
                            e++;
                            if (e>5000) {
                                printf("Too many errors, not printing any more.\n");
                                goto abort;
                            }
                        } else {
//                            printf("Compare: m=%d, b=%d, x=%d, y=%d, golden=%g, test=%g, delta=%g\n",m,b,x,y,(double)golden, (double)test, (double)delta);
                        }
                    }
                }
            }
        }

    abort:
        if(e) {
            printf("Compare errors :(\n");
            return false;
        } else {
            printf("Compare OK.\n");
            return true;
        }
    }

    // Quantization related functions
    // This one returns a vector of pairs of min/max values.
    // If per_channel is true, the length of the vector is equal to the s
    template <bool per_channel=false>
    std::vector<std::pair<T, T> > get_range() {
        unsigned int l = per_channel ? N : 1;

        std::vector<std::pair<T, T> > r;

        for (unsigned int i=0; i<l; i++) {
            r.emplace_back(std::make_pair(std::numeric_limits<T>::max(), std::numeric_limits<T>::min()));
        }

        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        int n = per_channel ? x : 0;

                        T v = safe_read(m, b, y, x);

                        if (v < r[n].first) {
                            r[n].first = v;
                        }

                        if (v > r[n].second) {
                            r[n].second = v;
                        }
                    }
                }
            }
        }

        return r;
    }

    template<typename T_from>
    void Quantize(std::shared_ptr<Matrix<T_from> > from, QuantizeParameters<T, T_from> &params) {
        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        data[(m*multi_stride) + (b*batch_stride) + (y * stride) + x] = params.quantize(from->safe_read(m, b, y, x));
                    }
                }
            }
        }
    }

    template<typename T_from>
    void Quantize(std::shared_ptr<Matrix<T_from> > from, std::vector< QuantizeParameters<T, T_from> > &params) {
        if (params.size() == 1) {
            return Quantize(from, params[0]);
        }

        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        data[(m*multi_stride) + (b*batch_stride) + (y * stride) + x] = params[x].quantize(from->safe_read(m, b, y, x));
                    }
                }
            }
        }
    }

    template<typename Tint>
    void QuantizeRound(QuantizeParameters<Tint, T> &params) {
        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        T &v = data[(m*multi_stride) + (b*batch_stride) + (y * stride) + x];
                        v = params.dequantize(params.quantize(v));
                    }
                }
            }
        }
    }

    template<typename Tint>
    void QuantizeRound(std::vector<QuantizeParameters<Tint, T> > &params) {
        if (params.size() == 1) {
            return QuantizeRound(params[0]);
        }
        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        T &v = data[(m*multi_stride) + (b*batch_stride) + (y * stride) + x];
                        v = params[x].dequantize(params[x].quantize(v));
                    }
                }
            }
        }
    }

    template<typename T_from, typename T_float>
    void Dequantize(std::shared_ptr<Matrix<T_from> > from, QuantizeParameters<T_from, T_float> &params) {
        for (int m=0; m<multis; m++) {
            for (int b=0; b<batches; b++) {
                for (int y=0; y<M; y++) {
                    for (int x=0; x<N; x++) {
                        data[(m*multi_stride) + (b*batch_stride) + (y * stride) + x] = params.dequantize(from->safe_read(m, b, y, x));
                    }
                }
            }
        }
    }

    // Return submatrix with a subset of rows
    std::shared_ptr<Matrix<T>> split_rows(int M_start, int num_rows) {
        return std::make_shared<Matrix<T>>(data + (M_start * stride), num_rows, N, batches, multis, stride, batch_stride, multi_stride);
    }

    // Return submatrix with a subset of columns
    std::shared_ptr<Matrix<T>> split_cols(int N_start, int num_cols) {
        return std::make_shared<Matrix<T>>(data + N_start, M, num_cols, batches, multis, stride, batch_stride, multi_stride);
    }
};

// Enum used to identify which type of quantization is used by a particular kernel.
enum class QuantizationType
{
    NONE,
    INTEGER,
    FLOAT
};

class Requantize32 : public kai::ops::Requantize32 {
private:
    std::vector<int32_t> mults;
    std::vector<int32_t> left_shifts;
    std::vector<int32_t> right_shifts;

    template<typename Tfloat>
    static std::pair<int32_t, int32_t> get_mul_shift(Tfloat rescale) {
        const Tfloat shiftf = ::round(::log2(0.5f / rescale));
        const Tfloat multf = ::exp2(31.0f + shiftf)*rescale;

        int64_t shift = static_cast<int64_t>(shiftf);
        int64_t mult = static_cast<int64_t>(multf);

        if (mult == (1ll << 31))
        {
            mult /= 2;
            shift--;
        }

//        assert(shift >= 0);
        assert(mult <= std::numeric_limits<int32_t>::max());

        return std::pair<int32_t, int32_t>(-shift, mult);
    }

public:
    template<typename Tfloat, typename Tint>
    void set_multipliers(Tfloat multiplier, const std::vector<QuantizeParameters<Tint, Tfloat> > &qp_B) {
        if (qp_B.size()==1) {
            auto p = get_mul_shift(multiplier * qp_B[0].m_scale);
            this->per_channel_requant = false;
            this->per_layer_right_shift = std::min(p.first, 0);
            this->per_layer_left_shift = std::max(p.first, 0);
            this->per_layer_mul = p.second;
        } else {
            auto rounded_size = round_up_to_nearest_multiple(qp_B.size(), 32);
            bool need_left_shift = false;
            this->per_channel_requant = true;
            for (auto &&i : qp_B) {
                auto p = get_mul_shift(multiplier * i.m_scale);
                left_shifts.push_back(std::max(p.first, 0));
                right_shifts.push_back(std::min(p.first, 0));
                if (p.first > 0) {
                    need_left_shift=true;
                }
                mults.push_back(p.second);
            }
            left_shifts.resize(rounded_size);
            right_shifts.resize(rounded_size);
            mults.resize(rounded_size);
            this->per_channel_right_shifts=right_shifts.data();
            this->per_channel_left_shifts=need_left_shift ? left_shifts.data() : nullptr;
            this->per_channel_muls=mults.data();
        }
    }
};


// Cache flush
//
// Flush every CACHE_LINE_SIZE sized block overlapping the stated [base, base+len) region
#ifdef BARE_METAL

#define CACHE_LINE_SIZE 64

static inline void do_cache_flush(void *base, size_t len) {
    uintptr_t mask = (CACHE_LINE_SIZE - 1);
    uintptr_t start = reinterpret_cast<uintptr_t>(base);
    uintptr_t end = start + len;

    if (len==0) {
        return;
    }

    // Align blocks we clear to cache lines; DC CIVAC does not require an
    // aligned address but aligning here is the easiest way of making sure
    // we hit all the correct blocks.
    start &= ~mask;

    for (uintptr_t addr = start; addr < end; addr+=CACHE_LINE_SIZE) {
        __asm(
            "dc civac, %[addr]"
            :
            : [addr] "r" (addr)
            : "memory"
        );
    }

    __asm("dsb sy" ::: "memory");
}

#endif // BARE_METAL

/* Prototypes from test_suite.cpp */
void print_test_suites();
bool get_test_config(GemmProblem *p, std::string name, unsigned int ID, std::string implname, bool wg_ok, bool conv_ok);

/* Prototypes from file_dumper.cpp */
FILE *write_dump(const GemmProblem *p, const char *filename);
FILE *read_dump(GemmProblem *p, const char *filename);

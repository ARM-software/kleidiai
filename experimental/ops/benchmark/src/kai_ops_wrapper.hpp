//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <time.h>

#include <cassert>
#include <cstring>
#include <type_traits>

#include "barrier.hpp"
#include "gemm_lib.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "scheduler.hpp"
#include "utils.hpp"  // iceildiv

#include "barrier.hpp"

#undef TASKMASTER

// (Templated) glue to make it work with current infrastructure
template <typename Tlop, typename Trop, typename Tret, QuantizationType quantized=QuantizationType::NONE>
class kai_ops_wrapper {
private:
    struct free_delete {
        void operator()(void *x) { free_aligned_memory(x); }
    };

    // Quantized support
    std::shared_ptr<Matrix<int32_t>> bias;
    Requantize32 qp;

    const GemmProblem * const p;

    const unsigned int nthreads;

    std::unique_ptr<void, free_delete> pretransposed_data={};
    std::unique_ptr<void, free_delete> working_data={};

    /* im2row support */
    std::shared_ptr<Matrix<Tlop>>    im2row_buf;

    /* indirect support */
    std::shared_ptr<Matrix<const Tlop *>>       indirect_buf;
    std::vector<Tlop>                     padding_buf;

    std::unique_ptr<const Tlop * const *, free_delete>  indirect_arg;

    kai::ops::UniqueGemmCommon<Tlop, Trop, Tret> gemm;
    std::unique_ptr<scheduler> sched;

    // Need to keep this for timing transform.
    std::shared_ptr<Matrix<Trop>> B;

    // Need these for fixed format cases - B_transformed also used for transpose B cases
    std::shared_ptr<Matrix<Trop>> B_transformed;
    std::shared_ptr<Matrix<bfloat16>> B_transformed_bf16;

    // Either B or B_transformed, depending on whether transpose
    std::shared_ptr<Matrix<Trop>> B_pt;

    bool B_pt_is_transposed = false;

    kai::ops::barrier barrier;

public:
    static int get_m_block() { return 1; }
    static int get_n_block() { return 1; }
    static int get_k_block() { return 1; }

    typedef Tlop  lhs_operand_type;
    typedef Trop rhs_operand_type;
    typedef Tret result_type;

    kai_ops_wrapper(kai_ops_wrapper &) = delete;
    kai_ops_wrapper(kai_ops_wrapper &&) = default;
    kai_ops_wrapper & operator=(kai_ops_wrapper &) = delete;

    // Constructor: non-quantized
    kai_ops_wrapper(std::shared_ptr<Matrix<Tlop>> &A,
              std::shared_ptr<Matrix<Trop>> &B,
              std::shared_ptr<Matrix<Tret>> &C,
              std::shared_ptr<Matrix<Tret>> &bias,
              GemmProblem *p, unsigned int nthreads, int do_init) : p(p), nthreads(nthreads), B(B), barrier(nthreads) {
        assert(quantized==QuantizationType::NONE || quantized==QuantizationType::FLOAT);

        if (p->groups != 1) {
            printf("kai_ops_wrapper: Groups not supported, aborting.\n");
            exit(1);
        }

        const unsigned int output_hw = p->output_height * p->output_width;

        // K parameter for the GEMM - input channels is correct for non-convolution cases, indirect and convolution
        // GEMMs.  For im2row we update this later.
        unsigned int gemm_k = p->input_channels;
        unsigned int gemm_sections = 1;
        bool indirect_flag = false;

        const CPUInfo *ci = get_CPUInfo();

        kai::ops::GemmConfig cfg;

        cfg.filter           = p->kernel_filter;
        cfg.inner_block_size = p->inner_block_size;
        cfg.outer_block_size = p->outer_block_size;
        cfg.weight_format    = p->weight_format;

        if (!p->is_basic_gemm()) {
            if (p->strategy == ConvStrategy::indirect || p->strategy == ConvStrategy::convolution) {
                indirect_flag = true;
                gemm_sections = p->kernel_height * p->kernel_width;
            } else {
                // For im2row GEMMs, we pass in the "full" K value.
                gemm_k = p->kernel_height * p->kernel_width * p->input_channels;
            }
        }

        auto args = kai::ops::GemmArgs(ci, output_hw, p->output_channels, gemm_k, gemm_sections, p->batches, p->multis, indirect_flag, p->act, nthreads, p->fixed_format, p->fast_mode, p->accumulate, &cfg);

        gemm = kai::ops::gemm<Tlop, Trop, Tret>(args);
        if (gemm.get() == nullptr) {
            printf("kai_ops_wrapper: No GEMM matching stated requirements, aborting.\n");
            exit(1);
        }

        do_common_setup(A, B, C, bias, p, nthreads, do_init, 0);
    }

    // Constructor: dequantized
    kai_ops_wrapper(std::shared_ptr<Matrix<Tlop>> &A,
              std::shared_ptr<Matrix<Trop>> &B,
              std::shared_ptr<Matrix<Tret>> &C,
              std::shared_ptr<Matrix<Tret>> &bias,
              kai::ops::DequantizeFloat &scale,
              GemmProblem *p, unsigned int nthreads, int do_init) : p(p), nthreads(nthreads), B(B), barrier(nthreads) {
        assert(quantized==QuantizationType::FLOAT);

        if (p->groups != 1) {
            printf("kai_ops_wrapper: Groups not supported, aborting.\n");
            exit(1);
        }

        const unsigned int output_hw = p->output_height * p->output_width;

        // K parameter for the GEMM - input channels is correct for non-convolution cases, indirect and convolution
        // GEMMs.  For im2row we update this later.
        unsigned int gemm_k = p->input_channels;
        unsigned int gemm_sections = 1;
        bool indirect_flag = false;

        const CPUInfo *ci = get_CPUInfo();

        kai::ops::GemmConfig cfg;

        cfg.filter           = p->kernel_filter;
        cfg.inner_block_size = p->inner_block_size;
        cfg.outer_block_size = p->outer_block_size;
        cfg.weight_format    = p->weight_format;

        if (!p->is_basic_gemm()) {
            if (p->strategy == ConvStrategy::indirect || p->strategy == ConvStrategy::convolution) {
                indirect_flag = true;
                gemm_sections = p->kernel_height * p->kernel_width;
            } else {
                // For im2row GEMMs, we pass in the "full" K value.
                gemm_k = p->kernel_height * p->kernel_width * p->input_channels;
            }
        }

        auto args = kai::ops::GemmArgs(ci, output_hw, p->output_channels, gemm_k, gemm_sections, p->batches, p->multis, indirect_flag, p->act, nthreads, p->fixed_format, p->fast_mode, p->accumulate, &cfg);

        gemm = kai::ops::gemm<Tlop, Trop, Tret, kai::ops::DequantizeFloat>(args, scale);

        if (gemm.get() == nullptr) {
            printf("kai_ops_wrapper: No GEMM matching stated requirements, aborting.\n");
            exit(1);
        }

        do_common_setup(A, B, C, bias, p, nthreads, do_init, 0);
    }

    // Constructor: quantized
    template<typename Tfloat>
    kai_ops_wrapper(std::shared_ptr<Matrix<Tlop>> &A,
              std::shared_ptr<Matrix<Trop>> &B,
              std::shared_ptr<Matrix<Tlop>> &C,
              std::shared_ptr<Matrix<int32_t>> &bias,
              QuantizeParameters<Tlop, Tfloat> &A_qp,
              std::vector<QuantizeParameters<Trop, Tfloat> > &B_qp,
              QuantizeParameters<Tlop, Tfloat> &C_qp,
              GemmProblem *p, unsigned int nthreads, int do_init) : bias(bias), p(p), nthreads(nthreads), B(B), barrier(nthreads) {
        assert(quantized==QuantizationType::INTEGER);

        if (p->groups != 1) {
            printf("kai_ops_wrapper: Groups not supported, aborting.\n");
            exit(1);
        }

        if (p->in_stride_h != 1 || p->in_stride_w != 1) {
            printf("kai_ops_wrapper: in_stride/dilation not supported, aborting.\n");
            exit(1);
        }

        if (p->accumulate) {
            printf("kai_ops_wrapper: Accumulation not supported, aborting.\n");
            exit(1);
        }

        const CPUInfo *ci = get_CPUInfo();

        const unsigned int output_hw = p->output_height * p->output_width;

        // K parameter for the GEMM - input channels is correct for non-convolution cases, indirect and convolution
        // GEMMs.  For im2row we update this later.
        unsigned int gemm_k = p->input_channels;
        unsigned int gemm_sections = 1;
        bool indirect_flag = false;

        kai::ops::GemmConfig cfg;

        cfg.filter           = p->kernel_filter;
        cfg.inner_block_size = p->inner_block_size;
        cfg.outer_block_size = p->outer_block_size;
        cfg.weight_format    = p->weight_format;

        /* Set quantization parameters */
        qp.bias     = (bias ? bias->data : nullptr);
        qp.bias_multi_stride = bias ? bias->multi_stride : 0;
        qp.a_offset = A_qp.m_zeropt;
        qp.b_offset = B_qp[0].m_zeropt;
        qp.c_offset = C_qp.m_zeropt;
        qp.minval   = C_qp.m_minval;
        qp.maxval   = C_qp.m_maxval;

        qp.set_multipliers(A_qp.m_scale / C_qp.m_scale, B_qp);

        if (!p->is_basic_gemm()) {
            if (p->strategy == ConvStrategy::indirect || p->strategy == ConvStrategy::convolution) {
                indirect_flag = true;
                gemm_sections = p->kernel_height * p->kernel_width;
            } else {
                // For im2row GEMMs, we pass in the "full" K value.
                gemm_k = p->kernel_height * p->kernel_width * p->input_channels;
            }
        }

        // Don't (currently) apply activation or accumulation to quantized GEMMs.
        auto args = kai::ops::GemmArgs(ci, output_hw, p->output_channels, gemm_k, gemm_sections, p->batches, p->multis, indirect_flag, Activation(), nthreads, p->fixed_format, p->fast_mode, false, &cfg);

        gemm = kai::ops::gemm<Tlop, Trop, Tret, kai::ops::Requantize32>(args, qp);

        if (gemm.get() == nullptr) {
            printf("kai_ops_wrapper: No GEMM matching stated requirements, aborting.\n");
            exit(1);
        }

        // The "normal" bias is always null in quantized cases - need a null
        // one to pass on to do_common_setup()
        auto null_bias_ptr = std::shared_ptr<Matrix<Tret>>(nullptr);

        do_common_setup(A, B, C, null_bias_ptr, p, nthreads, do_init, A_qp.m_zeropt);
    }

    void do_common_setup(std::shared_ptr<Matrix<Tlop>> &A,
              std::shared_ptr<Matrix<Trop>> &B,
              std::shared_ptr<Matrix<Tret>> &C,
              std::shared_ptr<Matrix<Tret>> &bias,
              GemmProblem *p, unsigned int nthreads, int do_init, Tlop zeropt)
    {
        auto my_cfg = gemm->get_config();

#ifndef SILENT
        printf("GEMM configuration: kernel=%s inner_block=%u outer_block=%u weight_format=%x\n", my_cfg.filter.c_str(), my_cfg.inner_block_size, my_cfg.outer_block_size, static_cast<uint32_t>(my_cfg.weight_format));
#endif
        const unsigned int output_hw = p->output_height * p->output_width;
        const unsigned int kernel_hwi = p->kernel_height * p->kernel_width * p->input_channels;
        const unsigned int kernel_hw = p->kernel_height * p->kernel_width;

        // Values to pass in to describe the B argument in set_arrays()
        // For pretransposed cases these don't matter, but we need to populate them for fixed format.
        // This is done this way because in BF16 cases we need to do some icky casting.
        const Trop *setargs_B_ptr = nullptr;
        size_t setargs_B_stride = 0;
        size_t setargs_B_multi_stride = 0;

        // Do the B transform if needed
        if (p->fixed_format) {
            // If we requested a fixed format kernel, it is required that a valid weight_format is provided.
            if (my_cfg.weight_format == kai::ops::WeightFormat::UNSPECIFIED) {
                printf("Error: Requested fixed format GEMM but no weight format specified.  Aborting.\n");
                exit(1);
            }

            // Weight formats are bitfield encoded, so we can extract the interleave/block parameters by casting and
            // masking.
            uint32_t i_wf = static_cast<uint32_t>(my_cfg.weight_format);

            auto interleave_by = (i_wf >> 8) & 0xFFF;
            auto block_by = (i_wf >> 20) & 0xF;

            // By default, assume we will be packing data per kernel point -
            // so we will need to round up the channel count according to
            // "block_by" for each point.  Basic GEMM cases only have a
            // single point so this makes no difference.
            auto kernel_points = p->kernel_height * p->kernel_width;
            auto rounded_input_channels = roundup<unsigned int>(p->input_channels, block_by);
            auto copy_input_channels = p->input_channels;
            auto rounded_height = iceildiv(B->N, interleave_by);

            // Except that if we are going to use "im2row" style
            // convolution, we shouldn't do the per-kernel-point packing
            // after all - adjust values accordingly so we pack all the
            // points at once.
            if (!p->is_basic_gemm() && p->strategy==ConvStrategy::im2row) {
                kernel_points = 1;
                rounded_input_channels = roundup<unsigned int>(p->input_channels * p->kernel_height * p->kernel_width, block_by);
                copy_input_channels = p->input_channels * p->kernel_height * p->kernel_width;
            }

            // Each row of transformed output contains "interleave_by" channels and also needs to allow for each
            // kernel point being padded based on the input channel blacking.
            auto transformed_width = kernel_points * rounded_input_channels * interleave_by;

            // If the bf16 flag is set, the weights need to be prepared in bf16.
            if (((i_wf >> 4) & 0xf) == 1) {
                B_transformed_bf16 = std::make_shared<Matrix<bfloat16>>(rounded_height, transformed_width, transformed_width, 1, B->multis);

                // To make sure we get per-kernel point padding right, slice the input and output by kernel point and
                // transform each one individually.
                for(int i=0; i<kernel_points; i++) {
                    auto input_slice = B->split_rows(i * copy_input_channels, copy_input_channels);
                    auto output_slice = B_transformed_bf16->split_cols(i * rounded_input_channels * interleave_by, rounded_input_channels * interleave_by);

                    output_slice->Interleave_Blocked_Transposed(input_slice, interleave_by, block_by);
                }

                // In BF16 cases, this pointer is immediately cast back to a bf16 * on the other side.
                setargs_B_ptr = reinterpret_cast<Trop *>(B_transformed_bf16->data);
                setargs_B_stride = B_transformed_bf16->stride;
                setargs_B_multi_stride = B_transformed_bf16->multi_stride;
            } else {
                B_transformed = std::make_shared<Matrix<Trop>>(rounded_height, transformed_width, transformed_width, 1, B->multis);

                // To make sure we get per-kernel point padding right, slice the input and output by kernel point and
                // transform each one individually.
                for(int i=0; i<kernel_points; i++) {
                    auto input_slice = B->split_rows(i * copy_input_channels, copy_input_channels);
                    auto output_slice = B_transformed->split_cols(i * rounded_input_channels * interleave_by, rounded_input_channels * interleave_by);

                    output_slice->Interleave_Blocked_Transposed(input_slice, interleave_by, block_by);
                }

                setargs_B_ptr = B_transformed->data;
                setargs_B_stride = B_transformed->stride;
                setargs_B_multi_stride = B_transformed->multi_stride;
            }
        }

        // Set the arrays for the GEMM, or for the convolution type GEMMs do
        // whatever handling is necessary.
        if (p->is_basic_gemm()) {
            gemm->set_arrays(A->data, A->stride, A->batch_stride, A->multi_stride,
                             setargs_B_ptr, setargs_B_stride, setargs_B_multi_stride,
                             C->data, C->stride, C->batch_stride, C->multi_stride,
                             (bias ? bias->data : nullptr), (bias ? bias->multi_stride : 0));
        } else {
            switch (p->strategy) {
                default:
                case ConvStrategy::fail:
                    printf("Convolution problem requested but no strategy specified - aborting.\n");
                    exit(1);
                    break;

                case ConvStrategy::im2row:
#ifndef SILENT
                    printf("Configuring im2row GEMM.\n");
#endif
                    {
                        im2row_buf = std::make_shared<Matrix <Tlop> >(output_hw, kernel_hwi, kernel_hwi, p->batches, p->multis);

                        if (im2row_buf.get() == nullptr) {
                            printf("Unable to allocate im2row buffer.\n");
                            exit(1);
                        }

                        for (int64_t m=0; m<p->multis; m++) {
                            for (int64_t b=0; b<p->batches; b++) {
                                for (int64_t output_y=0; output_y<p->output_height; output_y++) {
                                    for (int64_t output_x=0; output_x<p->output_width; output_x++) {
                                        int64_t output_xy = (output_y * p->output_width) + output_x;
                                        int64_t output_pos = 0;

                                        for (int64_t kernel_y=0; kernel_y<p->kernel_height; kernel_y++) {
                                            for (int64_t kernel_x=0; kernel_x<p->kernel_width; kernel_x++) {
                                                int64_t input_x = (output_x * p->out_stride_w) + (kernel_x * p->in_stride_w) - p->padding_left;
                                                int64_t input_y = (output_y * p->out_stride_h) + (kernel_y * p->in_stride_h) - p->padding_top;

                                                if (input_x < 0 || input_x >= p->input_width || input_y < 0 || input_y >= p->input_height) {
                                                    for (int64_t inch=0; inch < p->input_channels; inch++) {
                                                        im2row_buf->safe_set(m, b, output_xy, output_pos, zeropt);
                                                        output_pos++;
                                                    }
                                                } else {
                                                    int64_t input_xy = (input_y * p->input_width) + input_x;

                                                    for (int64_t inch = 0; inch < p->input_channels; inch++) {
                                                        im2row_buf->safe_set(m, b, output_xy, output_pos, A->safe_read(m, b, input_xy, inch));
                                                        output_pos++;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        gemm->set_arrays(im2row_buf->data, im2row_buf->stride, im2row_buf->batch_stride, im2row_buf->multi_stride,
                                         setargs_B_ptr, setargs_B_stride, setargs_B_multi_stride,
                                         C->data, C->stride, C->batch_stride, C->multi_stride,
                                         (bias ? bias->data : nullptr), (bias ? bias->multi_stride : 0));
                    }
                    break;

                case ConvStrategy::indirect:
#ifndef SILENT
                    printf("Configuring indirect GEMM.\n");
#endif
                    {
                        indirect_buf = std::make_shared<Matrix <const Tlop *> >(kernel_hw, output_hw, output_hw, p->batches, p->multis);
                        padding_buf  = std::vector<Tlop>(p->input_channels, zeropt);

                        for (int64_t m=0; m<p->multis; m++) {
                            for (int64_t b=0; b<p->batches; b++) {
                                for (int64_t output_y=0; output_y<p->output_height; output_y++) {
                                    for (int64_t output_x=0; output_x<p->output_width; output_x++) {
                                        int64_t output_xy = (output_y * p->output_width) + output_x;

                                        for (int64_t kernel_y=0; kernel_y<p->kernel_height; kernel_y++) {
                                            for (int64_t kernel_x=0; kernel_x<p->kernel_width; kernel_x++) {
                                                int64_t input_x = (output_x * p->out_stride_w) + (kernel_x * p->in_stride_w) - p->padding_left;
                                                int64_t input_y = (output_y * p->out_stride_h) + (kernel_y * p->in_stride_h) - p->padding_top;
                                                int64_t kernel_xy = (kernel_y * p->kernel_width) + kernel_x;
                                                int64_t input_xy = (input_y * p->input_width) + input_x;

                                                if (input_x < 0 || input_x >= p->input_width || input_y < 0 || input_y >= p->input_height) {
                                                    indirect_buf->safe_set(m, b, kernel_xy, output_xy, padding_buf.data());
                                                } else {
                                                    indirect_buf->safe_set(m, b, kernel_xy, output_xy, A->data + (m * A->multi_stride + b * A->batch_stride + input_xy * A->stride));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        indirect_arg = std::unique_ptr<const Tlop * const *, free_delete>(reinterpret_cast<const Tlop * const **>(allocate_aligned_memory(64, sizeof(Tlop **) * p->kernel_height * p->kernel_width * p->multis * p->batches)));

                        int64_t pos=0;
                        for (int64_t m=0;m<p->multis;m++) {
                            for (int64_t b=0;b<p->batches;b++) {
                                for (int64_t kernel_xy=0; kernel_xy<p->kernel_width * p->kernel_height; kernel_xy++) {
                                    (indirect_arg.get())[pos++] = indirect_buf->data + m * indirect_buf->multi_stride + b * indirect_buf->batch_stride + kernel_xy * indirect_buf->stride;
                                }
                            }
                        }

                        gemm->set_arrays(nullptr, 0, 0, 0,
                                         setargs_B_ptr, setargs_B_stride, setargs_B_multi_stride,
                                         C->data, C->stride, C->batch_stride, C->multi_stride,
                                         (bias ? bias->data : nullptr), (bias ? bias->multi_stride : 0));

                        gemm->set_indirect_parameters( p->input_channels, indirect_arg.get() );
                    }
                    break;

                case ConvStrategy::convolution:
#ifndef SILENT
                    printf("Configuring convolution GEMM.\n");
#endif
                    gemm->set_arrays(A->data, A->stride, A->batch_stride, A->multi_stride,
                                     setargs_B_ptr, setargs_B_stride, setargs_B_multi_stride,
                                     C->data, C->stride, C->batch_stride, C->multi_stride,
                                     (bias ? bias->data : nullptr), (bias ? bias->multi_stride : 0));

                    kai::ops::ConvolutionParameters cp = {
                        p->input_width, p->input_height, p->input_channels,
                        p->kernel_width, p->kernel_height, p->output_width, p->output_height,
                        p->out_stride_w, p->out_stride_h, p->in_stride_w, p->in_stride_h,
                        p->padding_top, p->padding_left, static_cast<float>(zeropt) };

                    gemm->set_convolution_parameters(cp);
                    break;
            }
        }


        if (gemm->B_is_pretransposed()) {
            assert(!p->fixed_format); // Fixed format and pretransposed makes no sense.

            if (p->transposed_b && gemm->B_pretranspose_supports_transpose()) {
#ifndef SILENT
                printf("Generating transposed form of B matrix.\n");
#endif
                B_transformed = std::make_shared<Matrix<Trop>>(B->N, B->M, std::max<int>(B->M, p->b_stride), 1, B->multis);
                B_transformed->Transpose(B);

                B_pt = B_transformed;
                B_pt_is_transposed = true;
            } else {
                B_pt = B;
                B_pt_is_transposed = false;
            }

#ifndef SILENT
            printf("Allocating pretranspose data: %zu bytes\n",gemm->get_B_pretransposed_array_size());
#endif
            pretransposed_data = std::unique_ptr<void, free_delete>(allocate_aligned_memory(64, gemm->get_B_pretransposed_array_size()));
            if (!p->time_weight_transform) {
#ifndef BARE_METAL
                struct timespec ts1, ts2;
                clock_gettime(CLOCK_MONOTONIC, &ts1);
                gemm->pretranspose_B_array(pretransposed_data.get(), B_pt->data, B_pt->stride, B_pt->multi_stride, B_pt_is_transposed);
                clock_gettime(CLOCK_MONOTONIC, &ts2);
#ifndef SILENT
                int64_t res=(ts2.tv_sec - ts1.tv_sec) * 1000000000 + (ts2.tv_nsec - ts1.tv_nsec);
                printf("Pretranspose: %" PRId64 " ns.\n", res);
#endif
#else // BARE_METAL
                gemm->pretranspose_B_array(pretransposed_data.get(), B_pt->data, B_pt->stride, B_pt->multi_stride, B_pt_is_transposed);
                if (p->cache_flush) {
                    do_cache_flush(pretransposed_data.get(), gemm->get_B_pretransposed_array_size());
                }
#endif
            }
        }

        if (gemm->get_working_size()) {
#ifndef SILENT
            printf("Allocating working data: %zu bytes\n",gemm->get_working_size());
#endif
            working_data = std::unique_ptr<void, free_delete>(allocate_aligned_memory(64, gemm->get_working_size()));
            gemm->set_working_space(working_data.get());
        }

        auto winsize = gemm->get_window_size().total_size();

        if (winsize < nthreads) {
            gemm->set_nthreads(winsize);
            nthreads = winsize;
        }

#ifndef NO_MULTI_THREADING
        if (p->dynamic_scheduling && gemm->supports_dynamic_scheduling()) {
//            printf("Setting up dynamic scheduling (granule=%u)\n", p->dynamic_granule_count);
#ifdef TASKMASTER
            sched = std::unique_ptr<scheduler> (new taskmaster_scheduler(gemm.get(), nthreads, p->dynamic_granule_count));
#else
            sched = std::unique_ptr<scheduler> (new dynamic_scheduler(gemm.get(), nthreads, p->dynamic_granule_count));
#endif
        } else {
#else
        if (1) {
#endif
//            printf("Setting up static scheduling.\n");
            sched = std::unique_ptr<scheduler> (new static_scheduler(gemm.get(), nthreads, p->schedule_shape_override));
        }
    }

    __attribute__ ((noinline)) void Run(unsigned int threadid) {
        if (gemm->B_is_pretransposed() && p->time_weight_transform) {
            const size_t wsize = gemm->get_B_pretranspose_window_size();
            const size_t start = (threadid * wsize) / nthreads;
            const size_t end = ((threadid + 1) * wsize) / nthreads;

            if (start < end) {
                gemm->pretranspose_B_array_part(pretransposed_data.get(), B_pt->data, B_pt->stride, B_pt->multi_stride, B_pt_is_transposed, start, end);
            }
            barrier.arrive_and_wait();
        }
        sched->execute(threadid);
    }

    static std::vector<kai::ops::KernelDescription> get_kernels(kai::ops::GemmArgs args) {
        using OutputStage = typename std::conditional<quantized == QuantizationType::NONE, kai::ops::Nothing, typename std::conditional<quantized == QuantizationType::INTEGER, kai::ops::Requantize32, kai::ops::DequantizeFloat>::type>::type;
        return kai::ops::get_compatible_kernels<Tlop, Trop, Tret, OutputStage>(args);
    }

    /* Dump valid kernel name list. */
    static void print_kernels (GemmProblem *p, unsigned int nthreads) {
        const unsigned int output_hw = p->output_height * p->output_width;

        // K parameter for the GEMM - input channels is correct for non-convolution cases, indirect and convolution
        // GEMMs.  For im2row we update this later.
        unsigned int gemm_k = p->input_channels;
        unsigned int gemm_sections = 1;
        bool indirect_flag = false;

        const CPUInfo *ci = get_CPUInfo();

        kai::ops::GemmConfig cfg;

        cfg.filter           = p->kernel_filter;
        cfg.inner_block_size = p->inner_block_size;
        cfg.outer_block_size = p->outer_block_size;
        cfg.weight_format    = p->weight_format;

        if (p->groups != 1) {
            printf("kai_ops_wrapper: Groups not supported, aborting.\n");
            exit(1);
        }

        // Accumulation mode is not supported for integer requantizing GEMMS.
        if (p->accumulate && quantized==QuantizationType::INTEGER) {
            printf("kai_ops_wrapper: Accumulation not supported, aborting.\n");
            exit(1);
        }

        if (!p->is_basic_gemm()) {
            if (p->strategy == ConvStrategy::indirect || p->strategy == ConvStrategy::convolution) {
                indirect_flag = true;
                gemm_sections = p->kernel_height * p->kernel_width;
            } else {
                // For im2row GEMMs, we pass in the "full" K value.
                gemm_k = p->kernel_height * p->kernel_width * p->input_channels;
            }
        }

        auto args = kai::ops::GemmArgs(ci, output_hw, p->output_channels, gemm_k, gemm_sections, p->batches, p->multis, indirect_flag, p->act, nthreads, p->fixed_format, p->fast_mode, p->accumulate, &cfg);

        auto kernel_list = get_kernels(args);

        if (kernel_list.size() > 0) {
            printf("Available kernels:\n");
            for (auto &&i : kernel_list) {
                // Only print kernels which match the filter (if one has been provided)
                if (cfg.filter != "" && !strstr(i.name.c_str(), cfg.filter.c_str())) {
                    continue;
                }
                if (i.cycle_estimate == UINT64_MAX) {
                    printf("\t%s%-35s (not preferred)\n", i.is_default?"*":" ", i.name.c_str());
                } else if (i.cycle_estimate == 0) {
                    printf("\t%s%-35s (preferred)\n", i.is_default?"*":" ", i.name.c_str());
                } else {
                    printf("\t%s%-35s (%llu estimated cycles)\n", i.is_default?"*":" ", i.name.c_str(), (unsigned long long)i.cycle_estimate);
                }
            }
        } else {
            printf("No kernels available for selected parameters.\n");
        }
    }
};

template<typename Tlop, typename Trop, typename Tret>
using kai_ops_quantized = kai_ops_wrapper<Tlop, Trop, Tret, QuantizationType::INTEGER>;

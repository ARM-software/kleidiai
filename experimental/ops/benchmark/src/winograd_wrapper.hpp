//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cinttypes>
#include <cstring>

#include "barrier.hpp"
#include "gemm_lib.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "scheduler.hpp"
#include "utils.hpp"  // iceildiv

#undef TASKMASTER

#include "kai/ops/conv/winograd.hpp"

// (Templated) glue to make it work with current infrastructure
template <typename Top, typename Tret>
class winograd_wrapper {
private:
    struct free_delete {
        void operator()(void *x) { free_aligned_memory(x); }
    };

    const GemmProblem * const p;

    const unsigned int nthreads;

    std::unique_ptr<void, free_delete> winograd_input={};
    std::unique_ptr<void, free_delete> winograd_output={};
    std::unique_ptr<void, free_delete> winograd_weights={};
    std::unique_ptr<void, free_delete> gemm_pretransposed_data={};

    std::unique_ptr<void, free_delete> working_data={};

    std::unique_ptr<scheduler> sched;
    std::unique_ptr<kai::ops::barrier> barr;

    std::shared_ptr<Matrix<Top>> A;
    std::shared_ptr<Matrix<Tret>> C;
    std::shared_ptr<Matrix<Tret>> bias;

    kai::ops::ConvolutionArgs conv_args;
    kai::ops::winograd::WinogradImpl winograd_impl;
    kai::ops::UniqueGemmCommon<Top, Top, Top> gemm;

public:
    static int get_m_block() { return 1; }
    static int get_n_block() { return 1; }
    static int get_k_block() { return 1; }

    typedef Top lhs_operand_type;
    typedef Top rhs_operand_type;
    typedef Tret result_type;

    winograd_wrapper(winograd_wrapper &) = delete;
    winograd_wrapper(winograd_wrapper &&) = default;
    winograd_wrapper & operator=(winograd_wrapper &) = delete;

    // Constructor: non-quantized
    winograd_wrapper(
              std::shared_ptr<Matrix<Top>> &A,
              std::shared_ptr<Matrix<Top>> &B,
              std::shared_ptr<Matrix<Tret>> &C,
              std::shared_ptr<Matrix<Tret>> &bias,
              GemmProblem *p, unsigned int nthreads, int do_init
    ) : p(p), nthreads(nthreads), barr(new kai::ops::barrier(nthreads)),
        A(A), C(C), bias(bias),
        conv_args(
          p->batches,
          { static_cast<unsigned int>(p->input_height),
            static_cast<unsigned int>(p->input_width) },
          p->input_channels,
          p->padding_top, p->padding_left,
          { static_cast<unsigned int>(p->output_height),
            static_cast<unsigned int>(p->output_width) },
          p->output_channels,
          { static_cast<unsigned int>(p->kernel_height),
            static_cast<unsigned int>(p->kernel_width) },
          p->act
        )
    {
        const CPUInfo *ci = get_CPUInfo();

        if (p->groups != 1) {
            printf("Winograd: Groups not supported, aborting.\n");
            // TODO
            exit(1);
        }

        // Get a winograd layer appropriate for the kernel size.
        if (p->out_stride_h != 1 || p->out_stride_w != 1) {
            printf("Winograd: strided convolution not supported\n");
            exit(1);
        }
        if (p->in_stride_h != 1 || p->in_stride_w != 1) {
            printf("Winograd: dilated convolution not supported\n");
            exit(1);
        }

        // Get configuration arguments for Winograd
        kai::ops::winograd::WinogradConfig winograd_cfg;
        winograd_cfg.output_rows = p->winograd_args.output_tile_rows;
        winograd_cfg.output_cols = p->winograd_args.output_tile_cols;
        winograd_cfg.input_transform_filter = p->winograd_args.input_transform_filter;
        winograd_cfg.weight_transform_filter = p->winograd_args.weight_transform_filter;
        winograd_cfg.output_transform_filter = p->winograd_args.output_transform_filter;

        // Get arguments for the GEMM
        kai::ops::GemmConfig cfg;
        cfg.filter = p->kernel_filter;
        cfg.inner_block_size = p->inner_block_size;
        cfg.outer_block_size = p->outer_block_size;

        const bool success = kai::ops::winograd::get_implementation<Top>(
          winograd_impl, ci, conv_args, nthreads, p->fast_mode, &winograd_cfg, &cfg
        );

        if (!success)
        {
          printf("Winograd: Unsupported kernel size: %" PRId64 "x%" PRId64 ".\n",p->kernel_height, p->kernel_width);
          exit(1);
        }
        else
        {
          printf("Using input transform: %s\n", winograd_impl.input_transform->get_name().c_str());
          printf("Using weight transform: %s\n", winograd_impl.weight_transform->get_name().c_str());
          printf("Using output transform: %s\n", winograd_impl.output_transform->get_name().c_str());
        }

        // Construct the GEMM
        gemm = kai::ops::gemm<Top, Top, Top>(*winograd_impl.gemm_args);

#ifndef SILENT
        auto my_cfg = gemm->get_config();

        printf("GEMM configuration: kernel=%s inner_block=%u outer_block=%u\n", my_cfg.filter.c_str(), my_cfg.inner_block_size, my_cfg.outer_block_size);
#endif
        // Determine how much working space is required, allocate it.
        size_t working_space_size = std::max({
          winograd_impl.input_transform->get_working_space_size(conv_args, nthreads),
          winograd_impl.output_transform->get_working_space_size(conv_args, nthreads),
          gemm->get_working_size()
        });
        working_data = std::unique_ptr<void, free_delete>(allocate_aligned_memory(64, working_space_size));
        gemm->set_working_space(working_data.get());

        // Allocate memory for the Winograd-domain representation of the problem
        const auto &wds = winograd_impl.winograd_spec;
        winograd_input = std::unique_ptr<void, free_delete>(allocate_aligned_memory(64, wds.input_matrix_size_bytes));
        winograd_weights = std::unique_ptr<void, free_delete>(allocate_aligned_memory(64, wds.weight_matrix_size_bytes));
        winograd_output = std::unique_ptr<void, free_delete>(allocate_aligned_memory(64, wds.output_matrix_size_bytes));

        // Prepare the GEMM
        gemm->set_arrays_generic(
          winograd_input.get(), wds.input_ld_row, wds.input_ld_batch, wds.input_ld_matrix,
          winograd_weights.get(), wds.weight_ld_row, wds.weight_ld_matrix,
          winograd_output.get(), wds.output_ld_row, wds.output_ld_batch, wds.output_ld_matrix,
          nullptr, 0  // No bias in the GEMM
        );

        // Prepare weights
        winograd_impl.weight_transform->execute(
          conv_args,
          B->data, p->kernel_width * p->input_channels * B->stride, p->input_channels * B->stride, B->stride,
          winograd_weights.get(), winograd_impl.winograd_spec,
          0, 1  // Thread 1 of 1
        );
        if (gemm->B_is_pretransposed())
        {
          // Pretranspose the weights if required to by the GEMM
          gemm_pretransposed_data = std::unique_ptr<void, free_delete>(allocate_aligned_memory(64, gemm->get_B_pretransposed_array_size()));
          gemm->pretranspose_B_array_generic(
             gemm_pretransposed_data.get(),
             winograd_weights.get(),
             winograd_impl.winograd_spec.weight_ld_row,
             winograd_impl.winograd_spec.weight_ld_matrix,
             false
          );
          gemm->set_pretransposed_B_data(gemm_pretransposed_data.get());
        }

        sched = std::unique_ptr<scheduler> (new static_scheduler(gemm.get(), nthreads, p->schedule_shape_override));
    }

    __attribute__ ((noinline)) void Run(unsigned int threadid) {
        {
            // Input transform
            winograd_impl.input_transform->execute(
              conv_args,
              A->data, A->batch_stride, p->input_width * A->stride, A->stride,
              winograd_input.get(), winograd_impl.winograd_spec,
              working_data.get(), threadid, nthreads
            );
        }
        barr->arrive_and_wait();

        // GEMM
        {
            sched->execute(threadid);
            barr->arrive_and_wait();
        }

        // Output transform
        {
            winograd_impl.output_transform->execute(
              conv_args,
              winograd_output.get(), winograd_impl.winograd_spec,
              (bias == nullptr) ? nullptr : bias->data,
              C->data, C->batch_stride, p->output_width * C->stride, C->stride,
              working_data.get(), threadid, nthreads
            );
            barr->arrive_and_wait();
        }
    }

    /* Dump valid kernel name list. */
    static void print_kernels (GemmProblem *p, unsigned int nthreads) {
        printf("Kernel list not supported yet.\n");
    }
};

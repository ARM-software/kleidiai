//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cinttypes>
#include <cstring>
#include <type_traits>

#include "kai/ops/gemm/kai_ops.hpp"

#include "scheduler.hpp"
#include "utils.hpp" // iceildiv
#include "gemm_lib.hpp"

#undef TASKMASTER

#include "kai/ops/conv/depthwise.hpp"

// (Templated) glue to make it work with current infrastructure
template <typename Ta, typename Tb, typename Tret, QuantizationType quantized=QuantizationType::NONE>
class depthwise_wrapper {
  struct free_delete {
    void operator()(void *x) { free_aligned_memory(x); }
  };

  // Quantized support
  std::shared_ptr<Matrix<int32_t>> bias;
  Requantize32 qp;

  const GemmProblem *p;
  const unsigned int n_threads;

  std::shared_ptr<Matrix<Ta>> input;
  std::shared_ptr<Matrix<Tret>> output;

  kai::ops::depthwise::UniqueDepthwiseCommon<Ta, Tb, Tret> depthwise_operator = {};

  std::unique_ptr<void, free_delete> working_data = {};
  std::unique_ptr<void, free_delete> packed_params = {};

  public:
  static int get_m_block(void) { return 1; }
  static int get_n_block(void) { return 1; }
  static int get_k_block(void) { return 1; }

  typedef Ta lhs_operand_type;
  typedef Tb rhs_operand_type;
  typedef Tret result_type;

  depthwise_wrapper(depthwise_wrapper &) = delete;
  depthwise_wrapper(depthwise_wrapper &&) = default;
  depthwise_wrapper& operator=(depthwise_wrapper &) = delete;

  // Constructor (non-quantized)
  depthwise_wrapper(
    std::shared_ptr<Matrix<Ta>> &input,
    std::shared_ptr<Matrix<Tb>> &weights,
    std::shared_ptr<Matrix<Tret>> &output,
    std::shared_ptr<Matrix<Tret>> &bias,
    GemmProblem *p,
    unsigned int n_threads,
    int do_init
  ) : p(p), n_threads(n_threads), input(input), output(output) {
    const CPUInfo *ci = get_CPUInfo();

    // Check for a valid problem
    if (p->input_channels != p->groups)
    {
      printf("Depthwise: Input channels (%" PRId64 ") != number of groups (%" PRId64 ")\n", p->input_channels, p->groups);
      exit(1);
    }

    if (p->output_channels % p->groups)
    {
      printf("Depthwise: Output channels not a multiple of the number of groups.\n");
      exit(1);
    }

    if (p->multis != 1)
    {
      printf("Depthwise: Multis not supported.\n");
      exit(1);
    }

    if (p->accumulate)
    {
      printf("Depthwise: Accumulation not supported.\n");
      exit(1);
    }

    if (p->kernel_height==1 && p->kernel_width==1) {
      printf("Depthwise: 1x1 doesn't look like a real depthwise problem.\n");
      exit(1);
    }

    // Construct the depthwise operator.
    const int pad_rows = std::max<int>(
      0, (p->output_height - 1) * p->out_stride_h + p->kernel_height - p->input_height
    );
    const int pad_cols = std::max<int>(
      0, (p->output_width - 1) * p->out_stride_w + p->kernel_width - p->input_width
    );
    int padding_bottom = pad_rows - p->padding_top;
    int padding_right = pad_cols - p->padding_left;

    kai::ops::depthwise::DepthwiseConfig cfg = {};
    cfg.filter = p->kernel_filter;

    kai::ops::depthwise::DepthwiseArgs args(
      ci,
      p->kernel_height, p->kernel_width,
      p->out_stride_h, p->out_stride_w,
      p->in_stride_h, p->in_stride_w,
      p->batches, p->input_height, p->input_width,
      p->input_channels,
      p->output_height, p->output_width,
      p->output_channels / p->groups,
      { static_cast<unsigned int>(p->padding_left),
        static_cast<unsigned int>(p->padding_top),
        static_cast<unsigned int>(padding_right),
        static_cast<unsigned int>(padding_bottom) },
      p->act,
      &cfg
    );

    args.fast_mode = p->fast_mode;

    depthwise_operator = kai::ops::depthwise::depthwise<Ta, Tb, Tret>(args);
    if (depthwise_operator == nullptr)
    {
      printf("Depthwise: no valid kernel implementation found.\n");
      exit(1);
    }

#ifndef SILENT
    printf("Using kernel: %s\n", depthwise_operator->name().c_str());
#endif

    // Initialise working space, and space for packed weights
    working_data = std::unique_ptr<void, free_delete>(
      allocate_aligned_memory(64, depthwise_operator->get_working_size(n_threads))
    );
    packed_params = std::unique_ptr<void, free_delete>(
      allocate_aligned_memory(64, depthwise_operator->get_storage_size())
    );

    // Pack parameters
    depthwise_operator->pack_parameters(
      packed_params.get(),
      (bias.get() == nullptr) ? nullptr : bias->data,
      weights->data
    );
  }

  // Constructor: quantized
  template<typename Tfloat>
  depthwise_wrapper(
    std::shared_ptr<Matrix<Ta>> &input,
    std::shared_ptr<Matrix<Tb>> &weights,
    std::shared_ptr<Matrix<Tret>> &output,
    std::shared_ptr<Matrix<int32_t>> &bias,
    QuantizeParameters<Ta, Tfloat> &A_qp,
    std::vector<QuantizeParameters<Tb, Tfloat> > &B_qp,
    QuantizeParameters<Ta, Tfloat> &C_qp,
    GemmProblem *p,
    unsigned int n_threads,
    int do_init
  ) : p(p), n_threads(n_threads), input(input), output(output)
  {
    assert(quantized==QuantizationType::INTEGER);
    const CPUInfo *ci = get_CPUInfo();

    // Check for a valid problem
    if (p->input_channels != p->groups)
    {
      printf("Depthwise: Input channels != number of groups\n");
      exit(1);
    }

    if (p->output_channels % p->groups)
    {
      printf("Depthwise: Output channels not a multiple of the number of groups.\n");
      exit(1);
    }

    if (p->multis != 1)
    {
      printf("Depthwise: Multis not supported.\n");
      exit(1);
    }

    if (p->accumulate)
    {
      printf("Depthwise: Accumulation not supported.\n");
      exit(1);
    }

    // Construct the depthwise operator.
    const int pad_rows = std::max<int>(
      0, (p->output_height - 1) * p->out_stride_h + p->kernel_height - p->input_height
    );
    const int pad_cols = std::max<int>(
      0, (p->output_width - 1) * p->out_stride_w + p->kernel_width - p->input_width
    );
    int padding_bottom = pad_rows - p->padding_top;
    int padding_right = pad_cols - p->padding_left;

    kai::ops::depthwise::DepthwiseConfig cfg = {};
    cfg.filter = p->kernel_filter;

    kai::ops::depthwise::DepthwiseArgs args(
      ci,
      p->kernel_height, p->kernel_width,
      p->out_stride_h, p->out_stride_w,
      p->in_stride_h, p->in_stride_w,
      p->batches, p->input_height, p->input_width,
      p->input_channels,
      p->output_height, p->output_width,
      p->output_channels / p->groups,
      { static_cast<unsigned int>(p->padding_left),
        static_cast<unsigned int>(p->padding_top),
        static_cast<unsigned int>(padding_right),
        static_cast<unsigned int>(padding_bottom) },
      p->act,
      &cfg
    );

    qp.bias = (bias ? bias->data : nullptr);
    qp.bias_multi_stride = bias ? bias->multi_stride : 0;
    qp.a_offset = A_qp.m_zeropt;
    qp.b_offset = B_qp[0].m_zeropt;
    qp.c_offset = C_qp.m_zeropt;
    qp.minval = C_qp.m_minval;
    qp.maxval = C_qp.m_maxval;
    qp.set_multipliers(A_qp.m_scale / C_qp.m_scale, B_qp);

    depthwise_operator = kai::ops::depthwise::depthwise<Ta, Tb, Tret, kai::ops::Requantize32>(args, qp);
    if (depthwise_operator == nullptr)
    {
      printf("Depthwise: no valid kernel implementation found.\n");
      exit(1);
    }

#ifndef SILENT
    const auto kernel_list = kai::ops::depthwise::get_compatible_kernels<Ta, Tb, Tret, kai::ops::Requantize32>(args, qp);
    if (kernel_list.size() > 0) {
        for (auto &&i : kernel_list) {
            // Print the selected kernel
            if (i.is_default) {
              printf("Using kernel: %s\n", i.name.c_str());
              break;
            }
        }
    }
#endif

    // Initialise working space, and space for packed weights
    working_data = std::unique_ptr<void, free_delete>(
      allocate_aligned_memory(64, depthwise_operator->get_working_size(n_threads))
    );
    packed_params = std::unique_ptr<void, free_delete>(
      allocate_aligned_memory(64, depthwise_operator->get_storage_size())
    );

    // Pack parameters
    depthwise_operator->pack_parameters(
      packed_params.get(),
      (bias.get() == nullptr) ? nullptr : bias->data,
      weights->data
    );
  }

  __attribute__ ((noinline)) void Run(const unsigned int thread_id)
  {
    depthwise_operator->execute(
      input->data, input->stride, input->stride * p->input_width, input->batch_stride,
      packed_params.get(),
      output->data, output->stride, output->stride * p->output_width, output->batch_stride,
      working_data.get(),
      thread_id, n_threads
    );
  }

  static void print_kernels(const GemmProblem *const p, const unsigned int n_threads)
  {
    const CPUInfo *ci = get_CPUInfo();

    // Check for a valid problem
    if (p->input_channels != p->groups)
    {
      printf("Depthwise: Input channels != number of groups\n");
      exit(1);
    }

    if (p->output_channels % p->groups)
    {
      printf("Depthwise: Output channels not a multiple of the number of groups.\n");
      exit(1);
    }

    if (p->multis != 1)
    {
      printf("Depthwise: Multis not supported.\n");
      exit(1);
    }

    if (p->accumulate)
    {
      printf("Depthwise: Accumulation not supported.\n");
      exit(1);
    }

    // Construct the depthwise operator.
    const int pad_rows = std::max<int>(
      0, (p->output_height - 1) * p->out_stride_h + p->kernel_height - p->input_height
    );
    const int pad_cols = std::max<int>(
      0, (p->output_width - 1) * p->out_stride_w + p->kernel_width - p->input_width
    );
    int padding_bottom = pad_rows - p->padding_top;
    int padding_right = pad_cols - p->padding_left;

    kai::ops::depthwise::DepthwiseConfig cfg = {};
    cfg.filter = p->kernel_filter;

    kai::ops::depthwise::DepthwiseArgs args(
      ci,
      p->kernel_height, p->kernel_width,
      p->out_stride_h, p->out_stride_w,
      p->in_stride_h, p->in_stride_w,
      p->batches, p->input_height, p->input_width,
      p->input_channels,
      p->output_height, p->output_width,
      p->output_channels / p->groups,
      { static_cast<unsigned int>(p->padding_left),
        static_cast<unsigned int>(p->padding_top),
        static_cast<unsigned int>(padding_right),
        static_cast<unsigned int>(padding_bottom) },
      p->act,
      &cfg
    );

    args.fast_mode = p->fast_mode;

    using OutputStage = typename std::conditional<quantized == QuantizationType::NONE, kai::ops::Nothing, kai::ops::Requantize32>::type;
    const auto kerns = kai::ops::depthwise::get_compatible_kernels<Ta, Tb, Tret, OutputStage>(args);

    if (kerns.empty())
    {
      printf("No kernels available for selected parameters.\n");
    }
    else
    {
      printf("Available kernels:\n");
      for (auto &&impl : kerns)
      {
        if (cfg.filter != "" && !std::strstr(impl.name.c_str(), cfg.filter.c_str()))
        {
          // Don't print filtered kernels
          continue;
        }

        if (impl.cycle_estimate == UINT32_MAX)
        {
          printf("\t%s%-35s (not preferred)\n", impl.is_default ? "*" : " ", impl.name.c_str());
        }
        else if (impl.cycle_estimate == 0)
        {
          printf("\t%s%-35s (preferred)\n", impl.is_default ? "*" : " ", impl.name.c_str());
        }
        else
        {
          printf("\t%s%-35s (%llu estimated cycles)\n", impl.is_default ? "*" : " ", impl.name.c_str(), (unsigned long long) impl.cycle_estimate);
        }
      }
    }
  }
};

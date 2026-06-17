//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>

#include "gemm_lib.hpp" // for transpose_matrix()
#include "kai/ops/bfloat.hpp"

/*
 * Type to use for accumulation - for some type combinations, we need to
 * accumulate in something higher precision and downconvert at the end.
 *
 * This mechanism supports this by defining an accumulation type based on
 * the output type.  At present, we just use the output type except for
 * BF16, where we accumulate in float.
 */
namespace {

template <typename T>
struct accumulator_type {
    typedef T type;
};

template<>
struct accumulator_type<bfloat16> {
    typedef float type;
};

template<>
struct accumulator_type<__fp16> {
    typedef float type;
};

} // anonymous namespace

template <typename To, typename Tr>
struct gemm_transposeB {
private:
    const GemmProblem * const p;

    std::shared_ptr<Matrix <To> > A_trans={};
    std::shared_ptr<Matrix <To> > B_trans={};
    std::shared_ptr<Matrix <Tr> > C;
    std::shared_ptr<Matrix <Tr> > bias;

public:
    static int get_m_block() { return 1; }
    static int get_n_block() { return 1; }
    static int get_k_block() { return 1; }

    typedef To operand_type;
    typedef Tr result_type;

    gemm_transposeB(std::shared_ptr<Matrix <To> > &A,
             std::shared_ptr<Matrix <To> > &B,
             std::shared_ptr<Matrix <Tr> > &C,
             std::shared_ptr<Matrix <Tr> > &bias,
             const GemmProblem * const p, bool do_init) : p(p), C(C), bias(bias) {
        // Create transposed weight matrix - the "width" of this is the squashed combination of 'HWI' (with I
        // divided by the number of groups), the height is 'O' (output channels)
        auto gemm_K = (p->input_channels / p->groups) * p->kernel_height * p->kernel_width;

        B_trans = std::make_shared<Matrix <To> >(p->output_channels, gemm_K, gemm_K, p->batches, p->multis);

        if (do_init) {
            B_trans->Transpose(B);
        }

        A_trans = A;
    }

    int64_t Run() {
	int lda = A_trans->stride;
	int ldb = B_trans->stride;
	int ldc = C->stride;

	// Note that this function now computes a noddy 9-loop grouped 2D convolution operation rather than a GEMM.
	//
	// For pure MM cases, the parameters will be set such that the two are equivalent.

	const int64_t group_inch = (p->input_channels / p->groups);
	const int64_t group_outch = (p->output_channels / p->groups);

        typename accumulator_type<Tr>::type max_pos=0;
        typename accumulator_type<Tr>::type max_neg=0;

	for (int64_t m=0; m<p->multis; m++) {
	    for (int64_t b=0; b<p->batches; b++) {
	        const To *a_base = (A_trans->data + (m * A_trans->multi_stride) + (b * A_trans->batch_stride));
	        const To *b_base = (B_trans->data + (m * B_trans->multi_stride));
	        Tr *c_base = (C->data + (m * C->multi_stride) + (b * C->batch_stride));

	        for (int64_t out_y=0; out_y<p->output_height; out_y++) {
	            for (int64_t out_x=0; out_x<p->output_width; out_x++) {
	                // GEMM 'm' equivalent - squash output height/width into a single dimension
	                int64_t m_val = (out_y * p->output_width) + out_x;

	                for (int64_t group=0; group<p->groups; group++) {
                            for (int64_t out_ch=(group * group_outch); out_ch<((group + 1) * group_outch); out_ch++) {
                                typename accumulator_type<Tr>::type v=0, pos=0, neg=0;
                                Tr &out_ref = c_base[(m_val * ldc) + out_ch];

                                if (p->add_problem) {
                                    // Adds do a pointwise add of A to C, then multiply by the value at the top of
                                    // the corresponding column of B (which turns into a row as this is a transposed
                                    // implementation).  This multiply is the first half of a fused batchnorm (the
                                    // second half is handled by the bias add below).
                                    int64_t input_y = (out_y * p->out_stride_h);
                                    int64_t input_x = (out_x * p->out_stride_w);

                                    // Squash input x/y into single 'row' value (for (MN)HWC format; M,N taken care of by outer loops)
                                    int64_t in_m_val = (input_y * p->input_width) + input_x;

                                    v = out_ref;
                                    v += a_base[(in_m_val * lda) + out_ch];
                                    v *= b_base[(out_ch * ldb)];
                                } else {
                                    if (p->accumulate) {
                                        v = out_ref;
                                    }

                                    for (int64_t kern_y=0; kern_y<p->kernel_height; kern_y++) {
                                        for (int64_t kern_x=0; kern_x<p->kernel_width; kern_x++) {
                                            int64_t input_y = (out_y * p->out_stride_h) + kern_y * p->in_stride_h - p->padding_top;
                                            int64_t input_x = (out_x * p->out_stride_w) + kern_x * p->in_stride_w - p->padding_left;

                                            // Check for out of bounds cases.
                                            if (input_y < 0 || input_y >= p->input_height || input_x < 0 || input_x >= p->input_width) {
                                                // TODO: may need to perform some padding multiplies in the non-zero padding case.
                                                // For now, just skip this kernel point entirely.
                                                continue;
                                            }

                                            // Squash input x/y into single 'row' value (for (MN)HWC format; M,N taken care of by outer loops)
                                            int64_t in_m_val = (input_y * p->input_width) + input_x;

                                            for (int64_t in_ch=(group * group_inch); in_ch<((group + 1) * group_inch); in_ch++) {
                                                // GEMM 'k' equivalent - squash kernel position and input channel into a single dimension
                                                // Use this to access weights/B.
                                                int64_t k_val = ((kern_y * p->kernel_width) + kern_x) * group_inch + (in_ch - (group * group_inch));

                                                typename accumulator_type<Tr>::type val = a_base[(in_m_val * lda) + in_ch] * b_base[(out_ch * ldb) + k_val];

                                                if (val > 0) {
                                                    pos += val;
                                                } else {
                                                    neg += val;
                                                }

                                                v += val;
                                            }
                                        }
                                    }
                                }

                                if (bias) {
                                    typename accumulator_type<Tr>::type bd = bias->data[(m * bias->multi_stride) + out_ch];
                                    v += bd;

                                    if (bd > 0) {
                                        pos += bd;
                                    } else {
                                        neg += bd;
                                    }
                                }

                                if (pos > max_pos) {
                                    max_pos = pos;
                                }
                                if (neg < max_neg) {
                                    max_neg = neg;
                                }


                                v = activate(p->act, v);

                                out_ref = v;
                            }
                        }
                    }
                }
            }
        }

#ifndef SILENT
        printf("Reference computed: maximum positive contribution %f, negative %f\n", (float)max_pos, (float)max_neg);
#endif

        return std::max<int64_t>(max_pos, -max_neg);
    }
};

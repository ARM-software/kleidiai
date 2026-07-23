//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Routine to dump a problem to a file, and read it back.

#include <cinttypes>
#include <cstdint>
#include <cstdio>

#include "kai/ops/gemm/kai_ops.hpp"
#include "gemm_lib.hpp"

#define MAGIC 0x123456789abc0001

struct problem_dump {
    int64_t magic;
    int64_t input_height;
    int64_t input_width;
    int64_t input_channels;
    int64_t kernel_height;
    int64_t kernel_width;
    int64_t output_height;
    int64_t output_width;
    int64_t output_channels;
    int64_t padding_top;
    int64_t padding_left;
    int64_t in_stride_h;
    int64_t in_stride_w;
    int64_t out_stride_h;
    int64_t out_stride_w;
    int64_t groups;
    int64_t batches;
    int64_t multis;
    Activation act;
    bool use_bias;
};

FILE *write_dump(const GemmProblem *p, const char *filename) {
    FILE *fp;

    if ((fp = fopen(filename, "wb")) == nullptr) {
        printf("Error: can't open dump file %s.\n", filename);
        exit(1);
    }

    problem_dump pd;

    pd.magic = MAGIC;
    pd.input_height = p->input_height;
    pd.input_width = p->input_width;
    pd.input_channels = p->input_channels;
    pd.kernel_height = p->kernel_height;
    pd.kernel_width = p->kernel_width;
    pd.output_height = p->output_height;
    pd.output_width = p->output_width;
    pd.output_channels = p->output_channels;
    pd.padding_top = p->padding_top;
    pd.padding_left = p->padding_left;
    pd.in_stride_h = p->in_stride_h;
    pd.in_stride_w = p->in_stride_w;
    pd.out_stride_h = p->out_stride_h;
    pd.out_stride_w = p->out_stride_w;
    pd.groups = p->groups;
    pd.batches = p->batches;
    pd.multis = p->multis;
    pd.act = p->act;
    pd.use_bias = p->use_bias;

    fwrite(&pd, sizeof(problem_dump), 1, fp);

    return fp;
}

FILE *read_dump(GemmProblem *p, const char *filename) {
    FILE *fp;

    if ((fp = fopen(filename, "rb")) == nullptr) {
        printf("Error: can't open dump file %s.\n", filename);
        exit(1);
    }

    problem_dump pd;

    fread(&pd, sizeof(problem_dump), 1, fp);

    if (pd.magic != MAGIC) {
        printf("Bad dump magic: %" PRIx64 " (not %" PRIx64 ").\n", static_cast<uint64_t>(pd.magic), static_cast<uint64_t>(MAGIC));
        printf("This probably means the dump was made with an incompatible version of gemm-linux.\n");
        exit(1);
    }

    p->input_height = pd.input_height;
    p->input_width = pd.input_width;
    p->input_channels = pd.input_channels;
    p->kernel_height = pd.kernel_height;
    p->kernel_width = pd.kernel_width;
    p->output_height = pd.output_height;
    p->output_width = pd.output_width;
    p->output_channels = pd.output_channels;
    p->padding_top = pd.padding_top;
    p->padding_left = pd.padding_left;
    p->in_stride_h = pd.in_stride_h;
    p->in_stride_w = pd.in_stride_w;
    p->out_stride_h = pd.out_stride_h;
    p->out_stride_w = pd.out_stride_w;
    p->groups = pd.groups;
    p->batches = pd.batches;
    p->multis = pd.multis;
    p->act = pd.act;
    p->use_bias = pd.use_bias;

#ifndef SILENT
    printf ("===============================================================================\n");
    printf ("Configuration loaded from dump:\n");
    printf (" Convolution: filter %" PRId64 "x%" PRId64 ", dilation %" PRId64 "x%" PRId64 ", stride %" PRId64 "x%" PRId64 ", output %" PRId64 "x%" PRId64 " (%" PRId64 " -> %" PRId64 " channels)\n",
            p->kernel_height, p->kernel_width, p->in_stride_h, p->in_stride_w, p->out_stride_h, p->out_stride_w,
            p->output_height, p->output_width, p->input_channels, p->output_channels);
    printf (" Input: %" PRId64 "x%" PRId64 ", padding %" PRId64 "x%" PRId64 ", groups %" PRId64 ", batches %" PRId64 ", multis %" PRId64 ", bias %d\n",
            p->input_height, p->input_width, p->padding_top, p->padding_left, p->groups, p->batches, p->multis, p->use_bias);
    printf ("===============================================================================\n");
#endif

    return fp;
}

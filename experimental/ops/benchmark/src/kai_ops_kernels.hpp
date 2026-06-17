//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "depthwise_wrapper.hpp"
#include "kai/ops/bfloat.hpp"
#include "kai_ops_wrapper.hpp"
#include "winograd_wrapper.hpp"

// FP types.
typedef kai_ops_wrapper<float, float, float> sgemm_new;
typedef kai_ops_wrapper<__fp16, __fp16, __fp16> hgemm_new;
typedef kai_ops_wrapper<bfloat16, bfloat16, float> gemm_bf16_new;
typedef kai_ops_wrapper<bfloat16, bfloat16, bfloat16> gemm_bf16bf16;
typedef kai_ops_wrapper<__fp16, __fp16, float> gemm_fp16fp32;

typedef winograd_wrapper<float, float> winograd_fp32;

#ifdef __aarch64__
typedef winograd_wrapper<__fp16, __fp16> winograd_fp16;

typedef depthwise_wrapper<float, float, float> depthwise_fp32;
typedef depthwise_wrapper<__fp16, __fp16, __fp16> depthwise_fp16;

// FP64 is AArch64 only
typedef kai_ops_wrapper<double, double, double> dgemm_new;

// int types (AArch64 only).
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
#endif

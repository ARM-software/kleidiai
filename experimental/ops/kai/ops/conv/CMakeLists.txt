#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------
message(STATUS "Entered conv directory")
list(APPEND CMAKE_MESSAGE_INDENT " ")

# -------------------------------------------------------------------------------------------------

set(KLEIDIAI_OPS_CONV_WINOGRAD_CORE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/a64_fp32_6x6.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/arm_fp32_1x8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/arm_fp32_4x4.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/arm_fp32_6x6.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms_fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_1x2_1x7.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_1x4_1x5.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_1x6_1x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_2x2_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_2x2_5x5.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_4x4_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms_fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/arm_fp32_2x2_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/arm_fp32_2x2_5x5.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/arm_fp32_4x4_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/cpp_fp32_1x2_1x7.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/cpp_fp32_1x4_1x5.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/cpp_fp32_1x6_1x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms_fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/winograd_fp32.cpp)

set(KLEIDIAI_OPS_CONV_WINOGRAD_SVE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/sme_fp32_mla_6x6.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/sve_fp32_6x6.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/sme_fp32_mopa_4x4_3x3.cpp)

set(KLEIDIAI_OPS_CONV_WINOGRAD_FP16_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/a64_fp16_6x6.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms_fp16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/a64_fp16_4x4_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms_fp16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/a64_fp16_4x4_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms_fp16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/winograd_fp16.cpp)

set(KLEIDIAI_OPS_CONV_DEPTHWISE_CORE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/addressing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/depthwise_bf16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/depthwise_common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/depthwise_fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/depthwise_s8q.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/depthwise_strategies_common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/depthwise_u8q.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/depthwise_u8s8u8q.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/interleaves/a64_s8q_3x3_dot.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/interleaves/a64_u8q_3x3_dot.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/interleaves/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/interleaves/generic_quantized_dot_product.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_nhwc_generic_output9_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp32_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_s8q_nhwc_5x5_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_s8q_nhwc_generic_output9_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_s8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8q_nhwc_5x5_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8q_nhwc_generic_output4_mla_depthfirst_intrinsics/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8q_nhwc_generic_output9_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8qa_nhwc_3x3_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8qa_nhwc_3x3_s2_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8qa_nhwc_5x5_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8s8u8q_nhwc_3x3_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8s8u8q_nhwc_3x3_s2_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8s8u8q_nhwc_5x5_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8s8u8q_nhwc_generic_output9_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_u8s8u8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/premultiply.cpp)

set(KLEIDIAI_OPS_CONV_DEPTHWISE_SVE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/interleaves/sme2_s8q_3x3_dot.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/interleaves/sme2_u8q_3x3_dot.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/interleaves/sme_s8q_3x3_dot.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/interleaves/sme_u8q_3x3_dot.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/interleaves/sve_s8q_3x3_dot.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/interleaves/sve_u8q_3x3_dot.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_bf16_planar_3x3_s1_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_bf16_planar_3x3_s2_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_bf16_planar_5x5_s1_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_bf16_planar_5x5_s2_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_planar_3x3_s1_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_planar_3x3_s2_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_planar_5x5_s1_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp16_planar_5x5_s2_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_planar_3x3_s1_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_planar_3x3_s2_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_planar_5x5_s1_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32_planar_5x5_s2_4rows_mla_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32bf16fp32_planar_3x3_s1_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32bf16fp32_planar_3x3_s2_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32bf16fp32_planar_5x5_s1_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_fp32bf16fp32_planar_5x5_s2_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_s8q_planar_3x3_s1_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_s8q_planar_3x3_s2_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_s8q_planar_5x5_s1_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_s8q_planar_5x5_s2_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_u8q_planar_3x3_s1_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_u8q_planar_3x3_s2_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_u8q_planar_5x5_s1_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_u8q_planar_5x5_s2_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_u8s8u8q_planar_3x3_s1_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_u8s8u8q_planar_3x3_s2_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_u8s8u8q_planar_5x5_s1_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme2_u8s8u8q_planar_5x5_s2_4rows_dot_za/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_nhwc_generic_output9_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_s8q_nhwc_5x5_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_u8q_nhwc_5x5_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_u8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sme_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_nhwc_generic_output9_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_fp32_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_s8q_nhwc_5x5_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_u8q_nhwc_5x5_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_u8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_u8s8u8q_nhwc_3x3_s1_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_u8s8u8q_nhwc_3x3_s2_output2x2_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/sve_u8s8u8q_nhwc_5x5_s1_output2x2_mla_depthfirst/generic.cpp
)

set(KLEIDIAI_OPS_CONV_DEPTHWISE_FP16_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/depthwise_fp16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_direct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst/generic_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_nhwc_generic_output9_mla_depthfirst/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/depthwise/kernels/a64_fp16_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst/generic.cpp
)

# -------------------------------------------------------------------------------------------------

set(KLEIDIAI_OPS_CONV_PRIVATE_INCLUDES
    # Internal sources
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../
    # Public headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)

# -------------------------------------------------------------------------------------------------

# Helper to create an OBJECT lib, apply common compile options/compile definitions/include
# directories, and arch
function(arm_add_obj_lib OBJ_NAME)
    # Parse flags
    set(PREFIX ARG)
    set(OPTIONS)
    set(SINGLE_VALUE_ARGS ARCH)
    set(MULTI_VALUE_ARGS SOURCES)
    cmake_parse_arguments(
        ${PREFIX}
        "${OPTIONS}"
        "${SINGLE_VALUE_ARGS}"
        "${MULTI_VALUE_ARGS}"
        ${ARGN})

    # Sanity check
    if(NOT ${PREFIX}_SOURCES)
        message(FATAL_ERROR "No SOURCES provided for '${OBJ_NAME}'")
    endif()

    # Define the object library
    add_library(${OBJ_NAME} OBJECT ${${PREFIX}_SOURCES})
    target_include_directories(${OBJ_NAME} PRIVATE ${KLEIDIAI_OPS_CONV_PRIVATE_INCLUDES})
    target_compile_definitions(${OBJ_NAME} PRIVATE ${KLEIDIAI_OPS_COMPILE_DEFINES})
    target_compile_options(${OBJ_NAME} PRIVATE ${KLEIDIAI_OPS_CCXX_FLAGS})

    if(${PREFIX}_ARCH)
        target_compile_options(${OBJ_NAME} PRIVATE ${${PREFIX}_ARCH})
    else()
        message(STATUS "Architecture flag unset for '${OBJ_NAME}'")
    endif()

    set_target_properties(
        ${OBJ_NAME}
        PROPERTIES C_STANDARD ${KLEIDIAI_OPS_C_STANDARD}
                   C_STANDARD_REQUIRED ON
                   C_EXTENSIONS OFF
                   CXX_STANDARD ${KLEIDIAI_OPS_CXX_STANDARD}
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   POSITION_INDEPENDENT_CODE ON)
endfunction()

# -------------------------------------------------------------------------------------------------

# Winograd sources; non-SVE, non-FP16
arm_add_obj_lib(
    conv_winograd_core
    SOURCES ${KLEIDIAI_OPS_CONV_WINOGRAD_CORE_SOURCES}
    ARCH ${KLEIDIAI_OPS_ARCH})

# Winograd sources; AArch64-only buckets
if(NOT KLEIDIAI_OPS_ENABLE_AARCH32)
    arm_add_obj_lib(
        conv_winograd_sve
        SOURCES ${KLEIDIAI_OPS_CONV_WINOGRAD_SVE_SOURCES}
        ARCH ${KLEIDIAI_OPS_SVE_ARCH})

    arm_add_obj_lib(
        conv_winograd_fp16
        SOURCES ${KLEIDIAI_OPS_CONV_WINOGRAD_FP16_SOURCES}
        ARCH ${KLEIDIAI_OPS_FP16_ARCH})
endif()

# Depthwise sources; only on non-AArch32 systems
if(NOT KLEIDIAI_OPS_ENABLE_AARCH32)
    # Depthwise sources; non-SVE, non-FP16
    arm_add_obj_lib(
        conv_depthwise_core
        SOURCES ${KLEIDIAI_OPS_CONV_DEPTHWISE_CORE_SOURCES}
        ARCH ${KLEIDIAI_OPS_ARCH})

    # Depthwise sources; combined SVE/SME/SME2 bucket
    arm_add_obj_lib(
        conv_depthwise_sve
        SOURCES ${KLEIDIAI_OPS_CONV_DEPTHWISE_SVE_SOURCES}
        ARCH ${KLEIDIAI_OPS_SVE_ARCH})

    # Depthwise sources; FP16
    arm_add_obj_lib(
        conv_depthwise_fp16
        SOURCES ${KLEIDIAI_OPS_CONV_DEPTHWISE_FP16_SOURCES}
        ARCH ${KLEIDIAI_OPS_FP16_ARCH})
endif()

# -------------------------------------------------------------------------------------------------

target_sources(
    kleidiai_ops
    PRIVATE $<TARGET_OBJECTS:conv_winograd_core>
            $<$<TARGET_EXISTS:conv_winograd_sve>:$<TARGET_OBJECTS:conv_winograd_sve>>
            $<$<TARGET_EXISTS:conv_winograd_fp16>:$<TARGET_OBJECTS:conv_winograd_fp16>>)

if(NOT KLEIDIAI_OPS_ENABLE_AARCH32)
    target_sources(
        kleidiai_ops
        PRIVATE $<TARGET_OBJECTS:conv_depthwise_core> $<TARGET_OBJECTS:conv_depthwise_sve>
                $<TARGET_OBJECTS:conv_depthwise_fp16>)
endif()

message(STATUS "Leaving conv directory")
# -------------------------------------------------------------------------------------------------

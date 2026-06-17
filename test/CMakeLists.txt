#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

include(FetchGTest)
enable_testing()
include(GoogleTest)

add_library(kleidiai_test_framework
    test/common/bfloat16.cpp
    test/common/bfloat16_asm.S
    test/common/buffer.cpp
    test/common/compare.cpp
    test/common/cpu_info.cpp
    test/common/data_format.cpp
    test/common/data_type.cpp
    test/common/float16.cpp
    test/common/float16_asm.S
    test/common/int2.cpp
    test/common/int4.cpp
    test/common/matmul_test_common.cpp
    test/common/matrix_portion.cpp
    test/common/memory.cpp
    test/common/printer.cpp
    test/common/rect.cpp
    test/common/round.cpp
    test/common/round_asm.S
    test/common/seed.cpp
    $<$<NOT:$<BOOL:${MSVC}>>:test/common/sme.cpp>
    test/common/sve.cpp

    test/reference/binary_elementwise.cpp
    test/reference/cast.cpp
    test/reference/clamp.cpp
    test/reference/dwconv.cpp
    test/reference/fill.cpp
    test/reference/matmul.cpp
    test/reference/matmul_pack.cpp
    test/reference/pack.cpp
    test/reference/pad.cpp
    test/reference/quantize.cpp
    test/reference/reduce.cpp
    test/reference/reorder.cpp
    test/reference/transpose.cpp

    test/nextgen/common/test_config.cpp
    test/nextgen/common/test_registry.cpp
    test/nextgen/reference/unary_elementwise.cpp
    test/nextgen/reference/binary_elementwise.cpp
    test/nextgen/reference/reduce.cpp
    test/nextgen/reference/quantize.cpp
    test/nextgen/reference/dequantize.cpp
    test/nextgen/reference/clamp.cpp
    test/nextgen/reference/matmul.cpp
    test/nextgen/reference/print.cpp
    test/nextgen/reference/pack.cpp
    test/nextgen/reference/compare.cpp
    test/nextgen/format/plain_format.cpp
    test/nextgen/format/block2d_row_format.cpp
    test/nextgen/quantization/asymm_linear_quantizer.cpp
    test/nextgen/quantization/symm_linear_quantizer.cpp
    test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_wrapper_registry.cpp
    test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_dq_wrapper.cpp
    test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_fp_wrapper.cpp
    test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_ukerapi_wrapper.cpp
    test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper_registry.cpp
    test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_quant_wrapper.cpp
    test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_fp_nt_wrapper.cpp
    test/nextgen/operators/matmul/matmul/matmul_wrapper_registry.cpp
    test/nextgen/operators/matmul/matmul/matmul_dq_wrapper.cpp
    test/nextgen/operators/matmul/matmul/matmul_fp_wrapper.cpp
    test/nextgen/operators/matmul/matmul/matmul_ukerapi_wrapper.cpp
    test/nextgen/operators/matmul/matmul_bias_mode.cpp
    test/nextgen/operators/matmul/matmul_operator.cpp
    test/nextgen/operators/matmul/matmul_tb.cpp
)

target_compile_options(kleidiai_test_framework
    PUBLIC ${KLEIDIAI_WARNING_FLAGS}
    PUBLIC $<$<NOT:$<BOOL:${MSVC}>>:-march=armv8-a${KLEIDIAI_INTERNAL_EXTRA_ARCH}>
    PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>
)

if(MSVC)
    set_source_files_properties(
        test/common/bfloat16_asm.S
        test/common/float16_asm.S
        test/common/round_asm.S
        PROPERTIES LANGUAGE ASM_MARMASM)
else()
    set_source_files_properties(test/common/float16_asm.S PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16${KLEIDIAI_INTERNAL_EXTRA_ARCH}")
    set_source_files_properties(test/common/bfloat16_asm.S PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16+bf16${KLEIDIAI_INTERNAL_EXTRA_ARCH}")
    set_source_files_properties(test/common/sme.cpp PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve${KLEIDIAI_INTERNAL_EXTRA_ARCH}")
endif()

target_link_libraries(kleidiai_test_framework
    PUBLIC  kleidiai
    PRIVATE GTest::gtest
)

if(MSVC)
    set(KLEIDIAI_FILES_TESTS
        test/tests/bfloat16_test.cpp
        test/tests/float16_test.cpp
        test/tests/matmul_clamp_f32_qai8dxp_qsu2cxp_test.cpp
        test/tests/matmul_clamp_f32_qsi8d32p_qai4c32p_test.cpp
        test/tests/matmul_clamp_bf16_qai8dxp_qsi4cxp_test.cpp
        test/tests/matmul_clamp_qai8dxp_qsi4c32p_test.cpp
    )
else()
    set(KLEIDIAI_FILES_TESTS
        test/tests/bfloat16_test.cpp
        test/tests/buffer_test.cpp
        test/tests/float16_test.cpp
        test/tests/imatmul_test.cpp
        test/tests/dwconv_test.cpp
        test/tests/matmul_clamp_bf16_qai8dxp_qsi4cxp_test.cpp
        test/tests/matmul_clamp_f16_bf16p_bf16p_test.cpp
        test/tests/matmul_clamp_f32_bf16p_bf16p_test.cpp
        test/tests/matmul_clamp_f32_qai8dxp_qsu2cxp_test.cpp
        test/tests/matmul_clamp_f32_f16p_qsi4c32p_test.cpp
        test/tests/matmul_clamp_f32_qai8dxp_qsi4cxp_test.cpp
        test/tests/matmul_clamp_f32_qai8dxp_qsi8cxp_test.cpp
        test/tests/matmul_clamp_f32_qsi8d32p_qsi4c32p_test.cpp
        test/tests/matmul_clamp_f32_qsi8d32p_qai4c32p_test.cpp
        test/tests/matmul_clamp_f16_qsi8d32p_qai4c32p_test.cpp
        test/tests/matmul_clamp_f16_qai8dxp_qsi4cxp_test.cpp
        test/tests/matmul_clamp_f16_qai8dxp_qsi8cxp_test.cpp
        test/tests/matmul_clamp_qai8_qai8p_qsi8cxp_test.cpp
        test/tests/matmul_clamp_qai8dxp_qsi4c32p_test.cpp
        test/tests/matmul_test.cpp

        test/nextgen/tests/matmul_test_next.cpp
        test/nextgen/tests/poly_test.cpp
    )

    set_source_files_properties(
        test/tests/imatmul_test.cpp
        test/tests/matmul_clamp_f16_bf16p_bf16p_test.cpp
        test/tests/matmul_clamp_f32_bf16p_bf16p_test.cpp
        test/tests/matmul_clamp_f32_qai8dxp_qsi4cxp_test.cpp
        test/tests/matmul_clamp_qai8_qai8p_qsi8cxp_test.cpp
        test/tests/matmul_test.cpp
        PROPERTIES COMPILE_FLAGS "-Wpedantic")
endif()

# Build individual test binaries one per source file
message(STATUS "Building individual test executables:")
foreach(KLEIDIAI_TEST_FILE IN LISTS KLEIDIAI_FILES_TESTS)
    get_filename_component(KLEIDIAI_TEST_NAME ${KLEIDIAI_TEST_FILE} NAME_WE)
    add_executable(${KLEIDIAI_TEST_NAME}
        ${KLEIDIAI_TEST_FILE}
        test/nextgen/main.cpp
    )
    target_link_libraries(${KLEIDIAI_TEST_NAME}
        PRIVATE kleidiai_test_framework
        PRIVATE GTest::gtest
    )
    gtest_discover_tests(${KLEIDIAI_TEST_NAME} DISCOVERY_MODE PRE_TEST)
    message(STATUS " ${KLEIDIAI_TEST_NAME}")
endforeach()

# Build monolithic test binary containing all tests
message(STATUS "Building monolithic test executable: kleidiai_test")
add_executable(kleidiai_test
    ${KLEIDIAI_FILES_TESTS}
    test/nextgen/main.cpp
)
target_link_libraries(kleidiai_test
    PRIVATE kleidiai_test_framework
    PRIVATE GTest::gtest
)

# Cross-compiling is a common use case which creates a conflict if DISCOVERY_MODE is set to POST_BUILD (by default)
# since the host platform does not match the target. Setting the mode to PRE_TEST avoids this conflict. This feature
# was added in CMake 3.18
if(CMAKE_CROSSCOMPILING)
    cmake_minimum_required(VERSION 3.18)
endif()
gtest_discover_tests(kleidiai_test DISCOVERY_MODE PRE_TEST)

#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(conv2d_imatmul_clamp_f16_f16_f16p_sme2 LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(KLEIDIAI_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)
include(FetchContent)

set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

fetchcontent_declare(
    kleidiai
    SOURCE_DIR ${KLEIDIAI_PATH}
)

fetchcontent_makeavailable(kleidiai)

add_executable(conv2d_imatmul_clamp_f16_f16_f16p_sme2
    conv2d_imatmul_clamp_f16_f16_f16p.cpp
)

target_link_libraries(conv2d_imatmul_clamp_f16_f16_f16p_sme2 PRIVATE kleidiai)

target_compile_options(conv2d_imatmul_clamp_f16_f16_f16p_sme2
    PRIVATE "-march=armv8.2-a+sve+sve2;-fno-tree-vectorize"
)

target_compile_definitions(conv2d_imatmul_clamp_f16_f16_f16p_sme2
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)

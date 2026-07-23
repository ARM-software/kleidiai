#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(matmul_clamp_f32_f8p_f8p_sme2
        LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(KAI_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)
include(FetchContent)

# Avoid pulling KleidiAI tests into this example build.
set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

# We treat KleidiAI as an external dependency here.
fetchcontent_declare(
    kleidiai
    SOURCE_DIR ${KAI_PATH}
)

# Make external project available.
fetchcontent_makeavailable(kleidiai)

add_executable(matmul_clamp_f32_f8p_f8p_sme2
    reference_fp8.cpp
    test_fp8_2d.cpp
)

target_link_libraries(matmul_clamp_f32_f8p_f8p_sme2 PRIVATE kleidiai)

# +sme-f8f32 is needed for the intrinsics
target_compile_options(matmul_clamp_f32_f8p_f8p_sme2
    PRIVATE
        "-Werror"
        "-march=armv9.2-a+sve+sve2+sme2+fp8+sme-f8f32"
        "-g"
        "-O3"
)

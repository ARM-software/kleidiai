#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(matmul_clamp_qai8_qai8p_qsi8cxp_sme2p1
        LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
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

add_executable(matmul_clamp_qai8_qai8p_qsi8cxp_sme2p1
    kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_2vsx32vs_sme2p1_mop4a_asm.S
    matmul_clamp_qai8_qai8p_qsi8cxp_sme2p1.cpp
    matmul_clamp_qai8_qai8p_qsi8cxp_sme2p1_benchmark.cpp
)

set_source_files_properties(
    matmul_clamp_qai8_qai8p_qsi8cxp_sme2p1.cpp
    matmul_clamp_qai8_qai8p_qsi8cxp_sme2p1_benchmark.cpp
    PROPERTIES LANGUAGE CXX
)

target_link_libraries(matmul_clamp_qai8_qai8p_qsi8cxp_sme2p1 PRIVATE kleidiai)

target_compile_options(matmul_clamp_qai8_qai8p_qsi8cxp_sme2p1
    PRIVATE
        "-Werror"
        "-march=armv9.4-a+sme2p1+sme-mop4"
        "-g"
        "-O3"
)

#
# SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(matmul_clamp_f32_qsi8d32p_qsi4c32p LANGUAGES C CXX ASM)

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

add_executable(matmul_clamp_f32_qsi8d32p_qsi4c32p
    matmul_clamp_f32_qsi8d32p_qsi4c32p.cpp
)

target_link_libraries(matmul_clamp_f32_qsi8d32p_qsi4c32p PRIVATE kleidiai)

target_compile_options(matmul_clamp_f32_qsi8d32p_qsi4c32p
    PRIVATE -march=armv8.2-a+dotprod+i8mm
)
